# Enhanced scalar matrix parallel lstm with finance-specific improvements

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.checkpoint import checkpoint

def asymmetric_directional_loss(y_pred, y_true, up_penalty=2.0, down_penalty=1.0):
    """
    Penalizes direction errors with asymmetric weighting.
    """
    # Handle case when either tensor has single item
    if y_pred.shape[1] <= 1 or y_true.shape[1] <= 1:
        return torch.tensor(0.0, device=y_pred.device)
        
    diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
    diff_true = y_true[:, 1:] - y_true[:, :-1]
    
    direction_match = (torch.sign(diff_pred) == torch.sign(diff_true))
    up_moves = diff_true > 0
    down_moves = diff_true < 0
    
    penalties = torch.ones_like(diff_true)
    penalties = torch.where(up_moves, penalties * up_penalty, penalties)
    penalties = torch.where(down_moves, penalties * down_penalty, penalties)
    
    weighted_errors = (~direction_match).float() * penalties
    return weighted_errors.mean()

class StructuredStateSpace(nn.Module):
    def __init__(self, d_state, d_model, discretization='bilinear', dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.discretization = discretization
        
        # Reduced parameter count
        self.log_lambda_real = nn.Parameter(torch.linspace(math.log(dt_min), math.log(dt_max), d_state))
        self.log_b = nn.Parameter(torch.randn(d_state, 1).uniform_(-0.5, 0.5))
        self.c = nn.Parameter(torch.randn(1, d_state) / math.sqrt(d_state))
        self.log_d = nn.Parameter(torch.zeros(1))
        self.log_step = nn.Parameter(torch.tensor(math.log(dt_min)))
        
        self.volatility_gate = nn.Parameter(torch.ones(d_state))
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.layer_norm = nn.LayerNorm(d_state)

    def calculate_trend_strength(self, x):
        batch_size, seq_len, _ = x.shape
        if seq_len <= 5:
            return torch.ones(batch_size, seq_len, 1, device=x.device) * 0.5
        
        # More memory-efficient calculation
        fast_ema = torch.zeros_like(x[:, :, :1])  # Just use first feature dimension
        slow_ema = torch.zeros_like(fast_ema)
        
        fast_ema[:, 0] = x[:, 0, :1]
        slow_ema[:, 0] = x[:, 0, :1]
        
        fast_alpha = 0.3
        slow_alpha = 0.05
        
        for t in range(1, seq_len):
            fast_ema[:, t] = fast_alpha * x[:, t, :1] + (1 - fast_alpha) * fast_ema[:, t-1]
            slow_ema[:, t] = slow_alpha * x[:, t, :1] + (1 - slow_alpha) * slow_ema[:, t-1]
        
        # Compute agreement more efficiently
        diff_sign = torch.sign(fast_ema[:, 1:] - fast_ema[:, :-1]) * torch.sign(slow_ema[:, 1:] - slow_ema[:, :-1])
        agreement = (diff_sign > 0).float()
        
        # Pad to match original sequence length
        padded_agreement = F.pad(agreement, (0, 0, 0, 1), "replicate")
        
        trend_strength = torch.sigmoid(5 * (padded_agreement - 0.5))
        return trend_strength.unsqueeze(-1)

    def forward(self, x, volatility_scale=None, trend_strength=None):
        batch, seq_len, _ = x.shape
        lambda_real = -torch.exp(self.log_lambda_real)
        b = torch.exp(self.log_b)
        c = self.c
        d = torch.exp(self.log_d)
        step = torch.exp(self.log_step)
        
        if self.discretization == 'zoh':
            a_discrete = torch.exp(lambda_real * step)
            b_discrete = (a_discrete - 1.0) / lambda_real * b.squeeze(-1)
        else:
            a_discrete = (2.0 + step * lambda_real) / (2.0 - step * lambda_real)
            b_discrete = step * (torch.ones_like(a_discrete) + a_discrete) * b.squeeze(-1) / 2.0
        
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        # Process in chunks for memory efficiency
        chunk_size = 16  # Process sequence in chunks
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            
            chunk_outputs = []
            for t in range(chunk_len):
                idx = chunk_start + t
                new_h = a_discrete.unsqueeze(0) * h + b_discrete.unsqueeze(0) * x[:, idx, :].mean(dim=1, keepdim=True)
                
                if trend_strength is not None:
                    h = new_h * (1 - trend_strength[:, idx]) + h * trend_strength[:, idx]
                else:
                    h = new_h
                
                h = self.layer_norm(h)
                
                if volatility_scale is not None:
                    vol_influence = torch.sigmoid(self.volatility_gate) * volatility_scale[:, idx]
                    h = h * (1.0 / (1.0 + vol_influence))
                
                y = torch.matmul(h.unsqueeze(1), c.transpose(0, 1)).squeeze(1) + d * x[:, idx, :]
                chunk_outputs.append(y)
            
            outputs.extend(chunk_outputs)
            
            # Free memory
            del chunk_outputs
            torch.cuda.empty_cache()
        
        y = torch.stack(outputs, dim=1)
        alpha = torch.sigmoid(self.alpha)
        
        return alpha * x + (1 - alpha) * y

class ParallelSMLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, matrix_size=4, stabilizer_threshold=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.matrix_size = matrix_size
        self.d = hidden_size
        self.stabilizer_threshold = stabilizer_threshold
        
        self.C_init = nn.Parameter(torch.zeros(1, hidden_size, hidden_size))
        self.n_init = nn.Parameter(torch.zeros(1, hidden_size))
        self.m_init = nn.Parameter(torch.zeros(1, 1))
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_o = nn.Linear(input_size, hidden_size)
        
        self.ssm = StructuredStateSpace(
            d_state=hidden_size,
            d_model=hidden_size,
            discretization='bilinear',
            dt_min=0.001,
            dt_max=0.1
        )
        
        self.regime_detector = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm_input = nn.LayerNorm(input_size)
        self.layer_norm_memory = nn.LayerNorm(hidden_size)
        
        self.W_f.bias.data.fill_(1.0)
        self.W_i.bias.data.fill_(0.0)

        self.regime_processors = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()),
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()),
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()),
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        ])

    def detect_volatility(self, x):
        batch_size, seq_len, _ = x.shape
        window_size = max(1, min(int(seq_len * 0.2), 10))  # Limit window size
        if seq_len <= 1:
            return torch.ones(batch_size, seq_len, 1, device=x.device)
            
        # More efficient volatility calculation
        mean_value = x.mean(dim=2, keepdim=True)  # Use mean instead of std
        volatility = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        for i in range(window_size, seq_len+1):
            window = mean_value[:, i-window_size:i]
            volatility[:, i-1] = torch.std(window, dim=1)
            
        # Fill beginning
        if window_size > 1:
            volatility[:, :window_size-1] = volatility[:, window_size-1:window_size]
        
        # Normalize
        volatility = volatility / (torch.mean(volatility, dim=1, keepdim=True) + 1e-6)
        
        return volatility
    
    def stabilize_gates(self, i_tilde, f_tilde, m_prev):
        batch_size, seq_len = i_tilde.shape[0], i_tilde.shape[1]
        
        if m_prev.dim() == 1:
            m_prev = m_prev.unsqueeze(1)
        
        m_prev_exp = m_prev.unsqueeze(1).expand(batch_size, seq_len, 1).clamp(-100, 100)
        m_t = torch.maximum(f_tilde, m_prev_exp + f_tilde)
        
        i = torch.exp(torch.clamp(i_tilde - m_t, -15.0, 15.0))
        f = torch.exp(torch.clamp(f_tilde + m_prev_exp - m_t, -15.0, 15.0))
        
        return i, f, m_t

    def store_key_value(self, key, value, i):
        batch_size, seq_len, d = key.shape
        
        # Scale keys and values
        key_norm = F.normalize(key, dim=2) * math.sqrt(self.d)
        value_norm = value / (torch.norm(value, dim=2, keepdim=True) + 1e-6) * math.sqrt(self.d)
        
        # Ensure same sequence length
        min_seq = min(key_norm.shape[1], value_norm.shape[1], i.shape[1])
        key_norm = key_norm[:, :min_seq]
        value_norm = value_norm[:, :min_seq]
        i = i[:, :min_seq]
        
        # Smaller chunk size for less memory usage
        chunk_size = 8  # Reduced from 32 for memory efficiency
        num_chunks = (min_seq + chunk_size - 1) // chunk_size
        
        result_chunks = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, min_seq)
            
            chunk_key = key_norm[:, start_idx:end_idx]
            chunk_value = value_norm[:, start_idx:end_idx]
            chunk_i = i[:, start_idx:end_idx]
            
            # Special handling for small batch sizes
            if batch_size * (end_idx - start_idx) == 0:
                continue
                
            # Compute outer product for this chunk
            outer_product = torch.bmm(
                chunk_value.reshape(batch_size * (end_idx - start_idx), d, 1),
                chunk_key.reshape(batch_size * (end_idx - start_idx), 1, d)
            ).reshape(batch_size, end_idx - start_idx, d, d)
            
            # Apply input gate
            i_clamped = torch.clamp(chunk_i, 0.0, 1.0)
            result_chunks.append(i_clamped.unsqueeze(-1) * outer_product)
            
            # Free memory
            del outer_product
            torch.cuda.empty_cache()
            
        if len(result_chunks) == 0:
            # Handle edge case: create empty tensor with correct shape
            return torch.zeros(batch_size, min_seq, d, d, device=key.device)
            
        return torch.cat(result_chunks, dim=1)

    def parallel_gate_computation(self, x_norm):
        batch_size, seq_len, _ = x_norm.shape
        
        # Compute gates in chunks for memory efficiency
        q = self.W_q(x_norm)
        k = self.W_k(x_norm) / math.sqrt(self.d)
        v = self.W_v(x_norm)
        i_tilde = self.W_i(x_norm).view(batch_size, seq_len, 1)
        f_tilde = self.W_f(x_norm).view(batch_size, seq_len, 1)
        o = torch.sigmoid(self.W_o(x_norm))
        
        # Detect regime in chunks to save memory
        chunk_size = 16
        regime_weights_list = []
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_regime = self.regime_detector(x_norm[:, i:end_idx])
            regime_weights_list.append(chunk_regime)
            
        regime_weights = torch.cat(regime_weights_list, dim=1)
        
        return q, k, v, i_tilde, f_tilde, o, regime_weights

    def apply_structured_memory_mixing(self, C_t, regime_weights, volatility_scale):
        # Ensure C_t is 4D [batch, seq, d, d]
        if len(C_t.shape) != 4:
            shape = C_t.shape
            batch_size = shape[0]
            
            if len(shape) == 5:  # [batch, seq, d, d, 1]
                C_t = C_t.squeeze(-1)
            elif len(shape) == 3:  # [batch, d, d]
                seq_len = regime_weights.shape[1]
                C_t = C_t.unsqueeze(1).expand(-1, seq_len, -1, -1)
            else:
                raise ValueError(f"Cannot handle C_t shape {shape}")
        
        batch_size, seq_len, d, _ = C_t.shape
        
        # Use a simplified approach for large matrices
        if d > 128:
            return C_t  # Skip processing for large dimensions
            
        # Process memory matrix in smaller chunks to save memory
        chunk_size = 4  # Process very small chunks for memory efficiency
        num_chunks = (d + chunk_size - 1) // chunk_size
        
        C_mixed = torch.zeros_like(C_t)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, d)
            
            # Process just this chunk of rows
            for i_row in range(start_idx, end_idx):
                try:
                    C_row = C_t[:, :, i_row, :]  # [batch, seq, d]
                    # Process row with SSM
                    C_row_mixed = self.ssm(C_row, volatility_scale)
                    # Store result
                    C_mixed[:, :, i_row, :] = C_row_mixed
                except Exception as e:
                    # On error, just copy the original row
                    print(f"Error processing row {i_row}: {e}")
                    C_mixed[:, :, i_row, :] = C_t[:, :, i_row, :]
                    
                # Free memory
                torch.cuda.empty_cache()
        
        # Final memory mixing
        mixing_factor = regime_weights[:, :, 0:1].unsqueeze(-1)
        result = C_mixed * mixing_factor + C_t * (1 - mixing_factor)
        
        return result

    def calculate_trend_strength(self, x):
        batch_size, seq_len, dim = x.shape
        if seq_len <= 5:
            return torch.zeros(batch_size, 1, 1, 1, device=x.device)
        
        # More efficient trend strength calculation
        # Just use first feature dimension to save memory
        signal = x.mean(dim=2, keepdim=True)
        signal_shifted = torch.cat([signal[:, :1], signal[:, :-1]], dim=1)
        
        # Calculate on device to avoid transfers
        signal_mean = signal.mean(dim=1, keepdim=True)
        centered_signal = signal - signal_mean
        centered_shifted = signal_shifted - signal_mean
        
        numerator = (centered_signal * centered_shifted).sum(dim=1, keepdim=True)
        denominator = torch.sqrt((centered_signal**2).sum(dim=1, keepdim=True) * 
                                (centered_shifted**2).sum(dim=1, keepdim=True) + 1e-8)
        
        autocorr = numerator / denominator
        trend_strength = (autocorr.abs() + 1) / 2
        
        return trend_strength.unsqueeze(-1).unsqueeze(-1)

    def process_memory_by_regime(self, C_t, regime_weights):
        # First, normalize C_t to ensure it's 4D [batch, seq, d, d]
        shape = list(C_t.shape)
        if len(shape) != 4:
            # Handle different input dimensions
            if len(shape) == 5:  # [batch, seq, d, d, 1]
                C_t = C_t.squeeze(-1)
                shape = list(C_t.shape)
            elif len(shape) == 3:  # [batch, d, d]
                C_t = C_t.unsqueeze(1)  # Add sequence dimension
                shape = list(C_t.shape)
            else:
                print(f"Warning: Cannot process tensor with shape {shape}")
                # Return input unchanged if we can't fix it
                return C_t
        
        # Now we know it's 4D
        batch_size, seq_len, d, _ = shape
        
        # Perform a simplified processing if dimensions are too large
        if d > 128 or batch_size * seq_len > 2048:
            # Just apply weighted average
            result = C_t.clone()
            for i_regime in range(4):  # Assuming 4 regimes
                regime_weight = regime_weights[:, :, i_regime:i_regime+1].unsqueeze(-1)
                result = result * (1 - regime_weight * 0.1)  # Small adjustment
            return result
        
        processed_memories = []
        for i_regime, processor in enumerate(self.regime_processors):
            regime_weight = regime_weights[:, :, i_regime:i_regime+1].unsqueeze(-1)
            processed_memory = C_t.clone()
            
            # Handle a lower level of batching to save memory
            # Process row by row to avoid OOM
            for row in range(d):
                row_data = C_t[:, :, row, :]  # [batch, seq, d]
                # Reshape to 2D for processing
                flat_row_data = row_data.reshape(batch_size * seq_len, d)
                
                # Process in very small sub-batches
                sub_batch_size = 16
                for b_start in range(0, batch_size * seq_len, sub_batch_size):
                    b_end = min(b_start + sub_batch_size, batch_size * seq_len)
                    
                    if b_start >= b_end:
                        continue
                        
                    # Extract small batch
                    small_batch = flat_row_data[b_start:b_end]
                    # Process and save back
                    small_result = processor(small_batch)
                    # Update the row
                    flat_processed = processed_memory.reshape(batch_size * seq_len, d, d)
                    flat_processed[b_start:b_end, row] = small_result
            
            # Apply regime weighting
            processed_memories.append(processed_memory * regime_weight)
            
            # Free memory
            del processed_memory
            torch.cuda.empty_cache()
        
        # Sum all processed memories
        final_memory = sum(processed_memories)
        
        return final_memory

    def parallel_memory_update(self, v, k, i, f, C_prev, seq_len, regime_weights, volatility_scale):
        batch_size = v.shape[0]
        d = self.d
        
        # Shape normalization for f - ensure it's 3D [batch, seq, 1]
        if f.dim() > 3:
            f = f.squeeze(-1)
        
        # Ensure all tensors have consistent sequence dimensions
        min_seq_len = min(v.shape[1], k.shape[1], i.shape[1], f.shape[1])
        v = v[:, :min_seq_len]
        k = k[:, :min_seq_len]
        i = i[:, :min_seq_len]
        f = f[:, :min_seq_len]
        regime_weights = regime_weights[:, :min_seq_len]
        volatility_scale = volatility_scale[:, :min_seq_len]
        seq_len = min_seq_len
        
        # Compute cumprod of forget gate with strict dimension control
        f_cum = torch.cumprod(f, dim=1)  # [batch, seq, 1]
        # Add dimensions needed for later computation but avoid excess dimensions
        f_cum = f_cum.view(batch_size, min_seq_len, 1, 1)  # Explicitly make it [batch, seq, 1, 1]
        
        # Generate memory updates
        update_matrices = self.store_key_value(k, v, i)
        
        if update_matrices.shape[1] != min_seq_len:
            update_matrices = update_matrices[:, :min_seq_len]
        
        # Compute cumulative updates
        C_updates = torch.cumsum(update_matrices, dim=1)
        
        # Shape C_prev correctly for memory updates
        if C_prev.dim() != 3:  # Should be [batch, d, d]
            print(f"Warning: Reshaping C_prev from {C_prev.shape}")
            C_prev = C_prev.view(batch_size, d, d)  # Force to [batch, d, d]
        
        # Expand C_prev along sequence dimension
        C_prev_expanded = C_prev.unsqueeze(1).expand(-1, min_seq_len, -1, -1)  # [batch, seq, d, d]
        
        # Combine previous memory (forget gate) with updates
        C_t = f_cum * C_prev_expanded + C_updates  # [batch, seq, d, d]
        
        # Calculate trend influence
        trend_strength = self.calculate_trend_strength(v)
        
        # Ensure trend_strength is compatible with C_t
        if trend_strength.dim() > 4:
            trend_strength = trend_strength.squeeze(-1)
        
        # Initialize trend memory (previous timestep's memory)
        trend_memory = torch.zeros_like(C_t)
        
        # Implement trend memory (persistence)
        if seq_len > 1:
            # Shift memory by 1 timestep
            trend_memory[:, 1:] = C_t[:, :-1]
            trend_memory[:, 0] = C_prev
        else:
            trend_memory = C_prev.unsqueeze(1)
        
        # Apply trend-based memory mixing
        C_t = C_t * (1 - trend_strength) + trend_memory * trend_strength
        
        # Make sure C_t is exactly 4D
        if C_t.dim() != 4:
            print(f"Warning: Reshaping C_t from {C_t.shape} to 4D")
            C_t = C_t.view(batch_size, min_seq_len, d, d)
        
        # Skip memory processing if there are issues
        try:
            C_t = self.process_memory_by_regime(C_t, regime_weights)
        except Exception as e:
            print(f"Error in process_memory_by_regime: {e}, using unprocessed memory")
        
        try:
            C_t = self.apply_structured_memory_mixing(C_t, regime_weights, volatility_scale)
        except Exception as e:
            print(f"Error in apply_structured_memory_mixing: {e}, using unmixed memory")
        
        # Final normalization to maintain stable dynamic range
        norm = torch.norm(C_t, dim=(-2, -1), keepdim=True) + 1e-6
        C_t = C_t / norm * math.sqrt(self.d)
        
        return C_t

    def parallel_forward(self, x, state):
        batch_size, seq_len, _ = x.size()
        C_prev, n_prev, m_prev = state
        
        volatility_scale = self.detect_volatility(x)
        x_norm = self.layer_norm_input(x)
        
        q, k, v, i_tilde, f_tilde, o, regime_weights = self.parallel_gate_computation(x_norm)
        i, f, m_t = self.stabilize_gates(i_tilde, f_tilde, m_prev)
        
        i = i.view(batch_size, seq_len, 1)
        f = f.view(batch_size, seq_len, 1)
        
        C_t = self.parallel_memory_update(v, k, i, f, C_prev, seq_len, regime_weights, volatility_scale)
        
        # Make sure C_t is exactly 4D [batch, seq, d, d]
        if len(C_t.shape) != 4:
            # Handle 5D case (extra dimension)
            if len(C_t.shape) == 5:
                C_t = C_t.squeeze(-1)
            elif len(C_t.shape) == 3:
                C_t = C_t.unsqueeze(1)  # Add sequence dimension
            else:
                print(f"Warning: Unexpected C_t shape {C_t.shape}, attempting to reshape")
                C_t = C_t.view(batch_size, seq_len, self.d, self.d)
        
        actual_seq_len = C_t.shape[1]
        n_prev_expanded = n_prev.unsqueeze(1).expand(-1, actual_seq_len, -1)
        
        q_actual = q[:, :actual_seq_len]
        k_actual = k[:, :actual_seq_len]
        i_actual = i[:, :actual_seq_len]
        f_actual = f[:, :actual_seq_len]
        o_actual = o[:, :actual_seq_len]
        
        i_k_product = i_actual * k_actual
        f_cum = torch.cumprod(f_actual, dim=1)
        
        n_t = torch.cumsum(i_k_product, dim=1) + f_cum * n_prev_expanded
        
        # Safe einsum with explicit dimension check
        try:
            h_tilde = torch.einsum('bsde,bse->bsd', C_t, q_actual)
        except RuntimeError as e:
            print(f"Einsum error with shapes C_t: {C_t.shape}, q_actual: {q_actual.shape}")
            # Fallback implementation
            h_tilde = torch.zeros(batch_size, actual_seq_len, self.d, device=x.device)
            for b in range(batch_size):
                for s in range(actual_seq_len):
                    h_tilde[b, s] = torch.matmul(C_t[b, s], q_actual[b, s])
        
        q_n_dot = torch.sum(n_t * q_actual, dim=-1)
        
        denominator = torch.maximum(torch.abs(q_n_dot), torch.tensor(self.stabilizer_threshold, device=x.device)).unsqueeze(-1)
        h_tilde = h_tilde / denominator
        
        h_t = o_actual * h_tilde
        
        # Ensure final C_t has the right shape for the state
        final_C_t = C_t[:, -1]  # [batch, d, d]
        
        final_state = (final_C_t, n_t[:, -1], m_t[:, -1])
        
        return h_t, final_state

class ParallelSMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, matrix_size=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.matrix_size = matrix_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.cells = nn.ModuleList([
            ParallelSMLSTMCell(
                input_size if i == 0 else hidden_size, 
                hidden_size,
                matrix_size=matrix_size
            )
            for i in range(num_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        self.trend_detector = nn.Linear(hidden_size, 3)

    def init_state(self, batch_size, device):
        states = []
        for cell in self.cells:
            C = torch.randn(batch_size, cell.d, cell.d, device=device) * 0.01
            n = torch.zeros(batch_size, cell.d, device=device)
            m = torch.zeros(batch_size, 1, device=device)
            states.append((C, n, m))
        return states
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        states = self.init_state(batch_size, device)
        layer_input = x
        
        for layer_idx, cell in enumerate(self.cells):
            if seq_len > 100 and self.training:
                def run_cell(inp, st):
                    return cell.parallel_forward(inp, st)
                layer_output, states[layer_idx] = checkpoint(run_cell, layer_input, states[layer_idx])
            else:
                layer_output, states[layer_idx] = cell.parallel_forward(layer_input, states[layer_idx])
            
            if layer_idx > 0 and layer_input.shape[-1] == layer_output.shape[-1]:
                layer_output = layer_output + layer_input
            
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            layer_input = layer_output
        
        trend_logits = self.trend_detector(layer_output.mean(dim=1))
        
        return layer_output, trend_logits

class ParallelMLSTMBlock(nn.Module):
    def __init__(self, input_dim, expanded_dim=None, num_layers=2, dropout=0.2, 
                 expansion_factor=1, lstm_class=ParallelSMLSTM, matrix_size=4):
        super().__init__()
        expanded_dim = expanded_dim or input_dim
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.up_proj = nn.Sequential(nn.Linear(input_dim, expanded_dim), nn.GELU())
        self.mlstm = lstm_class(expanded_dim, expanded_dim // 2, matrix_size, num_layers, dropout)
        self.down_proj = nn.Linear(expanded_dim // 2, input_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_up = self.up_proj(x_norm)
        x_mlstm, trend_logits = self.mlstm(x_up)
        x_down = self.down_proj(x_mlstm)
        
        if x.shape != x_down.shape:
            x_down = x_down[:, :x.shape[1], :x.shape[2]]
        
        skip_strength = torch.sigmoid(self.skip_scale)
        
        return x + skip_strength * x_down, trend_logits

class ParallelExtendedSMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16, matrix_size=4, num_layers=2, dropout=0.2, expansion_factor=4):
        super().__init__()
        self.pre_fc_dim = hidden_size * matrix_size * matrix_size
        self.input_projection = nn.Linear(input_size, self.pre_fc_dim)
        self.layer_norm = nn.LayerNorm(self.pre_fc_dim)
        self.mlstm_block = ParallelMLSTMBlock(
            input_dim=self.pre_fc_dim,
            expanded_dim=self.pre_fc_dim * expansion_factor,
            num_layers=num_layers,
            dropout=dropout,
            expansion_factor=expansion_factor,
            lstm_class=ParallelSMLSTM,
            matrix_size=matrix_size
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.pre_fc_dim, self.pre_fc_dim)
        self.output_projection.weight.data *= 0.1
        self.regime_detector = nn.Linear(self.pre_fc_dim, 4)
        self.signal_generator = nn.Linear(self.pre_fc_dim, 3)
        self.price_head = nn.Linear(self.pre_fc_dim, 5)
        self.direction_head = nn.Linear(self.pre_fc_dim, 5)
        self.volatility_head = nn.Linear(self.pre_fc_dim, 5)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.layer_norm(x)
        x, trend_logits = self.mlstm_block(x)
        x = F.gelu(x)
        x = self.dropout(x)
        outputs = self.output_projection(x)
        regime = F.softmax(self.regime_detector(outputs.mean(dim=1)), dim=-1)
        signals = F.softmax(self.signal_generator(outputs.mean(dim=1)), dim=-1)
        
        # Change: Return logits directly for direction, apply sigmoid in loss function
        results = {
            'price': self.price_head(outputs.mean(dim=1)),
            'direction_logits': self.direction_head(outputs.mean(dim=1)),  # Raw logits
            'direction': torch.sigmoid(self.direction_head(outputs.mean(dim=1))),  # For compatibility
            'volatility': F.softplus(self.volatility_head(outputs.mean(dim=1)))
        }
        return results, regime, signals

class DirectionalEnsembleModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[16, 32], matrix_sizes=[3, 4], sequence_lengths=[48, 64, 96], dropout=0.2):
        super().__init__()
        self.sequence_lengths = sequence_lengths
        self.models = nn.ModuleList()
        for hidden_s in hidden_sizes:
            for mat_s in matrix_sizes:
                model = ParallelExtendedSMLSTM(
                    input_size=input_size,
                    hidden_size=hidden_s,
                    matrix_size=mat_s,
                    num_layers=2,
                    dropout=dropout,
                    expansion_factor=2
                )
                self.models.append(model)
        num_models = len(self.models)
        self.price_meta = nn.Linear(num_models * 5, 5)
        self.direction_meta = nn.Linear(num_models * 5, 5)
        self.volatility_meta = nn.Linear(num_models * 5, 5)
        self.regime_meta = nn.Linear(num_models * 4, 4)
        self.signal_meta = nn.Linear(num_models * 3, 3)

    def forward(self, x_dict):
        all_price_preds = []
        all_direction_logits = []
        all_volatility_preds = []
        all_regime_preds = []
        all_signal_preds = []
        for i, model in enumerate(self.models):
            seq_idx = i % len(self.sequence_lengths)
            seq_len = self.sequence_lengths[seq_idx]
            x = x_dict[f'seq_{seq_len}']
            out, reg, sig = model(x)
            all_price_preds.append(out['price'])
            all_direction_logits.append(out['direction'])
            all_volatility_preds.append(out['volatility'])
            all_regime_preds.append(reg)
            all_signal_preds.append(sig)
        all_prices = torch.cat(all_price_preds, dim=1)
        all_directions = torch.cat(all_direction_logits, dim=1)
        all_volatilities = torch.cat(all_volatility_preds, dim=1)
        all_regimes = torch.cat(all_regime_preds, dim=1)
        all_signals = torch.cat(all_signal_preds, dim=1)
        final_price = self.price_meta(all_prices)
        final_direction_logits = self.direction_meta(all_directions)
        final_direction = torch.sigmoid(final_direction_logits)
        final_volatility = F.softplus(self.volatility_meta(all_volatilities))
        final_regime = F.softmax(self.regime_meta(all_regimes), dim=-1)
        final_signals = F.softmax(self.signal_meta(all_signals), dim=-1)
        return {
            'price': final_price,
            'direction': final_direction,
            'direction_logits': final_direction_logits,
            'volatility': final_volatility,
            'regime': final_regime,
            'signals': final_signals
        }

def train_with_accumulation(model, loader, optimizer, criterion, accumulation_steps=2):
    model.train()
    optimizer.zero_grad()
    for i, (X_batch, y_batch) in enumerate(loader):
        with torch.amp.autocast('cuda'):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    return model