import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ScalarMatrixLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ScalarMatrixLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate weights and biases
        self.weight_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forget gate weights and biases
        self.weight_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # Cell state weights and biases
        self.weight_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output gate weights and biases
        self.weight_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_o = nn.Parameter(torch.Tensor(hidden_size))
        
        # Scalar multipliers
        self.scalar_ii = nn.Parameter(torch.Tensor(1))
        self.scalar_hi = nn.Parameter(torch.Tensor(1))
        self.scalar_if = nn.Parameter(torch.Tensor(1))
        self.scalar_hf = nn.Parameter(torch.Tensor(1))
        self.scalar_ig = nn.Parameter(torch.Tensor(1))
        self.scalar_hg = nn.Parameter(torch.Tensor(1))
        self.scalar_io = nn.Parameter(torch.Tensor(1))
        self.scalar_ho = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # Forget gate bias initialized to 1 to encourage remembering
        self.bias_f.data.fill_(1.0)
        # Initialize scalars to 1
        self.scalar_ii.data.fill_(1.0)
        self.scalar_hi.data.fill_(1.0)
        self.scalar_if.data.fill_(1.0)
        self.scalar_hf.data.fill_(1.0)
        self.scalar_ig.data.fill_(1.0)
        self.scalar_hg.data.fill_(1.0)
        self.scalar_io.data.fill_(1.0)
        self.scalar_ho.data.fill_(1.0)
    
    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            cx = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        else:
            hx, cx = hx
        
        # Input gate
        i = F.sigmoid(
            self.scalar_ii * F.linear(x, self.weight_ii, self.bias_i) + 
            self.scalar_hi * F.linear(hx, self.weight_hi)
        )
        
        # Forget gate
        f = F.sigmoid(
            self.scalar_if * F.linear(x, self.weight_if, self.bias_f) + 
            self.scalar_hf * F.linear(hx, self.weight_hf)
        )
        
        # Cell state
        g = F.tanh(
            self.scalar_ig * F.linear(x, self.weight_ig, self.bias_g) + 
            self.scalar_hg * F.linear(hx, self.weight_hg)
        )
        
        # Output gate
        o = F.sigmoid(
            self.scalar_io * F.linear(x, self.weight_io, self.bias_o) + 
            self.scalar_ho * F.linear(hx, self.weight_ho)
        )
        
        # New cell state and hidden state
        cy = f * cx + i * g
        hy = o * F.tanh(cy)
        
        return hy, cy

class ScalarMatrixLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(ScalarMatrixLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_cells.append(ScalarMatrixLSTMCell(input_dim, hidden_size))
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, hx=None):
        """
        Input shape: (batch_size, seq_len, input_size)
        Output shape: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        if hx is None:
            h_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h_states, c_states = hx
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                if layer > 0 and self.dropout_layer is not None:
                    x_t = self.dropout_layer(x_t)
                
                h_states[layer], c_states[layer] = self.lstm_cells[layer](x_t, (h_states[layer], c_states[layer]))
                x_t = h_states[layer]
            
            outputs.append(h_states[-1])
        
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        return outputs, (h_states, c_states)

class FinancialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(FinancialLSTM, self).__init__()
        
        self.lstm = ScalarMatrixLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Move input to device
        x = x.to(device)
        
        # Run through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply output layer to each time step
        output = self.output_layer(lstm_out)
        
        return output
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
    def to_device(self):
        self.to(device)
        return self

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, forecast_horizon=5, 
                 num_layers=1, dropout=0.0):
        super(MultiStepLSTM, self).__init__()
        
        self.forecast_horizon = forecast_horizon
        
        self.lstm = ScalarMatrixLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Move input to device
        x = x.to(device)
        
        batch_size, seq_len, _ = x.size()
        
        # Run through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get immediate forecast (t+1)
        outputs = [self.output_layer(lstm_out[:, -1, :])]
        
        # For multi-step forecasting, use previous prediction as input
        last_hidden = h_n
        last_cell = c_n
        last_output = outputs[0]
        
        # Create multi-step forecasts
        for i in range(1, self.forecast_horizon):
            # Prepare input for next step prediction
            next_input = last_output.unsqueeze(1)  # Shape: (batch_size, 1, output_size)
            
            # Feed through LSTM again
            lstm_out, (last_hidden, last_cell) = self.lstm(next_input, (last_hidden, last_cell))
            
            # Get next prediction
            next_output = self.output_layer(lstm_out[:, -1, :])
            outputs.append(next_output)
            
            # Update last output for next iteration
            last_output = next_output
        
        # Stack all outputs
        outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, forecast_horizon, output_size)
        
        return outputs
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
            
    def to_device(self):
        self.to(device)
        return self

# Custom loss functions
def directional_loss(y_pred, y_true, up_weight=1.0, down_weight=1.0):
    """
    Computes directional loss, penalizing incorrect predictions of movement direction.
    
    Args:
        y_pred: Predicted values tensor
        y_true: Target values tensor
        up_weight: Weight for upward movements (default: 1.0)
        down_weight: Weight for downward movements (default: 1.0)
    
    Returns:
        Directional loss tensor
    """
    # Calculate directions
    pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    true_diff = y_true[:, 1:] - y_true[:, :-1]
    
    # Get signs of directions
    pred_sign = torch.sign(pred_diff)
    true_sign = torch.sign(true_diff)
    
    # Create masks for up and down movements
    up_mask = (true_sign > 0).float()
    down_mask = (true_sign < 0).float()
    
    # Calculate directional accuracy
    correct_dir = (pred_sign == true_sign).float()
    incorrect_dir = 1.0 - correct_dir
    
    # Apply weights
    weighted_incorrect = incorrect_dir * (up_mask * up_weight + down_mask * down_weight)
    
    # Return mean loss
    return torch.mean(weighted_incorrect)

def combined_loss(y_pred, y_true, mse_weight=1.0, dir_weight=1.0, 
                 up_weight=1.0, down_weight=1.0):
    """
    Combines MSE loss with directional loss for financial time series.
    
    Args:
        y_pred: Predicted values tensor
        y_true: Target values tensor
        mse_weight: Weight for MSE component (default: 1.0)
        dir_weight: Weight for directional component (default: 1.0)
        up_weight: Weight for upward movements in directional loss (default: 1.0)
        down_weight: Weight for downward movements in directional loss (default: 1.0)
    
    Returns:
        Combined loss tensor
    """
    mse = F.mse_loss(y_pred, y_true)
    dir_loss = directional_loss(y_pred, y_true, up_weight, down_weight)
    
    return mse_weight * mse + dir_weight * dir_loss

# Helper functions for batching and sequence creation
def create_sequences(data, seq_length, target_column=0):
    """
    Creates sequences from time series data for LSTM input.
    
    Args:
        data: Input data tensor
        seq_length: Sequence length for LSTM
        target_column: Column index to use as target variable (default: 0)
        
    Returns:
        X sequences and y targets
    """
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, target_column]
        xs.append(x)
        ys.append(y)
    
    return torch.stack(xs), torch.stack(ys)

def create_multistep_sequences(data, seq_length, forecast_horizon, target_column=0):
    """
    Creates sequences for multi-step forecasting.
    
    Args:
        data: Input data tensor
        seq_length: Input sequence length
        forecast_horizon: Number of future steps to predict
        target_column: Column index to use as target variable (default: 0)
        
    Returns:
        X sequences and y targets for multiple steps
    """
    xs, ys = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length):(i+seq_length+forecast_horizon), target_column]
        xs.append(x)
        ys.append(y)
    
    return torch.stack(xs), torch.stack(ys)

def batch_generator(X, y, batch_size, shuffle=True):
    """
    Generates batches of data for training.
    
    Args:
        X: Input sequences
        y: Target values
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data (default: True)
        
    Yields:
        Batches of (X, y) pairs
    """
    indices = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    
    for start_idx in range(0, len(indices), batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx].to(device), y[batch_idx].to(device)
