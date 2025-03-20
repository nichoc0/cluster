# TPU-Accelerated Scalar-Matrix LSTM for Financial Time Series

## Introduction

This repository contains an advanced implementation of a memory-efficient Scalar-Matrix LSTM (SM-LSTM) model specifically designed for financial time series analysis. The model combines elements from modern architectural innovations including:

1. State Space Models (SSMs)
2. Matrix-based memory representations
3. Regime-aware processing
4. Volatility-adaptive computations
5. Trend detection and exploitation

This document provides an in-depth mathematical explanation of the model's components and operations.

## Mathematical Foundations

### 1. Scalar-Matrix LSTM Cell Structure

The core innovation of the SM-LSTM cell is its use of matrix-based memory instead of traditional vector memory. While standard LSTMs use hidden states $h_t \in \mathbb{R}^d$, our SM-LSTM uses $C_t \in \mathbb{R}^{d \times d}$, enabling richer state representation.

#### Core Memory Update Equation

$$C_t = f_t \odot C_{t-1} + i_t \cdot (v_t \otimes k_t)$$

Where:
- $C_t \in \mathbb{R}^{d \times d}$ is the cell state matrix
- $f_t \in \mathbb{R}$ is the forget gate scalar
- $i_t \in \mathbb{R}$ is the input gate scalar
- $v_t \in \mathbb{R}^d$ is the value vector
- $k_t \in \mathbb{R}^d$ is the key vector
- $\otimes$ denotes the outer product $(v_t \otimes k_t) \in \mathbb{R}^{d \times d}$

#### Hidden State Computation

The hidden state is computed as:

$$h_t = o_t \odot \frac{C_t q_t}{max(|q_t^T n_t|, \theta)}$$

Where:
- $o_t \in \mathbb{R}^d$ is the output gate
- $q_t \in \mathbb{R}^d$ is the query vector
- $n_t \in \mathbb{R}^d$ is the normalization vector updated as: $n_t = f_t \odot n_{t-1} + i_t \cdot k_t$
- $\theta$ is a stabilization threshold (typically 1.0)

### 2. Structured State Space Component

The model incorporates a structured state space module defined by:

$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

In discrete form with step size $\Delta t$:

#### Bilinear Discretization:

$$A_d = \frac{2 + \Delta t \cdot \lambda}{2 - \Delta t \cdot \lambda}$$
$$B_d = \frac{\Delta t \cdot (1 + A_d)}{2} \cdot b$$

#### Zero-Order Hold (ZOH) Discretization:

$$A_d = e^{\lambda \Delta t}$$
$$B_d = \frac{A_d - 1}{\lambda} \cdot b$$

The SSM implementation uses reduced parameters:
- $\lambda \in \mathbb{R}^{d_{state}}$ (eigenvalues)
- $b \in \mathbb{R}^{d_{state} \times 1}$ (input projection)
- $c \in \mathbb{R}^{1 \times d_{state}}$ (output projection)

### 3. Parallel Memory Update

The efficient parallel implementation for sequence processing uses cumulative products and sums:

$$f_{cum}[t] = \prod_{i=1}^{t} f[i]$$
$$C_{updates}[t] = \sum_{i=1}^{t} i[i] \cdot (v[i] \otimes k[i])$$
$$C_t[t] = f_{cum}[t] \cdot C_{prev} + C_{updates}[t]$$

### 4. Financial-Specific Loss Functions

#### Asymmetric Directional Loss

$$L_{dir}(y_{pred}, y_{true}) = \frac{1}{T-1}\sum_{t=1}^{T-1} \mathbf{1}_{\{sign(y_{pred,t+1} - y_{pred,t}) \neq sign(y_{true,t+1} - y_{true,t})\}} \cdot w_t$$

Where $w_t$ is:
- $w_t = w_{up}$ when $y_{true,t+1} - y_{true,t} > 0$
- $w_t = w_{down}$ when $y_{true,t+1} - y_{true,t} < 0$

This penalizes direction prediction errors asymmetrically, with typically $w_{up} > w_{down}$.

## Key Components and Their Mathematical Details

### 1. Gate Stabilization

To prevent numerical instability from exponential operations, we use the following stabilization technique:

$$m_t = \max(f_t, m_{t-1} + f_t)$$
$$i_t = \exp(i_t - m_t)$$
$$f_t = \exp(f_t + m_{t-1} - m_t)$$

### 2. Volatility Detection

The model detects volatility using a rolling standard deviation approach:

$$\sigma_t = \sqrt{\frac{1}{W} \sum_{i=t-W}^{t} (x_i - \bar{x}_{t-W:t})^2}$$

Where $W$ is the window size and $\bar{x}_{t-W:t}$ is the mean over the window. The normalized volatility is:

$$\hat{\sigma}_t = \frac{\sigma_t}{\bar{\sigma} + \epsilon}$$

### 3. Trend Strength Calculation

The model calculates trend strength through autocorrelation:

$$r = \frac{\sum_{t=1}^{T-1} (x_t - \bar{x})(x_{t+1} - \bar{x})}{\sqrt{\sum_{t=1}^{T} (x_t - \bar{x})^2 \sum_{t=1}^{T} (x_{t+1} - \bar{x})^2}}$$

$$\text{trend strength} = \frac{|r| + 1}{2}$$

For greater efficiency, the implementation also uses fast and slow exponential moving averages (EMAs):

$$\text{fast EMA}_{t} = \alpha_{\text{fast}} \cdot x_t + (1-\alpha_{\text{fast}}) \cdot \text{fast EMA}_{t-1}$$

$$\text{slow EMA}_{t} = \alpha_{\text{slow}} \cdot x_t + (1-\alpha_{\text{slow}}) \cdot \text{slow EMA}_{t-1}$$

### 4. Memory Mixing

The model employs a sophisticated memory mixing approach:

1. **Regime-based Processing**: The memory matrix $C_t$ is processed differently based on detected regime probabilities:
   
   $$C_t^{\text{processed}} = \sum_{r=1}^{4} w_r \cdot \text{Processor}_r(C_t)$$
   
   Where $w_r$ are regime weights and $\text{Processor}_r$ are regime-specific neural networks.

2. **Structured Memory Mixing**:
   
   $$C_t^{\text{mixed}} = \text{SSM}(C_t, \sigma_t)$$
   
   The SSM applies row-wise state space dynamics to the memory matrix, conditioned by volatility.

3. **Trend-based Memory Persistence**:
   
   $$C_t^{\text{final}} = (1 - \text{trend strength}) \cdot C_t + \text{trend strength} \cdot C_{t-1}$$
   
   This allows the model to maintain persistent memory during strong trends.

### 5. Ensemble Model Architecture

The ensemble model combines multiple SM-LSTM models using different configurations:

$$y_{\text{ensemble}} = \text{MetaNetwork}([y_1, y_2, ..., y_n])$$

Where each $y_i$ is the output of an individual model operating on a different time scale or with different hyperparameters.

## Implementation Optimizations

### Memory Efficiency Techniques

1. **Chunked Processing**: Operations are performed in small chunks to reduce peak memory usage:

   ```python
   chunk_size = 8  # Reduced from 32 for memory efficiency
   num_chunks = (min_seq + chunk_size - 1) // chunk_size
   
   for chunk_idx in range(num_chunks):
       # Process chunk
       # Free memory after chunk processing
   ```

2. **Gradient Checkpointing**: For long sequences, we use PyTorch's gradient checkpointing:

   ```python
   if seq_len > 100 and self.training:
       def run_cell(inp, st):
           return cell.parallel_forward(inp, st)
       layer_output, states[layer_idx] = checkpoint(run_cell, layer_input, states[layer_idx])
   ```

3. **Dimension-wise Processing**: For large matrices, row-by-row processing is used:

   ```python
   for row in range(d):
       row_data = C_t[:, :, row, :]  # [batch, seq, d]
       # Process row
   ```

## Understanding Key Functions

### 1. ParallelSMLSTMCell.parallel_forward

This is the main entry point for processing a sequence through the cell. It:
1. Normalizes the input
2. Computes gates (i, f, o)
3. Transforms the input into queries, keys, and values
4. Updates the memory matrix
5. Computes the output

### 2. ParallelSMLSTMCell.parallel_memory_update

This function efficiently updates the matrix memory in parallel across the sequence dimension using:
1. Cumulative products of forget gates
2. Cumulative sums of memory updates
3. Regime-based memory processing
4. Structured state space mixing
5. Trend-based memory mixing

### 3. ParallelSMLSTMCell.stabilize_gates

This function implements the numerical stabilization technique described above, ensuring that the exponential operations in the gate computations don't cause overflow or underflow.

### 4. StructuredStateSpace.forward

This implements the structured state space model, which processes temporal data using:
1. Discretized state space dynamics
2. Volatility-conditioned state transitions
3. Trend-aware update steps

## Conclusion

The TPU-accelerated Scalar-Matrix LSTM represents a significant advancement in memory-based models for financial time series. By combining matrix memory, structured state spaces, and financial domain-specific optimizations, it achieves both expressiveness and efficiency. The implementation is carefully optimized for memory usage and computational efficiency, making it suitable for deployment in resource-constrained environments.


# Understanding TPU-Accelerated LSTM for Financial Time Series: main.py Explained

This document provides an in-depth explanation of `main.py`, which is the primary execution script for the TPU-accelerated Scalar-Matrix LSTM model for financial time series analysis.

## Overview

The `main.py` script orchestrates the entire workflow, including:

1. Data acquisition and preprocessing
2. Feature engineering
3. Model configuration and initialization
4. Training and evaluation with TPU acceleration
5. Performance monitoring

## Data Acquisition and Processing

### Time Series Data Collection

The script implements a robust data collection strategy that handles the rate-limited nature of financial APIs:

```python
def safe_fetch(ticker, start, end, retries=15, delay=10):
```

This function employs:
1. **Exponential backoff**: The delay between retries increases exponentially to avoid API rate limits
2. **Error handling**: Specific handling for rate limiting errors
3. **Data validation**: Checks to ensure the returned data is valid

For larger datasets, the script uses a chunking strategy:

```python
def fetch_chunk(ticker, start, end):
```

This breaks the request into manageable time periods and concatenates them later:

```python
def get_stock_data(ticker, start_date, end_date, max_workers=3):
```

The chunking approach is mathematically optimized using:

$$\text{chunk\_size} = \min(\text{max\_days}, \frac{\text{total\_days}}{\text{optimal\_chunks}})$$

where `optimal_chunks` is determined based on API rate limits.

### Data Merging and Alignment

A critical function for financial time series analysis is `merge_datasets()`, which aligns data from different frequencies (daily and hourly):

```python
def merge_datasets(daily_df, hourly_df):
```

This function uses interpolation and forward-filling techniques to create a consistent time series:

$$X_{interpolated}(t) = X_{last} + \frac{t - t_{last}}{t_{next} - t_{last}} \cdot (X_{next} - X_{last})$$

## Feature Engineering

### Technical Indicators

The script generates a comprehensive set of technical indicators through the `prepare_data()` and `add_technical_indicators()` functions:

```python
def prepare_data(df):
    # ...
    df['Returns'] = close.pct_change()
    df['EMA_Short'] = close.ewm(span=12, adjust=False).mean() / close - 1
    # ...
```

Mathematically, the EMA (Exponential Moving Average) is calculated as:

$$\text{EMA}_t = \alpha \cdot \text{price}_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$

where $\alpha = \frac{2}{span+1}$

The RSI (Relative Strength Index) is calculated as:

$$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$

where $\text{RS} = \frac{\text{average gain}}{\text{average loss}}$

### Sequence Generation and Augmentation

Time series data is transformed into sequence-based training examples using:

```python
def create_sequences(scaled_data, target_scaled, seq_length, augment=True):
```

With data augmentation applied using controlled noise injection:

$$X_{augmented} = X + \mathcal{N}(0, 0.001 \cdot \sigma_X)$$

This preserves the signal's basic structure while improving model generalization.

## Model Training with TPU Acceleration

### TPU-Optimized Data Loading

A critical component for TPU acceleration is the data loading pipeline:

```python
def create_tpu_dataloader(dataset, batch_size, shuffle=True):
```

This configures the dataloader specifically for TPU execution by:
1. Disabling multiprocessing workers (`num_workers=0`)
2. Ensuring proper batch sizes for TPU memory tiles
3. Managing dataset prefetching

### Learning Rate Scheduling

The training loop implements a sophisticated learning rate schedule with warmup and decay:

```python
def lr_schedule(epoch):
    if epoch < 5:
        # Warmup phase - linear increase from 0.1x to 1x
        return 0.1 + 0.9 * epoch / 5
    else:
        # Decay phase - cosine decay
        return 0.1 + 0.9 * (1 + math.cos(math.pi * (epoch - 5) / (epochs - 5))) / 2
```

This is mathematically defined as:

$$\eta_t = \begin{cases}
\eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{t}{T_{warmup}} & \text{if } t < T_{warmup} \\
\eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{1 + \cos(\pi \cdot \frac{t - T_{warmup}}{T_{total} - T_{warmup}})}{2} & \text{otherwise}
\end{cases}$$

### TPU-Specific Optimization

The training loop includes several TPU-specific optimizations:

1. **Mark_step for execution**: `xm.mark_step()` ensures TPU execution barriers are properly set
2. **Gradient accumulation**: To optimize memory usage, gradients are accumulated across mini-batches
3. **Host-device synchronization**: Critical tensors are explicitly moved to host memory to avoid device memory accumulation

```python
# TPU-friendly backward pass
loss.backward()

if (batch_idx + 1) % 2 == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_VALUE)
    optimizer.step()
    optimizer.zero_grad()
    # Critical for TPU execution
    xm.mark_step()
```

### Multi-Objective Loss Functions

The training process uses multiple specialized loss functions tailored for financial forecasting:

```python
criterion_price = nn.HuberLoss(delta=1.0)
pos_weight = torch.tensor([2.0]).to(device)
criterion_direction = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_volatility = nn.MSELoss()
```

1. **Huber Loss**: For price predictions, uses a hybrid approach that reduces sensitivity to outliers:

$$L_{\delta}(y, f(x)) = \begin{cases}
\frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)| \leq \delta, \\
\delta \cdot (|y - f(x)| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

2. **Weighted BCE Loss**: For directional predictions, assigns higher weight to up movements:

$$L_{BCE} = -\frac{1}{N}\sum_i w_i[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

where $w_i$ is typically higher (2.0) for positive class (up movement) examples.

3. **MSE Loss**: For volatility prediction:

$$L_{MSE} = \frac{1}{N}\sum_i(y_i - \hat{y}_i)^2$$

### Performance Metrics

The model tracks multiple specialized metrics for financial forecasting:

1. **Price MAPE** (Mean Absolute Percentage Error):
   
$$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{A_t - F_t}{A_t} \right|$$

2. **Directional Accuracy**: The percentage of correct movement direction predictions:

$$\text{Directional Accuracy} = \frac{100\%}{n-1} \sum_{t=1}^{n-1} \mathbf{1}_{\{\text{sign}(P_{t+1} - P_t) = \text{sign}(\hat{P}_{t+1} - \hat{P}_t)\}}$$

3. **Regime Analysis**: Tracking the distribution of predicted market regimes for model interpretability:

$$\text{Regime Distribution} = \left[\frac{1}{n} \sum_{i=1}^{n} r_{i,1}, \ldots, \frac{1}{n} \sum_{i=1}^{n} r_{i,4}\right]$$

## Curriculum Learning Strategy

The implementation uses curriculum learning to gradually increase the complexity of the training objective:

```python
# Add curriculum learning - focus first on price, then direction
dir_weight = min(0.5 + epoch / 10, 2.0)  # Gradually increase direction weight
vol_weight = min(0.2 + epoch / 20, 0.5)  # Gradually increase volatility weight
```

This is mathematically defined as increasing the contribution of direction and volatility tasks over time:

$$L_{total} = L_{price} + \min\left(0.5 + \frac{\text{epoch}}{10}, 2.0\right) \cdot L_{direction} + \min\left(0.2 + \frac{\text{epoch}}{20}, 0.5\right) \cdot L_{volatility}$$

## Early Stopping with Delta Threshold

The implementation uses a sophisticated early stopping approach that includes both patience and a minimum improvement threshold:

```python
if val_loss < (best_val_loss - min_delta):
    best_val_loss = val_loss
    waiting = 0
else:
    waiting += 1
    if waiting >= patience:
        print(f'Early stopping triggered at epoch {epoch+1}')
        break
```

This improves upon standard early stopping by requiring a meaningful improvement (not just any improvement) to reset the patience counter, based on:

$$\Delta_{\text{val\_loss}} = \text{best\_val\_loss} - \text{current\_val\_loss} > \delta_{\min}$$

## Main Execution Flow

The `main()` function orchestrates the entire pipeline, covering:

1. Data loading and preparation
2. Cross-validation with time series split
3. Feature normalization
4. Sequence generation and sampling
5. Model initialization and training
6. Model saving and evaluation

Time series cross-validation is implemented to ensure proper training-validation separation:

```python
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
    # ...
```

This maintains the temporal ordering, ensuring that models are always trained on past data and evaluated on future data, preventing information leakage.

## Conclusion

The `main.py` script represents a sophisticated implementation of modern deep learning techniques for financial time series, optimized for TPU execution. It incorporates numerous mathematical and algorithmic optimizations specifically designed for the challenges of financial forecasting, including:

1. Robust data acquisition and preprocessing
2. Finance-specific feature engineering
3. TPU-optimized training procedures
4. Multi-objective learning with curriculum strategy
5. Specialized loss functions and metrics for financial time series
6. Memory-efficient implementation for long sequences

These elements combine to create a powerful framework for high-performance financial forecasting using Scalar-Matrix LSTM architectures.




# TPU-Accelerated Scalar-Matrix LSTM Training Results Analysis

This document analyzes the training results of the Scalar-Matrix LSTM model on financial time series data, focusing on the performance metrics and convergence patterns observed during training.

## Overview

The output shows training across 5 folds of cross-validation, each using time series data with increasing temporal range. The model contains approximately 1.93 million parameters and processes 26 financial features.

## Convergence Analysis

### Initial Adaptation Phase

The first pattern evident across all folds is the dramatic reduction in error metrics during the first 2-3 epochs:

| Fold | Initial MAPE | MAPE after 3 epochs | Improvement |
|------|--------------|---------------------|-------------|
| 1    | 90.87%       | 16.77%              | 81.5%       |
| 2    | 90.53%       | 9.23%               | 89.8%       |
| 3    | 92.23%       | 5.70%               | 93.8%       |
| 4    | 94.14%       | 13.12%              | 86.1%       |
| 5    | 95.13%       | 24.81%              | 73.9%       |

This indicates that the model quickly adapts to capture the fundamental price patterns in the data, even before fine-tuning begins.

### Error Metric Progression

The Mean Absolute Percentage Error (MAPE) shows consistent improvement throughout training:

- **Fold 1**: 90.87% → 7.31% (final)
- **Fold 2**: 90.53% → 4.81% (best)
- **Fold 3**: 92.23% → 3.59% (best)
- **Fold 4**: 94.14% → 3.72% (best)
- **Fold 5**: 95.13% → 3.51% (best)

Later folds achieve lower MAPE values, suggesting that the model benefits from increasing data availability or possibly that more recent financial data (used in later folds) has more predictable patterns.

### MSE and MAE Trends

The RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) show similar improvements:

- **Fold 2 (Best)**: RMSE: 2.19 → 0.14, MAE: 2.18 → 0.11
- **Fold 3 (Best)**: RMSE: 1.59 → 0.16, MAE: 1.59 → 0.12
- **Fold 5 (Best)**: RMSE: 3.44 → 0.16, MAE: 3.44 → 0.12

These metrics confirm the model's increasing accuracy in price prediction across all folds.

## Regime Detection Analysis

The regime distribution shows interesting patterns during training:

### Initial Regime Specialization

In early training, each fold shows a strong preference for specific regimes:

- **Fold 2**: By epoch 5, regime 3 dominates (54%)
- **Fold 3**: By epoch 6, regime 4 dominates (79%)
- **Fold 4**: By epoch 6, regime 3 dominates (43%)
- **Fold 5**: By epoch 6, regime 2 dominates (89%)

This suggests the model initially identifies a dominant market regime for each time period.

### Regime Balancing

As training progresses, regime distributions consistently balance toward approximately 25% each:

```
Epoch 15 (Fold 3): [0.2591144 0.24519493 0.24478552 0.25090513]
Epoch 16 (Fold 4): [0.24918784 0.24459712 0.24935074 0.25686428]
Epoch 19 (Fold 5): [0.25001785 0.25751257 0.24698563 0.24548395]
```

This balanced distribution suggests that after sufficient training, the model learns to identify multiple market regimes with similar frequency, indicating a more nuanced understanding of market states rather than overfitting to a single regime explanation.

## Signal Distribution Patterns

The model produces three signal categories, which also show evolution during training:

### Early Signal Specialization

Similar to regimes, early training shows a preference for one signal:

- **Fold 2 (Epoch 5)**: Signal 3 dominates (66%)
- **Fold 3 (Epoch 6)**: Signal 3 dominates (79%)
- **Fold 5 (Epoch 6)**: Signal 2 dominates (51%)

### Signal Balancing

By the end of training, signals are more evenly distributed:

```
Epoch 24 (Fold 2): [0.33942443 0.32072926 0.3398463]
Epoch 15 (Fold 3): [0.32974148 0.33232832 0.3379302]
Epoch 19 (Fold 5): [0.32863995 0.3293468 0.34201327]
```

This evolution suggests that while the model initially focuses on strong signals that explain most of the variance, it later develops more nuanced signal processing capabilities.

## Learning Dynamics

### Optimal Learning Rate

The learning rate starts at 0.000003 and peaks at 0.000010 around epochs 5-7, before gradually decreasing. Significant performance improvements correlate with this peak learning rate period, suggesting this range effectively balances exploration and exploitation.

### Early Stopping Pattern

Folds exhibit different early stopping points:
- Fold 1: Stops at epoch 10
- Fold 2: Stops at epoch 25
- Fold 3: Stops at epoch 15
- Fold 4: Stops at epoch 16
- Fold 5: Continues through epoch 19

Later folds generally train longer before early stopping triggers, suggesting more complex patterns in recent data requiring additional training.

## TPU Efficiency Observations

The TPU-specific optimizations show their value through:

1. Consistent batch processing without OOM errors
2. Smooth execution across all folds
3. Effective memory management with large parameter count (1.93M parameters)
4. Stable convergence patterns across folds

## Conclusion

The Scalar-Matrix LSTM model demonstrates strong predictive capabilities with extremely low error rates (MAPE as low as 3.5%), while showing evidence of learning meaningful market regimes and signals. The high-parameter count (1.93M) is efficiently managed by the TPU implementation.


# Prediction Timeframe Analysis for the Scalar-Matrix LSTM Model

## Overview of Prediction Structure

The TPU-accelerated Scalar-Matrix LSTM is designed to predict **5 future time steps** based on a historical sequence. This is evident in both the data preparation and model architecture:

### Input Sequence Length

In `main.py`, the model's configuration defines:

```python
SEQ_LENGTH = 64
```

This means each input sample consists of 64 consecutive time steps of financial data.

### Target Output Length

The model generates predictions for 5 future steps, as seen in the sequence creation function:

```python
def create_sequences(scaled_data, target_scaled, seq_length, augment=True):
    X, y = [], []
    for i in range(seq_length, len(scaled_data) - 5):
        seq = scaled_data[i - seq_length:i]
        # ...
        y.append(target_scaled[i:i+5].flatten())  # 5 future steps
    return np.array(X), np.array(y)
```

This is further confirmed by the model's output heads, all designed with 5 output units:

```python
self.price_head = nn.Linear(self.pre_fc_dim, 5)
self.direction_head = nn.Linear(self.pre_fc_dim, 5)
self.volatility_head = nn.Linear(self.pre_fc_dim, 5)
```

## Time Resolution

The model operates on **hourly data**, as indicated by the data fetching functions:

```python
def get_hourly_data(ticker, start_date, end_date, max_workers=3):
    file_path = f"{ticker}_hourly_{start_date}_{end_date}.csv"
    # ...
```

And the data download function:

```python
data = download(
    ticker,
    start=start.strftime("%Y-%m-%d"), 
    end=end.strftime("%Y-%m-%d"),
    interval='1h',  # Hourly interval
    auto_adjust=True
)
```

## Prediction Timeframe Summary

Based on the code analysis:

1. **Input Window**: 64 hours (approximately 2.7 days) of historical financial data
2. **Prediction Horizon**: 5 hours into the future
3. **Resolution**: Hourly price predictions

## Evaluation Methodology

When evaluating the model, the predictions are compared to the actual prices for those 5 future hours. This is reflected in the evaluation metrics:

```python
price_mape = torch.mean(torch.abs((all_price_targets - all_price_outputs) / (all_price_targets + 1e-8))) * 100
price_rmse = torch.sqrt(torch.mean((all_price_targets - all_price_outputs) ** 2))
price_mae = torch.mean(torch.abs(all_price_targets - all_price_outputs))
```

The directional accuracy specifically evaluates whether the model correctly predicted price movements between consecutive hours:

```python
# Calculate price differences between consecutive predictions
diff_pred = all_price_outputs[:, 1:] - all_price_outputs[:, :-1]
diff_true = all_price_targets[:, 1:] - all_price_targets[:, :-1]
            
# Convert to binary UP (>0) matching the label creation logic
binary_pred_up = (diff_pred > 0).float()
binary_true_up = (diff_true > 0).float()
```

## Training-Testing Configuration

The model uses a time series cross-validation approach, which ensures that training data always precedes validation data, maintaining the temporal integrity of financial series:

```python
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
    # ...
```

This approach progressively increases the amount of training data while always validating on unseen future data.

The progression of training shows an initial rapid adaptation phase followed by more nuanced refinement, with the model ultimately achieving balanced regime and signal distributions that suggest it has learned genuine market patterns rather than overfitting to specific temporal artifacts.

The significant reduction in MAPE from ~90% to ~3-7% across all folds indicates that the model has successfully captured the underlying financial time series dynamics.