import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time
import argparse
from datetime import datetime
import json
import gc

# Import custom LSTM models for CUDA

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Memory management functions
def free_memory():
    """Free up GPU memory by clearing cache"""
    gc.collect()
    torch.cuda.empty_cache()

def print_memory_stats():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Data preprocessing functions
def load_financial_data():
    """Load both daily and hourly financial data from CSV"""
    # Load daily data
    daily_path = '/kaggle/input/msftdat/MSFT_daily_2010-01-01_2023-12-31.csv'
    print(f"Loading daily data from {daily_path}...")
    daily_data = pd.read_csv(daily_path, 
                         skiprows=[1,2],  # Skip the metadata rows
                         index_col=0,
                         parse_dates=True)
    
    # Load hourly data
    hourly_path = '/kaggle/input/msftdat/MSFT_hourly_2023-02-24_2025-02-16.csv'
    print(f"Loading hourly data from {hourly_path}...")
    hourly_data = pd.read_csv(hourly_path, 
                          index_col=0, 
                          parse_dates=True)
    
    # Print column names to debug
    print(f"Daily data columns: {daily_data.columns.tolist()}")
    print(f"Hourly data columns: {hourly_data.columns.tolist()}")
    
    print(f"Daily data loaded with shape: {daily_data.shape}")
    print(f"Hourly data loaded with shape: {hourly_data.shape}")
    
    return daily_data, hourly_data

def prepare_data(df, target_column='Price', feature_columns=None, 
                 window_size=60, forecast_horizon=1, train_ratio=0.8,
                 val_ratio=0.1, scale_data=True):
    """
    Prepare data for LSTM training:
    - Scale features
    - Create sequences
    - Split into train/val/test sets
    """
    if feature_columns is None:
        # Use all numeric columns by default
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ensure target_column is in feature_columns
    if target_column not in feature_columns:
        feature_columns.append(target_column)
    
    # Extract features
    data = df[feature_columns].values
    
    # Scale data
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
    else:
        scaler = None
    
    # Convert to PyTorch tensor
    data_tensor = torch.FloatTensor(data)
    
    # Create sequences
    if forecast_horizon == 1:
        X, y = create_sequences(data_tensor, window_size, 
                               feature_columns.index(target_column))
    else:
        X, y = create_multistep_sequences(data_tensor, window_size, 
                                         forecast_horizon, 
                                         feature_columns.index(target_column))
    
    # Split data
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            scaler, feature_columns.index(target_column))

# Model training functions
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, 
                epochs=100, learning_rate=0.001, patience=10, min_delta=0.0001,
                loss_fn=None, optimizer=None, scheduler=None, verbose=1,
                checkpoint_dir='checkpoints'):
    """
    Train the LSTM model with early stopping
    """
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Initialize loss function if not provided
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    # Initialize optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize scheduler if not provided
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, 
            verbose=True, min_lr=1e-6
        )
    
    # Move model to device
    model = model.to(device)
    
    # Training loop setup
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Early stopping variables
    counter = 0
    early_stop = False
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        if early_stop:
            break
        
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in batch_generator(X_train, y_train, batch_size):
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(batch_X)
            
            # Compute loss
            loss = loss_fn(y_pred, batch_y.unsqueeze(-1) if batch_y.dim() == 1 else batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in batch_generator(X_val, y_val, batch_size, shuffle=False):
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y.unsqueeze(-1) if batch_y.dim() == 1 else batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(X_val) // batch_size + 1)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Print progress
        if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                  f"Time: {time.time() - start_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
            if torch.cuda.is_available():
                print_memory_stats()
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
        
        # Early stopping check
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            early_stop = True
        
        # Free up memory
        if torch.cuda.is_available():
            free_memory()
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    return model, train_losses, val_losses

# Evaluation functions
def evaluate_model(model, X, y, batch_size=32, scaler=None, target_idx=0, multi_step=False):
    """
    Evaluate model performance on test data
    """
    model.eval()
    
    # Generate predictions
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_pred = model(batch_X)
            
            # For multi-step, we're interested in all predictions
            # For single-step, we're only interested in the last time step
            if not multi_step and batch_pred.dim() == 3:
                batch_pred = batch_pred[:, -1, :]
                
            predictions.append(batch_pred.cpu())
    
    # Concatenate predictions
    y_pred = torch.cat(predictions, dim=0)
    
    # Convert to numpy for evaluation
    y_np = y.numpy()
    y_pred_np = y_pred.numpy()
    
    # Reshape for single-step if needed
    if y_pred_np.ndim == 3:
        y_pred_np = y_pred_np.squeeze(-1)
    
    # Flatten for single-step metrics if necessary
    if not multi_step and y_np.ndim == 1 and y_pred_np.ndim > 1:
        y_pred_np = y_pred_np.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_np, y_pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_np, y_pred_np)
    
    # Calculate directional accuracy
    if multi_step:
        # For multi-step, calculate direction accuracy across the forecast horizon
        true_dir = np.sign(y_np[:, 1:] - y_np[:, :-1])
        pred_dir = np.sign(y_pred_np[:, 1:] - y_pred_np[:, :-1])
        dir_match = (true_dir == pred_dir)
        dir_accuracy = np.mean(dir_match) * 100
    else:
        # For single-step, we need at least 2 predictions to calculate direction
        if len(y_np) > 1:
            true_dir = np.sign(y_np[1:] - y_np[:-1])
            pred_dir = np.sign(y_pred_np[1:] - y_pred_np[:-1])
            dir_match = (true_dir == pred_dir)
            dir_accuracy = np.mean(dir_match) * 100
        else:
            dir_accuracy = 0.0
    
    # Return results
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'dir_accuracy': dir_accuracy
    }
    
    return metrics, y_pred_np

def predict_future(model, X_last, forecast_steps=30, scaler=None, target_idx=0):
    """
    Generate future predictions using the last available data point
    """
    model.eval()
    
    # Make sure X_last is on device
    X_last = X_last.to(device)
    
    # For multi-step model
    if isinstance(model, MultiStepLSTM):
        with torch.no_grad():
            predictions = model(X_last.unsqueeze(0))
            predictions = predictions.cpu().numpy()[0]
        
        # If we need more steps than the model's horizon
        if forecast_steps > predictions.shape[0]:
            # Need to recursively predict
            remaining_steps = forecast_steps - predictions.shape[0]
            for _ in range(remaining_steps):
                # Create new input from predictions
                new_input = torch.cat([
                    X_last[1:], 
                    torch.FloatTensor(predictions[-1]).unsqueeze(0)
                ], dim=0).unsqueeze(0).to(device)
                
                # Predict next step
                next_pred = model(new_input)
                next_pred = next_pred.cpu().numpy()[0, 0]
                
                # Append to predictions
                predictions = np.vstack([predictions, next_pred])
                
                # Update X_last
                X_last = torch.cat([
                    X_last[1:], 
                    torch.FloatTensor(next_pred).unsqueeze(0)
                ], dim=0)
    
    # For single-step model
    else:
        predictions = []
        current_input = X_last.unsqueeze(0)
        
        for _ in range(forecast_steps):
            with torch.no_grad():
                next_pred = model(current_input)
                if next_pred.dim() == 3:
                    next_pred = next_pred[:, -1, :]
                predictions.append(next_pred.cpu().numpy()[0])
                
                # Update input for next prediction
                new_input = torch.cat([
                    current_input[:, 1:, :],
                    next_pred.unsqueeze(1)
                ], dim=1)
                current_input = new_input
        
        predictions = np.array(predictions)
    
    # Inverse transform if needed
    if scaler is not None:
        # Create dummy array with correct shape
        dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
        # Fill target column with predictions
        dummy[:, target_idx] = predictions.flatten()
        # Inverse transform
        dummy = scaler.inverse_transform(dummy)
        # Extract target column
        predictions = dummy[:, target_idx]
    
    return predictions

# Plotting functions
def plot_training_history(train_losses, val_losses, title='Training History'):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

def plot_predictions(y_true, y_pred, title='Model Predictions'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

def plot_future_predictions(historical_data, future_predictions, n_historical=100, title='Future Predictions'):
    """Plot historical data with future predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    if n_historical > 0 and len(historical_data) > n_historical:
        hist_data = historical_data[-n_historical:]
        plt.plot(range(len(hist_data)), hist_data, label='Historical')
    else:
        hist_data = historical_data
        plt.plot(range(len(hist_data)), hist_data, label='Historical')
    
    # Plot future predictions
    plt.plot(
        range(len(hist_data)-1, len(hist_data) + len(future_predictions)-1),
        future_predictions,
        label='Predicted',
        linestyle='--'
    )
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

# Main function to run training and evaluation
def main(args=None):
    """Main function to train and evaluate model"""
    # Handle the case when running interactively without command line args
    if args is None:
        # Create a parser but don't parse from command line
        parser = create_parser()
        # Create an empty namespace
        args = parser.parse_args([])
        # Set defaults for other common arguments
        if not hasattr(args, 'output_dir') or not args.output_dir:
            args.output_dir = 'output'
        if not hasattr(args, 'target_column') or not args.target_column:
            args.target_column = 'Price'
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Log parameters
    params = vars(args)
    params_path = os.path.join(args.output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Load both datasets
    daily_data, hourly_data = load_financial_data()
    
    # Determine which dataset to use for this run
    if args.data_type == 'daily':
        df = daily_data
        print("Using daily dataset for training")
    else:
        df = hourly_data
        print("Using hourly dataset for training")
    
    # Check if target_column exists in the DataFrame
    # If not, use the first column as target
    if args.target_column not in df.columns:
        print(f"WARNING: Target column '{args.target_column}' not found in data!")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Try to find a price-related column
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
        if price_columns:
            args.target_column = price_columns[0]
            print(f"Using '{args.target_column}' as target column instead.")
        else:
            args.target_column = df.columns[0]
            print(f"Using first column '{args.target_column}' as target column instead.")
    
    # Determine feature columns if not specified
    if args.feature_columns is None or len(args.feature_columns) == 0:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        feature_columns = args.feature_columns.split(',')
    
    # Ensure target_column is in feature_columns
    if args.target_column not in feature_columns:
        feature_columns.append(args.target_column)
    
    print(f"Target column: {args.target_column}")
    print(f"Feature columns: {feature_columns}")
    
    # Prepare data
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, target_idx) = prepare_data(
        df, args.target_column, feature_columns, 
        args.window_size, args.forecast_horizon, 
        args.train_ratio, args.val_ratio, 
        args.scale_data
    )
    
    # Determine input size (number of features)
    input_size = X_train.shape[2]
    
    # Create model
    if args.forecast_horizon > 1:
        model = MultiStepLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=1,  # Single feature prediction
            forecast_horizon=args.forecast_horizon,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        multi_step = True
    else:
        model = FinancialLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=1,  # Single feature prediction
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        multi_step = False
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create loss function
    if args.directional_loss_weight > 0:
        loss_fn = lambda y_pred, y_true: combined_loss(
            y_pred, y_true, 
            mse_weight=1.0, 
            dir_weight=args.directional_loss_weight,
            up_weight=args.up_weight,
            down_weight=args.down_weight
        )
    else:
        loss_fn = nn.MSELoss()
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        loss_fn=loss_fn,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
        verbose=args.verbose
    )
    
    # Plot training history
    history_plot = plot_training_history(train_losses, val_losses)
    history_plot.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate on test set
    test_metrics, test_preds = evaluate_model(
        model, X_test, y_test, 
        batch_size=args.batch_size, 
        scaler=scaler, 
        target_idx=target_idx,
        multi_step=multi_step
    )
    
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Plot test predictions
    if multi_step:
        # For multi-step, plot horizons separately
        for i in range(args.forecast_horizon):
            pred_plot = plot_predictions(
                y_test[:, i].numpy(), 
                test_preds[:, i], 
                title=f'Test Predictions (Horizon: {i+1})'
            )
            pred_plot.savefig(os.path.join(args.output_dir, f'test_predictions_h{i+1}.png'))
    else:
        # For single-step
        pred_plot = plot_predictions(y_test.numpy(), test_preds, title='Test Predictions')
        pred_plot.savefig(os.path.join(args.output_dir, 'test_predictions.png'))
    
    # Generate future predictions
    if args.predict_future > 0:
        print(f"\nGenerating {args.predict_future} future predictions...")
        
        # Use the last window of data for predictions
        last_window = X_test[-1].clone()
        
        # Generate predictions
        future_preds = predict_future(
            model, last_window, 
            forecast_steps=args.predict_future,
            scaler=scaler,
            target_idx=target_idx
        )
        
        # Get historical data for plotting
        historical_data = df[args.target_column].values[-100:]
        
        # Plot future predictions
        future_plot = plot_future_predictions(
            historical_data, future_preds, 
            title='Future Predictions'
        )
        future_plot.savefig(os.path.join(args.output_dir, 'future_predictions.png'))
        
        # Save predictions
        future_df = pd.DataFrame({
            'date': pd.date_range(
                start=df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(), 
                periods=args.predict_future+1
            )[1:],
            'prediction': future_preds
        })
        future_df.to_csv(os.path.join(args.output_dir, 'future_predictions.csv'), index=False)
    
    print(f"\nResults saved to {args.output_dir}")

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description='Financial Time Series Forecasting with LSTM')
    
    # Removed data_path - we'll load both datasets automatically
    parser.add_argument('--data_type', type=str, default='daily', choices=['daily', 'hourly'], 
                        help='Type of data to use (daily or hourly)')
    parser.add_argument('--target_column', type=str, default='Price', help='Target column to predict')
    parser.add_argument('--feature_columns', type=str, default='', help='Comma-separated list of feature columns')
    parser.add_argument('--window_size', type=int, default=60, help='Sequence length for LSTM')
    parser.add_argument('--forecast_horizon', type=int, default=1, help='Number of future steps to predict')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Proportion of data for validation')
    parser.add_argument('--scale_data', type=bool, default=True, help='Whether to scale data')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64, help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change for early stopping')
    parser.add_argument('--directional_loss_weight', type=float, default=0.2, help='Weight for directional loss')
    parser.add_argument('--up_weight', type=float, default=1.0, help='Weight for upward movements')
    parser.add_argument('--down_weight', type=float, default=1.0, help='Weight for downward movements')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--predict_future', type=int, default=30, help='Number of future steps to predict after training')
    
    return parser

# Modified to work better in Kaggle/Jupyter environment
if __name__ == "__main__":
    try:
        # For Kaggle/Jupyter environment, ignore unknown arguments
        parser = create_parser()
        args, unknown = parser.parse_known_args()
        
        main(args)
    except Exception as e:
        print(f"Error running script: {e}")
        print("Falling back to interactive mode...")
        
        # Create default args object
        args = argparse.Namespace(
            data_type='daily',  # Use daily data by default
            target_column='Close',  # Changed from 'Price' to 'Close' as it's more likely to exist
            feature_columns='',
            window_size=60,
            forecast_horizon=1,
            train_ratio=0.8,
            val_ratio=0.1,
            scale_data=True,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            patience=10,
            min_delta=0.0001,
            directional_loss_weight=0.2,
            up_weight=1.0,
            down_weight=1.0,
            seed=42,
            output_dir='output',
            verbose=1,
            predict_future=30
        )
        
        main(args)
