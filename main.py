#Next Steps
#For your practical implementation, start with:

#Define a small set of stocks (e.g., 5-10 major stocks) to test the universal model
#Implement the MultiStockDataset and data preparation functions
#Create the UniversalStockModel with stock embeddings
#Update the training loop to handle the new data format
#Test the model on multiple stocks
#After you've verified the basic universal model works, you can begin adding:

#Macroeconomic features
#Options data
#Factor model exposures
#Would you like me to continue with specific code for adding macroeconomic features next, or would you prefer to implement the multi-stock foundation first?

import torch
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset, DataLoader
import ta
from torch.optim import lr_scheduler, AdamW
import concurrent.futures
import time
from requests.exceptions import RequestException
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
# Memory optimization settings
torch.backends.cudnn.benchmark = True

# Configurations - REDUCED DIMENSIONS
SEQ_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 35
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
INPUT_DIM = 128


def safe_fetch(ticker, start, end, retries=15, delay=10):
    """
    Attempt to download data with retries and exponential backoff.
    If YFRateLimitError is encountered, wait even longer.
    """
    for attempt in range(retries):
        try:
            data = download(
                ticker,
                start=start.strftime("%Y-%m-%d"), 
                end=end.strftime("%Y-%m-%d"),
                interval='1h',
                auto_adjust=True
            )
            if data is not None and not data.empty:
                return data
        except Exception as yfle:
            print(f"YFRateLimitError on attempt {attempt+1}: {yfle}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        # Exponential backoff
        sleep_time = delay * (attempt + 1)
        print(f"Sleeping for {sleep_time} seconds before next attempt...")
        time.sleep(sleep_time)
    return None


def fetch_chunk(ticker, start, end):
    """
    Fetch a single chunk of data using safe_fetch.
    Adds an extra delay between chunks.
    """
    print(f"Fetching data from {start.date()} to {end.date()}")
    chunk = safe_fetch(ticker, start, end, retries=5, delay=10)
    # Extra delay to reduce request frequency
    time.sleep(5)
    if chunk is not None and not chunk.empty:
        try:
            # Remove multi-index if present
            chunk.columns = chunk.columns.droplevel(1)
        except Exception:
            pass
        return chunk
    else:
        print(f"Warning: no data from {start.date()} to {end.date()}")
        return None

def get_stock_data(ticker, start_date, end_date, max_workers=3):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    delta = pd.Timedelta(days=8)
    
    date_ranges = []
    current_start = start_date
    while (current_start < end_date):
        current_end = min(current_start + delta, end_date)
        date_ranges.append((current_start, current_end))
        current_start = current_end

    data_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_chunk, ticker, s, e) for s, e in date_ranges]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                data_list.append(result)
                
    if data_list:
        try:
            data = pd.concat(data_list)
            data.sort_index(inplace=True)
            return data
        except ValueError as ve:
            print("No objects to concatenate:", ve)
            return pd.DataFrame()
    else:
        print("No data was fetched!")
        return pd.DataFrame()
    
def get_daily_data(ticker, start_date='2010-01-01', end_date='2023-12-31'):
    file_path = f"{ticker}_daily_{start_date}_{end_date}.csv"
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("Loaded daily data from file:", file_path)
    except FileNotFoundError:
        print("File not found, downloading daily data.")
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        data.to_csv(file_path)
        print("Saved daily data to file:", file_path)
    try:
        data.columns = data.columns.droplevel(1)
    except Exception:
        pass
    data.sort_index(inplace=True)
    return data


def get_hourly_data(ticker, start_date, end_date, max_workers=3):
    file_path = f"{ticker}_hourly_{start_date}_{end_date}.csv"
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("Loaded hourly data from file:", file_path)
    except FileNotFoundError:
        print("File not found, downloading hourly data.")
        data = get_stock_data(ticker, pd.to_datetime(start_date), pd.to_datetime(end_date), max_workers)
        data.to_csv(file_path)
        print("Saved hourly data to file:", file_path)
    return data

def merge_datasets(daily_df, hourly_df):
    # Read daily data properly, skipping metadata rows
    if 'Ticker' in daily_df.index:
        # Reset index and skip metadata rows
        daily_df = daily_df.reset_index()
        daily_df = daily_df[daily_df['Price'].str.match(r'\d{4}-\d{2}-\d{2}$', na=False)]
        daily_df = daily_df.set_index('Price')
    
    # Ensure indices are datetime
    daily_df.index = pd.to_datetime(daily_df.index)
    hourly_df.index = pd.to_datetime(hourly_df.index)
    
    # Remove timezone info if present
    if hasattr(daily_df.index, "tz") and daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_localize(None)
    if hasattr(hourly_df.index, "tz") and hourly_df.index.tz is not None:
        hourly_df.index = hourly_df.index.tz_localize(None)
    
    # Drop any duplicate indices
    daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
    hourly_df = hourly_df[~hourly_df.index.duplicated(keep='first')]
    
    # Sort both dataframes by index
    daily_df = daily_df.sort_index()
    hourly_df = hourly_df.sort_index()
    
    # Resample daily data to hourly frequency
    daily_up = daily_df.resample('1h').ffill()
    
    # Merge the datasets
    merged = pd.concat([daily_up, hourly_df])
    
    # Handle duplicates keeping the most recent data
    merged = merged[~merged.index.duplicated(keep='last')]
    
    # Final sort
    merged = merged.sort_index()
    
    return merged

def prepare_multi_stock_data(tickers, start_date='2010-01-01', end_date='2023-12-31', 
                          seq_length=64, pred_steps=5, balance_directions=True):
    """
    Prepare data for multiple stocks for universal model training
    
    Args:
        tickers: List of ticker symbols to include
        start_date, end_date: Date range for data
        seq_length: Length of input sequences
        pred_steps: Number of prediction steps
        balance_directions: Whether to balance up/down samples
        
    Returns:
        Dictionary mapping tickers to processed data
    """
    print(f"Preparing data for {len(tickers)} stocks")
    
    # Get daily data for all tickers in parallel
    print("Fetching daily data...")
    ticker_data = {}
    
    for ticker in tickers:
        try:
            # Get daily data
            daily_data = get_daily_data(ticker, start_date, end_date)
            
            # Get hourly data for recent period
            hourly_start = pd.to_datetime(end_date) - pd.Timedelta(days=365)
            hourly_start_str = hourly_start.strftime('%Y-%m-%d')
            hourly_data = get_hourly_data(ticker, hourly_start_str, end_date)
            
            # Skip tickers with insufficient data
            if daily_data.empty or hourly_data.empty:
                print(f"Skipping {ticker}: insufficient data")
                continue
                
            # Merge daily and hourly data
            merged_data = merge_datasets(daily_data, hourly_data)
            merged_data['Ticker'] = ticker
            
            # Add technical indicators and prepare features
            processed_data = prepare_data(merged_data)
            ticker_data[ticker] = processed_data
            
            print(f"Processed {ticker}: {len(processed_data)} data points")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Create sequences for each ticker
    stock_data_dict = {}
    
    for ticker, df in ticker_data.items():
        try:
            # Extract features and target
            features = df.drop(columns=['Close', 'Ticker'])
            target = df['Close'].values.reshape(-1, 1)
            
            # Get some basic static features for each stock
            # In a full implementation, you would add more static features
            static_features = {
                'avg_volume': df['Volume'].mean(),
                'avg_volatility': df['Returns'].std() if 'Returns' in df.columns else 0.01,
                # Add more static features as needed
            }
            
            # Scale the data
            feature_scaler = RobustScaler()
            target_scaler = RobustScaler()
            
            X_scaled = feature_scaler.fit_transform(features)
            y_scaled = target_scaler.fit_transform(target)
            
            # Create sequences
            X_seq, y_seq, raw_seq = create_sequences(
                X_scaled, y_scaled, raw_prices=target,
                seq_length=seq_length, balance_directions=balance_directions
            )
            
            # Skip tickers with too few sequences
            if len(X_seq) < 100:
                print(f"Skipping {ticker}: only {len(X_seq)} sequences")
                continue
                
            # Store the data
            stock_data_dict[ticker] = {
                'X': torch.FloatTensor(X_seq),
                'y_price': torch.FloatTensor(y_seq),
                'raw_prices': torch.FloatTensor(raw_seq),
                'static_features': static_features,
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler
            }
            
            print(f"Created {len(X_seq)} sequences for {ticker}")
            
        except Exception as e:
            print(f"Error creating sequences for {ticker}: {e}")
    
    return stock_data_dict
def prepare_data(df):
    df = df.copy()
    print("Initial data size:", len(df))
    
    # Handle NaN and infinity values
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # PRICE DYNAMICS
    df['Returns'] = close.pct_change()
    df['Returns'] = df['Returns'].clip(lower=-0.5, upper=0.5)
    
    # TREND INDICATORS
    df['EMA_Short'] = close.ewm(span=12, adjust=False).mean() / close - 1
    df['EMA_Long'] = close.ewm(span=50, adjust=False).mean() / close - 1
    
    # Add crossover signals - explicit direction indicators
    df['EMA_Cross'] = (df['EMA_Short'] > 0).astype(float)
    
    # MOMENTUM INDICATORS
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi() / 100
    
    # Add RSI direction
    df['RSI_Direction'] = df['RSI'].diff().apply(lambda x: 1 if x > 0 else 0)
    
    # VOLATILITY INDICATORS
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df['ATR_Norm'] = atr.average_true_range() / close
        # VOLUME INDICATORS
    df['Vol_Change'] = volume.pct_change().clip(-5, 5)
    
    # Volume trend indicator (is volume increasing or decreasing)
    df['Vol_Trend'] = volume.rolling(window=5).mean().pct_change().apply(lambda x: 1 if x > 0 else 0)
    
    # CYCLICAL FEATURES
    df['DayOfWeek'] = df.index.dayofweek / 6
    df['HourOfDay'] = df.index.hour / 23
    
    
    print("After technical indicators:", len(df))
    df = df.dropna().ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    
    return df

def add_technical_indicators(df):
    if len(df) < 20:  
        print("Data is too short, skipping indicators.")
        return df.copy()

    # Convert string columns to numeric
    numeric_columns = ['Close', 'High', 'Low', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.copy().ffill().bfill()    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Rest of your indicators code...
    rsi_indicator = ta.momentum.RSIIndicator(close=close)
    df['RSI'] = rsi_indicator.rsi()

    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(close=close)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close)
    df['ATR'] = atr.average_true_range()

    df['Returns'] = close.pct_change()
    df['Log_Returns'] = np.log1p(df['Returns'].replace([-np.inf, np.inf], np.nan))
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    df['Volume_MA'] = volume.rolling(window=20).mean()
    df['Volume_STD'] = volume.rolling(window=20).std()

    df.index = pd.to_datetime(df.index)
    df['DayOfWeek'] = df.index.dayofweek
    df['MonthOfYear'] = df.index.month

    df = df.dropna().ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    return df


class MultiStockDataset(Dataset):
    """Dataset that handles multiple stocks with price, direction, and volatility labels"""
    def __init__(self, stock_data_dict, seq_length=64, pred_steps=5):
        """
        Initialize the multi-stock dataset
        
        Args:
            stock_data_dict: Dict mapping ticker symbols to dict containing:
                - 'X': feature tensor (n_samples, seq_len, n_features)
                - 'y_price': price targets 
                - 'y_direction': direction targets (optional)
                - 'y_volatility': volatility targets (optional)
                - 'raw_prices': unscaled original prices (optional)
                - 'static_features': stock-specific attributes (optional)
            seq_length: length of input sequences
            pred_steps: number of prediction steps
        """
        self.stock_data = stock_data_dict
        self.tickers = list(stock_data_dict.keys())
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.tickers)}
        
        # Count total number of sequences
        self.sequence_counts = {}
        self.total_sequences = 0
        self.ticker_indices = []
        self.X_indices = []
        
        # Create unified indices for accessing data
        for ticker, data in stock_data_dict.items():
            n_sequences = len(data['X'])
            self.sequence_counts[ticker] = n_sequences
            self.total_sequences += n_sequences
            
            # Store ticker and sequence indices for each sample
            ticker_idx = self.ticker_to_idx[ticker]
            for i in range(n_sequences):
                self.ticker_indices.append(ticker_idx)
                self.X_indices.append(i)
        
        # Convert to tensors for efficiency
        self.ticker_indices = torch.tensor(self.ticker_indices, dtype=torch.long)
        self.X_indices = torch.tensor(self.X_indices, dtype=torch.long)
        
        print(f"Created dataset with {self.total_sequences} sequences from {len(self.tickers)} stocks")
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Get ticker and sequence index for this sample
        ticker_idx = self.ticker_indices[idx].item()
        X_idx = self.X_indices[idx].item()
        ticker = self.tickers[ticker_idx]
        
        # Get the data for this ticker
        stock_data = self.stock_data[ticker]
        
        # Get the sequence and targets
        X = stock_data['X'][X_idx]
        
        # Build target dict with available targets
        targets = {}
        if 'y_price' in stock_data:
            targets['price'] = stock_data['y_price'][X_idx]
        if 'y_direction' in stock_data:
            targets['direction'] = stock_data['y_direction'][X_idx]
        if 'y_volatility' in stock_data:
            targets['volatility'] = stock_data['y_volatility'][X_idx]
        
        # Add static features if available
        if 'static_features' in stock_data:
            static_features = stock_data['static_features']
            targets['static_features'] = static_features
        
        # Return the features, targets, and ticker index for embedding lookup
        return X, targets, ticker_idx
def create_tpu_dataloader(dataset, batch_size, shuffle=True):
    """Create a DataLoader optimized for TPU"""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0  # TPU works best with 0 workers
    )
    return loader
def create_universal_dataloaders(train_data, val_data, batch_size=32):
    """Create DataLoaders for universal model training"""
    # Create datasets
    train_dataset = MultiStockDataset(train_data)
    val_dataset = MultiStockDataset(val_data)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Adjust based on your system
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return train_loader, val_loader
# Enhanced FunnyMachine with directional prediction
class UniversalStockModel(nn.Module):
    def __init__(self, input_size, num_stocks, embedding_dim=32, hidden_size=16, 
                 matrix_size=4, dropout=0.2, use_static_features=False):
        super().__init__()
        
        # Stock embedding layer
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)
        
        # Static feature processing (optional)
        self.use_static_features = use_static_features
        if use_static_features:
            self.static_feature_layer = nn.Linear(5, embedding_dim)  # Adjust 5 based on static feature count
        
        # Size of combined input (original features + embedding)
        # Each timestep in the sequence gets concatenated with the embedding
        self.combined_input_size = input_size + embedding_dim
        
        # Main model
        self.extended_mlstm = ParallelExtendedSMLSTM(
            input_size=self.combined_input_size,  # Increased to include embedding
            hidden_size=hidden_size,
            matrix_size=matrix_size,
            num_layers=2,
            dropout=dropout,
            expansion_factor=2
        )
        
        # Regime adaptation
        self.regime_adapter = nn.Linear(4, 5)
        
        # Stock-specific output adaptation
        self.stock_output_adapter = nn.Linear(embedding_dim, 5)

    def forward(self, x, stock_idx, static_features=None):
        batch_size, seq_len, feat_dim = x.size()
        
        # Get stock embeddings
        stock_emb = self.stock_embedding(stock_idx)  # [batch_size, embedding_dim]
        
        # Expand embedding to match sequence length
        # From [batch_size, embedding_dim] to [batch_size, seq_len, embedding_dim]
        stock_emb_expanded = stock_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine input features with stock embedding
        combined_input = torch.cat([x, stock_emb_expanded], dim=2)
        
        # Process through main model
        results, regime, signals = self.extended_mlstm(combined_input)
        
        # Apply regime conditioning
        regime_adjustment = self.regime_adapter(regime) * 0.1
        
        # Apply stock-specific adjustment based on embedding
        stock_adjustment = self.stock_output_adapter(stock_emb) * 0.05
        
        # Ensure results is a dictionary
        if not isinstance(results, dict):
            results = {
                'price': results, 
                'direction_logits': results,
                'volatility': F.softplus(results)
            }
        elif 'direction' in results and 'direction_logits' not in results:
            results['direction_logits'] = results['direction']
        
        # Apply adjustments to price
        results['price'] = results['price'] + regime_adjustment + stock_adjustment
        
        return results, regime, signals

def create_sequences(scaled_data, target_scaled, raw_prices=None, seq_length=64, augment=True, balance_directions=True):
    X, y, raw_price_seqs = [], [], []
    
    # Calculate directions first if balancing is requested
    if balance_directions and raw_prices is not None:
        directions = np.sign(np.diff(raw_prices, axis=0)).flatten()
        up_indices = np.where(directions > 0)[0]
        down_indices = np.where(directions <= 0)[0]
        min_count = min(len(up_indices), len(down_indices))
        
        # If severe imbalance, use undersampling of majority class
        if min_count * 10 < max(len(up_indices), len(down_indices)):
            if len(up_indices) > len(down_indices):
                # Undersample UP
                selected_up = np.random.choice(up_indices, size=min_count*3, replace=False)
                selected_indices = np.concatenate([selected_up, down_indices])
            else:
                # Undersample DOWN
                selected_down = np.random.choice(down_indices, size=min_count*3, replace=False)
                selected_indices = np.concatenate([up_indices, selected_down])
            print(f"Balanced to {len(selected_indices)} samples from {len(directions)}")
            balanced_indices = True
        else:
            balanced_indices = False
    else:
        balanced_indices = False
    
    for i in range(seq_length, len(scaled_data) - 5):
        # Skip if using balanced indices and this index isn't in our balanced set
        if balanced_indices and i not in selected_indices:
            continue
            
        seq = scaled_data[i - seq_length:i]
        
        # Simplified augmentation strategy
        if augment and np.random.random() < 0.3:
            noise = np.random.normal(0, 0.001 * np.std(seq), seq.shape)
            seq = seq + noise
        
        X.append(seq)
        y.append(target_scaled[i:i+5].flatten())
        
        if raw_prices is not None:
            raw_price_seqs.append(raw_prices[i:i+5].flatten())
            
    if raw_prices is not None:
        return np.array(X), np.array(y), np.array(raw_price_seqs)
    else:
        return np.array(X), np.array(y)

def create_multi_stock_train_val_split(stock_data_dict, val_ratio=0.15, time_based=True):
    """
    Create train/validation split for multi-stock data
    
    Args:
        stock_data_dict: Dict of stock data as returned by prepare_multi_stock_data
        val_ratio: Ratio of data to use for validation (0.0-1.0)
        time_based: If True, use the most recent data for validation
                    If False, randomly sample from each stock
    
    Returns:
        train_data_dict, val_data_dict: Dictionaries for training and validation
    """
    train_data = {}
    val_data = {}
    
    for ticker, data in stock_data_dict.items():
        n_sequences = len(data['X'])
        val_size = int(n_sequences * val_ratio)
        
        if time_based:
            # Time-based split (most recent data for validation)
            train_idx = slice(0, n_sequences - val_size)
            val_idx = slice(n_sequences - val_size, None)
        else:
            # Random split
            indices = torch.randperm(n_sequences)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]
        
        # Create train data for this ticker
        train_data[ticker] = {
            'X': data['X'][train_idx],
            'y_price': data['y_price'][train_idx],
            'raw_prices': data['raw_prices'][train_idx],
            'static_features': data['static_features'],
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler']
        }
        
        # Create validation data for this ticker
        val_data[ticker] = {
            'X': data['X'][val_idx],
            'y_price': data['y_price'][val_idx],
            'raw_prices': data['raw_prices'][val_idx],
            'static_features': data['static_features'],
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler']
        }
        
        # Now generate direction and volatility labels for both sets
        for dataset in [train_data[ticker], val_data[ticker]]:
            # Create direction labels based on raw prices
            y_direction = torch.zeros_like(dataset['y_price'])
            raw_prices = dataset['raw_prices']
            
            for i in range(len(raw_prices)):
                for j in range(1, min(5, raw_prices.shape[1])):
                    y_direction[i, j] = (raw_prices[i, j] > raw_prices[i, 0]).float()
            
            # Create volatility labels as absolute price changes
            y_volatility = torch.zeros_like(dataset['y_price'])
            if dataset['y_price'].shape[1] > 1:
                y_volatility[:, 1:] = torch.abs(dataset['y_price'][:, 1:] - dataset['y_price'][:, :-1])
            
            # Store the labels
            dataset['y_direction'] = y_direction
            dataset['y_volatility'] = y_volatility
    
    return train_data, val_data
def get_cluster_data(tickers, start_date='2010-01-01', end_date='2023-12-31', max_workers=3):
    """
    Fetch and prepare data for a cluster of stocks.
    
    Args:
        tickers: List of stock ticker symbols in the cluster
        start_date: Start date for data collection
        end_date: End date for data collection
        max_workers: Number of parallel workers for downloading
        
    Returns:
        Dictionary mapping each ticker to its processed dataframe
    """
    print(f"Getting data for cluster of {len(tickers)} stocks: {', '.join(tickers)}")
    
    cluster_data = {}
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            # Get daily data
            daily_data = get_daily_data(ticker, start_date, end_date)
            
            # Get hourly data (using a shorter timeframe for hourly data to manage size)
            hourly_start = pd.to_datetime(end_date) - pd.Timedelta(days=365)  # Last year only
            hourly_start_str = hourly_start.strftime('%Y-%m-%d')
            hourly_data = get_hourly_data(ticker, hourly_start_str, end_date, max_workers)
            
            if daily_data.empty or hourly_data.empty:
                print(f"Warning: Missing data for {ticker}, skipping")
                continue
                
            # Merge daily and hourly data
            merged_data = merge_datasets(daily_data, hourly_data)
            
            # Add ticker identifier column
            merged_data['Ticker'] = ticker
            
            # Add technical indicators
            processed_data = prepare_data(merged_data)
            
            cluster_data[ticker] = processed_data
            print(f"Successfully processed {ticker}: {len(processed_data)} datapoints")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    return cluster_data

def create_cluster_sequences(cluster_data, seq_length=64, pred_steps=5, include_tickers=None):
    """
    Create sequences from a cluster of stocks for training.
    
    Args:
        cluster_data: Dictionary mapping tickers to dataframes
        seq_length: Length of input sequences
        pred_steps: Number of steps to predict
        include_tickers: List of tickers to include (None = all)
        
    Returns:
        X, y data suitable for the model
    """
    all_X = []
    all_y = []
    all_raw_prices = []
    all_tickers = []
    
    # Filter tickers if specified
    tickers = include_tickers if include_tickers else list(cluster_data.keys())
    
    for ticker in tickers:
        if ticker not in cluster_data:
            print(f"Warning: {ticker} not found in cluster data")
            continue
            
        df = cluster_data[ticker]
        
        # Skip tickers with insufficient data
        if len(df) < seq_length + pred_steps:
            print(f"Warning: {ticker} has insufficient data ({len(df)} points)")
            continue
            
        # Scale the data
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Extract features and target
        features = df.drop(columns=['Close', 'Ticker'])
        target = df['Close'].values.reshape(-1, 1)
        
        # Scale data
        X_scaled = feature_scaler.fit_transform(features)
        y_scaled = target_scaler.fit_transform(target)
        
        # Create sequences
        X_seq, y_seq, raw_seq = create_sequences(
            X_scaled, y_scaled, raw_prices=target, 
            seq_length=seq_length, augment=False, balance_directions=False
        )
        
        # Create ticker info array (same ticker for all sequences)
        ticker_info = np.array([ticker] * len(X_seq))
        
        # Append to overall arrays
        all_X.append(X_seq)
        all_y.append(y_seq)
        all_raw_prices.append(raw_seq)
        all_tickers.append(ticker_info)
    
    # Combine all data
    X = np.concatenate(all_X, axis=0) if all_X else np.array([])
    y = np.concatenate(all_y, axis=0) if all_y else np.array([])
    raw_prices = np.concatenate(all_raw_prices, axis=0) if all_raw_prices else np.array([])
    tickers = np.concatenate(all_tickers, axis=0) if all_tickers else np.array([])
    
    print(f"Created {len(X)} sequences from {len(tickers)} stocks")
    return X, y, raw_prices, tickers



def train_universal_model(model, train_loader, val_loader, target_scaler, epochs=100):
    device = xm.xla_device()
    print(f"Using device: {device}")
    model.to(device)
    
    # Wrap with parallel loader for TPU
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)
    
    # Loss functions
    criterion_price = nn.HuberLoss(delta=1.0)
    pos_weight = torch.tensor([2.0]).to(device)
    criterion_direction = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_volatility = nn.MSELoss()
    
    # Optimizer with warmup learning rate
    optimizer = AdamW(
        model.parameters(), 
        lr=1e-5,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # Learning rate scheduler
    def lr_schedule(epoch):
        if epoch < 5:
            # Warmup phase - linear increase from 0.1x to 1x
            return 0.1 + 0.9 * epoch / 5
        else:
            # Decay phase - cosine decay
            return 0.1 + 0.9 * (1 + math.cos(math.pi * (epoch - 5) / (epochs - 5))) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Training variables
    best_val_loss = float('inf')
    patience = 8
    min_delta = 1e-4
    waiting = 0
    GRAD_CLIP_VALUE = 0.5
    
    regime_history = []  # Track market regime predictions
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        # Curriculum learning - focus first on price, then direction
        dir_weight = min(1.0 + epoch / 5, 5.0)
        vol_weight = min(0.2 + epoch / 20, 0.5)
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, (X_batch, y_batch, ticker_idx) in enumerate(train_loader):
            if batch_idx == 0 and epoch == 0:
                print(f"X_batch shape: {X_batch.shape}")
                print(f"ticker_idx shape: {ticker_idx.shape}")
                for k, v in y_batch.items():
                    print(f"{k} shape: {v.shape}")
            
            X_batch = X_batch.to(device)
            ticker_idx = ticker_idx.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            
            try:
                # Forward pass
                results, regime, signals = model(X_batch, ticker_idx)
                
                # Ensure results is properly formatted
                if not isinstance(results, dict) or 'price' not in results:
                    if isinstance(results, torch.Tensor):
                        results = {
                            'price': results,
                            'direction_logits': results,
                            'volatility': F.softplus(results)
                        }
                
                # Calculate losses
                price_loss = criterion_price(results['price'], y_batch['price'])
                
                if 'direction_logits' in results:
                    direction_loss = criterion_direction(results['direction_logits'], y_batch['direction'])
                else:
                    direction = results['direction'].to(device)
                    direction_logits = torch.log(direction / (1 - direction + 1e-7))
                    direction_loss = criterion_direction(direction_logits, y_batch['direction'])
                    
                volatility_loss = criterion_volatility(results['volatility'], y_batch['volatility'])
                
                # Combined loss
                loss = price_loss + dir_weight * direction_loss + vol_weight * volatility_loss

                # Regime balance regularization
                regime_target = torch.ones(regime.shape[1], device=device) / regime.shape[1]
                regime_balance_loss = F.kl_div(
                    F.log_softmax(regime.mean(dim=0, keepdim=True), dim=1),
                    regime_target.unsqueeze(0),
                    reduction='batchmean'
                )
                loss = loss + 0.1 * regime_balance_loss
                
                # Diversity regularization
                if epoch > 5:
                    regime = regime.to(device)
                    signals = signals.to(device)
                    regime_entropy = -(regime * torch.log(regime + 1e-8)).sum(dim=1).mean()
                    signals_entropy = -(signals * torch.log(signals + 1e-8)).sum(dim=1).mean()
                    loss = loss - 0.01 * (regime_entropy + signals_entropy)
                    
                loss = loss / 2  # Scale for gradient accumulation
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
                
            # Backward pass
            loss.backward()
            
            # Update weights every 2 batches
            if (batch_idx + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_VALUE)
                optimizer.step()
                optimizer.zero_grad()
                xm.mark_step()  # Critical for TPU execution
            
            train_loss += loss.item() * 2
            num_batches += 1
        
        # Handle remaining gradients
        if (batch_idx + 1) % 2 != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_VALUE)
            optimizer.step()
            optimizer.zero_grad()
            xm.mark_step()
        
        # Evaluation
        model.eval()
        val_loss = 0
        all_price_outputs = []
        all_direction_outputs = []
        all_volatility_outputs = []
        all_price_targets = []
        all_direction_targets = []
        all_volatility_targets = []
        all_regimes = []
        all_signals = []
        all_ticker_indices = []
                
        with torch.no_grad():
            for X_val, y_val, ticker_idx in val_loader:
                X_val = X_val.to(device)
                ticker_idx = ticker_idx.to(device)
                y_val = {k: v.to(device) for k, v in y_val.items()}
                
                # Forward pass
                results, regime, signals = model(X_val, ticker_idx)
                
                # Calculate losses
                val_loss += criterion_price(results['price'], y_val['price']).item()
                
                if 'direction_logits' in results:
                    val_loss += criterion_direction(results['direction_logits'], y_val['direction']).item()
                else:
                    direction_logits = torch.log(results['direction'] / (1 - results['direction'] + 1e-7))
                    val_loss += criterion_direction(direction_logits, y_val['direction']).item()
                    
                val_loss += criterion_volatility(results['volatility'], y_val['volatility']).item()
                
                # Collect results
                all_price_outputs.append(results['price'].cpu())
                all_direction_outputs.append(results['direction'].cpu() if 'direction' in results else 
                                            torch.sigmoid(results['direction_logits']).cpu())
                all_volatility_outputs.append(results['volatility'].cpu())
                
                all_price_targets.append(y_val['price'].cpu())
                all_direction_targets.append(y_val['direction'].cpu())
                all_volatility_targets.append(y_val['volatility'].cpu())
                
                all_regimes.append(regime.cpu())
                all_signals.append(signals.cpu())
                all_ticker_indices.append(ticker_idx.cpu())
                
                xm.mark_step()
        
        # Combine results
        all_price_outputs = torch.cat(all_price_outputs, dim=0)
        all_direction_outputs = torch.cat(all_direction_outputs, dim=0)
        all_volatility_outputs = torch.cat(all_volatility_outputs, dim=0)
        
        all_price_targets = torch.cat(all_price_targets, dim=0)
        all_direction_targets = torch.cat(all_direction_targets, dim=0)
        all_volatility_targets = torch.cat(all_volatility_targets, dim=0)
        
        all_regimes = torch.cat(all_regimes, dim=0)
        all_signals = torch.cat(all_signals, dim=0)
        all_ticker_indices = torch.cat(all_ticker_indices, dim=0)
        
        # Calculate overall metrics
        price_mape = torch.mean(torch.abs((all_price_targets - all_price_outputs) / (all_price_targets + 1e-8))) * 100
        price_rmse = torch.sqrt(torch.mean((all_price_targets - all_price_outputs) ** 2))
        price_mae = torch.mean(torch.abs(all_price_targets - all_price_outputs))
        
        # Direction metrics
        direction_true_percent = torch.mean(all_direction_targets) * 100
        print(f"Direction targets distribution: {direction_true_percent:.2f}% UP")
        directional_accuracy = 0.0

        # Calculate directional accuracy
        if all_price_targets.shape[1] > 1:
            diff_pred = all_price_outputs[:, 1:] - all_price_outputs[:, :-1]
            diff_true = all_price_targets[:, 1:] - all_price_targets[:, :-1]
            
            binary_pred_up = (diff_pred > 0).float()
            binary_true_up = (diff_true > 0).float()
            
            directional_accuracy = torch.mean((binary_pred_up == binary_true_up).float()) * 100
            print(f"Overall directional accuracy: {directional_accuracy:.2f}%")
            
            # Per-step accuracies
            for step in range(binary_pred_up.shape[1]):
                step_accuracy = torch.mean((binary_pred_up[:, step] == binary_true_up[:, step]).float()) * 100
                print(f"  Step {step+1} dir accuracy: {step_accuracy:.2f}%")
        else:
            directional_accuracy = torch.mean((all_direction_outputs > 0.5).float() == all_direction_targets) * 100
        
        # Calculate per-stock metrics
        unique_tickers = torch.unique(all_ticker_indices)
        print("\nPer-stock performance:")
        for ticker_idx in unique_tickers:
            idx = ticker_idx.item()
            mask = (all_ticker_indices == idx)
            
            # Skip if no samples
            if not mask.any():
                continue
                
            ticker_outputs = all_price_outputs[mask]
            ticker_targets = all_price_targets[mask]
            
            ticker_mape = torch.mean(torch.abs((ticker_targets - ticker_outputs) / (ticker_targets + 1e-8))) * 100
            ticker_rmse = torch.sqrt(torch.mean((ticker_targets - ticker_outputs) ** 2))
            
            print(f"  Stock {idx}: MAPE={ticker_mape:.2f}%, RMSE={ticker_rmse:.4f}")
        
        # Signal and regime distributions
        signal_distribution = all_signals.mean(dim=0)
        regime_distribution = all_regimes.mean(dim=0)
        
        # Store regime history
        regime_history.append(regime_distribution.numpy())
        
        # Finalize epoch
        train_loss /= num_batches
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'MAPE: {price_mape:.2f}%, RMSE: {price_rmse:.4f}, MAE: {price_mae:.4f}, '
              f'Directional Accuracy: {directional_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
        print(f'Regime Distribution: {regime_distribution.numpy()}')
        print(f'Signal Distribution: {signal_distribution.numpy()}')

        # Early stopping check
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            waiting = 0
            print("Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mape': price_mape,
                'rmse': price_rmse,
                'mae': price_mae,
                'directional_accuracy': directional_accuracy,
                'regime_distribution': regime_distribution,
                'signal_distribution': signal_distribution,
                'regime_history': regime_history,
            }, 'universal_stock_model_checkpoint.pth')
            
            xm.mark_step()
        else:
            waiting += 1
            if waiting >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

    return model
def main():
    # Define a set of stocks for the universal model
    tickers = [
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'AMZN',  # Amazon
        'GOOGL', # Google/Alphabet
        'META',  # Meta/Facebook
        'TSLA',  # Tesla
        'NVDA',  # NVIDIA
        'JPM',   # JPMorgan Chase
        'V',     # Visa
        'WMT'    # Walmart
    ]
    
    print(f"Building universal model for {len(tickers)} stocks: {', '.join(tickers)}")
    
    # Prepare data for all stocks - note this might take a while
    start_date = '2018-01-01'  # Using a more recent start date to keep data size manageable
    end_date = '2023-12-31'
    
    # Get data for all stocks
    stock_data_dict = prepare_multi_stock_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        seq_length=SEQ_LENGTH,
        balance_directions=True
    )
    
    # Check how many stocks we have usable data for
    print(f"Successfully processed data for {len(stock_data_dict)} stocks")
    if len(stock_data_dict) == 0:
        print("No usable stock data. Exiting.")
        return
    
    # Get the feature dimension from the first stock
    first_ticker = list(stock_data_dict.keys())[0]
    feature_dim = stock_data_dict[first_ticker]['X'].shape[2]
    
    # Create train/validation split
    train_data, val_data = create_multi_stock_train_val_split(
        stock_data_dict, 
        val_ratio=0.15,
        time_based=True  # Use most recent data for validation
    )
    
    # Create data loaders
    train_loader, val_loader = create_universal_dataloaders(
        train_data, 
        val_data, 
        batch_size=BATCH_SIZE
    )
    
    # Initialize the universal model
    model = UniversalStockModel(
        input_size=feature_dim,
        num_stocks=len(stock_data_dict),
        embedding_dim=32,
        hidden_size=16,
        matrix_size=4,
        dropout=0.3,
        use_static_features=True
    )
    
    # Get device
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Get a sample batch to inspect
    X_sample, targets_sample, ticker_idx_sample = next(iter(train_loader))
    print(f"X shape: {X_sample.shape}")
    print(f"Target keys: {list(targets_sample.keys())}")
    print(f"Ticker indices shape: {ticker_idx_sample.shape}")
    
    # Ticker mapping for reference
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(stock_data_dict.keys())}
    idx_to_ticker = {idx: ticker for ticker, idx in ticker_to_idx.items()}
    print("Ticker mapping:")
    for idx, ticker in idx_to_ticker.items():
        print(f"  {idx}: {ticker}")
    
    # Forward pass to test model
    try:
        # Move to device
        X_sample = X_sample.to(device)
        ticker_idx_sample = ticker_idx_sample.to(device)
        
        # Forward pass
        with torch.no_grad():
            model = model.to(device)
            results, regime, signals = model(X_sample, ticker_idx_sample)
        
        print("Model test forward pass successful!")
        print(f"Results keys: {list(results.keys())}")
        print(f"Regime shape: {regime.shape}")
        print(f"Signals shape: {signals.shape}")
    except Exception as e:
        print(f"Error during model test: {e}")
        return
    
    # Get a reference scaler for output conversion
    target_scaler = stock_data_dict[first_ticker]['target_scaler']
    
    # Train the model
    trained_model = train_universal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_scaler=target_scaler,
        epochs=EPOCHS
    )
    
    print("Training complete!")
    
    # Save the model and ticker mapping
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'ticker_to_idx': ticker_to_idx,
        'idx_to_ticker': idx_to_ticker
    }, 'universal_stock_model.pth')
    
    print("Model saved to universal_stock_model.pth")

if __name__ == "__main__":
    main()
