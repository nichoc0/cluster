import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import logging
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define stock cluster
TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC']
# Define date ranges
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'
HOURLY_LOOKBACK_DAYS = 730  # ~2 years of hourly data (Yahoo limit is ~2 years)

# Define paths
DATA_DIR = Path("data")
PRICE_DIR = DATA_DIR / "price"
OPTIONS_DIR = DATA_DIR / "options"

def ensure_directories():
    """Create all necessary directories"""
    for directory in [DATA_DIR, PRICE_DIR, OPTIONS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)

def safe_fetch(ticker, start, end, interval='1h', retries=15, delay=10):
    """Fetch data with exponential backoff for rate limiting"""
    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                start=start if isinstance(start, str) else start.strftime("%Y-%m-%d"), 
                end=end if isinstance(end, str) else end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            if data is not None and not data.empty:
                return data
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
        
        # Exponential backoff
        sleep_time = delay * (2 ** attempt)
        logger.info(f"Sleeping for {sleep_time} seconds before next attempt...")
        time.sleep(sleep_time)
    
    logger.error(f"Failed to fetch data for {ticker} after {retries} attempts")
    return pd.DataFrame()

def fetch_hourly_chunks(ticker, start_date, end_date):
    """Fetch hourly data in chunks to avoid timeouts"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Check if requested date range is within Yahoo's limit (730 days)
    today = datetime.now()
    max_lookback = today - timedelta(days=730)
    
    if start < max_lookback:
        logger.warning(f"Yahoo Finance only provides hourly data for the last 730 days. Adjusting start date from {start.date()} to {max_lookback.date()}")
        start = max_lookback
    
    if start >= end:
        logger.warning(f"No valid date range for hourly data after adjusting for Yahoo's 730-day limit")
        return pd.DataFrame()
    
    # Use smaller chunks for more reliability (60 days)
    chunk_size = timedelta(days=60)
    chunks = []
    
    current_start = start
    while current_start < end:
        current_end = min(current_start + chunk_size, end)
        logger.info(f"Fetching {ticker} hourly data from {current_start.date()} to {current_end.date()}")
        
        try:
            # Use a more conservative retry strategy for hourly data
            for attempt in range(3):
                try:

                    chunk = yf.download(
                        ticker,
                        start=current_start.strftime("%Y-%m-%d"),
                        end=current_end.strftime("%Y-%m-%d"),
                        interval='1h',
                        auto_adjust=True,
                        progress=False
                    )
                    
                    if not chunk.empty:
                        chunks.append(chunk)
                        break
                except Exception as e:
                    logger.warning(f"Hourly chunk attempt {attempt+1} failed: {e}")
                    time.sleep(5 * (attempt + 1))  # Increasing delay
            
            if not chunk.empty:
                logger.info(f"Successfully fetched {len(chunk)} rows of hourly data")
            else:
                logger.warning(f"Empty chunk for period {current_start.date()} to {current_end.date()}")
                
        except Exception as e:
            logger.error(f"Error fetching hourly chunk for {ticker}: {e}")
        
        # Add delay between chunks
        time.sleep(10)
        current_start = current_end
    
    if chunks:
        combined = pd.concat(chunks)
        logger.info(f"Combined {len(chunks)} chunks into {len(combined)} rows of hourly data")
        return combined
    
    logger.warning(f"No hourly data retrieved for {ticker}")
    return pd.DataFrame()

def fetch_options_data(ticker, lookback_days=60, forward_days=365):
    """
    Fetch options data for a ticker for multiple expiration dates
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Only consider options with expirations after this many days in the past
        forward_days: Only consider options with expirations within this many days in the future
        
    Returns:
        Dictionary mapping expiration dates to DataFrames with calls and puts
    """
    logger.info(f"Fetching options data for {ticker}")
    options_data = {}
    
    # Create ticker object
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get available expiration dates
        try:
            expiration_dates = ticker_obj.options
            
            if not expiration_dates:
                logger.warning(f"No options data available for {ticker}")
                return options_data
                
            logger.info(f"Found {len(expiration_dates)} expiration dates for {ticker}")
            
            # Filter dates based on lookback and forward days criteria
            today = datetime.now().date()
            min_date = today - timedelta(days=lookback_days)
            max_date = today + timedelta(days=forward_days)
            
            filtered_dates = []
            for date_str in expiration_dates:
                exp_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if min_date <= exp_date <= max_date:
                    filtered_dates.append(date_str)
            
            logger.info(f"Filtered to {len(filtered_dates)} expiration dates for {ticker}")
            
            # Fetch option chain for each date
            for exp_date in filtered_dates:
                try:
                    logger.info(f"Fetching {ticker} options for expiration {exp_date}")
                    option_chain = ticker_obj.option_chain(exp_date)
                    
                    # Save calls and puts
                    options_data[exp_date] = {
                        'calls': option_chain.calls,
                        'puts': option_chain.puts
                    }
                    
                    # Add delay between requests
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching {ticker} options for {exp_date}: {e}")
            
        except Exception as e:
            logger.error(f"Error getting expiration dates for {ticker}: {e}")
            
    except Exception as e:
        logger.error(f"Error creating ticker object for {ticker}: {e}")
    
    return options_data

def save_options_data(ticker, options_data):
    """Save options data to files"""
    if not options_data:
        logger.warning(f"No options data to save for {ticker}")
        return
        
    # Create ticker directory
    ticker_dir = OPTIONS_DIR / ticker
    ticker_dir.mkdir(exist_ok=True)
    
    # Save data for each expiration date
    for exp_date, data in options_data.items():
        try:
            # Save calls
            calls_file = ticker_dir / f"calls_{exp_date}.csv"
            data['calls'].to_csv(calls_file)
            
            # Save puts
            puts_file = ticker_dir / f"puts_{exp_date}.csv"
            data['puts'].to_csv(puts_file)
            
            logger.info(f"Saved {ticker} options for {exp_date}")
        except Exception as e:
            logger.error(f"Error saving {ticker} options for {exp_date}: {e}")

def load_options_data(ticker):
    """Load options data from files"""
    ticker_dir = OPTIONS_DIR / ticker
    
    if not ticker_dir.exists():
        logger.warning(f"No options directory for {ticker}")
        return {}
        
    options_data = {}
    
    # Get all CSV files in the ticker directory
    for file_path in ticker_dir.glob("*.csv"):
        try:
            file_name = file_path.name
            
            # Parse filename to get type and expiration date
            if file_name.startswith("calls_"):
                option_type = "calls"
                exp_date = file_name[6:-4]  # Remove "calls_" and ".csv"
            elif file_name.startswith("puts_"):
                option_type = "puts"
                exp_date = file_name[5:-4]  # Remove "puts_" and ".csv"
            else:
                continue
                
            # Load data
            data = pd.read_csv(file_path, index_col=0)
            
            # Initialize expiration date entry if needed
            if exp_date not in options_data:
                options_data[exp_date] = {}
                
            # Store data
            options_data[exp_date][option_type] = data
            
        except Exception as e:
            logger.error(f"Error loading options data from {file_path}: {e}")
    
    return options_data

def extract_options_features(ticker, options_data):
    """
    Extract useful features from options data for machine learning
    
    Returns:
        DataFrame with options-derived features (IV skew, put/call ratio, etc.)
    """
    if not options_data:
        logger.warning(f"No options data to extract features from for {ticker}")
        return pd.DataFrame()
        
    # Initialize lists to store data
    dates = []
    features = []
    
    # Process each expiration date
    for exp_date, data in options_data.items():
        try:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            
            # Skip if missing data
            if 'calls' not in data or 'puts' not in data:
                continue
                
            calls = data['calls']
            puts = data['puts']
            
            # Calculate days to expiration
            dte = (exp_datetime - datetime.now()).days
            
            # Basic features
            put_call_volume_ratio = puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0
            put_call_oi_ratio = puts['openInterest'].sum() / calls['openInterest'].sum() if calls['openInterest'].sum() > 0 else 0
            
            # Filter for ATM options (within 5% of current price)
            current_stock_price = None
            
            # Yahoo Finance option chains have 'lastPrice' for underlying stock
            if 'underlyingPrice' in calls.columns:
                current_stock_price = calls['underlyingPrice'].iloc[0]
            elif 'underlyingPrice' in puts.columns:
                current_stock_price = puts['underlyingPrice'].iloc[0]
            else:
                # Fallback: try to get from current stock data
                try:
                    current_stock_data = yf.Ticker(ticker).history(period="1d")
                    if not current_stock_data.empty:
                        current_stock_price = current_stock_data['Close'].iloc[0]
                except Exception as e:
                    logger.warning(f"Could not get current price for {ticker}: {e}")
            
            # Only proceed if we have a valid stock price
            if current_stock_price and current_stock_price > 0:
                # Use the correct price to find ATM options
                atm_calls = calls[(calls['strike'] >= 0.95 * current_stock_price) & 
                                 (calls['strike'] <= 1.05 * current_stock_price)]
                atm_puts = puts[(puts['strike'] >= 0.95 * current_stock_price) & 
                              (puts['strike'] <= 1.05 * current_stock_price)]
                
                # Add the underlying price to features

                # IV calculations (if available)
                call_iv = atm_calls['impliedVolatility'].mean() if 'impliedVolatility' in atm_calls.columns else 0
                put_iv = atm_puts['impliedVolatility'].mean() if 'impliedVolatility' in atm_puts.columns else 0
                iv_skew = put_iv - call_iv
                
                # Store date and features
                dates.append(exp_datetime)
                features.append({
                    'dte': dte,
                    'underlying_price': current_stock_price,  # Add this
                    'put_call_volume_ratio': put_call_volume_ratio,
                    'put_call_oi_ratio': put_call_oi_ratio,
                    'call_iv': call_iv,
                    'put_iv': put_iv,
                    'iv_skew': iv_skew
                })
            
        except Exception as e:
            logger.error(f"Error extracting features for {ticker} options on {exp_date}: {e}")
    
    # Create DataFrame if we have data
    if features:
        df = pd.DataFrame(features, index=dates)
        return df
    else:
        return pd.DataFrame()

def find_fung_hsieh_file():
    """Find the Fung-Hsieh factors file in various possible locations"""
    # Possible locations to check
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # Gets directory where the script is located
    
    possible_locations = [
        DATA_DIR / "factors" / "fung_hsieh_factors.csv",
        Path("fung_hsieh_factors.csv"),
        DATA_DIR / "fung_hsieh_factors.csv",
        Path("data") / "fung_hsieh_factors.csv",
        DATA_DIR / "factors" / "TF-Fac.xls",
        Path("TF-Fac.xls"),
        # Add these additional paths relative to the script location
        script_dir / "data" / "factors" / "fung_hsieh_factors.csv",
        script_dir / "fung_hsieh_factors.csv",
        script_dir / "data" / "fung_hsieh_factors.csv",
        script_dir / "TF-Fac.xls",
        # Check in cluster directory explicitly
        Path("cluster") / "data" / "factors" / "fung_hsieh_factors.csv",
        Path("cluster") / "fung_hsieh_factors.csv"
    ]
    
    # Check each location
    for location in possible_locations:
        logger.info(f"Checking for Fung-Hsieh file at: {location.absolute()}")
        if location.exists():
            logger.info(f"Found Fung-Hsieh factors file at: {location}")
            return location
    
    logger.warning("Could not find Fung-Hsieh factors file in any expected location")
    return None
def merge_options_with_stock_data(stock_data, options_features):
    """
    Merge options features with stock price data
    """
    if options_features is None or options_features.empty or stock_data is None:
        logger.warning("Cannot merge options: missing data")
        return stock_data
    
    # Clone the stock data
    merged_data = stock_data.copy()
    
    # Ensure the index is correctly formatted - FIX FOR THE ERROR
    try:
        # Check if the index has "Ticker" which indicates a problem
        if isinstance(merged_data.index, pd.Index) and any('ticker' in str(x).lower() for x in merged_data.index):
            logger.warning("Found 'Ticker' in index - fixing data format")
            # The CSV might have been saved with wrong headers - try to fix
            # Assume the first column should be the date
            merged_data = pd.read_csv(daily_file, header=0)
            # Use first column as index
            if len(merged_data.columns) > 1:
                merged_data = merged_data.set_index(merged_data.columns[0])
    
        # Now convert to datetime safely
        if not isinstance(merged_data.index, pd.DatetimeIndex):
            try:
                merged_data.index = pd.to_datetime(merged_data.index, errors='coerce')
                # Drop rows with NaT (not a time) values that couldn't be converted
                merged_data = merged_data.loc[~merged_data.index.isna()]
            except Exception as e:
                logger.error(f"Error converting index to datetime: {e}")
                return stock_data
    except Exception as e:
        logger.error(f"Error preprocessing data before merge: {e}")
        return stock_data
    
    # Clone the stock data
    # merged_data = stock_data.copy()   
    
    # Convert stock data index to datetime if needed
    if not isinstance(merged_data.index, pd.DatetimeIndex):
        merged_data.index = pd.to_datetime(merged_data.index)
    
    # For each stock data date, find the nearest option expiration
    for stock_date in merged_data.index:
        # Find options expiring after this date
        future_options = options_features[options_features.index > stock_date]
        
        if not future_options.empty:
            # Get the nearest expiration
            nearest_exp = future_options.index.min()
            nearest_features = future_options.loc[nearest_exp]
            
            # Add options features to the stock data
            for feature, value in nearest_features.items():
                merged_data.loc[stock_date, f'option_{feature}'] = value
    
    return merged_data
def fetch_stock_data(ticker, start_date=START_DATE, end_date=END_DATE, fetch_options=True):
    """Fetch both daily and hourly data for a ticker"""
    ensure_directories()
    
    # File paths
    daily_file = PRICE_DIR / f"{ticker}_daily_{start_date}_{end_date}.csv"
    
    # 1. Fetch daily data
    try:
        if daily_file.exists():
            logger.info(f"Loading {ticker} daily data from file")
            daily_data = pd.read_csv(daily_file, index_col=0, parse_dates=True)
        else:
            logger.info(f"Downloading {ticker} daily data from {start_date} to {end_date}")
            daily_data = safe_fetch(ticker, start_date, end_date, interval='1d')
            if not daily_data.empty:
                daily_data.to_csv(daily_file)
                logger.info(f"Saved {ticker} daily data ({len(daily_data)} rows)")
            else:
                logger.error(f"No daily data found for {ticker}")
                return False
    except Exception as e:
        logger.error(f"Error fetching daily data for {ticker}: {e}")
        return False
    
    # 2. Fetch hourly data (using current date as reference)
    try:
        # Calculate hourly data range based on Yahoo's 730-day limit
        today = datetime.now()
        hourly_start_dt = today - timedelta(days=720)  # Leave some margin
        hourly_end_dt = today
        
        hourly_start = hourly_start_dt.strftime('%Y-%m-%d')
        hourly_end = hourly_end_dt.strftime('%Y-%m-%d')
        
        hourly_file = PRICE_DIR / f"{ticker}_hourly_recent.csv"
        
        if hourly_file.exists():
            # Check if we need to update hourly data (if file is older than 1 day)
            file_age = (datetime.now() - datetime.fromtimestamp(hourly_file.stat().st_mtime)).days
            
            if file_age > 1:
                logger.info(f"Hourly data is {file_age} days old - refreshing")
                needs_refresh = True
            else:
                logger.info(f"Loading recent {ticker} hourly data from file")
                hourly_data = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
                needs_refresh = False
        else:
            needs_refresh = True
        
        if needs_refresh:
            logger.info(f"Downloading {ticker} recent hourly data from {hourly_start} to {hourly_end}")
            hourly_data = fetch_hourly_chunks(ticker, hourly_start, hourly_end)
            if not hourly_data.empty:
                hourly_data.to_csv(hourly_file)
                logger.info(f"Saved {ticker} recent hourly data ({len(hourly_data)} rows)")
            else:
                logger.warning(f"No hourly data found for {ticker}")
    except Exception as e:
        logger.error(f"Error fetching hourly data for {ticker}: {e}")
    

    # 3. Fetch options data if requested
    if fetch_options:
        try:
            # Check if we already have options data
            existing_options = load_options_data(ticker)
            
            if existing_options:
                logger.info(f"Found existing options data for {ticker}")
                options_data = existing_options
                
                # Check if we need to refresh (get new expirations)
                oldest_exp = min([datetime.strptime(date, '%Y-%m-%d').date() for date in existing_options.keys()])
                if oldest_exp < datetime.now().date():
                    logger.info(f"Refreshing options data for {ticker}")
                    new_options = fetch_options_data(ticker)
                    
                    # Merge with existing data
                    for exp_date, data in new_options.items():
                        options_data[exp_date] = data
                    
                    # Save updated data
                    save_options_data(ticker, new_options)
            else:
                # Fetch all options data
                logger.info(f"Fetching new options data for {ticker}")
                options_data = fetch_options_data(ticker)
                save_options_data(ticker, options_data)
            
            # Extract features from options data
            options_features = extract_options_features(ticker, options_data)
            
            if not options_features.empty:
                # Save features
                features_file = OPTIONS_DIR / f"{ticker}_options_features.csv"
                options_features.to_csv(features_file)
                logger.info(f"Saved {ticker} options features ({len(options_features)} rows)")
            
        except Exception as e:
            logger.error(f"Error processing options data for {ticker}: {e}")
    # Add to your fetch_stock_data function after merging factor data
    # Merge options features with daily price data
    #try:
     #   options_file = OPTIONS_DIR / f"{ticker}_options_features.csv"
      #  if options_file.exists():
       #     options_features = pd.read_csv(options_file, index_col=0, parse_dates=True)
        #    
         #   # Only merge if we have both daily data and options features
          #  # Only merge if we have both daily data and options features
           # if not options_features.empty:
            #    daily_with_options = merge_options_with_stock_data(
             #       daily_data,  # Use the available daily_data 
              #      options_features
               # )
                ## Save the merged data
                #full_merged_file = PRICE_DIR / f"{ticker}_daily_with_options_{START_DATE}_{END_DATE}.csv"
                #daily_with_options.to_csv(full_merged_file)
                #logger.info(f"Saved {ticker} daily data with options ({len(daily_with_options)} rows)")
   # except Exception as e:
    #        logger.error(f"Error processing options data for {ticker}: {e}")
    logger.info(f"Completed data fetch for {ticker}")
    return True

def fetch_fundamental_data(ticker):
    """Fetch fundamental data like balance sheet, income statement, etc."""
    logger.info(f"Fetching fundamental data for {ticker}")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Create directory
        fundamental_dir = DATA_DIR / "fundamentals" / ticker
        fundamental_dir.mkdir(exist_ok=True, parents=True)
        
        # Get various fundamental data
        try:
            # Balance sheet
            balance_sheet = ticker_obj.balance_sheet
            if not balance_sheet.empty:
                balance_sheet.to_csv(fundamental_dir / "balance_sheet.csv")
                logger.info(f"Saved {ticker} balance sheet data")
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            
        try:
            # Income statement
            income_stmt = ticker_obj.income_stmt
            if not income_stmt.empty:
                income_stmt.to_csv(fundamental_dir / "income_stmt.csv")
                logger.info(f"Saved {ticker} income statement data")
        except Exception as e:
            logger.error(f"Error fetching income statement for {ticker}: {e}")
            
        try:
            # Cash flow
            cash_flow = ticker_obj.cashflow
            if not cash_flow.empty:
                cash_flow.to_csv(fundamental_dir / "cash_flow.csv")
                logger.info(f"Saved {ticker} cash flow data")
        except Exception as e:
            logger.error(f"Error fetching cash flow for {ticker}: {e}")
            
        try:
            # Quarterly financials
            quarterly = ticker_obj.quarterly_financials
            if not quarterly.empty:
                quarterly.to_csv(fundamental_dir / "quarterly_financials.csv")
                logger.info(f"Saved {ticker} quarterly financials")
        except Exception as e:
            logger.error(f"Error fetching quarterly financials for {ticker}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fetching fundamental data for {ticker}: {e}")
        return False


def fetch_factor_data(start_date=START_DATE, end_date=END_DATE):
    """
    Fetch and process both Fung-Hsieh 7-factor model data and Fama-French factors
    """
    logger.info("Fetching factor model data")
    
    # Create factor directory
    factor_dir = DATA_DIR / "factors"
    factor_dir.mkdir(exist_ok=True)
    
    # File paths
    fh_file = factor_dir / "fung_hsieh_factors.csv"
    ff_file = factor_dir / "fama_french_factors.csv"
    combined_file = factor_dir / "combined_factors.csv"
    
    fh_data = None
    ff_data = None
    
    # 1. Fung-Hsieh factors
    try:
        if fh_file.exists():
            logger.info("Loading existing Fung-Hsieh factor data")
            fh_data = pd.read_csv(fh_file, index_col=0, parse_dates=True)
        else:
            # Try to find the file
            found_file = find_fung_hsieh_file()
            
            if found_file is not None:
                logger.info(f"Processing Fung-Hsieh data from {found_file}")
                
                # Check if it's an Excel file
                if found_file.suffix.lower() == '.xls' or found_file.suffix.lower() == '.xlsx':
                    try:
                        # Read Excel file
                        fh_data = pd.read_excel(found_file, skiprows=1)
                        # Process columns
                        fh_data.columns = ['date', 'ptfsbd', 'ptfsfx', 'ptfscom', 
                                          'ptfsir', 'ptfsstk', 'snpmrf', 'scmlc']
                        # Convert date
                        fh_data['date'] = pd.to_datetime(fh_data['date'], format='%Y%m')
                        fh_data = fh_data.set_index('date')
                    except Exception as e:
                        logger.error(f"Error processing Excel file: {e}")
                else:
                    # Read CSV file
                    try:
                        # The file has comments and headers at the top - need to skip them
                        # Try different numbers of rows to skip until we find the actual data
                        for skiprows in range(20):  # Try up to 20 rows to skip
                            try:
                                temp_df = pd.read_csv(found_file, skiprows=skiprows)
                                # Check if this looks like the data we want (has yyyymm column)
                                if any('yyyymm' in col.lower() for col in temp_df.columns):
                                    logger.info(f"Found data by skipping {skiprows} rows")
                                    fh_data = temp_df
                                    
                                    # Get the date column (likely named 'yyyymm')
                                    date_col = [col for col in fh_data.columns if 'yyyymm' in col.lower()][0]
                                    
                                    # Convert dates
                                    fh_data[date_col] = pd.to_datetime(fh_data[date_col], format='%Y%m')
                                    fh_data = fh_data.set_index(date_col)
                                    break
                            except Exception as e:
                                continue
                                
                        if fh_data is None:
                            logger.error("Could not process Fung-Hsieh CSV file - invalid format")
                    except Exception as e:
                        logger.error(f"Error processing CSV file: {e}")
                # Save to standard location
                if fh_data is not None:
                    fh_data.to_csv(fh_file)
                    logger.info(f"Saved processed Fung-Hsieh data ({len(fh_data)} rows)")
            else:
                # Display download instructions
                logger.info("Downloading Fung-Hsieh factor data")
                logger.info("Please download the Fung-Hsieh factor data manually:")
                logger.info("1. Visit: http://faculty.fuqua.duke.edu/~dah7/DataLibrary/TF-Fac.xls")
                logger.info(f"2. Save the file to: {factor_dir / 'TF-Fac.xls'}")
                logger.info("3. Run this script again to process the downloaded file")
    except Exception as e:
        logger.error(f"Error with Fung-Hsieh factors: {e}")
        

    
    # Fix for the Fama-French factors section in fetch_factor_data
    try:
        if ff_file.exists():
            logger.info("Loading existing Fama-French factor data")
            ff_data = pd.read_csv(ff_file, index_col=0, parse_dates=True)
        else:
            logger.info("Downloading Fama-French factor data")
            
            # French's data library URL for 5-factor model (includes momentum)
            ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
            
            # Create a temporary directory for the download
            import tempfile
            import zipfile
            import urllib.request
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_path = Path(tmpdirname) / "ff_data.zip"
                
                # Download the zip file
                urllib.request.urlretrieve(ff_url, zip_path)
                
                # Extract the contents
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdirname)
                
                # Find the CSV file (might need to adjust based on actual archive structure)
                csv_files = list(Path(tmpdirname).glob("*.CSV"))
                if csv_files:
                    # Read the first CSV file found (adjust as needed)
                    try:
                        # First attempt to read with standard format
                        ff_data = pd.read_csv(csv_files[0], skiprows=3)
                        
                        # Check if the data was loaded correctly - if not, try alternative format
                        if 'date' not in ff_data.columns and len(ff_data.columns) >= 1:
                            # Try a different approach for the file format
                            ff_data = pd.read_csv(csv_files[0], skiprows=3, header=None)
                            
                            # Rename columns based on expected structure
                            # First column is date, followed by factor returns
                            column_names = ['date', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']
                            if len(ff_data.columns) >= len(column_names):
                                ff_data.columns = column_names + [f'extra_{i}' for i in range(len(ff_data.columns) - len(column_names))]
                            else:
                                # Use available columns
                                ff_data.columns = column_names[:len(ff_data.columns)]
                        
                        # Clean column names if they exist
                        ff_data.columns = [str(col).strip().lower() for col in ff_data.columns]
                        
                        # Convert date format (usually in YYYYMM format)
                        if 'date' in ff_data.columns:
                            # Try to handle date conversion - may need adjustment based on actual format
                            try:
                                ff_data['date'] = pd.to_datetime(ff_data['date'].astype(str), format='%Y%m')
                            except ValueError:
                                # Alternative format handling (e.g., if dates are different format)
                                ff_data['date'] = pd.to_datetime(ff_data['date'].astype(str), errors='coerce')
                            
                            # Remove any rows with invalid dates
                            ff_data = ff_data.dropna(subset=['date'])
                            ff_data = ff_data.set_index('date')
                            
                            # Filter by date range
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date)
                            ff_data = ff_data[(ff_data.index >= start_dt) & (ff_data.index <= end_dt)]
                            
                            # Save processed data
                            ff_data.to_csv(ff_file)
                            logger.info(f"Saved processed Fama-French data ({len(ff_data)} rows)")
                        else:
                            logger.error("Could not find date column in Fama-French data")
                    except Exception as e:
                        logger.error(f"Error processing Fama-French data: {e}")
                else:
                    logger.error("No CSV files found in the Fama-French data archive")
    except Exception as e:
        logger.error(f"Error with Fama-French factors: {e}")
    
    # 3. Create combined factor dataset if both are available
    if fh_data is not None and ff_data is not None:
        try:
            logger.info("Creating combined factor dataset")
            
            # Ensure both datasets have the same frequency (monthly)
            # Resample if needed
            if fh_data.index.freq != ff_data.index.freq:
                # Determine the common frequency (likely monthly)
                # This is a simplification - might need adjustment
                fh_data = fh_data.resample('M').last()
                ff_data = ff_data.resample('M').last()
            
            # Merge the datasets on the date index
            combined_data = pd.merge(
                fh_data, 
                ff_data, 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
            
            # Fill missing values (if any)
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            
            # Save combined data
            combined_data.to_csv(combined_file)
            logger.info(f"Saved combined factor data ({len(combined_data)} rows)")
            
            return {
                'fung_hsieh': fh_data,
                'fama_french': ff_data,
                'combined': combined_data
            }
        except Exception as e:
            logger.error(f"Error creating combined factor dataset: {e}")
    
    # Return whatever data we have
    return {
        'fung_hsieh': fh_data,
        'fama_french': ff_data,
        'combined': None
    }


def load_csv_with_robust_parsing(file_path):
    """Load CSV with better handling of date indexes and headers"""
    try:
        # Try standard pandas read_csv first
        df = pd.read_csv(file_path)
        
        # Check if the file has a header row that should be skipped
        if df.columns[0].lower() in ['ticker', 'symbol', 'stock']:
            # This might be a header row - check if we can use the first column as dates
            try:
                # Try to convert the first column to datetime
                dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                if not dates.isna().all():
                    # First column can be dates - use it as index
                    df = df.set_index(df.columns[0])
                    df.index = pd.to_datetime(df.index, errors='coerce')
                else:
                    # First column isn't dates - use default 0-based index
                    df = df.reset_index(drop=True)
            except:
                # If conversion fails, just use default index
                df = df.reset_index(drop=True)
        else:
            # Try to find a date column to use as index
            date_cols = []
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'day', 'yyyymm']):
                    date_cols.append(col)
            
            if date_cols:
                # Use the first date column as index
                df = df.set_index(date_cols[0])
                df.index = pd.to_datetime(df.index, errors='coerce')
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV with robust parsing: {e}")
        return pd.DataFrame()
def check_for_manual_factor_files():
    """Check if user has manually downloaded factor data files"""
    factor_dir = DATA_DIR / "factors"
    factor_dir.mkdir(exist_ok=True)
    
    # Check for Fama-French data
    ff_manual = factor_dir / "FF_5_Factors.csv"
    if ff_manual.exists():
        logger.info("Found manually downloaded Fama-French data")
        try:
            ff_data = pd.read_csv(ff_manual)
            
            # Save to standard location with proper formatting
            ff_file = factor_dir / "fama_french_factors.csv"
            
            # Process columns and dates as needed
            # ... (format processing)
            
            ff_data.to_csv(ff_file)
            logger.info(f"Processed manually downloaded Fama-French data ({len(ff_data)} rows)")
            return True
        except Exception as e:
            logger.error(f"Error processing manual Fama-French data: {e}")
    
    # Instructions for manual download
    logger.info("\nFor more reliable data loading, you can manually download:")
    logger.info("1. Fama-French 5-Factor data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
    logger.info("   - Download the CSV and save as: data/factors/FF_5_Factors.csv")
    logger.info("2. Fung-Hsieh factor data: http://faculty.fuqua.duke.edu/~dah7/DataLibrary/TF-Fac.xls")
    logger.info("   - Save as: data/factors/TF-Fac.xls")
    
    return False

def merge_factors_with_stock_data(stock_data, factors_data, freq='D'):
    """
    Merge factor data with stock price data
    
    Args:
        stock_data: DataFrame with stock price data (datetime index)
        factors_data: DataFrame with factor data (datetime index)
        freq: Frequency to resample factors to ('D' for daily, 'H' for hourly)
        
    Returns:
        DataFrame with stock data and factor data merged
    """
    if factors_data is None or stock_data is None:
        logger.warning("Cannot merge factors: missing data")
        return stock_data
        
    # Clone the stock data to avoid modifying the original
    merged_data = stock_data.copy()
    
    # Convert stock data index to datetime if needed
    if not isinstance(merged_data.index, pd.DatetimeIndex):
        merged_data.index = pd.to_datetime(merged_data.index)
    
    # Resample factors to the desired frequency
    # For daily data, we forward-fill the monthly factors
    resampled_factors = factors_data.resample(freq).ffill()
    
    # Align the factor data to the stock data dates
    aligned_factors = pd.DataFrame(index=merged_data.index)
    
    for col in resampled_factors.columns:
        # Use reindex to align with stock data dates
        aligned_factors[col] = resampled_factors[col].reindex(
            aligned_factors.index, method='ffill'
        )
    
    # Merge with stock data
    for col in aligned_factors.columns:
        merged_data[f'factor_{col}'] = aligned_factors[col]
    
    return merged_data

def load_factor_data():
    """Load factor data from saved files"""
    factor_dir = DATA_DIR / "factors"
    
    # Initialize empty dict for results
    factors = {
        'fung_hsieh': None,
        'fama_french': None,
        'combined': None
    }
    
    # Try to load each file
    files_to_check = {
        'fung_hsieh': factor_dir / "fung_hsieh_factors.csv",
        'fama_french': factor_dir / "fama_french_factors.csv",
        'combined': factor_dir / "combined_factors.csv"
    }
    
    for key, file_path in files_to_check.items():
        if file_path.exists():
            try:
                factors[key] = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded {key} factor data ({len(factors[key])} rows)")
            except Exception as e:
                logger.error(f"Error loading {key} factor data: {e}")
    
    return factors

# In your main function
def main():
    """Main function to fetch data for all stocks"""
    logger.info(f"Starting data collection for {len(TECH_STOCKS)} stocks")
    
    successful = 0
    failed = []
    
    # Create data directories
    ensure_directories()
    
    # Fetch factor data 
    factor_data = fetch_factor_data(START_DATE, END_DATE)
    
    # Process sequentially to respect rate limits
    for ticker in TECH_STOCKS:
        logger.info(f"\n{'='*40}\nProcessing {ticker}\n{'='*40}")
        
        # Fetch price data
        if fetch_stock_data(ticker):
            # Also fetch fundamental data
            fetch_fundamental_data(ticker)
            
            # In the main function
            try:
                # First load daily data
                daily_file = PRICE_DIR / f"{ticker}_daily_{START_DATE}_{END_DATE}.csv"
                if daily_file.exists():
                    try:
                        # Try to load with automatic parsing
                        daily_data = load_csv_with_robust_parsing(daily_file)
                        
                        # Check if the index looks valid (not containing strings like "Ticker")
                        if isinstance(daily_data.index[0], str) and not daily_data.index[0].replace('-','').isdigit():
                            logger.warning(f"Index doesn't look like dates: {daily_data.index[0]}")
                            # Try alternative loading without index_col
                            daily_data = pd.read_csv(daily_file)
                            # If first column looks like dates, use it as index
                            if daily_data.shape[1] > 1:
                                date_col = daily_data.columns[0]
                                daily_data = daily_data.set_index(date_col)
                                daily_data.index = pd.to_datetime(daily_data.index, errors='coerce')
                    except Exception as e:
                        logger.error(f"Error loading daily data: {e}")
                        continue  # Skip to next ticker
                        
                    # Make sure we have a valid DatetimeIndex
                    if not isinstance(daily_data.index, pd.DatetimeIndex):
                        try:
                            daily_data.index = pd.to_datetime(daily_data.index, errors='coerce')
                            # Remove rows with invalid dates
                            daily_data = daily_data.loc[~daily_data.index.isna()]
                        except Exception as e:
                            logger.error(f"Could not convert index to datetime: {e}")
                            continue  # Skip to next ticker
                    # Add factors if available
                    daily_with_factors = daily_data
                    if factor_data['combined'] is not None:
                        daily_with_factors = merge_factors_with_stock_data(
                            daily_data, 
                            factor_data['combined'], 
                            freq='D'
                        )
                        
                    # Add options features if available
                    options_file = OPTIONS_DIR / f"{ticker}_options_features.csv"
                    if options_file.exists():
                        options_features = pd.read_csv(options_file, index_col=0, parse_dates=True)
                        if not options_features.empty:
                            daily_complete = merge_options_with_stock_data(
                                daily_with_factors,
                                options_features
                            )
                            # Save complete data
                            complete_file = PRICE_DIR / f"{ticker}_daily_complete_{START_DATE}_{END_DATE}.csv"
                            daily_complete.to_csv(complete_file)
                            logger.info(f"Saved {ticker} complete data with factors and options")
            except Exception as e:
                logger.error(f"Error integrating data for {ticker}: {e}")
            
            successful += 1
        else:
            failed.append(ticker)
        
        # Add delay between tickers
        time.sleep(10)
    
    logger.info(f"\nData collection complete: {successful} successful, {len(failed)} failed")
    if failed:
        logger.info(f"Failed tickers: {', '.join(failed)}")
if __name__ == "__main__":
    main()


