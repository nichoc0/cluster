import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import concurrent.futures
from datetime import datetime, timedelta

# Define stock cluster
TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC']
# Define date ranges
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'
HOURLY_LOOKBACK_DAYS = 730  # ~2 years of hourly data (Yahoo limit is ~2 years)

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
            print(f"Attempt {attempt+1} failed for {ticker}: {e}")
        
        # Exponential backoff
        sleep_time = delay * (2 ** attempt)
        print(f"Sleeping for {sleep_time} seconds before next attempt...")
        time.sleep(sleep_time)
    
    print(f"Failed to fetch data for {ticker} after {retries} attempts")
    return pd.DataFrame()

def fetch_hourly_chunks(ticker, start_date, end_date):
    """Fetch hourly data in chunks to avoid timeouts"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Yahoo Finance limits hourly data to ~2 years, so we'll chunk by ~90 days
    chunk_size = timedelta(days=90)
    chunks = []
    
    current_start = start
    while current_start < end:
        current_end = min(current_start + chunk_size, end)
        print(f"Fetching {ticker} hourly data from {current_start.date()} to {current_end.date()}")
        
        chunk = safe_fetch(
            ticker, 
            current_start, 
            current_end, 
            interval='1h'
        )
        
        if not chunk.empty:
            chunks.append(chunk)
        
        # Add delay between chunks
        time.sleep(5)
        current_start = current_end
    
    if chunks:
        return pd.concat(chunks)
    return pd.DataFrame()

def fetch_stock_data(ticker, start_date=START_DATE, end_date=END_DATE):
    """Fetch both daily and hourly data for a ticker"""
    os.makedirs('data', exist_ok=True)
    
    # File paths
    daily_file = f"data/{ticker}_daily_{start_date}_{end_date}.csv"
    
    # Calculate hourly data start date (limited lookback)
    end_dt = pd.to_datetime(end_date)
    hourly_start_dt = end_dt - timedelta(days=HOURLY_LOOKBACK_DAYS)
    hourly_start = hourly_start_dt.strftime('%Y-%m-%d')
    hourly_file = f"data/{ticker}_hourly_{hourly_start}_{end_date}.csv"
    
    # 1. Fetch daily data
    try:
        if os.path.exists(daily_file):
            print(f"Loading {ticker} daily data from file")
            daily_data = pd.read_csv(daily_file, index_col=0, parse_dates=True)
        else:
            print(f"Downloading {ticker} daily data from {start_date} to {end_date}")
            daily_data = safe_fetch(ticker, start_date, end_date, interval='1d')
            if not daily_data.empty:
                daily_data.to_csv(daily_file)
                print(f"Saved {ticker} daily data ({len(daily_data)} rows)")
            else:
                print(f"No daily data found for {ticker}")
                return False
    except Exception as e:
        print(f"Error fetching daily data for {ticker}: {e}")
        return False
    
    # 2. Fetch hourly data
    try:
        if os.path.exists(hourly_file):
            print(f"Loading {ticker} hourly data from file")
            hourly_data = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
        else:
            print(f"Downloading {ticker} hourly data from {hourly_start} to {end_date}")
            hourly_data = fetch_hourly_chunks(ticker, hourly_start, end_date)
            if not hourly_data.empty:
                hourly_data.to_csv(hourly_file)
                print(f"Saved {ticker} hourly data ({len(hourly_data)} rows)")
            else:
                print(f"No hourly data found for {ticker}")
    except Exception as e:
        print(f"Error fetching hourly data for {ticker}: {e}")
    
    print(f"Completed data fetch for {ticker}")
    return True

def main():
    print(f"Starting data collection for {len(TECH_STOCKS)} stocks")
    
    successful = 0
    failed = []
    
    # Process sequentially to respect rate limits
    for ticker in TECH_STOCKS:
        print(f"\n{'='*40}\nProcessing {ticker}\n{'='*40}")
        if fetch_stock_data(ticker):
            successful += 1
        else:
            failed.append(ticker)
        
        # Add delay between tickers
        time.sleep(10)
    
    print(f"\nData collection complete: {successful} successful, {len(failed)} failed")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")

if __name__ == "__main__":
    main()