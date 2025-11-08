"""
Binance Data Downloader - Memory Optimized
Downloads tick data without loading everything into RAM
Uses streaming and chunked processing
"""

import os
import io
import zipfile
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Suppress pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
SYMBOL = "BTCUSDT"
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
DAILY_DIR = BASE_DIR / "daily"

# Binance CSV columns
EXPECTED_COLUMNS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]

# Create directory structure
RAW_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)


def read_binance_csv(file_obj):
    """
    Reads Binance CSV file efficiently.
    Returns DataFrame with datetime index and date column.
    """
    try:
        content = file_obj.read()
        
        if not content or len(content) == 0:
            return None
        
        # Decode content
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = content.decode('latin-1')
        
        text_content = text_content.replace('\ufeff', '')
        
        if not text_content.strip():
            return None
        
        # Check for header
        lines = text_content.strip().split('\n')
        if not lines:
            return None
            
        first_line = lines[0].strip()
        has_header = 'id' in first_line.lower() and 'price' in first_line.lower()
        
        from io import StringIO
        
        # Read CSV with efficient dtypes
        dtype_dict = {
            'id': 'int64',
            'price': 'float32',  # float32 saves 50% memory vs float64
            'qty': 'float32',
            'quote_qty': 'float32',
            'time': 'int64',
            'is_buyer_maker': 'bool',
            'is_best_match': 'bool'
        }
        
        if has_header:
            df = pd.read_csv(StringIO(text_content), dtype=dtype_dict, skipinitialspace=True)
        else:
            df = pd.read_csv(
                StringIO(text_content), 
                header=None, 
                names=EXPECTED_COLUMNS,
                dtype=dtype_dict,
                skipinitialspace=True
            )
        
        if df.empty:
            return None
        
        # Convert boolean columns if needed
        for bool_col in ['is_buyer_maker', 'is_best_match']:
            if bool_col in df.columns and df[bool_col].dtype == 'object':
                df[bool_col] = df[bool_col].astype(str).str.strip().str.lower()
                df[bool_col] = df[bool_col].map({'true': True, 'false': False})
        
        # Create datetime columns efficiently
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df['date'] = df['datetime'].dt.date
        
        # Drop invalid rows
        df = df.dropna(subset=['datetime', 'time'])
        
        if df.empty:
            return None
        
        return df
        
    except Exception as e:
        return None


def download_and_save_day(date_str, debug=False):
    """
    Downloads data for a single day and saves immediately to disk.
    Does NOT keep data in memory.
    
    Returns:
        Tuple of (date_str, record_count, status_message, file_path)
    """
    url = f"{BASE_URL}/{SYMBOL}/{SYMBOL}-trades-{date_str}.zip"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            return (date_str, 0, "Not found", None)
        
        if response.status_code != 200:
            return (date_str, 0, f"HTTP {response.status_code}", None)
        
        # Extract and read CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = zip_file.namelist()[0]
            
            with zip_file.open(csv_filename) as csv_file:
                df = read_binance_csv(csv_file)
        
        if df is None or df.empty:
            return (date_str, 0, "Failed to parse CSV", None)
        
        record_count = len(df)
        
        # Save immediately to disk and release memory
        daily_file = DAILY_DIR / f"{SYMBOL}_{date_str}.parquet"
        df.to_parquet(daily_file, index=False, compression='snappy')
        
        # Clear DataFrame from memory
        del df
        
        return (date_str, record_count, f"{record_count:,} records", daily_file)
        
    except requests.exceptions.Timeout:
        return (date_str, 0, "Timeout", None)
    except Exception as e:
        return (date_str, 0, f"Error: {str(e)}", None)


def merge_daily_to_yearly(year, daily_files):
    """
    Merge daily parquet files into yearly file using streaming.
    Memory efficient - processes in batches.
    """
    print(f"\n{'='*60}")
    print(f"MERGING TO YEARLY FILE - {year}")
    print(f"{'='*60}\n")
    
    yearly_file = RAW_DIR / f"{SYMBOL}_{year}.parquet"
    
    # Use PyArrow for efficient merging
    # This reads files in chunks and writes incrementally
    writer = None
    total_rows = 0
    
    try:
        # Sort files by date for chronological order
        sorted_files = sorted(daily_files)
        
        for i, file_path in enumerate(sorted_files, 1):
            # Read one file at a time
            table = pq.read_table(file_path)
            total_rows += len(table)
            
            # Initialize writer with schema from first file
            if writer is None:
                writer = pq.ParquetWriter(yearly_file, table.schema, compression='snappy')
            
            # Write this chunk
            writer.write_table(table)
            
            # Progress indicator
            if i % 30 == 0 or i == len(sorted_files):
                print(f"  ✓ Merged {i}/{len(sorted_files)} days ({total_rows:,} records so far)")
            
            # Clear from memory
            del table
        
        if writer:
            writer.close()
        
        file_size_mb = yearly_file.stat().st_size / 1024 / 1024
        
        print(f"\n{'='*60}")
        print(f"✅ YEARLY FILE CREATED")
        print(f"{'='*60}")
        print(f"File: {yearly_file}")
        print(f"Total records: {total_rows:,}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging files: {e}")
        if writer:
            writer.close()
        return False


def download_year(year):
    """
    Downloads all data for a given year.
    Memory optimized - saves to disk immediately, never loads all data at once.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING {SYMBOL} - YEAR {year}")
    print(f"{'='*60}\n")
    
    # Generate date range
    start_date = f"{year}-01-01"
    
    if year == datetime.now().year:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = f"{year}-12-31"
    
    date_range = pd.date_range(start_date, end_date, freq="D")
    date_strings = [d.strftime("%Y-%m-%d") for d in date_range]
    
    print(f"Downloading {len(date_strings)} days...\n")
    
    # Download and save each day immediately
    successful_files = []
    total_records = 0
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:  # Reduced workers to save memory
        future_to_date = {
            executor.submit(download_and_save_day, date): date 
            for date in date_strings
        }
        
        for future in as_completed(future_to_date):
            date_str, record_count, message, file_path = future.result()
            
            if file_path:
                successful_files.append(file_path)
                total_records += record_count
                successful += 1
                print(f"✅ {date_str}: {message}")
            else:
                failed += 1
                if message != "Not found":
                    print(f"⚠️ {date_str}: {message}")
    
    print(f"\n📊 Summary: {successful} successful, {failed} failed/missing")
    print(f"📊 Total records downloaded: {total_records:,}")
    print(f"📁 Daily files saved to: {DAILY_DIR}/")
    
    if not successful_files:
        print(f"\n❌ No valid data downloaded for {year}")
        return False
    
    # Merge daily files into yearly file
    success = merge_daily_to_yearly(year, successful_files)
    
    return success


def verify_data_structure(year):
    """
    Verify the downloaded data has correct structure.
    Uses chunked reading to avoid memory issues.
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING DATA STRUCTURE - {year}")
    print(f"{'='*60}\n")
    
    # Check yearly file
    yearly_file = RAW_DIR / f"{SYMBOL}_{year}.parquet"
    if not yearly_file.exists():
        print(f"❌ Yearly file not found: {yearly_file}")
        return
    
    # Read metadata without loading all data
    parquet_file = pq.ParquetFile(yearly_file)
    
    print(f"📊 Yearly File Analysis:")
    print(f"  ✓ Total rows: {parquet_file.metadata.num_rows:,}")
    print(f"  ✓ Columns: {parquet_file.schema.names}")
    print(f"  ✓ File size: {yearly_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  ✓ Row groups: {parquet_file.num_row_groups}")
    
    # Read just first batch to check structure
    first_batch = parquet_file.read_row_group(0).to_pandas()
    
    print(f"\n  Data types:")
    for col, dtype in first_batch.dtypes.items():
        print(f"    - {col}: {dtype}")
    
    print(f"\n  Date range (from first batch):")
    print(f"    - First: {first_batch['date'].iloc[0]}")
    print(f"    - Last: {first_batch['date'].iloc[-1]}")
    
    print(f"\n  First row sample:")
    print(f"    {first_batch.head(1).to_dict('records')[0]}")
    
    # Check daily files
    daily_files = list(DAILY_DIR.glob(f"{SYMBOL}_*.parquet"))
    if daily_files:
        print(f"\n📁 Daily Files:")
        print(f"  ✓ Total files: {len(daily_files)}")
        print(f"  ✓ Location: {DAILY_DIR}/")
        
        # Check size of daily files
        total_daily_size = sum(f.stat().st_size for f in daily_files)
        print(f"  ✓ Total size: {total_daily_size / 1024 / 1024:.2f} MB")
    
    print(f"\n{'='*60}\n")


def clean_daily_files(year):
    """
    Optional: Remove daily files after yearly merge to save disk space.
    """
    print(f"\n🧹 Cleaning up daily files for {year}...")
    
    # Check yearly file exists first
    yearly_file = RAW_DIR / f"{SYMBOL}_{year}.parquet"
    if not yearly_file.exists():
        print(f"❌ Yearly file not found, keeping daily files")
        return
    
    # Get daily files for this year
    pattern = f"{SYMBOL}_{year}-*.parquet"
    daily_files = list(DAILY_DIR.glob(pattern))
    
    if not daily_files:
        print(f"No daily files found for {year}")
        return
    
    # Delete daily files
    for file_path in daily_files:
        try:
            file_path.unlink()
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path.name}: {e}")
    
    print(f"✅ Deleted {len(daily_files)} daily files for {year}")
    print(f"   (Yearly file preserved at {yearly_file})")


if __name__ == "__main__":
    import sys
    
    # Debug mode
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        print("🔍 DEBUG MODE: Testing single day download\n")
        date_str, count, message, file_path = download_and_save_day("2023-01-01", debug=True)
        print(f"\nResult: {message}")
        if file_path:
            print(f"File saved: {file_path}")
            df = pd.read_parquet(file_path)
            print(f"\nDataFrame info:")
            print(df.info())
            print(f"\nFirst 5 rows:")
            print(df.head())
        sys.exit(0)
    
    # Verify mode
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
        verify_data_structure(year)
        sys.exit(0)
    
    # Clean mode (remove daily files after merge)
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
        clean_daily_files(year)
        sys.exit(0)
    
    # Normal mode: download all years
    years = [2023, 2024, 2025]
    
    print("\n" + "="*60)
    print("BINANCE DATA DOWNLOADER - MEMORY OPTIMIZED")
    print("="*60)
    print(f"Symbol: {SYMBOL}")
    print(f"Years: {years}")
    print(f"Output structure:")
    print(f"  📁 {RAW_DIR}/     → Yearly files")
    print(f"  📁 {DAILY_DIR}/   → Daily partitions")
    print(f"\n💡 Memory optimization:")
    print(f"  - Uses float32 instead of float64 (50% less memory)")
    print(f"  - Streams data to disk immediately")
    print(f"  - Never loads full year in RAM")
    print("="*60)
    
    for year in years:
        try:
            success = download_year(year)
            if success:
                verify_data_structure(year)
        except KeyboardInterrupt:
            print("\n\n⚠️ Download interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error for year {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("🎉 ALL DOWNLOADS COMPLETED!")
    print("="*60)
    print("\nUseful commands:")
    print("  python download_data.py verify 2023    # Verify data")
    print("  python download_data.py clean 2023     # Remove daily files (keep yearly)")
    print("="*60 + "\n")