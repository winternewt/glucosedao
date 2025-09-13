import pandas as pd
from pathlib import Path
import typer
import logging
import chardet

# BOM (Byte Order Mark) constants for encoding normalization
UTF8_BOM = b'\xef\xbb\xbf'
UTF16_LE_BOM = b'\xff\xfe'  
UTF16_BE_BOM = b'\xfe\xff'

# Common encoding artifacts and their fixes
ENCODING_ARTIFACTS = {
    # Double-encoded BOM in quotes: "ïººº¿"
    b'\x22\xc3\xaf\xc2\xbb\xc2\xbf\x22': UTF8_BOM,
    
    # Triple-encoded BOM (enterprise nightmare)
    b'\x22\xc3\x83\xc2\xaf\xc3\x82\xc2\xbb\xc3\x82\xc2\xbf\x22': UTF8_BOM,
    
    # Double-encoded BOM without quotes
    b'\xc3\xaf\xc2\xbb\xc2\xbf': UTF8_BOM,
    
    # Quoted BOM (some systems do this)
    b'\x22\xef\xbb\xbf\x22': UTF8_BOM,
}

# Encoding fallback order for maximum compatibility
ENCODING_FALLBACKS = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

def normalize_encoding_artifacts(raw_data: bytes) -> bytes:
    """
    Normalize encoding artifacts at the raw byte level.
    
    Converts double/triple encoded BOMs back to proper UTF-8 BOM so that
    standard tools like chardet can work normally.
    
    Args:
        raw_data: Raw file bytes
        
    Returns:
        Normalized bytes with proper BOM
    """
    normalized_data = raw_data
    
    for corrupted_pattern, proper_bom in ENCODING_ARTIFACTS.items():
        if normalized_data.startswith(corrupted_pattern):
            print(f"🔧 Normalized corrupted BOM: {corrupted_pattern.hex()} -> {proper_bom.hex()}")
            normalized_data = proper_bom + normalized_data[len(corrupted_pattern):]
            break
    
    return normalized_data

def detect_encoding_and_read_csv(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Universal CSV reader that handles BOM, encoding detection, and other encoding horrors.
    
    This is the "homogeneous-blended-standardized IOStream" solution that automatically
    detects encoding and strips BOM, giving you a clean DataFrame regardless of the input mess.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        Clean pandas DataFrame with properly decoded content
    """
    # Read raw data
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    # Normalize encoding artifacts first
    normalized_data = normalize_encoding_artifacts(raw_data)
    
    # Now detect encoding on normalized data
    sample = normalized_data[:10000]  # First 10KB for detection
    detected = chardet.detect(sample)
    encoding = detected['encoding']
    confidence = detected['confidence']
    
    print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
    
    # Check for proper BOM after normalization
    has_utf8_bom = normalized_data.startswith(UTF8_BOM)
    has_utf16_bom = (normalized_data.startswith(UTF16_LE_BOM) or 
                     normalized_data.startswith(UTF16_BE_BOM))
    
    print(f"BOM status after normalization: UTF-8={has_utf8_bom}, UTF-16={has_utf16_bom}")
    
    # Write normalized data to temp file for pandas to read
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(normalized_data)
        temp_path = tmp_file.name
    
    try:
        # Build encoding priority list
        encodings_to_try = []
        
        # Use utf-8-sig if we have BOM, otherwise use detected encoding
        if has_utf8_bom:
            encodings_to_try.append('utf-8-sig')
        elif encoding:
            encodings_to_try.append(encoding)
            
        # Add standard fallbacks
        encodings_to_try.extend(['utf-8-sig'] + ENCODING_FALLBACKS)
        
        # Remove None and duplicates while preserving order
        encodings_to_try = list(dict.fromkeys([enc for enc in encodings_to_try if enc]))
        
        for enc in encodings_to_try:
            try:
                print(f"Trying encoding: {enc}")
                df = pd.read_csv(temp_path, encoding=enc, **kwargs)
                
                # Minimal column cleaning (should be much cleaner now)
                df.columns = (df.columns
                             .str.strip()                # Remove whitespace
                             .str.replace('"', '')       # Remove any remaining quotes
                             .str.replace('"', ''))      # Remove fancy quotes
                
                print(f"✅ Successfully read with encoding: {enc}")
                return df
                
            except (UnicodeDecodeError, UnicodeError) as e:
                print(f"❌ Failed with {enc}: {e}")
                continue
            except Exception as e:
                print(f"💥 Unexpected error with {enc}: {e}")
                continue
        
        raise ValueError(f"Could not read file with any encoding. Tried: {encodings_to_try}")
        
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

def process_csv(
        input_dir: Path,
        output_file: Path,
        event_type_filter: str = 'egv',
        drop_duplicates: bool = True,
        time_diff_minutes: int = 1,
        chunk_size: int = 1000,
) -> pd.DataFrame:

    try:
        # Use our universal CSV reader - no more BOM pain!
        df = detect_encoding_and_read_csv(input_dir, low_memory=False)
        
        print(f"Available columns: {list(df.columns)}")
        
        # Check if required columns exist with flexible matching
        required_columns = {
            'index': ['Index', 'index'],
            'timestamp': ['Timestamp (YYYY-MM-DDThh:mm:ss)', 'timestamp', 'Timestamp'],
            'glucose': ['Glucose Value (mg/dL)', 'glucose', 'Glucose Value', 'glucose_value'],
            'event_type': ['Event Type', 'event_type', 'Event Type'],
            'event_subtype': ['Event Subtype', 'event_subtype', 'Event Subtype']
        }
        
        # Map actual column names
        column_mapping = {}
        for key, possible_names in required_columns.items():
            found_column = None
            for possible_name in possible_names:
                if possible_name in df.columns:
                    found_column = possible_name
                    break
            if found_column:
                column_mapping[key] = found_column
            else:
                print(f"Warning: Could not find column for {key}. Available columns: {list(df.columns)}")
        
        # Check if we have the minimum required columns
        if 'event_type' not in column_mapping:
            raise ValueError("Could not find Event Type column. Please check your CSV format.")
            
        if 'timestamp' not in column_mapping or 'glucose' not in column_mapping:
            raise ValueError("Could not find required timestamp or glucose columns. Please check your CSV format.")

        # Filter by Event Type and Event Subtype
        df = df[df[column_mapping['event_type']].str.lower() == event_type_filter]
        
        if 'event_subtype' in column_mapping:
            df = df[df[column_mapping['event_subtype']].isna()]

        # List of columns to keep (only those that exist)
        columns_to_keep = [column_mapping[key] for key in ['timestamp', 'glucose'] if key in column_mapping]
        
        # Add index column if it exists, otherwise create one
        if 'index' in column_mapping:
            columns_to_keep.insert(0, column_mapping['index'])
        else:
            # Create an index column
            df['Index'] = range(1, len(df) + 1)
            columns_to_keep.insert(0, 'Index')
            column_mapping['index'] = 'Index'

        # Keep only the specified columns
        df = df[columns_to_keep]

        # Rename columns to standard names
        column_rename = {}
        if 'index' in column_mapping:
            column_rename[column_mapping['index']] = 'id'
        if 'timestamp' in column_mapping:
            column_rename[column_mapping['timestamp']] = 'time'
        if 'glucose' in column_mapping:
            column_rename[column_mapping['glucose']] = 'gl'
            
        df = df.rename(columns=column_rename)

        # Ensure we have an id column
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
            
        # Convert id to int and handle NaN values
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df = df.dropna(subset=['id'])  # Drops rows where the index is NaN
        df['id'] = df['id'].astype(int)
        
        # Handle id assignment based on chunk_size
        if chunk_size is None or chunk_size == 0:
            df['id'] = 1  # Assign the same id to all rows
        else:
            df['id'] = (df.index // chunk_size).astype(int)

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['time'])

        # Calculate time difference and keep rows with at least the specified time difference
        df['time_diff'] = df['time'].diff()
        df = df[df['time_diff'].isna() | (df['time_diff'] >= pd.Timedelta(minutes=time_diff_minutes))]

        # Drop the temporary time_diff column
        df = df.drop(columns=['time_diff'])

        # Ensure glucose values are in float64
        df['gl'] = pd.to_numeric(df['gl'], errors='coerce')
        df = df.dropna(subset=['gl'])  # Remove rows with invalid glucose values
        df['gl'] = df['gl'].astype('float64')

        # Optionally drop duplicate rows based on time
        if drop_duplicates:
            df = df.drop_duplicates(subset=['time'], keep='first')

        # Write the modified dataframe to a new CSV file
        df.to_csv(output_file, index=False)

        print(f"CSV file has been successfully processed. Output shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        raise e


'''
def process_multiple_csv(
        input_dir: Path = typer.Argument('./raw_data/livia_unmerged', help="Directory containing the input CSV files."),
        output_file: Path = typer.Argument('./raw_data/livia_unmerged/livia_mini.csv', help="Path to save the processed CSV file."),
        event_type_filter: str = typer.Option('egv', help="Event type to filter by."),
        drop_duplicates: bool = typer.Option(True, help="Whether to drop duplicate timestamps."),
        time_diff_minutes: int = typer.Option(1, help="Minimum time difference in minutes to keep a row."),
        chunk_size: int = typer.Option(1000, help="Chunk size for the 'id' column increment. Set to 0 or None for a single id."),
):
    # Get all the CSV files in the specified directory
    all_files = list(input_dir.glob("*.csv"))

    # List to store the DataFrames
    df_list = []

    # Read each CSV file into a DataFrame and append to the list
    for filename in all_files:
        df = pd.read_csv(filename, low_memory=False)
        df_list.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    # Filter by Event Type and Event Subtype
    combined_df = combined_df[combined_df['Event Type'].str.lower() == event_type_filter]
    combined_df = combined_df[combined_df['Event Subtype'].isna()]

    # List of columns to keep
    columns_to_keep = [
        'Index',
        'Timestamp (YYYY-MM-DDThh:mm:ss)',
        'Glucose Value (mg/dL)',
    ]

    # Keep only the specified columns
    combined_df = combined_df[columns_to_keep]

    # Rename columns
    column_rename = {
        'Index': 'id',
        'Timestamp (YYYY-MM-DDThh:mm:ss)': 'time',
        'Glucose Value (mg/dL)': 'gl'
    }
    combined_df = combined_df.rename(columns=column_rename)

    # Sort the combined DataFrame by timestamp
    combined_df = combined_df.sort_values('time')

    # Handle id assignment based on chunk_size
    if chunk_size is None or chunk_size == 0:
        combined_df['id'] = 1  # Assign the same id to all rows
    else:
        combined_df['id'] = ((combined_df.index // chunk_size) % (combined_df.index.max() // chunk_size + 1)).astype(int)

    # Convert timestamp to datetime
    combined_df['time'] = pd.to_datetime(combined_df['time'])

    # Calculate time difference and keep rows with at least the specified time difference
    combined_df['time_diff'] = combined_df['time'].diff()
    combined_df = combined_df[combined_df['time_diff'].isna() | (combined_df['time_diff'] >= pd.Timedelta(minutes=time_diff_minutes))]

    # Drop the temporary time_diff column
    combined_df = combined_df.drop(columns=['time_diff'])

    # Ensure glucose values are in float64
    combined_df['gl'] = combined_df['gl'].astype('float64')

    # Optionally drop duplicate rows based on time
    if drop_duplicates:
        combined_df = combined_df.drop_duplicates(subset=['time'], keep='first')

    # Write the modified dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)

    typer.echo("CSV files have been successfully merged, modified, and saved.")
'''
if __name__ == "__main__":
    typer.run(process_csv)
