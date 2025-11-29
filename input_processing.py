"""
CGM Input Processing Module.

Unified interface for processing CGM data from multiple sources using the cgm_format library.
Supports:
- CSV file uploads (Dexcom, Libre, Unified formats)
- Base64 encoded data from web API
- Raw bytes from any source

This module provides a clean separation between data ingestion and processing.
"""

import base64
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import polars as pl
import pandas as pd

# Import from cgm_format common library
from cgm_format.format_converter import FormatParser
from cgm_format.interface.cgm_interface import (
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
)

# Set up logger
logger = logging.getLogger(__name__)


# ============================================================================
# Core Processing Functions
# ============================================================================

def process_raw_data(
    raw_data: bytes,
    return_format: str = "legacy"
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Optional[pl.DataFrame]]]:
    """
    Process raw CGM data bytes into structured format.
    
    This is the core processing function that all other functions delegate to.
    Automatically detects CGM format (Dexcom, Libre, Unified) and parses accordingly.
    
    Args:
        raw_data: Raw CSV data as bytes
        return_format: Output format:
            - "legacy": Returns DataFrame with ['time', 'gl', 'prediction'] columns (default)
            - "unified": Returns full UnifiedFormat DataFrame
            - "split": Returns tuple of (glucose_data, events_data)
        
    Returns:
        Processed data in requested format:
        - legacy: Single DataFrame with time, gl, prediction columns
        - unified: Full UnifiedFormat DataFrame with all columns
        - split: Tuple of (glucose_data, events_data) DataFrames
        
    Raises:
        UnknownFormatError: If format cannot be determined
        MalformedDataError: If data cannot be parsed
        ZeroValidInputError: If no valid glucose data found
    """
    logger.debug("Processing raw data with return_format=%s", return_format)
    
    # Use FormatParser from common library
    text_data = FormatParser.decode_raw_data(raw_data)
    format_type = FormatParser.detect_format(text_data)
    unified_df = FormatParser.parse_to_unified(text_data, format_type)
    
    if len(unified_df) == 0:
        raise ZeroValidInputError("No valid data found in input")
    
    logger.debug("Successfully processed %d rows", len(unified_df))
    
    # Return in requested format
    if return_format == "unified":
        return unified_df
    
    elif return_format == "split":
        return _convert_to_split_format(unified_df)
    
    else:  # legacy format (default)
        return _convert_to_legacy_format(unified_df)


def process_file(
    file_path: Union[str, Path],
    return_format: str = "legacy"
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Optional[pl.DataFrame]]]:
    """
    Process CGM data from a file.
    
    Convenience wrapper around process_raw_data() for file inputs.
    
    Args:
        file_path: Path to CGM data file (CSV format)
        return_format: Output format ("legacy", "unified", or "split")
        
    Returns:
        Processed data in requested format
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnknownFormatError: If format cannot be determined
        MalformedDataError: If data cannot be parsed
        ZeroValidInputError: If no valid glucose data found
    """
    logger.info("Processing file: %s", file_path)
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    return process_raw_data(raw_data, return_format=return_format)


def process_base64(
    base64_data: str,
    return_format: str = "legacy"
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Optional[pl.DataFrame]]]:
    """
    Process CGM data from base64 encoded string.
    
    Useful for web API endpoints that receive base64 encoded CSV data.
    
    Args:
        base64_data: Base64 encoded CSV data string
        return_format: Output format ("legacy", "unified", or "split")
        
    Returns:
        Processed data in requested format
        
    Raises:
        ValueError: If base64 decoding fails
        UnknownFormatError: If format cannot be determined
        MalformedDataError: If data cannot be parsed
        ZeroValidInputError: If no valid glucose data found
    """
    logger.debug("Processing base64 encoded data")
    
    try:
        # Decode base64 to bytes
        raw_data = base64.b64decode(base64_data)
    except Exception as e:
        logger.error("Failed to decode base64 data: %s", e)
        raise ValueError(f"Failed to decode base64 data: {e}")
    
    return process_raw_data(raw_data, return_format=return_format)


# ============================================================================
# Format Conversion Helpers
# ============================================================================

def _convert_to_legacy_format(unified_df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert UnifiedFormat DataFrame to legacy format.
    
    Legacy format has columns: ['time', 'gl', 'prediction']
    Only includes glucose readings (EGV_READ events).
    
    Args:
        unified_df: DataFrame in UnifiedFormat
        
    Returns:
        DataFrame with legacy column names
    """
    # Filter for glucose readings
    glucose_data = (unified_df
        .filter(pl.col("event_type") == "EGV_READ")
        .select([
            pl.col("datetime").alias("time"),
            pl.col("glucose").alias("gl"),
        ])
        .with_columns([
            pl.lit(0.0).alias("prediction")
        ])
    )
    
    return glucose_data


def _convert_to_split_format(
    unified_df: pl.DataFrame
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """
    Convert UnifiedFormat DataFrame to split format (glucose + events).
    
    Returns two DataFrames:
    - glucose_data: Time-series glucose readings with ['time', 'gl', 'prediction'] columns
    - events_data: Insulin, carbs, exercise events with ['time', 'event_type', 'value'] columns
    
    Args:
        unified_df: DataFrame in UnifiedFormat
        
    Returns:
        Tuple of (glucose_data, events_data or None)
    """
    # Extract glucose data
    glucose_data = _convert_to_legacy_format(unified_df)
    
    # Extract events data (insulin, carbs, exercise)
    events_df = unified_df.filter(pl.col("event_type") != "EGV_READ")
    
    if len(events_df) == 0:
        return glucose_data, None
    
    events_data = (events_df
        .select([
            pl.col("datetime").alias("time"),
            pl.col("event_type"),
            pl.when(pl.col("insulin_fast").is_not_null())
            .then(pl.col("insulin_fast"))
            .when(pl.col("insulin_slow").is_not_null())
            .then(pl.col("insulin_slow"))
            .when(pl.col("carbs").is_not_null())
            .then(pl.col("carbs"))
            .when(pl.col("exercise").is_not_null())
            .then(pl.col("exercise"))
            .otherwise(pl.lit(None))
            .alias("value")
        ])
    )
    
    return glucose_data, events_data


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def load_cgm_data(
    raw_data: bytes,
    add_metadata: bool = True
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """
    Load CGM data with optional metadata columns.
    
    DEPRECATED: Maintained for backward compatibility with cgm_processing.py
    New code should use process_raw_data() instead.
    
    Args:
        raw_data: Raw CSV bytes
        add_metadata: Add age and user_id columns (default: True)
        
    Returns:
        Tuple of (glucose_data, events_data)
        - glucose_data includes 'age' and 'user_id' columns if add_metadata=True
        
    Raises:
        UnknownFormatError: If format cannot be determined
        MalformedDataError: If data cannot be parsed
    """
    logger.debug("Loading CGM data with add_metadata=%s", add_metadata)
    
    glucose_data, events_data = process_raw_data(raw_data, return_format="split")
    
    # Add metadata columns if requested
    if add_metadata:
        glucose_data = glucose_data.with_columns([
            pl.lit(0).alias("age"),
            pl.lit(1).alias("user_id")
        ])
    
    return glucose_data, events_data


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert Polars DataFrame to Pandas DataFrame.
    
    Utility function for compatibility with pandas-based code.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        Pandas DataFrame
    """
    return df.to_pandas()


# ============================================================================
# High-Level Processing Functions
# ============================================================================

def process_csv_with_options(
    input_file: Path,
    output_file: Path,
    drop_duplicates: bool = True,
    time_diff_minutes: int = 1,
    chunk_size: Optional[int] = 1000,
) -> pd.DataFrame:
    """
    Process CSV file with post-processing options.
    
    This function provides the same interface as the old format_dexcom.process_csv()
    for backward compatibility.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        drop_duplicates: Remove duplicate timestamps (default: True)
        time_diff_minutes: Minimum time difference between readings in minutes (default: 1)
        chunk_size: Chunk size for ID assignment (default: 1000, None = single ID)
        
    Returns:
        Processed DataFrame with ['id', 'time', 'gl'] columns
    """
    logger.info("Processing CSV with options: input=%s, output=%s", input_file, output_file)
    
    # Process file to get glucose data
    glucose_data = process_file(input_file, return_format="legacy")
    
    # Apply post-processing
    
    # 1. Add ID column based on chunk_size
    if chunk_size is None or chunk_size <= 0:
        glucose_data = glucose_data.with_columns(pl.lit(1).alias('id'))
    else:
        glucose_data = glucose_data.with_row_index('_idx').with_columns(
            (pl.col('_idx') // chunk_size).alias('id')
        ).drop('_idx')
    
    # 2. Time filtering - keep only readings with sufficient time gap
    if time_diff_minutes > 0:
        glucose_data = glucose_data.with_columns(
            pl.col('time').diff().alias('time_diff')
        ).filter(
            pl.col('time_diff').is_null() | 
            (pl.col('time_diff') >= pl.duration(minutes=time_diff_minutes))
        ).drop('time_diff')
    
    # 3. Drop duplicates
    if drop_duplicates:
        glucose_data = glucose_data.unique(subset=['time'], keep='first')
    
    # 4. Select final columns and convert to pandas
    df = to_pandas(glucose_data.select(['id', 'time', 'gl']))
    
    # 5. Save to file
    df.to_csv(output_file, index=False)
    logger.info("Processed %d rows -> %s", len(df), output_file)
    print(f"Processed {len(df)} rows -> {output_file}")
    
    return df


def process_multiple_files(
    input_dir: Path,
    output_file: Path,
    drop_duplicates: bool = True,
    time_diff_minutes: int = 1,
    chunk_size: Optional[int] = 1000,
) -> pd.DataFrame:
    """
    Process and merge multiple CSV files from a directory.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to output merged CSV file
        drop_duplicates: Remove duplicate timestamps
        time_diff_minutes: Minimum time difference between readings
        chunk_size: Chunk size for ID assignment
        
    Returns:
        Merged DataFrame with all processed data
    """
    logger.info("Processing multiple files from directory: %s -> %s", input_dir, output_file)
    
    dfs = []
    
    for file_path in input_dir.glob("*.csv"):
        print(f"Processing {file_path.name}...")
        logger.debug("Processing file: %s", file_path.name)
        
        try:
            # Process file
            glucose_data = process_file(file_path, return_format="legacy")
            
            # Apply post-processing
            if chunk_size is None or chunk_size <= 0:
                glucose_data = glucose_data.with_columns(pl.lit(1).alias('id'))
            else:
                glucose_data = glucose_data.with_row_index('_idx').with_columns(
                    (pl.col('_idx') // chunk_size).alias('id')
                ).drop('_idx')
            
            if time_diff_minutes > 0:
                glucose_data = glucose_data.with_columns(
                    pl.col('time').diff().alias('time_diff')
                ).filter(
                    pl.col('time_diff').is_null() | 
                    (pl.col('time_diff') >= pl.duration(minutes=time_diff_minutes))
                ).drop('time_diff')
            
            if drop_duplicates:
                glucose_data = glucose_data.unique(subset=['time'], keep='first')
            
            dfs.append(glucose_data.select(['id', 'time', 'gl']))
            
        except Exception as e:
            logger.error("Error processing %s: %s", file_path.name, e)
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid files processed")
    
    # Combine all DataFrames
    combined = pl.concat(dfs).sort('time')
    result = to_pandas(combined)
    
    # Save to file
    result.to_csv(output_file, index=False)
    logger.info("Merged %d files -> %d rows -> %s", len(dfs), len(result), output_file)
    print(f"Merged {len(dfs)} files -> {len(result)} rows -> {output_file}")
    
    return result

