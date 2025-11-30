import pandas as pd
import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from typing import Union, Optional, List, Tuple
from pathlib import Path

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import data_formatter.utils as formatter_utils
import data_formatter.types as types
from utils.darts_dataset import SamplingDatasetInferenceDual
from utils.darts_processing import ScalerCustom

DataTypes = types.DataTypes
InputTypes = types.InputTypes

class GluformerModelConfig(BaseModel):
    """
    Configuration that matches Gluformer model arguments exactly.
    Used to instantiate the model directly using **model_dump().
    """
    model_config = {"frozen": True}
    
    d_model: int = Field(default=512, description="Model dimension")
    n_heads: int = Field(default=10, description="Number of attention heads")
    d_fcn: int = Field(default=1024, description="Fully connected layer dimension")
    num_enc_layers: int = Field(default=2, description="Number of encoder layers")
    num_dec_layers: int = Field(default=2, description="Number of decoder layers")
    
    len_seq: int = Field(..., description="Input sequence length (maps to input_chunk_length)")
    label_len: int = Field(..., description="Label length (usually len_seq // 3)")
    len_pred: int = Field(..., description="Prediction length (maps to output_chunk_length)")
    
    num_dynamic_features: int = Field(..., description="Number of dynamic features")
    num_static_features: int = Field(..., description="Number of static features")
    
    r_drop: float = Field(default=0.2, description="Dropout rate")
    activ: str = Field(default='gelu', description="Activation function")
    distil: bool = Field(default=True, description="Use distillation")

class GluformerInferenceConfig(BaseModel):
    """
    Input configuration for inference pipeline.
    Contains both processing parameters and base model architecture parameters.
    """
    model_config = {"frozen": True}

    # Architecture defaults (can be overridden to match weights)
    d_model: int = Field(default=512, description="Model dimension")
    n_heads: int = Field(default=10, description="Number of attention heads")
    d_fcn: int = Field(default=1024, description="Fully connected layer dimension")
    num_enc_layers: int = Field(default=2, description="Number of encoder layers")
    num_dec_layers: int = Field(default=2, description="Number of decoder layers")
    
    # Sequence Lengths
    input_chunk_length: int = Field(default=96, description="Length of input sequence")
    output_chunk_length: int = Field(default=12, description="Length of output sequence")
    
    # Feature Dimensions Defaults (Inferred from data during processing)
    num_dynamic_features: int = Field(default=6, description="Default number of dynamic features")
    num_static_features: int = Field(default=1, description="Default number of static features")
    
    # Data Processing
    gap_threshold: int = Field(default=45, description="Max gap in minutes to interpolate")
    min_drop_length: int = Field(default=12, description="Min length of segment to keep")
    interval_length: str = Field(default='5min', description="Interval length for interpolation")
    
    # Optional overrides for model defaults
    r_drop: float = Field(default=0.2, description="Dropout rate")
    activ: str = Field(default='gelu', description="Activation function")
    distil: bool = Field(default=True, description="Use distillation")

def create_inference_dataset_fast(
    data: Union[str, Path, pl.DataFrame],
    config: GluformerInferenceConfig = GluformerInferenceConfig(),
    scaler_target: Optional[ScalerCustom] = None,
    scaler_covs: Optional[ScalerCustom] = None
) -> Tuple[SamplingDatasetInferenceDual, ScalerCustom, GluformerModelConfig]:
    """
    Directly creates a SamplingDatasetInferenceDual from a CSV file or Polars DataFrame.
    
    Args:
        data: Path to the unified CSV file OR a Polars DataFrame (UnifiedFormat).
        config: GluformerInferenceConfig instance with processing parameters.
        scaler_target: Pre-fitted scaler for target. If None, fits new one.
        scaler_covs: Pre-fitted scaler for covariates. If None, fits new one.
                     
    Returns:
        dataset: SamplingDatasetInferenceDual ready for inference.
        scaler: The target scaler used.
        model_config: GluformerModelConfig instance ready to instantiate Gluformer.
    """
    
    # 1. Load Data
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
        # Ensure types
        df['time'] = pd.to_datetime(df['time'])
        if 'gl' in df.columns:
            df['gl'] = df['gl'].astype(np.float32)
            
    elif isinstance(data, pl.DataFrame):
        # Handle Polars DataFrame
        # Map UnifiedFormat columns to expected internal names
        mapping = {}
        if 'sequence_id' in data.columns:
            mapping['sequence_id'] = 'id'
        if 'datetime' in data.columns:
            mapping['datetime'] = 'time'
        if 'glucose' in data.columns:
            mapping['glucose'] = 'gl'
            
        if mapping:
            data = data.rename(mapping)
            
        # Convert to pandas
        df = data.to_pandas()
        
        # Ensure types
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        if 'gl' in df.columns:
            df['gl'] = df['gl'].astype(np.float32)
            
    else:
        raise ValueError("data must be a file path (str/Path) or a Polars DataFrame")
    
    # 2. Define minimal column definition for data_formatter utils
    column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('time', DataTypes.DATE, InputTypes.TIME),
        ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET)
    ]
    
    # 3. Interpolate
    # Use params from config
    df_interp, updated_col_def = formatter_utils.interpolate(
        df, 
        column_definition, 
        gap_threshold=config.gap_threshold,
        min_drop_length=config.min_drop_length,
        interval_length=config.interval_length
    )
    
    # 4. Encode Datetime Features
    date_features = ['day', 'month', 'year', 'hour', 'minute', 'second']
    df_encoded, final_col_def, _ = formatter_utils.encode(
        df_interp,
        updated_col_def,
        date=date_features
    )
    
    # 5. Create Darts TimeSeries
    target_series_list = []
    future_covariates_list = []
    
    # Identify columns
    target_col = 'gl'
    future_cols = [c for c in df_encoded.columns if any(f in c for f in date_features) and c not in ['id', 'time', 'gl', 'id_segment']]
    
    # Group by original ID and Segment ID
    # Note: interpolate adds 'id_segment' which is unique across the whole dataset usually?
    # Let's check utils.py: id_segment = len(output). Yes, it increments globally.
    # So we can just group by 'id_segment'.
    
    # However, we also need the original 'id' for static covariates if we want to keep it.
    
    groups = df_encoded.groupby('id_segment')
    
    for _, group in groups:
        group = group.sort_values('time')
        
        # Create Target Series
        ts_target = TimeSeries.from_dataframe(
            group, 
            time_col='time', 
            value_cols=[target_col],
            fill_missing_dates=False # Already handled by interpolate
        )
        
        # Create Future Covariates Series
        ts_future = TimeSeries.from_dataframe(
            group,
            time_col='time',
            value_cols=future_cols,
            fill_missing_dates=False
        )
        
        # Static covariates: just the ID
        # Darts expects static covariates to be a DataFrame/Series
        # We can add it to the target series
        original_id = group['id'].iloc[0]
        # Darts 0.24+ style static covariates
        static_cov_df = pd.DataFrame({'id': [original_id]})
        ts_target = ts_target.with_static_covariates(static_cov_df)
        
        target_series_list.append(ts_target)
        future_covariates_list.append(ts_future)

    # 6. Scaling
    # If scalers are provided, use them. Else fit new ones.
    if scaler_target is None:
        scaler_target = ScalerCustom()
        target_series_scaled = scaler_target.fit_transform(target_series_list)
    else:
        target_series_scaled = scaler_target.transform(target_series_list)
        
    if scaler_covs is None:
        scaler_covs = ScalerCustom()
        future_covariates_scaled = scaler_covs.fit_transform(future_covariates_list)
    else:
        future_covariates_scaled = scaler_covs.transform(future_covariates_list)
    
    # 7. Create Inference Dataset
    dataset = SamplingDatasetInferenceDual(
        target_series=target_series_scaled,
        covariates=future_covariates_scaled,
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        use_static_covariates=True,
        array_output_only=True
    )
    
    # 8. Create Model Config with inferred feature counts
    # Check first sample to determine feature dimensions
    if len(dataset) > 0:
        sample = dataset[0]
        # Check if future_covs exists and has shape
        if len(sample) > 2 and sample[2] is not None:
             num_dynamic = sample[2].shape[1]
        else:
             num_dynamic = config.num_dynamic_features # fallback

        # Check if static_covs exists and has shape
        if len(sample) > 3 and sample[3] is not None:
             num_static = sample[3].shape[1]
        else:
             num_static = config.num_static_features # fallback
    else:
        num_dynamic = config.num_dynamic_features
        num_static = config.num_static_features
        
    # Create the output model config mapping input params to model expected fields
    model_config = GluformerModelConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_fcn=config.d_fcn,
        num_enc_layers=config.num_enc_layers,
        num_dec_layers=config.num_dec_layers,
        
        len_seq=config.input_chunk_length,
        label_len=config.input_chunk_length // 3,
        len_pred=config.output_chunk_length,
        
        num_dynamic_features=num_dynamic,
        num_static_features=num_static,
        
        r_drop=config.r_drop,
        activ=config.activ,
        distil=config.distil
    )
    
    return dataset, scaler_target, model_config
