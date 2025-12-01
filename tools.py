import sys
import os
import pickle
import gzip
from pathlib import Path
import numpy as np
import polars as pl
import torch
from scipy import stats
from lib.gluformer.model import Gluformer
from utils.darts_processing import *
from utils.darts_dataset import *
import hashlib
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import gradio as gr
from typing import Tuple, List, Optional
from plotly.graph_objs._figure import Figure
from gradio.components import Slider
from gradio.components import Markdown
from cgm_format import FormatProcessor
from fast_inference import create_inference_dataset_fast, GluformerInferenceConfig


glucose = Path(os.path.abspath(__file__)).parent.resolve()
file_directory = glucose / "files"
       
def plot_forecast(forecasts: np.ndarray, filename: str,ind:int=10) -> Tuple[Path, Figure]:
    
    forecasts = (forecasts - scalers['target'].min_) / scalers['target'].scale_

    trues = [dataset_test_glufo.evalsample(i) for i in range(len(dataset_test_glufo))]
    trues = scalers['target'].inverse_transform(trues)

    trues = [ts.values() for ts in trues]  # Convert TimeSeries to numpy arrays
    trues = np.array(trues)

    inputs = [dataset_test_glufo[i][0] for i in range(len(dataset_test_glufo))]
    inputs = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_

    # Select a specific sample to plot

    samples = np.random.normal(
        loc=forecasts[ind, :],  # Mean (center) of the distribution
        scale=0.1,  # Standard deviation (spread) of the distribution
        size=(forecasts.shape[1], forecasts.shape[2])
    )
    

    # Create figure
    fig = go.Figure()

    # Plot predictive distribution
    for point in range(samples.shape[0]):
        kde = stats.gaussian_kde(samples[point,:])
        maxi, mini = 1.2 * np.max(samples[point, :]), 0.8 * np.min(samples[point, :])
        y_grid = np.linspace(mini, maxi, 200)
        x = kde(y_grid)

        # Create gradient color
        color = f'rgba(53, 138, 217, {(point + 1) / samples.shape[0]})'
        
        # Add filled area
        fig.add_trace(go.Scatter(
            x=np.concatenate([np.full_like(y_grid, point), np.full_like(y_grid, point - x * 15)[::-1]]),
            y=np.concatenate([y_grid, y_grid[::-1]]),
            fill='tonexty',
            fillcolor=color,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
    
    
    true_values = np.concatenate([inputs[ind, -12:], trues[ind, :]])
    true_values_flat=true_values.flatten()
    
    fig.add_trace(go.Scatter(
        x=list(range(-12, 12)),
        y=true_values_flat.tolist(),  # Convert to list explicitly
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        name='True Values'
    ))

    # Plot median
    forecast = samples[:, :]
    median = np.quantile(forecast, 0.5, axis=-1)

    last_true_value = true_values_flat[11]
    median_with_anchor = [last_true_value] + median.tolist()
    median_x = [-1] + list(range(12))

    fig.add_trace(go.Scatter(
        x=median_x,
        y=median_with_anchor,  # Include anchor to connect with history
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        name='Median Forecast'
    ))


    # Update layout
    fig.update_layout(
        title='Gluformer Prediction with Gradient for dataset',
        xaxis_title='Time (in 5 minute intervals)',
        yaxis_title='Glucose (mg/dL)',
        font=dict(size=14),
        showlegend=True,
        width=1000,
        height=600
    ) 

    # Save figure
    where = file_directory / filename
    fig.write_html(str(where.with_suffix('.html')))
    fig.write_image(str(where.with_suffix('.png')))

    return where.with_suffix('.png'), fig


def generate_filename_from_url(url: str, extension: str = "png") -> str:
    """
    :param url:
    :param extension:
    :return:
    """
    # Extract the last segment of the URL
    last_segment = urlparse(url).path.split('/')[-1]

    # Compute the hash of the URL
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()

    # Create the filename
    filename = f"{last_segment.replace('.','_')}_{url_hash}.{extension}"

    return filename


glufo = None
scalers = None
dataset_test_glufo = None
filename = None
cached_forecasts: Optional[np.ndarray] = None

def prep_predict_glucose_tool(unified_df: pl.DataFrame, model_name: str = "gluformer_1samples_10000epochs_10heads_32batch_geluactivation_livia_mini_weights.pth") -> Tuple[dict, dict]:
    """
    Function to predict future glucose of user.
    
    Args:
        unified_df: Unified format DataFrame (with sequence_id, datetime, glucose columns)
        model_name: Name of the model weights file
    """
    global scalers, glufo, dataset_test_glufo, filename, cached_forecasts
    
    model = "Livia-Zaharia/gluformer_models"
    model_path = hf_hub_download(repo_id=model, filename=model_name)
    
    # Filter to glucose-only events first (keeps sequence_id unlike to_data_only_df)
    #glucose_only_df, _ = FormatProcessor.split_glucose_events(unified_df)
    
    # Drop duplicate timestamps
    #glucose_only_df = glucose_only_df.unique(subset=['datetime'], keep='first')
    
    glucose_only_df = FormatProcessor.to_data_only_df(
        unified_df,
        drop_service_columns=False,
        drop_duplicates=True,
        glucose_only=True
    )

    # Initialize Config
    # Note: We hardcode model params here as they are tied to the specific weights file we download
    config = GluformerInferenceConfig(
        input_chunk_length=96,
        output_chunk_length=12,
        d_model=512,
        n_heads=10,
        d_fcn=1024,
        num_enc_layers=2,
        num_dec_layers=2,
        gap_threshold=45, # Overriding default if needed
        min_drop_length=12
    )
    
    # Use new fast inference pipeline directly with UnifiedFormat DataFrame
    dataset_test_glufo, target_scaler, model_config = create_inference_dataset_fast(
        data=glucose_only_df,
        config=config
    )
    
    # Update global scalers
    #plot_forecast only works with the target scaler
    scalers = {'target': target_scaler}
    filename = "uploaded_data"
    cached_forecasts = None  # Reset cache whenever a new dataset is prepared

    # Load Model
    global glufo
    glufo = Gluformer(**model_config.model_dump())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    glufo.load_state_dict(torch.load(str(model_path), map_location=torch.device(device)))

    max_index = len(dataset_test_glufo) - 1

    print(f"Total number of test samples: {max_index + 1}")
    
    return (
        gr.update(
            minimum=0,
            maximum=max_index,
            value=max_index,
            step=1,
            label="Select Sample Index",
            visible=True
        ),
        gr.update(value=f"Total number of test samples: {max_index + 1}", visible=True)
    )


def predict_glucose_tool(ind: int) -> Figure:
    global cached_forecasts

    if glufo is None or dataset_test_glufo is None or filename is None:
        raise RuntimeError("Prediction requested before model preparation.")

    cache_invalid = (
        cached_forecasts is None or
        cached_forecasts.shape[0] != len(dataset_test_glufo)
    )

    if cache_invalid:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
        forecasts, _ = glufo.predict(
            dataset_test_glufo,
            batch_size=16,
            num_samples=24,
            device=device
        )
        cached_forecasts = forecasts

    _, result = plot_forecast(cached_forecasts, filename, ind)

    return result
