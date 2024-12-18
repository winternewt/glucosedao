import sys
import os
import pickle
import gzip
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from gluformer.model import Gluformer
from utils.darts_processing import *
from utils.darts_dataset import *
import hashlib
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import gradio as gr
from format_dexcom import *
from typing import Tuple, Union, List
from plotly.graph_objs._figure import Figure
from gradio.components import Slider
from gradio.components import Markdown


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

    fig.add_trace(go.Scatter(
        x=list(range(12)),
        y=median.tolist(),  # Convert to list explicitly
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
    fig.write_image(str(where))

    return where, fig


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

def prep_predict_glucose_tool(file: Union[str, Path], model_name: str = "gluformer_1samples_10000epochs_10heads_32batch_geluactivation_livia_mini_weights.pth") -> Tuple[Slider, Markdown]:
    """
    Function to predict future glucose of user.
    """
    global formatter, series, scalers, glufo, dataset_test_glufo, filename
    
    model = "Livia-Zaharia/gluformer_models"
    model_path = hf_hub_download(repo_id=model, filename=model_name)
    
    formatter, series, scalers = load_data(
        url=str(file), 
        config_path=file_directory / "config.yaml",
        use_covs=True,
        cov_type='dual',
        use_static_covs=True
    )

    formatter.params['gluformer'] = {
        'in_len': 96,  # example input length, adjust as necessary
        'd_model': 512,  # model dimension
        'n_heads': 10,  # number of attention heads########################
        'd_fcn': 1024,  # fully connected layer dimension
        'num_enc_layers': 2,  # number of encoder layers
        'num_dec_layers': 2,  # number of decoder layers
        'length_pred': 12  # prediction length, adjust as necessary represents 1 h
    }

    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components

    global glufo
    glufo = Gluformer(
        d_model=formatter.params['gluformer']['d_model'],
        n_heads=formatter.params['gluformer']['n_heads'],
        d_fcn=formatter.params['gluformer']['d_fcn'],
        r_drop=0.2,
        activ='gelu',
        num_enc_layers=formatter.params['gluformer']['num_enc_layers'],
        num_dec_layers=formatter.params['gluformer']['num_dec_layers'],
        distil=True,
        len_seq=formatter.params['gluformer']['in_len'],
        label_len=formatter.params['gluformer']['in_len'] // 3,
        len_pred=formatter.params['length_pred'],
        num_dynamic_features=num_dynamic_features,
        num_static_features=num_static_features
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    glufo.load_state_dict(torch.load(str(model_path), map_location=torch.device(device)))

    global dataset_test_glufo
    dataset_test_glufo = SamplingDatasetInferenceDual(
        target_series=series['test']['target'],
        covariates=series['test']['future'],
        input_chunk_length=formatter.params['gluformer']['in_len'],
        output_chunk_length=formatter.params['length_pred'],
        use_static_covariates=True,
        array_output_only=True
    )

    global filename
    filename = generate_filename_from_url(file)

    max_index = len(dataset_test_glufo) - 1

    print(f"Total number of test samples: {max_index + 1}")
    
    return (
        gr.Slider(
            minimum=0,
            maximum=max_index-1,
            value=max_index,
            step=1,
            label="Select Sample Index",
            visible=True
        ),
        gr.Markdown(f"Total number of test samples: {max_index + 1}", visible=True)
    )


def predict_glucose_tool(ind: int) -> Figure:
       
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    forecasts, _ = glufo.predict(
      dataset_test_glufo,
      batch_size=16,#######
      num_samples=10,
      device=device
    )
    figure_path, result = plot_forecast(forecasts,filename,ind)
    
    return result
