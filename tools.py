import sys
import os
import pickle
import gzip
from pathlib import Path

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import torch
from scipy import stats

from gluformer.model import Gluformer
from utils.darts_processing import *
from utils.darts_dataset import *


import hashlib
from urllib.parse import urlparse

import numpy as np
import typer


glucose = Path(os.path.abspath(__file__)).parent.resolve()
file_directory = glucose / "files"


def plot_forecast(forecasts: np.ndarray, scalers: Any, dataset_test_glufo: Any, filename: str):
    filename=filename
    forecasts = (forecasts - scalers['target'].min_) / scalers['target'].scale_

    trues = [dataset_test_glufo.evalsample(i) for i in range(len(dataset_test_glufo))]
    trues = scalers['target'].inverse_transform(trues)

    trues = [ts.values() for ts in trues]  # Convert TimeSeries to numpy arrays
    trues = np.array(trues)

    inputs = [dataset_test_glufo[i][0] for i in range(len(dataset_test_glufo))]
    inputs = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_

    # Plot settings
    colors = ['#00264c', '#0a2c62', '#14437f', '#1f5a9d', '#2973bb', '#358ad9', '#4d9af4', '#7bb7ff', '#add5ff', '#e6f3ff']
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    sns.set_theme(style="whitegrid")

    # Generate the plot
    fig, ax = plt.subplots(figsize=(10, 6))


    # Select a specific sample to plot
    ind = 30  # Example index

    samples = np.random.normal(
        loc=forecasts[ind, :],  # Mean (center) of the distribution
        scale=0.1,  # Standard deviation (spread) of the distribution
        size=(forecasts.shape[1], forecasts.shape[2])
    )
    #samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    #print ("samples",samples.shape)

    # Plot predictive distribution
    for point in range(samples.shape[0]):
        kde = stats.gaussian_kde(samples[point,:])
        maxi, mini = 1.2 * np.max(samples[point, :]), 0.8 * np.min(samples[point, :])
        y_grid = np.linspace(mini, maxi, 200)
        x = kde(y_grid)
        ax.fill_betweenx(y_grid, x1=point, x2=point - x * 15,
                         alpha=0.7,
                         edgecolor='black',
                         color=cmap(point / samples.shape[0]))

    # Plot median
    forecast = samples[:, :]
    median = np.quantile(forecast, 0.5, axis=-1)
    ax.plot(np.arange(12), median, color='red', marker='o')

    # Plot true values
    ax.plot(np.arange(-12, 12), np.concatenate([inputs[ind, -12:], trues[ind, :]]), color='blue')

    # Add labels and title
    ax.set_xlabel('Time (in 5 minute intervals)')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_title(f'Gluformer Prediction with Gradient for dateset')

    # Adjust font sizes
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(18)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(14)

    # Save figure
    plt.tight_layout()
    where = file_directory /filename
    plt.savefig(str(where), dpi=300, bbox_inches='tight')

    return where,ax



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



def predict_glucose_tool(url: str= 'https://huggingface.co/datasets/Livia-Zaharia/glucose_processed/blob/main/livia_mini.csv',
                        model: str = 'https://huggingface.co/Livia-Zaharia/gluformer_models/blob/main/gluformer_1samples_10000epochs_10heads_32batch_geluactivation_livia_mini_weights.pth'
                    ) -> Figure:
    """
    Function to predict future glucose of user. It receives URL with users csv. It will run an ML and will return URL with predictions that user can open on her own..
    :param url: of the csv file with glucose values
    :param model: model that is used to predict the glucose
    :param explain if it should give both url and explanation
    :param if the person is diabetic when doing prediction and explanation
    :return:
    """

    formatter, series, scalers = load_data(url=str(url), config_path=file_directory / "config.yaml", use_covs=True,
                                           cov_type='dual',
                                           use_static_covs=True)

    filename = generate_filename_from_url(url)

    formatter.params['gluformer'] = {
        'in_len': 96,  # example input length, adjust as necessary
        'd_model': 512,  # model dimension
        'n_heads': 10,  # number of attention heads##############################################################################
        'd_fcn': 1024,  # fully connected layer dimension
        'num_enc_layers': 2,  # number of encoder layers
        'num_dec_layers': 2,  # number of decoder layers
        'length_pred': 12  # prediction length, adjust as necessary
    }

    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components

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
    weights = gr.Interface.load(model)
    assert f"weights for {model} should exist", weights.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    glufo.load_state_dict(torch.load(str(weights), map_location=torch.device(device), weights_only=False))

    # Define dataset for inference
    dataset_test_glufo = SamplingDatasetInferenceDual(
        target_series=series['test']['target'],
        covariates=series['test']['future'],
        input_chunk_length=formatter.params['gluformer']['in_len'],
        output_chunk_length=formatter.params['length_pred'],
        use_static_covariates=True,
        array_output_only=True
    )

    forecasts, _ = glufo.predict(
        dataset_test_glufo,
        batch_size=16,####################################################
        num_samples=10,
        device='cpu'
    )
    figure_path, result = plot_forecast(forecasts, scalers, dataset_test_glufo,filename)
    
    return result



if __name__ == "__main__":
    predict_glucose_tool()
