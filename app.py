import gradio as gr
from tools import *
from input_processing import process_csv_with_options
import tempfile
import os
from huggingface_hub import list_models
from typing import List, Tuple
from pathlib import Path
import plotly.graph_objects as go
from huggingface_hub import HfApi

def get_available_models() -> List[str]:
    """Get list of available gluformer models from HuggingFace."""
    api = HfApi()
    files = api.list_repo_files("Livia-Zaharia/gluformer_models")
    
    # Filter for .pth files
    gluformer_models = [
        file for file in files 
        if file.endswith('.pth') and "weights" in file.lower() and 'gluformer' in file.lower()
    ]
    
    return gluformer_models

AVAILABLE_MODELS = get_available_models()
print(AVAILABLE_MODELS)

def process_and_prepare(file: tempfile._TemporaryFileWrapper, model_name: str) -> Tuple[gr.Slider, gr.Markdown]:
    """Process the raw CSV and prepare it for prediction.
    
    Args:
        file: Uploaded temporary file object
        model_name: Name of the selected model
        
    Returns:
        Tuple containing:
        - Updated slider component
        - Sample count markdown component
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        processed_path = tmp_file.name
    
    process_csv_with_options(
        input_file=Path(file.name),
        output_file=Path(processed_path)
    )
    
    return prep_predict_glucose_tool(processed_path, model_name)


with gr.Blocks() as demo:
    gr.Markdown("# Glucose Prediction Tool")
    gr.Markdown("Upload a Dexcom CSV file to get predictions")
    
    model_selector = gr.Dropdown(
        choices=AVAILABLE_MODELS,
        value="gluformer_1samples_500epochs_10heads_32batch_geluactivation_livia_large_weights.pth",
        label="Select Model",
        interactive=True
    )
    file_input = gr.File(label="Upload Raw Dexcom CSV File")
    with gr.Row():
        index_slider = gr.Slider(
            minimum=0,
            maximum=100,  # This will be updated dynamically
            value=10,
            step=1,
            label="Select Sample Index",
            visible=False
        )
    sample_count = gr.Markdown(visible=False)
    plot_output = gr.Plot()
    
    # Update slider and show total samples when file is uploaded
    file_input.change(
        fn=process_and_prepare,
        inputs=[file_input, model_selector],
        outputs=[index_slider, sample_count],
        queue=True
    )
    
    # Only update plot after processing is complete
    index_slider.change(
        fn=predict_glucose_tool,
        inputs=[index_slider],
        outputs=plot_output,
        queue=True
    )

demo.launch(share=True)