import gradio as gr
from tools import *
from format_dexcom import process_csv
import tempfile
import os

def process_and_prepare(file):
    """Process the raw CSV and prepare it for prediction"""
    # Create a temporary file for the processed CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        processed_path = tmp_file.name
    
    # Process the CSV file
    process_csv(
        input_dir=file.name,
        output_file=processed_path
    )
    
    # Run the preparation step with processed file
    return prep_predict_glucose_tool(processed_path)

with gr.Blocks() as demo:
    gr.Markdown("# Glucose Prediction Tool")
    gr.Markdown("Upload a Dexcom CSV file to get predictions")
    
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
        inputs=[file_input],
        outputs=[index_slider, sample_count],
        queue=True
    )
    
    # Update visibility after processing
    file_input.change(
        fn=lambda: (gr.Slider(visible=True), gr.Markdown(visible=True)),
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

demo.launch()