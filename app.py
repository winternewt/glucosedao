import gradio as gr
from tools import *
from cgm_format import FormatParser, FormatProcessor
from cgm_format.interface import ProcessingWarning, WarningDescription
import tempfile
import os
from huggingface_hub import list_models
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
import plotly.graph_objects as go
from huggingface_hub import HfApi
import polars as pl

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


def _format_warnings_for_display(warning_flags: ProcessingWarning) -> Dict[str, Any]:
    """Format ProcessingWarning flags for Gradio UI display.
    
    Args:
        warning_flags: Combined ProcessingWarning flags from FormatProcessor
        
    Returns:
        Dictionary with UI-ready warning information
    """
    warnings_list = []
    
    if warning_flags & ProcessingWarning.TOO_SHORT:
        warnings_list.append(f"TOO_SHORT: {WarningDescription.TOO_SHORT.value}")
    
    if warning_flags & ProcessingWarning.CALIBRATION:
        warnings_list.append(f"CALIBRATION: {WarningDescription.CALIBRATION.value}")
    
    if warning_flags & ProcessingWarning.QUALITY:
        warnings_list.append(f"QUALITY: {WarningDescription.QUALITY.value}")
    
    if warning_flags & ProcessingWarning.IMPUTATION:
        warnings_list.append(f"IMPUTATION: {WarningDescription.IMPUTATION.value}")
    
    if warning_flags & ProcessingWarning.OUT_OF_RANGE:
        warnings_list.append(f"OUT_OF_RANGE: {WarningDescription.OUT_OF_RANGE.value}")
    
    if warning_flags & ProcessingWarning.TIME_DUPLICATES:
        warnings_list.append(f"TIME_DUPLICATES: {WarningDescription.TIME_DUPLICATES.value}")
    
    return {
        'warnings': warning_flags,
        'has_warnings': len(warnings_list) > 0,
        'warning_messages': warnings_list,
        'warning_count': len(warnings_list)
    }


def process_and_prepare(file: tempfile._TemporaryFileWrapper, model_name: str) -> Tuple[gr.Slider, gr.Markdown, Dict[str, Any]]:
    """Process the raw CSV and prepare it for prediction with quality checks.
    
    Args:
        file: Uploaded temporary file object
        model_name: Name of the selected model
        
    Returns:
        Tuple containing:
        - Updated slider component
        - Sample count markdown component
        - Warning information dictionary
    """
    print(f"🔍 Reading and checking data quality...")
    
    try:
        # Parse file using FormatParser
        unified_df = FormatParser.parse_file(file.name)
        print(f"✅ Parsed {len(unified_df)} rows to unified format")
        
        # Process and prepare for inference (with interpolation and quality checks)
        # Use 45 minutes to match old gap_threshold from config.yaml
        processor = FormatProcessor(
            expected_interval_minutes=5,
            small_gap_max_minutes=15
        )
        unified_df = processor.interpolate_gaps(unified_df)
        unified_df = processor.synchronize_timestamps(unified_df)
        
        # Prepare for inference with quality checks
        inference_df, warning_flags = processor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=15,  # 15 minutes minimum
            maximum_wanted_duration=24 * 60  # 24 hours maximum (1440 minutes)
        )
        
        # FormatParser.to_csv_file(inference_df, "inference_df.csv")
        
        # Format warnings for Gradio display
        warning_info = _format_warnings_for_display(warning_flags)
        print(f"✅ Quality check completed. Warnings: {warning_info['has_warnings']}")
        if warning_info['has_warnings']:
            print(f"⚠️  Warning messages: {warning_info['warning_messages']}")
        
        print(f"✅ Prepared {len(inference_df)} inference-ready rows")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        warning_info = {
            'has_warnings': True,
            'warning_messages': [f"Error during processing: {str(e)}"],
            'warning_count': 1
        }
        inference_df = None
    
    print(f"✅ Preparing prediction tool...")
    if inference_df is not None:
        # Pass unified DataFrame directly - conversion to legacy format happens in prep_predict_glucose_tool
        result = prep_predict_glucose_tool(inference_df, model_name)
    else:
        # If processing failed, we can't proceed - raise or return error
        raise RuntimeError("Failed to process input file. Check error messages above.")
    print(f"🔍 Returning: {result}")
    
    # Return slider, sample count, and warning info
    return result[0], result[1], warning_info


def create_warning_display(warning_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create warning indicator display components.
    
    Args:
        warning_info: Dictionary with warning information
        
    Returns:
        Dictionary with HTML for warning indicators
    """
    if not warning_info or not warning_info.get('has_warnings', False):
        # Green light - all good
        status_html = """
        <div style="padding: 15px; border-radius: 8px; background-color: #d4edda; border: 2px solid #28a745; margin: 10px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background-color: #28a745; box-shadow: 0 0 10px #28a745;"></div>
                <div style="flex: 1;">
                    <h3 style="margin: 0; color: #155724;">✅ Data Quality: GOOD</h3>
                    <p style="margin: 5px 0 0 0; color: #155724;">No quality issues detected. Data is ready for inference.</p>
                </div>
            </div>
        </div>
        """
        return gr.update(value=status_html, visible=True)
    else:
        # Red light - warnings present
        warning_messages = warning_info.get('warning_messages', [])
        warning_list_html = "".join([f"<li>{msg}</li>" for msg in warning_messages])
        
        status_html = f"""
        <div style="padding: 15px; border-radius: 8px; background-color: #f8d7da; border: 2px solid #dc3545; margin: 10px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background-color: #dc3545; box-shadow: 0 0 10px #dc3545;"></div>
                <div style="flex: 1;">
                    <h3 style="margin: 0; color: #721c24;">⚠️ Data Quality: WARNINGS DETECTED</h3>
                    <p style="margin: 5px 0; color: #721c24;"><strong>{warning_info.get('warning_count', 0)} warning(s) found:</strong></p>
                    <ul style="margin: 5px 0 0 20px; color: #721c24;">
                        {warning_list_html}
                    </ul>
                </div>
            </div>
        </div>
        """
        return gr.update(value=status_html, visible=True)


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
    
    # Warning display with flag lamps
    warning_display = gr.HTML(
        label="Data Quality Status",
        visible=False
    )
    
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
    
    # Store warning info in state
    warning_state = gr.State(value={})
    
    # Update slider and show total samples when file is uploaded
    upload_event = file_input.change(
        fn=process_and_prepare,
        inputs=[file_input, model_selector],
        outputs=[index_slider, sample_count, warning_state],
        queue=True
    )
    
    # Display warnings after processing
    upload_event.then(
        fn=create_warning_display,
        inputs=[warning_state],
        outputs=warning_display,
        queue=True
    )
    
    # Automatically generate plot after upload completes
    upload_event.then(
        fn=predict_glucose_tool,
        inputs=[index_slider],
        outputs=plot_output,
        queue=True
    )
    
    # Also allow manual update when slider changes
    index_slider.change(
        fn=predict_glucose_tool,
        inputs=[index_slider],
        outputs=plot_output,
        queue=True
    )

demo.launch(share=True)