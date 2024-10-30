import gradio as gr
from tools import *


with gr.Blocks() as demo:
    file_input = gr.File(label="Upload CSV File")
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
        fn=prep_predict_glucose_tool,
        inputs=[file_input],
        outputs=[index_slider, sample_count],
        queue=False
    )
        # Set visibility separately
    file_input.change(
        fn=lambda: (gr.Slider(visible=True), gr.Markdown(visible=True)),
        outputs=[index_slider, sample_count]
    )

    # Update plot when slider changes or file uploads
    file_input.change(
        fn=predict_glucose_tool,
        inputs=[index_slider],
        outputs=plot_output
    )
    index_slider.change(
        fn=predict_glucose_tool,
        inputs=[index_slider],
        outputs=plot_output
    )

demo.launch()