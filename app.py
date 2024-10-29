import gradio as gr
from tools import *


def gradio_output(file):
    return (predict_glucose_tool(file))

gr.Interface(fn=gradio_output,inputs=gr.File(label="Upload CSV File"),outputs="plot").launch()
