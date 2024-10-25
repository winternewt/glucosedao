import gradio as gr
from tools import *


def gradio_output():
    return (predict_glucose_tool())

gr.Interface(fn=gradio_output).launch()
