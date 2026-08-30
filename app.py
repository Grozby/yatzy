"""
Entry point for Hugging Face Spaces (Gradio SDK / Python).
Exposes the Yatzy AI Studio FastAPI application as 'app'.
"""

import gradio as gr
from src.gui.server import app as fastapi_app

# Create a Gradio Blocks instance and mount our full custom Yatzy FastAPI app
demo = gr.Blocks(title="Yatzy AI Studio")
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
