"""
Entry point for Hugging Face Spaces (Gradio SDK / Python).
Serves the Yatzy AI Studio application on port 7860.
"""

import os
import uvicorn
import gradio as gr
from src.gui.server import app as fastapi_app

# Define Gradio demo for Hugging Face Spaces integration and mount FastAPI
demo = gr.Blocks(title="Yatzy AI Studio")
app = gr.mount_gradio_app(fastapi_app, demo, path="/_gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
