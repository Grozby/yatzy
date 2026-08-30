"""
Entry point for Hugging Face Spaces (Gradio SDK / Python).
Mounts the Yatzy AI Studio FastAPI application onto Gradio and serves on port 7860.
"""

import os
import gradio as gr
import uvicorn
from src.gui.server import app as fastapi_app

# Create Gradio Blocks and mount our full custom Yatzy FastAPI app
demo = gr.Blocks(title="Yatzy AI Studio")
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
