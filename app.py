"""
Entry point for Hugging Face Spaces (Gradio SDK / ZeroGPU).
Mounts the Yatzy AI Studio FastAPI application onto Gradio and serves on port 7860.
"""

import os
import uvicorn
import gradio as gr

# Safe import of Hugging Face ZeroGPU spaces
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is None:
                return lambda f: f
            return fn

@spaces.GPU
def agent_inference_ready():
    """Hugging Face ZeroGPU heartbeat function."""
    return "Yatzy PPO Agent Ready"

from src.gui.server import app as fastapi_app

# Define Gradio demo and mount onto FastAPI
demo = gr.Blocks(title="Yatzy AI Studio")
app = gr.mount_gradio_app(fastapi_app, demo, path="/_gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
