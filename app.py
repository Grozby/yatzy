"""
Entry point for Hugging Face ZeroGPU Spaces.
Serves the Yatzy AI Studio application with real-time RL move suggestions.
"""

import os
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

from src.gui.server import app as fastapi_app

@spaces.GPU
def agent_inference_ready():
    """Hugging Face ZeroGPU heartbeat function."""
    return "Yatzy PPO Agent Ready"

# Define Gradio demo and mount our custom Yatzy FastAPI application
demo = gr.Blocks(title="Yatzy AI Studio")
app = gr.mount_gradio_app(fastapi_app, demo, path="/_gradio")

if __name__ == "__main__":
    demo.launch()
