"""
Entry point for Hugging Face Spaces (Gradio SDK / Python).
Serves the Yatzy AI Studio Web GUI on port 7860.
"""

import os
import uvicorn
from src.gui.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
