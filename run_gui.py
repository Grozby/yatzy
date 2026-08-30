"""
Launcher script for the Yatzy AI Web GUI.
Usage:
    uv run python run_gui.py [--port 8000] [--host 127.0.0.1] [--no-browser]
"""

import argparse
import sys
import threading
import time
import webbrowser
import uvicorn


def open_browser_delayed(url: str, delay: float = 1.0):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Yatzy RL Agent & Interactive Player GUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print(" 🎲 Yatzy AI Interactive GUI & Agent Visualizer 🎲")
    print("=" * 60)
    print(f"Server starting at: {url}")
    print(f"Press CTRL+C to stop the server.")
    print("=" * 60)

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url, 1.2), daemon=True).start()

    uvicorn.run("src.gui.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
