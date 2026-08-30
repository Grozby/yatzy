---
title: Yatzy AI Studio
emoji: 🎲
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Deep RL (PPO) agent for Yatzy with interactive Web GUI & AI suggestions
tags:
  - reinforcement-learning
  - ppo
  - pytorch
  - fastapi
  - game-ai
---

# 🎲 Yatzy AI Studio

A Deep Reinforcement Learning (PPO) agent for Yatzy with an interactive Web GUI and real-time AI assistant.

---

## 🌟 Overview

This project implements a complete Yatzy game environment, a Proximal Policy Optimization (PPO) reinforcement learning agent capable of scoring 230+ points, and an interactive Web GUI for human-AI play and agent visualization.

### Key Features
- **🎮 Interactive Web GUI**: 3D styled dice, interactive scorecard, roll animations, and synthesized sound effects.
- **💡 Real-Time AI Suggestions**: Keep probabilities on dice, category ranking badges, and expected score estimations from the PPO Critic value head.
- **🤖 Autonomous Agent Mode**: Watch the model play full 15-turn games end-to-end with adjustable speeds (Slow to Turbo) or advance step-by-step.
- **🔄 Checkpoint Discovery**: Auto-detects and switches between training checkpoints on the fly.

---

## 🚀 Quickstart

### 1. Install Dependencies
Using `uv` (recommended) or `pip`:
```bash
uv sync
# or
pip install -e .
```

### 2. Launch the Web GUI
```bash
uv run python run_gui.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🧠 Training the Agent

To train the PPO model from scratch or resume training:
```bash
uv run python -m src.training.train_ppo
```
Training runs and checkpoints are automatically saved to `experiments/run_<timestamp>/`.

To run quick evaluation episodes:
```bash
uv run python -m src.training.run_episodes
```

---

## 📁 Project Structure

```
├── run_gui.py              # Web GUI launcher script
├── ppo_yatzy.pt            # Pre-trained baseline model weights
├── pyproject.toml          # Project configuration and dependencies
├── experiments/            # Training runs, checkpoints, and metrics
└── src/
    ├── game/               # Game rules, scoring logic, and Gymnasium environment
    │   ├── environment.py  # YatzyEnvironment with observation and action spaces
    │   ├── scoring.py      # Category evaluation rules (Upper section, Yatzy, etc.)
    │   └── state.py        # Game state snapshot utilities
    ├── training/           # RL algorithm implementation
    │   ├── agent.py        # PPO Agent logic & action selection
    │   ├── ppo_network.py  # Actor-Critic neural network architecture
    │   ├── buffer.py       # Rollout experience buffer
    │   └── train_ppo.py    # Training loop with Hydra configuration
    └── gui/                # Web application backend and frontend
        ├── server.py       # FastAPI application endpoints
        ├── service.py      # Game session management & model analysis engine
        └── static/         # Web UI assets (HTML, CSS, JS)
```

---

## ⌨️ Controls & Shortcuts

| Action | Shortcut |
| :--- | :--- |
| **Roll Dice / Apply AI Move** | `Space` |
| **Toggle Dice 1 – 5** | `1`, `2`, `3`, `4`, `5` |
| **Toggle Auto Play** | `A` |
| **New Game** | `N` |
