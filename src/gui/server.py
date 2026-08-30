"""
FastAPI Server for Yatzy GUI with AI Assistant & Auto-Play.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.gui.service import GameService

app = FastAPI(
    title="Yatzy AI Player & Assistant",
    description="Interactive Web GUI for playing Yatzy with real-time AI move suggestions and full agent auto-play.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory setup
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared game service singleton
service = GameService()


# ----------------------------------------------------------------------
# Request / Response Schemas
# ----------------------------------------------------------------------
class NewGameRequest(BaseModel):
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducible dice rolls")


class RollRequest(BaseModel):
    keep_mask: List[bool] = Field(
        default=[False, False, False, False, False],
        description="Boolean array for each die (True = keep, False = reroll)",
    )


class CategoryRequest(BaseModel):
    category_index: int = Field(..., ge=0, le=14, description="Category index (0-14)")


class LoadCheckpointRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the .pt checkpoint file")


# ----------------------------------------------------------------------
# REST Endpoints
# ----------------------------------------------------------------------
@app.get("/")
def get_index():
    """Serve the single-page application frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_file)


@app.get("/api/checkpoints")
def list_checkpoints():
    """List all available model checkpoints and indicate the active one."""
    checkpoints = service.discover_checkpoints()
    return {
        "active_checkpoint": service.active_checkpoint_path,
        "checkpoints": checkpoints,
    }


@app.post("/api/checkpoints/load")
def load_checkpoint(req: LoadCheckpointRequest):
    """Load a specific model checkpoint."""
    success = service.load_checkpoint(req.path)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint from: {req.path}")
    return {
        "success": True,
        "active_checkpoint": service.active_checkpoint_path,
        "data": service.get_full_state(),
    }


@app.post("/api/game/new")
def new_game(req: Optional[NewGameRequest] = None):
    """Start a brand new Yatzy game."""
    seed = req.seed if req else None
    state = service.start_new_game(seed=seed)
    return {
        "success": True,
        "data": state,
    }


@app.get("/api/game/state")
def get_state():
    """Get current game state, scores, and AI suggestions."""
    return {
        "success": True,
        "data": service.get_full_state(),
    }


@app.post("/api/game/roll")
def roll_dice(req: RollRequest):
    """Re-roll unkept dice."""
    result = service.roll_dice(req.keep_mask)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Roll failed"))
    return result


@app.post("/api/game/select-category")
def select_category(req: CategoryRequest):
    """Assign current roll to a category."""
    result = service.select_category(req.category_index)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Category selection failed"))
    return result


@app.post("/api/game/step-ai")
def step_ai():
    """Execute a single optimal action chosen by the RL model."""
    result = service.step_ai()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "AI step failed"))
    return result
