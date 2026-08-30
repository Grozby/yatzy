"""
Game Service module for the Yatzy GUI.
Manages game state, agent interaction, model checkpoint discovery, and AI suggestions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.game.environment import ActionType, YatzyEnvironment
from src.training.agent import PPOAgent


class GameService:
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path(__file__).resolve().parents[2]
        self.env = YatzyEnvironment()
        self.obs_dim = self.env.observation_space.shape[0]
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            num_dices=self.env.num_dices,
            num_categories=self.env.num_categories,
            device="cpu",
        )
        self.active_checkpoint_path: Optional[str] = None
        self.last_step_info: Dict[str, Any] = {}
        self.is_game_over: bool = False
        self.game_history: List[Dict[str, Any]] = []

        # Auto-discover and load the best checkpoint
        self.load_best_checkpoint()
        # Initialize first game
        self.start_new_game()

    def discover_checkpoints(self) -> List[Dict[str, Any]]:
        """Find all available .pt model checkpoints in the workspace."""
        checkpoints: List[Dict[str, Any]] = []

        # 1. Root checkpoint (single deploy model)
        root_ckpt = self.workspace_root / "ppo_yatzy.pt"
        if root_ckpt.is_file():
            checkpoints.append(
                {
                    "path": str(root_ckpt.resolve()),
                    "name": "Trained PPO Model",
                    "relative_path": "ppo_yatzy.pt",
                    "score_mean": 226.4,
                    "info": "Trained PPO Policy",
                    "mtime": root_ckpt.stat().st_mtime,
                }
            )

        # 2. Experiments checkpoints
        exp_dir = self.workspace_root / "experiments"
        if exp_dir.is_dir():
            for pt_file in exp_dir.rglob("*.pt"):
                info_file = pt_file.parent / "best_model_info.json"
                score_mean = 0.0
                run_folder = pt_file.parent.name
                if pt_file.parent.name == "checkpoints":
                    run_folder = pt_file.parent.parent.name
                    info_file = pt_file.parent.parent / "best_model_info.json"

                info_text = pt_file.name
                if info_file.is_file():
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            score_mean = float(meta.get("best_running_mean", meta.get("best_rollout_mean", 0.0)))
                            timestamp = meta.get("timestamp", "")
                            epoch = meta.get("best_epoch", "")
                            info_text = f"Best Epoch {epoch} (Mean: {score_mean:.1f})"
                    except Exception:
                        pass

                rel_path = str(pt_file.relative_to(self.workspace_root))
                display_name = f"{run_folder} → {pt_file.name}"
                checkpoints.append(
                    {
                        "path": str(pt_file.resolve()),
                        "name": display_name,
                        "relative_path": rel_path,
                        "score_mean": score_mean,
                        "info": info_text,
                        "mtime": pt_file.stat().st_mtime,
                    }
                )

        # Sort: highest score_mean first, then latest modification time
        checkpoints.sort(key=lambda c: (c["score_mean"], c["mtime"]), reverse=True)
        return checkpoints

    def load_best_checkpoint(self) -> bool:
        """Load the best available checkpoint found."""
        checkpoints = self.discover_checkpoints()
        if checkpoints:
            return self.load_checkpoint(checkpoints[0]["path"])
        return False

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model weights from a specific path."""
        path = Path(checkpoint_path)
        if not path.is_file():
            return False

        try:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            self.agent.policy.load_state_dict(state_dict)
            self.agent.eval()
            self.active_checkpoint_path = str(path.resolve())
            return True
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return False

    def start_new_game(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment to start a fresh game."""
        import time
        actual_seed = seed if seed is not None else int(time.time() * 1000000) % 2_000_000
        obs, info = self.env.reset(seed=actual_seed)
        self.is_game_over = False
        self.last_step_info = {
            "message": "New game started. Roll 1 of 3.",
            "reward": 0.0,
            "action_type": "SELECT_DICE",
            "roll": 0,
            "turn": 0,
        }
        self.game_history = [
            {
                "type": "game_start",
                "message": "Game started. Initial roll: " + ", ".join(map(str, self.env.dices)),
                "turn": 1,
                "dices": self.env.dices.tolist(),
            }
        ]
        return self.get_full_state()

    def get_game_state(self) -> Dict[str, Any]:
        """Extract current game state for frontend display."""
        current_action_type_name = "SELECT_DICE" if self.env.current_action_type == ActionType.SELECT_DICE else "SELECT_CATEGORY"
        
        # Determine rolls remaining
        if self.env.current_action_type == ActionType.SELECT_DICE:
            rolls_remaining = max(0, self.env.max_rolls - (self.env.current_roll + 1))
        else:
            rolls_remaining = 0

        # Potential scores for current dice
        potential_scores = [
            int(cat.evaluate(self.env.dices)) if not self.env.category_filled[i] else int(self.env.category_scores[i])
            for i, cat in enumerate(self.env.scoring_categories)
        ]

        total_score = int(self.env.get_score())
        upper_sum = int(self.env.upper_section_sum)
        bonus_awarded = upper_sum >= self.env.upper_section_score_threshold

        return {
            "dices": self.env.dices.tolist(),
            "current_roll": int(self.env.current_roll),
            "max_rolls": int(self.env.max_rolls),
            "rolls_remaining": rolls_remaining,
            "turn_number": int(self.env.turn_number),
            "total_turns": int(self.env.num_categories),
            "current_action_type": current_action_type_name,
            "action_type_int": int(self.env.current_action_type),
            "categories": [
                {
                    "index": i,
                    "name": cat.name,
                    "display_name": self._format_category_name(cat.name),
                    "is_upper": bool(cat.is_upper_section),
                    "is_filled": bool(self.env.category_filled[i]),
                    "score": int(self.env.category_scores[i]) if self.env.category_filled[i] else None,
                    "potential_score": int(cat.evaluate(self.env.dices)),
                }
                for i, cat in enumerate(self.env.scoring_categories)
            ],
            "upper_section_sum": upper_sum,
            "upper_section_threshold": int(self.env.upper_section_score_threshold),
            "upper_section_bonus_points": int(self.env.upper_section_bonus_points),
            "has_upper_bonus": bonus_awarded,
            "total_score": total_score,
            "is_game_over": self.is_game_over or (self.env.turn_number >= self.env.num_categories),
        }

    def _format_category_name(self, name: str) -> str:
        """Convert CamelCase category name to readable title with spaces."""
        replacements = {
            "Ones": "Ones (1s)",
            "Twos": "Twos (2s)",
            "Threes": "Threes (3s)",
            "Fours": "Fours (4s)",
            "Fives": "Fives (5s)",
            "Sixes": "Sixes (6s)",
            "OnePair": "One Pair",
            "TwoPairs": "Two Pairs",
            "ThreeOfAKind": "Three of a Kind",
            "FourOfAKind": "Four of a Kind",
            "SmallStraight": "Small Straight (1-2-3-4-5)",
            "LargeStraight": "Large Straight (2-3-4-5-6)",
            "FullHouse": "Full House",
            "Chance": "Chance",
            "Yatzy": "Yatzy (5 of a kind)",
        }
        return replacements.get(name, name)

    def get_model_suggestions(self) -> Dict[str, Any]:
        """Compute comprehensive model suggestions for current state."""
        obs = self.env._get_observation()
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        self.agent.eval()
        with torch.no_grad():
            dice_logits, cat_logits, value = self.agent.policy(obs_t)
            dice_logits = dice_logits.squeeze(0)
            cat_logits = cat_logits.squeeze(0)
            expected_value = float(value.squeeze(0).item())

            # Dice keep suggestions
            keep_probs = torch.sigmoid(dice_logits).cpu().numpy().tolist()
            recommended_keep_mask = (dice_logits > 0.0).cpu().numpy().tolist()

            # Category ranking suggestions
            cat_probs = torch.softmax(cat_logits, dim=-1).cpu().numpy().tolist()

        # Build detailed category rankings for unfilled categories
        category_rankings = []
        for i, cat in enumerate(self.env.scoring_categories):
            is_filled = bool(self.env.category_filled[i])
            potential_pts = int(cat.evaluate(self.env.dices))
            prob = float(cat_probs[i]) if not is_filled else 0.0
            category_rankings.append(
                {
                    "index": i,
                    "name": cat.name,
                    "display_name": self._format_category_name(cat.name),
                    "is_filled": is_filled,
                    "potential_score": potential_pts,
                    "probability": prob,
                    "raw_logit": float(cat_logits[i].item()),
                }
            )

        # Sort unfilled by AI probability
        unfilled = [c for c in category_rankings if not c["is_filled"]]
        unfilled.sort(key=lambda x: x["probability"], reverse=True)
        for rank_idx, item in enumerate(unfilled, 1):
            item["rank"] = rank_idx

        best_category = unfilled[0] if unfilled else None

        # Build human-readable explanation
        if self.env.current_action_type == ActionType.SELECT_DICE:
            kept_indices = [i for i, k in enumerate(recommended_keep_mask) if k]
            kept_values = [int(self.env.dices[i]) for i in kept_indices]
            if len(kept_indices) == 5:
                explanation = "AI recommends keeping ALL dice (moving directly to category scoring)."
            elif len(kept_indices) == 0:
                explanation = "AI recommends re-rolling ALL dice for a better hand."
            else:
                explanation = f"AI recommends keeping dice {kept_values} (positions {[i+1 for i in kept_indices]})."
        else:
            if best_category:
                explanation = (
                    f"AI top choice: {best_category['display_name']} "
                    f"({best_category['probability'] * 100:.1f}% confidence, scores {best_category['potential_score']} pts)."
                )
            else:
                explanation = "Game complete."

        return {
            "expected_value": expected_value,
            "dice_keep_probs": keep_probs,
            "recommended_keep_mask": recommended_keep_mask,
            "category_rankings": category_rankings,
            "best_category": best_category,
            "explanation": explanation,
            "action_type": "SELECT_DICE" if self.env.current_action_type == ActionType.SELECT_DICE else "SELECT_CATEGORY",
        }

    def get_full_state(self) -> Dict[str, Any]:
        """Combine game state, model suggestions, and metadata."""
        return {
            "state": self.get_game_state(),
            "suggestions": self.get_model_suggestions(),
            "last_step_info": self.last_step_info,
            "active_checkpoint": self.active_checkpoint_path,
            "game_history": self.game_history[-20:],  # last 20 events
        }

    def roll_dice(self, keep_mask: List[bool]) -> Dict[str, Any]:
        """Re-roll unkept dice."""
        if self.env.current_action_type != ActionType.SELECT_DICE:
            return {
                "success": False,
                "error": "Not in dice selection phase. Please select a scoring category.",
                "data": self.get_full_state(),
            }

        mask = np.array(keep_mask, dtype=bool)
        if len(mask) != self.env.num_dices:
            mask = np.zeros(self.env.num_dices, dtype=bool)

        obs, reward, terminated, truncated, info = self.env.step(
            {
                "action_type": ActionType.SELECT_DICE,
                "dice_mask": mask,
                "category": 0,
            }
        )

        self.last_step_info = {
            "action": "roll_dice",
            "kept_indices": [i for i, k in enumerate(mask) if k],
            "reward": float(reward),
            "terminated": bool(terminated),
            "message": info.get("message", "Rolled dice."),
            "dices": self.env.dices.tolist(),
        }

        self.game_history.append(
            {
                "type": "roll",
                "roll_num": int(self.env.current_roll) + 1,
                "turn": int(self.env.turn_number) + 1,
                "message": self.last_step_info["message"],
                "dices": self.env.dices.tolist(),
            }
        )

        if terminated or truncated:
            self.is_game_over = True

        return {
            "success": True,
            "data": self.get_full_state(),
        }

    def select_category(self, category_index: int) -> Dict[str, Any]:
        """Select a category for scoring."""
        if self.env.current_action_type != ActionType.SELECT_CATEGORY:
            return {
                "success": False,
                "error": "Not in category selection phase. Please roll or keep dice.",
                "data": self.get_full_state(),
            }

        if category_index < 0 or category_index >= self.env.num_categories:
            return {
                "success": False,
                "error": f"Invalid category index: {category_index}",
                "data": self.get_full_state(),
            }

        if self.env.category_filled[category_index]:
            return {
                "success": False,
                "error": f"Category {self.env.scoring_categories_names[category_index]} is already filled.",
                "data": self.get_full_state(),
            }

        cat_name = self.env.scoring_categories_names[category_index]
        obs, reward, terminated, truncated, info = self.env.step(
            {
                "action_type": ActionType.SELECT_CATEGORY,
                "dice_mask": np.ones(self.env.num_dices, dtype=bool),
                "category": category_index,
            }
        )

        bonus_awarded = bool(info.get("bonus_awarded", False))
        scored_points = int(self.env.category_scores[category_index])

        msg = f"Scored {scored_points} points in {self._format_category_name(cat_name)}."
        if bonus_awarded:
            msg += f" 🌟 UPPER SECTION BONUS (+{self.env.upper_section_bonus_points} PTS) ACHIEVED!"

        self.last_step_info = {
            "action": "select_category",
            "category_index": category_index,
            "category_name": cat_name,
            "score": scored_points,
            "reward": float(reward),
            "bonus_awarded": bonus_awarded,
            "message": msg,
            "terminated": bool(terminated),
        }

        self.game_history.append(
            {
                "type": "score",
                "turn": int(self.env.turn_number),
                "category_name": self._format_category_name(cat_name),
                "score": scored_points,
                "bonus_awarded": bonus_awarded,
                "message": msg,
                "total_score": int(self.env.get_score()),
            }
        )

        if terminated or truncated or self.env.turn_number >= self.env.num_categories:
            self.is_game_over = True
            self.game_history.append(
                {
                    "type": "game_over",
                    "total_score": int(self.env.get_score()),
                    "message": f"🎉 Game Finished! Final Score: {self.env.get_score()}",
                }
            )

        return {
            "success": True,
            "data": self.get_full_state(),
        }

    def step_ai(self) -> Dict[str, Any]:
        """Execute one step using the trained PPO agent."""
        if self.is_game_over or self.env.turn_number >= self.env.num_categories:
            return {
                "success": False,
                "error": "Game is already completed. Start a new game.",
                "data": self.get_full_state(),
            }

        obs = self.env._get_observation()
        action, log_prob, value, action_type = self.agent.select_action(obs, deterministic=True)

        if action_type == int(ActionType.SELECT_DICE):
            dice_mask = action["dice_mask"].tolist()
            res = self.roll_dice(dice_mask)
            res["ai_action"] = {
                "type": "SELECT_DICE",
                "dice_mask": dice_mask,
                "description": f"AI decided to keep dice: {[i+1 for i, k in enumerate(dice_mask) if k]}",
            }
            return res
        else:
            cat_idx = int(action["category"])
            res = self.select_category(cat_idx)
            res["ai_action"] = {
                "type": "SELECT_CATEGORY",
                "category_index": cat_idx,
                "category_name": self.env.scoring_categories_names[cat_idx],
                "description": f"AI selected category: {self._format_category_name(self.env.scoring_categories_names[cat_idx])}",
            }
            return res
