"""
State representation utilities for Yatzy game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .environment import ActionType, YatzyEnvironment


class GameState:
    """Snapshot of the YatzyEnvironment state."""

    def __init__(
        self,
        num_dices: int,
        num_categories: int,
    ):
        self.num_dices = num_dices
        self.num_categories = num_categories

        self.dices = np.zeros(num_dices, dtype=np.int32)
        self.category_filled = np.zeros(num_categories, dtype=bool)
        self.category_scores = np.zeros(num_categories, dtype=np.int32)

        self.current_roll = 0
        self.turn_number = 0
        self.upper_section_sum = 0
        # Store as int (0/1) to avoid importing ActionType at runtime
        self.current_action_type: int = 0  # 0 = SELECT_DICE, 1 = SELECT_CATEGORY

    # ------------------------------------------------------------------
    # Constructors / sync helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: "YatzyEnvironment") -> "GameState":
        """Create a GameState snapshot from a YatzyEnvironment instance."""
        state = cls(
            num_dices=env.num_dices,
            num_categories=env.num_categories,
        )
        state.dices = env.dices.copy()
        state.category_filled = env.category_filled.copy()
        state.category_scores = env.category_scores.copy()
        state.current_roll = env.current_roll
        state.turn_number = env.turn_number
        state.upper_section_sum = env.upper_section_sum
        state.current_action_type = int(env.current_action_type)
        return state

    def apply_to_env(self, env: "YatzyEnvironment") -> None:
        """Overwrite a YatzyEnvironment's state with this snapshot."""
        assert env.num_dices == self.num_dices and env.num_categories == self.num_categories, "State/env shape mismatch"

        env.dices = self.dices.copy()
        env.category_filled = self.category_filled.copy()
        env.category_scores = self.category_scores.copy()
        env.current_roll = self.current_roll
        env.turn_number = self.turn_number
        env.upper_section_sum = self.upper_section_sum
        # Cast back to ActionType
        env.current_action_type = env.current_action_type.__class__(self.current_action_type)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def copy(self) -> "GameState":
        """Create a deep copy of the state."""
        new_state = GameState(self.num_dices, self.num_categories)
        new_state.dices = self.dices.copy()
        new_state.category_filled = self.category_filled.copy()
        new_state.category_scores = self.category_scores.copy()
        new_state.current_roll = self.current_roll
        new_state.turn_number = self.turn_number
        new_state.upper_section_sum = self.upper_section_sum
        new_state.current_action_type = self.current_action_type
        return new_state
