from enum import IntEnum
from typing import NamedTuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy import ndarray

from src.scoring import Scoring


class ActionType(IntEnum):
    """Type of action: dice selection or category selection"""

    SELECT_DICE = 0
    SELECT_CATEGORY = 1


class StepReturn(NamedTuple):
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: dict = None


class YatzyEnvironment(gym.Env):
    """
    Gym-like environment for Yatzy game.

    Game flow:
    1. Roll 5 dice (initial roll)
    2. Agent decides which dice to keep (action type: SELECT_DICE)
    3. Re-roll non-kept dice (up to 2 more rolls)
    4. Agent selects scoring category (action type: SELECT_CATEGORY)
    5. Score is recorded, move to next turn
    6. Repeat for 15 turns (one per category)
    """

    NUM_DICES = 5
    MAX_ROLLS = 3
    BONUS_THRESHOLD = 63  # Upper section sum needed for bonus
    BONUS_POINTS = 50
    NEGATIVE_REWARD = -1.0
    SMALL_POSITIVE_REWARD = 0.1

    def __init__(
        self,
        *,
        num_dices: int = NUM_DICES,
        max_rolls: int = MAX_ROLLS,
        scoring_categories: Optional[list[Scoring]] = None,
        upper_section_score_threshold=BONUS_THRESHOLD,
        upper_section_bonus_points=BONUS_POINTS,
    ):
        super().__init__()

        if scoring_categories is None:
            scoring_categories = self._init_default_scoring_categories()

        # Core configuration
        self.num_dices = num_dices
        self.max_rolls = max_rolls
        self.scoring_categories = scoring_categories
        self.num_categories = len(scoring_categories)
        self.scoring_categories_names = [s.name for s in scoring_categories]
        self.upper_section_score_threshold = upper_section_score_threshold
        self.upper_section_bonus_points = upper_section_bonus_points

        # Observation & action spaces
        self._init_observation_layout()
        self._init_action_space()

        # Game state
        self._reset_game_state()
        self._roll_all_dices()

    def _init_default_scoring_categories(self) -> list[Scoring]:
        from src.scoring import (
            Chance,
            Fives,
            FourOfAKind,
            Fours,
            FullHouse,
            LargeStraight,
            OnePair,
            Ones,
            Sixes,
            SmallStraight,
            ThreeOfAKind,
            Threes,
            TwoPairs,
            Twos,
            Yatzy,
        )

        return [
            Ones(),
            Twos(),
            Threes(),
            Fours(),
            Fives(),
            Sixes(),
            OnePair(),
            TwoPairs(),
            ThreeOfAKind(),
            FourOfAKind(),
            SmallStraight(),
            LargeStraight(),
            FullHouse(),
            Chance(),
            Yatzy(),
        ]

    # ------------------------------------------------------------------
    # Space initialization
    # ------------------------------------------------------------------
    def _init_observation_layout(self) -> None:
        """
        Layout of observation vector:

        [0 : num_dices)                      -> dice values (normalized)
        [num_dices : num_dices+num_cat)      -> category filled flags
        [.. + num_cat : .. + 2*num_cat)      -> category scores (normalized)
        [.. + 2*num_cat : .. + 3*num_cat)    -> POTENTIAL category scores (normalized)
        [roll_index]                         -> current roll (normalized)
        [upper_sum_index]                    -> upper section sum (normalized)
        [action_type_index]                  -> current action type (0 or 1)
        """
        self._dice_slice_start = 0
        self._dice_slice_end = self._dice_slice_start + self.num_dices

        self._counts_slice_start = self._dice_slice_end
        self._counts_slice_end = self._counts_slice_start + 6

        self._filled_slice_start = self._counts_slice_end
        self._filled_slice_end = self._filled_slice_start + self.num_categories

        self._scores_slice_start = self._filled_slice_end
        self._scores_slice_end = self._scores_slice_start + self.num_categories

        self._potential_scores_slice_start = self._scores_slice_end
        self._potential_scores_slice_end = self._potential_scores_slice_start + self.num_categories

        self._roll_index = self._potential_scores_slice_end
        self._upper_sum_index = self._roll_index + 1
        self._upper_needed_index = self._upper_sum_index + 1
        self._upper_remaining_index = self._upper_needed_index + 1
        self._turns_remaining_index = self._upper_remaining_index + 1
        self._action_type_index = self._turns_remaining_index + 1

        self.obs_size = self._action_type_index + 1

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

    def _init_action_space(self) -> None:
        """
        - action_type: Discrete(2)
            0 = SELECT_DICE
            1 = SELECT_CATEGORY

        - dice_mask: MultiBinary(num_dices)
            Only used when action_type == SELECT_DICE
            One boolean per die: True = keep, False = reroll

        - category: Discrete(num_categories)
            Only used when action_type == SELECT_CATEGORY
        """
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(2),
                "dice_mask": spaces.MultiBinary(self.num_dices),
                "category": spaces.Discrete(self.num_categories),
            }
        )

    # ------------------------------------------------------------------
    # Game state management
    # ------------------------------------------------------------------
    def _reset_game_state(self) -> None:
        """Initialize or clear all per-episode state."""
        self.dices = np.zeros(self.num_dices, dtype=np.int32)
        self.category_filled = np.zeros(self.num_categories, dtype=bool)
        self.category_scores = np.zeros(self.num_categories, dtype=np.int32)
        self.current_roll = 0
        self.turn_number = 0
        self.upper_section_sum = 0
        self.current_action_type = ActionType.SELECT_DICE

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state and roll starting dice."""
        super().reset(seed=seed)

        self._reset_game_state()
        self._roll_all_dices()

        info = {
            "turn": self.turn_number,
            "roll": self.current_roll,
            "action_type": "select_dice",
        }
        return self._get_observation(), info

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------
    def step(self, action: dict) -> tuple[ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: dict with keys
                - "action_type": 0 = select dice, 1 = select category
                - "dice_mask": MultiBinary(num_dices) (only if SELECT_DICE)
                - "category": int in [0, num_categories) (only if SELECT_CATEGORY)
        """
        action_type = ActionType(int(action["action_type"]))

        if action_type == ActionType.SELECT_DICE:
            dice_mask = np.asarray(action["dice_mask"], dtype=bool)
            reward, terminated, truncated, info = self._handle_select_dice(dice_mask)
        elif action_type == ActionType.SELECT_CATEGORY:
            category_index = int(action["category"])
            reward, terminated, truncated, info = self._handle_select_category(category_index)
        else:
            reward, terminated, truncated, info = self._handle_invalid_action_type(action_type)

        # Common info fields
        info["turn"] = self.turn_number
        info["action_type"] = self.current_action_type.name
        info["upper_section_sum"] = self.upper_section_sum

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Step subroutines
    # ------------------------------------------------------------------
    def _handle_select_dice(self, dice_mask: np.ndarray) -> StepReturn:
        """Process a SELECT_DICE action."""
        if self.current_action_type != ActionType.SELECT_DICE:
            # Wrong phase
            return StepReturn(
                reward=self.NEGATIVE_REWARD,
                info={"error": "Expected category selection"},
            )

        reward = 0.0
        if dice_mask.sum() == self.num_dices:
            # All dice chosen to keep: force category selection next
            self.current_action_type = ActionType.SELECT_CATEGORY
            message = "All dice kept, moving to select a scoring category"
        else:
            # Re-roll unkept dice
            self._reroll_dices(dice_mask)
            self.current_roll += 1
            # We don't add a positive reward for rerolling to avoid the agent farming rerolls.

            if self.current_roll >= self.max_rolls - 1:
                # After this roll, must choose a category
                self.current_action_type = ActionType.SELECT_CATEGORY
                message = "Max rolls reached, must select a scoring category"
            else:
                message = f"Rolled dice, roll {self.current_roll + 1}/{self.max_rolls}"

        return StepReturn(
            reward=reward,
            info={
                "dices": self.dices.copy(),
                "roll": self.current_roll,
                "message": message,
            },
        )

    def _handle_select_category(
        self,
        category_index: int,
    ) -> StepReturn:
        """Process a SELECT_CATEGORY action."""
        # Phase check
        if self.current_action_type != ActionType.SELECT_CATEGORY:
            return StepReturn(
                reward=self.NEGATIVE_REWARD,
                info={"error": "Expected category selection"},
            )

        # Basic validation
        ## Category out of range
        if category_index < 0 or category_index >= self.num_categories:
            return StepReturn(
                reward=self.NEGATIVE_REWARD,
                info={"error": f"Category index out of range: {category_index}"},
            )
        ## Category already selected
        if self.category_filled[category_index]:
            name = self.scoring_categories_names[category_index]
            return StepReturn(
                reward=self.NEGATIVE_REWARD,
                info={"error": f"Scoring category {name} already filled"},
            )

        # It is a valid action; proceed with scoring
        score = self.scoring_categories[category_index].evaluate(self.dices)
        self.category_scores[category_index] = score
        self.category_filled[category_index] = True

        previous_upper_sum = self.upper_section_sum
        # Upper section scores
        if self.scoring_categories[category_index].is_upper_section:
            self.upper_section_sum += score

        # Increment turn number
        self.turn_number += 1
        
        # Base reward is the score we just got
        reward = float(score)
        
        # Award bonus ONLY if we just crossed the threshold
        just_got_bonus = (
            previous_upper_sum < self.upper_section_score_threshold
            and self.upper_section_sum >= self.upper_section_score_threshold
        )
        if just_got_bonus:
            reward += self.upper_section_bonus_points

        # Check if the game is finished
        game_finished = self.turn_number >= self.num_categories
        if game_finished:
            total_score = self._compute_total_score()
            return StepReturn(
                reward=reward,
                terminated=True,
                info={"total_score": total_score, "bonus_awarded": just_got_bonus},
            )

        # Game not finished, we continue
        self._start_next_turn()

        return StepReturn(
            reward=reward,
            info={
                "message": (
                    f"Scored {score} in {self.scoring_categories_names[category_index]}, "
                    f"starting turn {self.turn_number + 1}"
                ),
                "bonus_awarded": just_got_bonus,
            },
        )

    def _handle_invalid_action_type(self, action_type: ActionType) -> StepReturn:
        """Penalty for invalid action type."""
        return StepReturn(
            reward=self.NEGATIVE_REWARD,
            info={"error": f"Invalid action type: {action_type}"},
        )

    def _compute_total_score(self) -> int:
        """Compute total score including bonus if threshold is met."""
        total = int(self.category_scores.sum())
        if self.upper_section_sum >= self.upper_section_score_threshold:
            total += self.upper_section_bonus_points
        return total

    def _start_next_turn(self) -> None:
        """Prepare state for the next turn."""
        self.current_roll = 0
        self.current_action_type = ActionType.SELECT_DICE
        self._roll_all_dices()

    # ------------------------------------------------------------------
    # Dice utilities
    # ------------------------------------------------------------------
    def _roll_all_dices(self):
        """Roll all dices."""
        self.dices = np.random.randint(1, 7, size=self.num_dices, dtype=np.int32)
        self.dices.sort()

    def _reroll_dices(self, keep_mask: np.ndarray) -> None:
        """Re-roll dices that are not in keep_mask.

        Args:
            keep_mask: Boolean array of length num_dices, True means keep the die.
        """
        num_reroll = int((~keep_mask).sum())
        self.dices[~keep_mask] = np.random.randint(1, 7, size=num_reroll, dtype=np.int32)
        self.dices.sort()

    # ------------------------------------------------------------------
    # Observation / rendering
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Build normalized observation vector."""
        obs = np.zeros(self.obs_size, dtype=np.float32)

        # Dice values (1..6 -> 0..1)
        obs[self._dice_slice_start : self._dice_slice_end] = (self.dices - 1) / 5.0

        # Dice counts histogram (counts for 1..6 -> 0..1)
        counts = np.bincount(self.dices, minlength=7)[1:7].astype(np.float32)
        obs[self._counts_slice_start : self._counts_slice_end] = counts / float(self.num_dices)

        # Category filled flags
        obs[self._filled_slice_start : self._filled_slice_end] = self.category_filled.astype(np.float32)

        # Category scores (normalize by 50 as a reasonable upper bound)
        obs[self._scores_slice_start : self._scores_slice_end] = self.category_scores.astype(np.float32) / 50.0

        # Potential category scores for CURRENT dice (normalize by 50)
        potential_scores = np.array([cat.evaluate(self.dices) for cat in self.scoring_categories], dtype=np.float32)
        potential_scores[self.category_filled] = 0.0
        obs[self._potential_scores_slice_start : self._potential_scores_slice_end] = potential_scores / 50.0

        # Roll number (0..max_rolls-1 -> 0..1)
        if self.max_rolls > 1:
            obs[self._roll_index] = self.current_roll / float(self.max_rolls - 1)
        else:
            obs[self._roll_index] = 0.0

        # Upper section sum (0..BONUS_THRESHOLD+ -> 0..1, clipped)
        obs[self._upper_sum_index] = min(self.upper_section_sum / float(self.BONUS_THRESHOLD), 1.0)

        # Upper section deficit to bonus threshold (0..1)
        needed = max(0, self.BONUS_THRESHOLD - self.upper_section_sum)
        obs[self._upper_needed_index] = needed / float(self.BONUS_THRESHOLD)

        # Upper categories remaining to be played (0..1)
        upper_left = (~self.category_filled[:6]).sum()
        obs[self._upper_remaining_index] = upper_left / 6.0

        # Total turns remaining in game (0..1)
        turns_left = max(0, self.num_categories - self.turn_number)
        obs[self._turns_remaining_index] = turns_left / float(self.num_categories)

        # Action type indicator
        obs[self._action_type_index] = float(self.current_action_type)

        return obs

    def render(self, mode: str = "human") -> None:
        """Render the current game state."""
        if mode != "human":
            return

        print(f"\n=== Turn {self.turn_number + 1}/{self.num_categories} ===")
        print(f"Dice: {self.dices}")
        print(f"Roll: {self.current_roll + 1}/{self.max_rolls}")
        print(
            f"Action: {'Select dice to keep' if self.current_action_type == ActionType.SELECT_DICE else 'Select category'}"
        )
        print("\nFilled categories:")
        for i, (name, filled, score) in enumerate(
            zip(
                self.scoring_categories_names,
                self.category_filled,
                self.category_scores,
            )
        ):
            status = f"✓ {score}" if filled else "○"
            print(f"  {i:2d}. {name:15s}: {status}")
        print(f"\nUpper section sum: {self.upper_section_sum}/{self.BONUS_THRESHOLD}")
        if self.upper_section_sum >= self.BONUS_THRESHOLD:
            print("  Bonus: ✓")
        print(f"Total score: {self.category_scores.sum()}")

    def get_score(self) -> int:
        """Get current total score (including bonus if any)."""
        return self._compute_total_score()
