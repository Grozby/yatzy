from collections import Counter

import numpy as np


class Scoring:
    def evaluate(self, dices: np.ndarray) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_upper_section(self) -> bool:
        return False


class UpperSectionScoring(Scoring):
    def __init__(self, dice_to_check: int):
        super().__init__()
        assert 1 <= dice_to_check <= 6
        self.dice_to_check = dice_to_check

    def evaluate(self, dices: np.ndarray) -> int:
        return int(dices[dices == self.dice_to_check].sum())

    @property
    def is_upper_section(self):
        return True


class Ones(UpperSectionScoring):
    def __init__(self):
        super().__init__(dice_to_check=1)


class Twos(UpperSectionScoring):
    def __init__(self):
        super().__init__(dice_to_check=2)


class Threes(UpperSectionScoring):
    def __init__(self):
        super().__init__(dice_to_check=3)


class Fours(UpperSectionScoring):
    def __init__(self):
        super().__init__(dice_to_check=4)


class Fives(UpperSectionScoring):
    def __init__(self):
        super().__init__(dice_to_check=5)


class Sixes(UpperSectionScoring):
    def __init__(self):
        super().__init__(dice_to_check=6)


class OnePair(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        counts = np.bincount(dices, minlength=7)  # indices 0..6, dice are 1..6
        pairs = np.where(counts >= 2)[0]
        return int(2 * pairs.max()) if pairs.size > 0 else 0


class TwoPairs(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        counts = np.bincount(dices, minlength=7)
        pairs = np.where(counts >= 2)[0]
        return int(2 * pairs[-1] + 2 * pairs[-2]) if pairs.size >= 2 else 0


class ThreeOfAKind(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        counts = np.bincount(dices, minlength=7)
        triples = np.where(counts >= 3)[0]
        return int(3 * triples.max()) if triples.size > 0 else 0


class FourOfAKind(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        counts = np.bincount(dices, minlength=7)
        quads = np.where(counts >= 4)[0]
        return int(4 * quads.max()) if quads.size > 0 else 0


class SmallStraight(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        # Exact multiset {1,2,3,4,5}
        return 15 if np.array_equal(np.sort(dices), np.arange(1, 6)) else 0


class LargeStraight(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        # Exact multiset {2,3,4,5,6}
        return 20 if np.array_equal(np.sort(dices), np.arange(2, 7)) else 0


class FullHouse(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        counts = np.bincount(dices, minlength=7)
        non_zero = counts[counts > 0]
        # one triple and one pair → counts {2,3}
        return int(dices.sum()) if np.array_equal(np.sort(non_zero), np.array([2, 3])) else 0


class Chance(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        return int(dices.sum())


class Yatzy(Scoring):
    def evaluate(self, dices: np.ndarray) -> int:
        return 50 if np.all(dices == dices[0]) else 0
