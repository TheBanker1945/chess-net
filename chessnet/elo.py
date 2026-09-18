"""Elo estimates from match results.

Uses the logistic Elo model: an expected score s corresponds to a rating
difference of -400 * log10(1/s - 1). The confidence interval comes from the
standard error of the per-game score (normal approximation, trinomial
W/D/L variance), mapped through the same formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def score_to_elo_diff(score: float) -> float:
    """Rating difference implied by an expected score in (0, 1)."""
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return 0.0 - 400.0 * math.log10(1.0 / score - 1.0)  # 0.0 - x avoids printing -0


def elo_diff_to_score(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


@dataclass
class EloEstimate:
    wins: int
    draws: int
    losses: int
    score: float  # mean points per game, 0..1
    diff: float  # Elo difference vs the opponent
    diff_low: float  # 95% CI lower bound
    diff_high: float  # 95% CI upper bound

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    def rating(self, anchor: float) -> tuple[float, float, float]:
        """(estimate, low, high) absolute rating given the opponent's rating."""
        return anchor + self.diff, anchor + self.diff_low, anchor + self.diff_high


def estimate(wins: int, draws: int, losses: int, z: float = 1.96) -> EloEstimate:
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games played")
    s = (wins + 0.5 * draws) / n
    var = (wins * (1.0 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s**2) / n
    stderr = math.sqrt(var / n)
    low = max(0.0, s - z * stderr)
    high = min(1.0, s + z * stderr)
    return EloEstimate(
        wins, draws, losses, s, score_to_elo_diff(s), score_to_elo_diff(low), score_to_elo_diff(high)
    )
