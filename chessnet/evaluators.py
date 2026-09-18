"""Registry of evaluation functions the search can be paired with.

Every evaluator has the signature ``evaluate(board) -> int`` (centipawns from
the side to move's point of view). Front-ends (CLI, UCI, match runner) look
evaluators up here by name; the search module never imports any of them.
"""

from __future__ import annotations

from chessnet.search import Evaluator

EVALUATORS = ("pesto",)
DEFAULT = "pesto"


def get_evaluator(name: str = DEFAULT) -> Evaluator:
    if name == "pesto":
        from chessnet.eval_pesto import evaluate

        return evaluate
    raise ValueError(f"unknown evaluator {name!r}; choose from {', '.join(EVALUATORS)}")
