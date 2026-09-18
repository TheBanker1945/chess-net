"""Registry of evaluation functions the search can be paired with.

Every evaluator has the signature ``evaluate(board) -> int`` (centipawns from
the side to move's point of view). Front-ends (CLI, UCI, match runner) look
evaluators up here by name; the search module never imports any of them.
"""

from __future__ import annotations

from chessnet.search import Evaluator

EVALUATORS = ("pesto", "nnue")
DEFAULT = "pesto"


def get_evaluator(name: str = DEFAULT) -> Evaluator:
    if name == "pesto":
        from chessnet.eval_pesto import evaluate

        return evaluate
    if name == "nnue":
        # Network file: $CHESSNET_NNUE if set, else models/nnue.npz.
        import os

        from chessnet.eval_nnue import DEFAULT_MODEL, NNUEEvaluator

        return NNUEEvaluator(os.environ.get("CHESSNET_NNUE", DEFAULT_MODEL))
    raise ValueError(f"unknown evaluator {name!r}; choose from {', '.join(EVALUATORS)}")
