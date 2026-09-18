"""Score an evaluator against held-out Stockfish evaluations.

Reports the same loss the network is trained on (MSE between win
probabilities, sigmoid(cp / 400)) plus mean absolute centipawn error, so the
handwritten and learned evaluations can be compared on identical positions.

Usage: python -m chessnet.nnue.score_val data/lichess/val.npz --eval pesto
"""

from __future__ import annotations

import argparse
import math

import chess
import numpy as np

from chessnet.evaluators import EVALUATORS, get_evaluator
from chessnet.nnue.features import PAD

SCALE = 400.0


def board_from_features(feats: np.ndarray, white_to_move: bool) -> chess.Board:
    board = chess.Board(None)
    for idx in feats[feats != PAD]:
        relation, rest = divmod(int(idx), 384)
        piece_type, square = divmod(rest, 64)
        board.set_piece_at(square, chess.Piece(piece_type + 1, relation == 0))
    board.turn = white_to_move
    return board


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("val", help="val.npz written by chessnet.nnue.prepare")
    parser.add_argument("--eval", choices=EVALUATORS, default="pesto")
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args(argv)

    evaluate = get_evaluator(args.eval)
    with np.load(args.val) as z:
        feats, stm, score = z["feats"][: args.limit], z["stm"][: args.limit], z["score"][: args.limit]

    sq_err = abs_err = 0.0
    for f, s, target in zip(feats, stm, score):
        pred = evaluate(board_from_features(f, bool(s)))
        p = 1 / (1 + math.exp(-max(-5000, min(5000, pred)) / SCALE))
        t = 1 / (1 + math.exp(-int(target) / SCALE))
        sq_err += (p - t) ** 2
        abs_err += abs(max(-2000, min(2000, pred)) - int(target))
    n = len(score)
    print(f"{args.eval}: {n:,} positions  loss {sq_err / n:.6f}  mean |cp error| {abs_err / n:.1f}")


if __name__ == "__main__":
    main()
