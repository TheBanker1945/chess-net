"""Input features shared by data preparation and local inference.

Each piece on the board activates one of 768 features, computed separately
for each side's perspective:

    index = relation * 384 + (piece_type - 1) * 64 + square

where relation is 0 for the perspective owner's pieces and 1 for the opponent's,
and the square is flipped vertically (sq ^ 56) for Black's perspective so both
sides see the board "from their own side". Training data stores only the
White-perspective indices; Black's are derived with ``flip``.

Index 768 is padding (positions have fewer than 32 pieces).
"""

from __future__ import annotations

import chess

NUM_FEATURES = 768
PAD = 768
MAX_PIECES = 32


def white_features(board: chess.Board) -> list[int]:
    """White-perspective feature indices of every piece on the board."""
    out = []
    white_occ = board.occupied_co[chess.WHITE]
    for pt, bb in (
        (chess.PAWN, board.pawns),
        (chess.KNIGHT, board.knights),
        (chess.BISHOP, board.bishops),
        (chess.ROOK, board.rooks),
        (chess.QUEEN, board.queens),
        (chess.KING, board.kings),
    ):
        base = (pt - 1) * 64
        while bb:
            low = bb & -bb
            sq = low.bit_length() - 1
            bb ^= low
            out.append(base + sq if low & white_occ else 384 + base + sq)
    return out


def flip(index: int) -> int:
    """White-perspective index -> Black-perspective index."""
    relation, rest = divmod(index, 384)
    return (1 - relation) * 384 + (rest ^ 56)
