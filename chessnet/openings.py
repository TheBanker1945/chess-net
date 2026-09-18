"""Short, roughly balanced opening lines for engine matches.

Each line is played twice with colours swapped so neither side profits from a
lopsided opening, and so games against a deterministic opponent differ.
"""

from __future__ import annotations

import chess

OPENINGS_SAN = [
    "e4 e5 Nf3 Nc6 Bb5 a6",  # Ruy Lopez
    "e4 e5 Nf3 Nc6 Bc4 Bc5",  # Italian
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",  # Scotch
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4",  # Petrov
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",  # Sicilian, open
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6",  # Sicilian, Taimanov
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6",  # Sicilian, accelerated dragon
    "e4 c5 Nc3 Nc6 g3 g6",  # Sicilian, closed
    "e4 e6 d4 d5 Nc3 Nf6",  # French, classical
    "e4 e6 d4 d5 e5 c5",  # French, advance
    "e4 c6 d4 d5 e5 Bf5",  # Caro-Kann, advance
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",  # Caro-Kann, classical
    "e4 d5 exd5 Qxd5 Nc3 Qa5",  # Scandinavian
    "e4 d6 d4 Nf6 Nc3 g6",  # Pirc
    "e4 Nf6 e5 Nd5 d4 d6",  # Alekhine
    "d4 d5 c4 e6 Nc3 Nf6",  # Queen's Gambit Declined
    "d4 d5 c4 c6 Nf3 Nf6",  # Slav
    "d4 d5 c4 dxc4 Nf3 Nf6",  # Queen's Gambit Accepted
    "d4 d5 Bf4 Nf6 e3 c5",  # London
    "d4 d5 Nf3 Nf6 e3 e6",  # Colle
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",  # King's Indian
    "d4 Nf6 c4 e6 Nc3 Bb4",  # Nimzo-Indian
    "d4 Nf6 c4 e6 Nf3 b6",  # Queen's Indian
    "d4 Nf6 c4 c5 d5 e6",  # Benoni
    "d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O",  # King's Indian Attack setup
    "d4 f5 g3 Nf6 Bg2 g6",  # Dutch, Leningrad
    "c4 e5 Nc3 Nf6 g3 d5",  # English, reversed Sicilian
    "c4 c5 Nf3 Nc6 Nc3 g6",  # English, symmetrical
    "c4 Nf6 Nc3 e5 Nf3 Nc6",  # English, four knights
    "Nf3 d5 g3 Nf6 Bg2 c6",  # Reti
]


def opening_moves(san_line: str) -> list[chess.Move]:
    board = chess.Board()
    moves = []
    for san in san_line.split():
        move = board.parse_san(san)
        moves.append(move)
        board.push(move)
    return moves


OPENINGS = [opening_moves(line) for line in OPENINGS_SAN]
