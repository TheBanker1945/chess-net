"""Perft against published node counts (chessprogramming.org/Perft_Results).

Default depths run in a few seconds. Set PERFT_DEEP=1 to also run the deeper
counts (millions of nodes, several minutes).
"""

import os
import unittest

import chess

from chessnet.perft import perft

DEEP = os.environ.get("PERFT_DEEP") == "1"

# (name, fen, {depth: nodes}, deepest depth run by default)
POSITIONS = [
    ("startpos", chess.STARTING_FEN,
     {1: 20, 2: 400, 3: 8_902, 4: 197_281, 5: 4_865_609}, 4),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     {1: 48, 2: 2_039, 3: 97_862, 4: 4_085_603}, 3),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     {1: 14, 2: 191, 3: 2_812, 4: 43_238, 5: 674_624, 6: 11_030_083}, 5),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     {1: 6, 2: 264, 3: 9_467, 4: 422_333, 5: 15_833_292}, 4),
    ("position4-mirrored", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
     {1: 6, 2: 264, 3: 9_467, 4: 422_333}, 4),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {1: 44, 2: 1_486, 3: 62_379, 4: 2_103_487}, 3),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {1: 46, 2: 2_079, 3: 89_890, 4: 3_894_594}, 3),
]


class PerftTest(unittest.TestCase):
    def test_known_counts(self):
        for name, fen, counts, default_depth in POSITIONS:
            board = chess.Board(fen)
            for depth, expected in sorted(counts.items()):
                if depth > default_depth and not DEEP:
                    continue
                with self.subTest(position=name, depth=depth):
                    self.assertEqual(perft(board, depth), expected)
                    self.assertEqual(board.fen(), fen, "perft must leave the board unchanged")


if __name__ == "__main__":
    unittest.main()
