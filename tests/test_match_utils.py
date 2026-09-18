import math
import unittest

import chess

from chessnet.elo import elo_diff_to_score, estimate, score_to_elo_diff
from chessnet.openings import OPENINGS, OPENINGS_SAN
from chessnet.uci import UciEngine, allocate_time


class EloTest(unittest.TestCase):
    def test_even_score_is_zero(self):
        self.assertEqual(score_to_elo_diff(0.5), 0.0)
        self.assertEqual(estimate(10, 0, 10).diff, 0.0)

    def test_known_values(self):
        # 75% expected score <=> ~191 Elo; 64% <=> ~100 Elo.
        self.assertAlmostEqual(score_to_elo_diff(0.75), 190.85, places=1)
        self.assertAlmostEqual(elo_diff_to_score(100), 0.64, places=2)

    def test_round_trip(self):
        for diff in (-400, -50, 0, 120, 350):
            self.assertAlmostEqual(score_to_elo_diff(elo_diff_to_score(diff)), diff, places=6)

    def test_interval_contains_estimate_and_shrinks(self):
        small = estimate(6, 4, 10)
        big = estimate(60, 40, 100)
        for est in (small, big):
            self.assertLess(est.diff_low, est.diff)
            self.assertGreater(est.diff_high, est.diff)
        self.assertLess(big.diff_high - big.diff_low, small.diff_high - small.diff_low)

    def test_perfect_score_is_unbounded(self):
        est = estimate(10, 0, 0)
        self.assertEqual(est.diff, math.inf)

    def test_rating_offsets_anchor(self):
        est = estimate(30, 40, 30)
        self.assertEqual(est.rating(1500)[0], 1500.0)


class OpeningsTest(unittest.TestCase):
    def test_all_lines_legal_and_distinct(self):
        self.assertEqual(len(OPENINGS), len(OPENINGS_SAN))
        fens = set()
        for moves in OPENINGS:
            board = chess.Board()
            for move in moves:
                self.assertIn(move, board.legal_moves)
                board.push(move)
            self.assertFalse(board.is_game_over())
            fens.add(board.board_fen())
        self.assertEqual(len(fens), len(OPENINGS))


class UciTest(unittest.TestCase):
    def test_allocate_time_is_bounded(self):
        self.assertLess(allocate_time(60, 0.6, None), 60 * 0.4)
        self.assertGreater(allocate_time(60, 0.6, None), 1.0)
        self.assertGreater(allocate_time(0.1, 0, None), 0)
        self.assertLessEqual(allocate_time(10, 0, 1), 4.0)

    def test_parse_position(self):
        board = UciEngine.parse_position("startpos moves e2e4 e7e5".split())
        self.assertEqual(board.fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
        board = UciEngine.parse_position(f"fen {fen} moves d1d8".split())
        self.assertTrue(board.is_checkmate())


if __name__ == "__main__":
    unittest.main()
