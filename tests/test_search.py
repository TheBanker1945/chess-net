import random
import unittest

import chess

from chessnet.eval_pesto import evaluate
from chessnet.search import MATE_BOUND, Searcher


def forces_mate(board: chess.Board, moves: int) -> bool:
    """Brute force: can the side to move force checkmate within ``moves`` moves?"""
    for move in board.legal_moves:
        board.push(move)
        try:
            if board.is_checkmate():
                return True
            if moves > 1 and not board.is_game_over() and all(
                _after_reply_forces_mate(board, reply, moves - 1) for reply in list(board.legal_moves)
            ):
                return True
        finally:
            board.pop()
    return False


def _after_reply_forces_mate(board: chess.Board, reply: chess.Move, moves: int) -> bool:
    board.push(reply)
    try:
        return forces_mate(board, moves)
    finally:
        board.pop()


def mates_in(board: chess.Board, move: chess.Move, moves: int) -> bool:
    """Does playing ``move`` force mate within ``moves`` moves (counting ``move``)?"""
    board = board.copy()
    board.push(move)
    if board.is_checkmate():
        return True
    if moves == 1 or board.is_game_over():
        return False
    return all(_after_reply_forces_mate(board, r, moves - 1) for r in list(board.legal_moves))


class MateTest(unittest.TestCase):
    CASES = [
        # (fen, mate in N moves)
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 1),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 1),
        ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1", 2),
        ("6k1/pp4p1/2p5/2bp4/8/P5Pb/1P3rrP/2BRRN1K b - - 0 1", 2),
        ("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 3),
    ]

    def test_finds_forced_mates(self):
        for fen, n in self.CASES:
            board = chess.Board(fen)
            with self.subTest(fen=fen):
                self.assertTrue(forces_mate(board.copy(), n), "test position is not a mate in N")
                result = Searcher(evaluate).search(board, max_depth=2 * n + 2, time_limit=60)
                self.assertTrue(mates_in(board, result.move, n), f"{result.move} does not mate in {n}")
                self.assertGreaterEqual(result.info.score, MATE_BOUND)
                self.assertEqual(result.info.mate_in, n)


class TacticsTest(unittest.TestCase):
    def test_takes_hanging_queen(self):
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p1q1/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1")
        result = Searcher(evaluate).search(board, max_depth=4)
        self.assertEqual(result.move, chess.Move.from_uci("c1g5"))

    def test_does_not_grab_poisoned_pawn(self):
        # Qxb7?? would lose the queen to ...Rb8 trapping lines is too deep; use
        # a direct recapture: QxP defended by a pawn.
        board = chess.Board("rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 1")
        result = Searcher(evaluate).search(board, max_depth=4)
        self.assertNotEqual(result.move, chess.Move.from_uci("f3f7"))

    def test_avoids_stalemate_when_winning(self):
        # Kxg... Qg6 would stalemate; engine must find a mate or keep playing.
        board = chess.Board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1")
        result = Searcher(evaluate).search(board, max_depth=6)
        after = board.copy()
        after.push(result.move)
        self.assertFalse(after.is_stalemate())
        self.assertGreaterEqual(result.info.score, MATE_BOUND)

    def test_finds_perpetual_check_when_losing(self):
        # White is a rook down and facing mating threats, but Qe8+ Kh7 Qh5+
        # Kg8 repeats the position: the engine should take the draw.
        board = chess.Board("6k1/6p1/8/7Q/8/r7/1q3PPP/6K1 w - - 0 1")
        result = Searcher(evaluate).search(board, max_depth=7)
        after = board.copy()
        after.push(result.move)
        self.assertTrue(after.is_check(), f"expected a checking move, got {result.move}")
        self.assertEqual(result.info.score, 0)


class SearchBehaviourTest(unittest.TestCase):
    def test_board_not_modified(self):
        board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        board.push_uci("e2a6")
        before = (board.fen(), list(board.move_stack))
        Searcher(evaluate).search(board, max_depth=3)
        self.assertEqual((board.fen(), list(board.move_stack)), before)

    def test_legal_moves_from_random_positions(self):
        rng = random.Random(1234)
        searcher = Searcher(evaluate)
        for _ in range(20):
            board = chess.Board()
            for _ in range(rng.randrange(0, 60)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if board.is_game_over():
                continue
            result = searcher.search(board, max_nodes=3000)
            self.assertIn(result.move, board.legal_moves)

    def test_time_limit_respected(self):
        board = chess.Board()
        result = Searcher(evaluate).search(board, time_limit=0.5)
        self.assertLess(result.info.elapsed, 0.7)
        self.assertIn(result.move, board.legal_moves)

    def test_no_legal_moves(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
        self.assertIsNone(Searcher(evaluate).search(board, max_depth=3).move)


class EvalTest(unittest.TestCase):
    def test_start_position_balanced(self):
        self.assertEqual(evaluate(chess.Board()), 0)

    def test_color_symmetry(self):
        rng = random.Random(42)
        for _ in range(200):
            board = chess.Board()
            for _ in range(rng.randrange(0, 80)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            with self.subTest(fen=board.fen()):
                self.assertEqual(evaluate(board), evaluate(board.mirror()))

    def test_side_to_move_perspective(self):
        board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        self.assertGreater(evaluate(board), 800)
        board.turn = chess.BLACK
        self.assertLess(evaluate(board), -800)


if __name__ == "__main__":
    unittest.main()
