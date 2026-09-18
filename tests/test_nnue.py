import json
import random
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from chessnet.nnue.features import MAX_PIECES, PAD, flip, white_features
from chessnet.nnue.prepare import SCORE_CLAMP, parse_chunk, parse_record

try:
    import torch

    from chessnet.nnue import train as nnue_train
except ImportError:  # inference must work without torch; training tests need it
    torch = None


def random_boards(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randrange(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def record(fen: str, pv: dict, depth: int = 30) -> bytes:
    return json.dumps({"fen": fen, "evals": [{"pvs": [pv], "knodes": 1, "depth": depth}]}).encode()


class FeatureTest(unittest.TestCase):
    def test_black_view_equals_white_view_of_mirror(self):
        for board in random_boards(100):
            black_view = sorted(flip(i) for i in white_features(board))
            self.assertEqual(black_view, sorted(white_features(board.mirror())))

    def test_indices_in_range_and_unique(self):
        for board in random_boards(50, seed=1):
            feats = white_features(board)
            self.assertEqual(len(feats), chess.popcount(board.occupied))
            self.assertEqual(len(set(feats)), len(feats))
            self.assertTrue(all(0 <= i < 768 for i in feats))

    def test_flip_is_involution(self):
        for i in range(768):
            self.assertEqual(flip(flip(i)), i)


class PrepareTest(unittest.TestCase):
    QUIET = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"

    def test_score_is_side_to_move_relative(self):
        feats, stm, score = parse_record(record(self.QUIET, {"cp": 35, "line": "e7e5 g1f3"}))
        self.assertFalse(stm)
        self.assertEqual(score, -35)  # Lichess gives White's view; Black is to move
        feats, stm, score = parse_record(record(chess.STARTING_FEN, {"cp": 20, "line": "e2e4"}))
        self.assertTrue(stm)
        self.assertEqual(score, 20)

    def test_deepest_eval_used(self):
        rec = {
            "fen": chess.STARTING_FEN,
            "evals": [
                {"pvs": [{"cp": 99, "line": "e2e4"}], "depth": 10},
                {"pvs": [{"cp": 15, "line": "d2d4"}], "depth": 40},
            ],
        }
        self.assertEqual(parse_record(json.dumps(rec).encode())[2], 15)

    def test_mates_and_clamping(self):
        self.assertEqual(parse_record(record(chess.STARTING_FEN, {"mate": 3, "line": "e2e4"}))[2], SCORE_CLAMP)
        self.assertEqual(parse_record(record(chess.STARTING_FEN, {"mate": -2, "line": "e2e4"}))[2], -SCORE_CLAMP)
        self.assertEqual(parse_record(record(chess.STARTING_FEN, {"cp": 5000, "line": "e2e4"}))[2], SCORE_CLAMP)

    def test_skips_non_quiet(self):
        in_check = "rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq -"
        self.assertEqual(parse_record(record(in_check, {"cp": 0, "line": "g8h6"})), "in_check")
        capture = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
        self.assertEqual(parse_record(record(capture, {"cp": 0, "line": "e4d5"})), "tactical_best_move")
        promo = "8/P6k/8/8/8/8/8/K7 w - -"
        self.assertEqual(parse_record(record(promo, {"cp": 900, "line": "a7a8q"})), "tactical_best_move")

    def test_skips_impossible_positions(self):
        too_many = "rnbqkbnr/pppppppp/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w - -"
        self.assertEqual(parse_record(record(too_many, {"cp": 0, "line": "e2e4"})), "invalid_position")
        no_black_king = "8/8/8/8/8/8/8/K7 w - -"
        self.assertEqual(parse_record(record(no_black_king, {"cp": 0, "line": "a1a2"})), "invalid_position")

    def test_chunk_layout(self):
        lines = [record(chess.STARTING_FEN, {"cp": 20, "line": "e2e4"}), b"not json"]
        feats, stm, score, skipped = parse_chunk(lines)
        self.assertEqual(feats.shape, (1, MAX_PIECES))
        self.assertEqual(feats.dtype, np.int16)
        self.assertEqual(skipped, {"malformed": 1})
        self.assertEqual(sorted(feats[0][feats[0] != PAD].tolist()), sorted(white_features(chess.Board())))


def write_shards(directory: Path, n_train: int, n_val: int, seed: int = 0) -> None:
    """Synthetic dataset whose target is the PeSTO eval, so it is learnable."""
    from chessnet.eval_pesto import evaluate

    boards = random_boards(n_train + n_val, seed)
    feats = np.full((len(boards), MAX_PIECES), PAD, dtype=np.int16)
    for i, b in enumerate(boards):
        f = white_features(b)
        feats[i, : len(f)] = f
    stm = np.array([b.turn for b in boards], dtype=np.uint8)
    score = np.array([max(-2000, min(2000, evaluate(b))) for b in boards], dtype=np.int16)
    np.savez_compressed(directory / "train-000.npz", feats=feats[:n_train], stm=stm[:n_train], score=score[:n_train])
    np.savez_compressed(directory / "val.npz", feats=feats[n_train:], stm=stm[n_train:], score=score[n_train:])


@unittest.skipIf(torch is None, "torch not installed")
class TrainingTest(unittest.TestCase):
    def test_numpy_inference_matches_torch(self):
        from chessnet.eval_nnue import NNUEEvaluator

        torch.manual_seed(0)
        model = nnue_train.NNUE()
        with torch.no_grad():  # make every layer non-trivial
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
            model.ft.weight[nnue_train.PAD].zero_()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "net.npz")
            nnue_train.export_npz(model, path, {})
            ev = NNUEEvaluator(path)
        for board in random_boards(40, seed=3):
            idx = torch.full((1, MAX_PIECES), PAD, dtype=torch.long)
            f = white_features(board)
            idx[0, : len(f)] = torch.tensor(f)
            with torch.no_grad():
                out = model(idx, nnue_train.flip_features(idx), torch.tensor([board.turn], dtype=torch.uint8))
            self.assertAlmostEqual(ev.raw(board), out.item() * nnue_train.SCALE, delta=0.05)
            # Perspective network: colour-mirrored positions evaluate identically.
            self.assertAlmostEqual(ev.raw(board), ev.raw(board.mirror()), delta=0.05)

    def test_train_resume_and_export(self):
        from chessnet.eval_nnue import NNUEEvaluator

        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "out"
            data.mkdir()
            write_shards(data, 3000, 500)
            common = ["--data", str(data), "--out", str(out), "--batch-size", "256", "--lr", "3e-3", "--device", "cpu"]

            nnue_train.main(common + ["--epochs", "3", "--max-minutes", "1e-9"])  # stops after one step
            ckpt = torch.load(out / "checkpoint.pt", weights_only=False)
            self.assertEqual((ckpt["epoch"], ckpt["step_in_epoch"]), (0, 1))

            nnue_train.main(common + ["--epochs", "3"])  # resumes mid-epoch and finishes
            history = json.loads((out / "history.json").read_text())
            self.assertEqual([h["epoch"] for h in history], [1, 2, 3])
            self.assertLess(history[-1]["val_loss"], history[0]["val_loss"])
            for name in ("nnue-best.npz", "nnue-latest.npz", "nnue-epoch003.npz"):
                self.assertTrue((out / name).exists(), name)

            ev = NNUEEvaluator(out / "nnue-best.npz")
            self.assertIsInstance(ev(chess.Board()), int)


if __name__ == "__main__":
    unittest.main()
