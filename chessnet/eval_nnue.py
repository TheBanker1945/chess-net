"""Stage 2 evaluation: the trained NNUE-style network, run on CPU with numpy.

Loads the .npz exported by chessnet/nnue/train.py (no torch needed) and
recomputes both accumulators from scratch for every call. Real NNUE engines
update accumulators incrementally as moves are made, but that would need
hooks inside the search; a full refresh keeps the evaluator a drop-in
``evaluate(board) -> int`` replacement for the handwritten one.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

from chessnet.nnue.features import white_features

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "nnue.npz"


class NNUEEvaluator:
    def __init__(self, path: str | Path = DEFAULT_MODEL, cache_size: int = 200_000):
        with np.load(path) as z:
            self.ft_w = z["ft_w"].astype(np.float32)
            self.ft_b = z["ft_b"].astype(np.float32)
            self.l1_w = z["l1_w"].T.astype(np.float32).copy()
            self.l1_b = z["l1_b"].astype(np.float32)
            self.l2_w = z["l2_w"].T.astype(np.float32).copy()
            self.l2_b = z["l2_b"].astype(np.float32)
            self.out_w = z["out_w"].reshape(-1).astype(np.float32)
            self.out_b = float(z["out_b"].reshape(-1)[0])
            self.scale = float(z["scale"])
            self.meta = json.loads(str(z["meta"])) if "meta" in z else {}
        hidden = self.ft_b.shape[0]
        # Row i holds [White-view weights of feature i | Black-view weights of
        # the same piece], so one gather + sum yields both accumulators.
        relation, rest = np.divmod(np.arange(768), 384)
        flipped = (1 - relation) * 384 + (rest ^ 56)
        self.ft_both = np.ascontiguousarray(np.concatenate((self.ft_w[:768], self.ft_w[flipped]), axis=1))
        self.ft_b2 = np.concatenate((self.ft_b, self.ft_b))
        # The first layer expects [side to move | other side]; for Black to
        # move, swap its input halves instead of swapping the accumulators.
        self.l1_w_black = np.ascontiguousarray(np.concatenate((self.l1_w[hidden:], self.l1_w[:hidden])))
        self.cache: dict = {}
        self.cache_size = cache_size

    def raw(self, board: chess.Board) -> float:
        """Network output in centipawns (side to move), unrounded."""
        x = self.ft_both[white_features(board)].sum(axis=0)
        x += self.ft_b2
        np.clip(x, 0.0, 1.0, out=x)
        l1_w = self.l1_w if board.turn == chess.WHITE else self.l1_w_black
        h = np.clip(x @ l1_w + self.l1_b, 0.0, 1.0)
        h = np.clip(h @ self.l2_w + self.l2_b, 0.0, 1.0)
        return (float(h @ self.out_w) + self.out_b) * self.scale

    def __call__(self, board: chess.Board) -> int:
        key = board._transposition_key()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        score = int(round(self.raw(board)))
        if len(self.cache) >= self.cache_size:
            self.cache.clear()
        self.cache[key] = score
        return score
