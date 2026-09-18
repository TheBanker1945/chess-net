"""Convert Lichess's Stockfish-evaluated positions into training shards.

Source: https://database.lichess.org/#evals (CC0), one JSON object per line:
    {"fen": ..., "evals": [{"pvs": [{"cp"|"mate": ..., "line": ...}], "depth": ...}, ...]}
Scores are from White's point of view.

For each position we take the deepest evaluation's principal variation and
skip positions that are in check or whose best move is a capture or
promotion. Those are not quiet, and the engine only calls the evaluation at
quiet quiescence leaves, so they would teach the net to predict tactics it
cannot see.

The 22 GB file is a single zstd frame, so it can only be read from the start.
By default it is streamed (curl | zstd -dc) and reading stops once enough
positions are collected, so the full file never touches the disk. Every
100th kept position goes to the validation set.

Output (in --out):
    train-000.npz ...  feats int16 [n, 32] White-perspective feature indices, padded with 768
                       stm   uint8 [n]     1 if White to move
                       score int16 [n]     centipawns from the side to move's point of view
    val.npz            same layout
    meta.json          counts, skip reasons, per-shard statistics

Usage:
    python -m chessnet.nnue.prepare --out data/lichess --positions 30000000
    python -m chessnet.nnue.prepare --input lichess_db_eval.jsonl.zst --out ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np

from chessnet.nnue.features import MAX_PIECES, PAD, white_features

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
SCORE_CLAMP = 2000  # centipawns; mates map to +-SCORE_CLAMP
VAL_EVERY = 100
CHUNK_LINES = 20_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_zstd() -> str:
    found = shutil.which("zstd")
    if found:
        return found
    local = PROJECT_ROOT / "tools" / "zstd" / "usr" / "bin" / "zstd"
    if local.exists():
        return str(local)
    sys.exit("zstd not found: install it (sudo apt install zstd) or unpack it into tools/zstd")


def parse_record(line: bytes):
    """Return (features, stm, score) or a skip-reason string."""
    rec = json.loads(line)
    evals = rec.get("evals")
    if not evals:
        return "no_eval"
    best = max(evals, key=lambda e: e.get("depth", 0))
    pv = best["pvs"][0]
    if "mate" in pv:
        white_cp = SCORE_CLAMP if pv["mate"] > 0 else -SCORE_CLAMP
    else:
        white_cp = max(-SCORE_CLAMP, min(SCORE_CLAMP, pv["cp"]))

    board = chess.Board(rec["fen"])
    # The database includes analysis-board setups that cannot arise in a game
    # (e.g. more than 32 pieces, missing kings).
    if board.status() != chess.STATUS_VALID or chess.popcount(board.occupied) > MAX_PIECES:
        return "invalid_position"
    if board.is_check():
        return "in_check"
    line = pv.get("line")
    if line:
        move = chess.Move.from_uci(line.split(" ", 1)[0])
        if move.promotion or board.is_capture(move):
            return "tactical_best_move"
    feats = white_features(board)
    stm = board.turn == chess.WHITE
    return feats, stm, white_cp if stm else -white_cp


def parse_chunk(lines: list[bytes]):
    n = len(lines)
    feats = np.full((n, MAX_PIECES), PAD, dtype=np.int16)
    stm = np.zeros(n, dtype=np.uint8)
    score = np.zeros(n, dtype=np.int16)
    skipped: dict[str, int] = {}
    k = 0
    for line in lines:
        try:
            parsed = parse_record(line)
        except (ValueError, KeyError, IndexError):
            parsed = "malformed"
        if isinstance(parsed, str):
            skipped[parsed] = skipped.get(parsed, 0) + 1
            continue
        f, s, sc = parsed
        feats[k, : len(f)] = f
        stm[k] = s
        score[k] = sc
        k += 1
    return feats[:k], stm[:k], score[:k], skipped


class ShardWriter:
    def __init__(self, out: Path, prefix: str, shard_size: int):
        self.out, self.prefix, self.shard_size = out, prefix, shard_size
        self.parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.pending = 0
        self.index = 0
        self.stats: list[dict] = []

    def add(self, feats, stm, score) -> None:
        if len(score) == 0:
            return
        self.parts.append((feats, stm, score))
        self.pending += len(score)
        while self.pending >= self.shard_size:
            self.flush(self.shard_size)

    def flush(self, limit: int | None = None) -> None:
        if not self.parts:
            return
        feats = np.concatenate([p[0] for p in self.parts])
        stm = np.concatenate([p[1] for p in self.parts])
        score = np.concatenate([p[2] for p in self.parts])
        limit = len(score) if limit is None else limit
        rest = (feats[limit:], stm[limit:], score[limit:])
        feats, stm, score = feats[:limit], stm[:limit], score[:limit]
        name = f"{self.prefix}-{self.index:03d}.npz" if self.prefix != "val" else "val.npz"
        np.savez_compressed(self.out / name, feats=feats, stm=stm, score=score)
        self.stats.append(
            {
                "file": name,
                "positions": int(len(score)),
                "mean_abs_score": round(float(np.abs(score.astype(np.float64)).mean()), 1),
                "mean_pieces": round(float((feats != PAD).sum(axis=1).mean()), 2),
                "white_to_move": round(float(stm.mean()), 3),
            }
        )
        print(f"  wrote {name}: {self.stats[-1]}", flush=True)
        self.index += 1
        self.parts = [rest] if len(rest[2]) else []
        self.pending = len(rest[2])


def open_source(path: str | None):
    """Yield the input's decompressed lines, plus a cleanup callback."""
    procs = []
    if path is None:
        curl = subprocess.Popen(["curl", "-sfL", URL], stdout=subprocess.PIPE)
        zstd = subprocess.Popen([find_zstd(), "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE)
        curl.stdout.close()
        procs = [zstd, curl]
        stream = zstd.stdout
    elif path.endswith(".zst"):
        zstd = subprocess.Popen([find_zstd(), "-dc", path], stdout=subprocess.PIPE)
        procs = [zstd]
        stream = zstd.stdout
    else:
        stream = open(path, "rb")

    def close() -> None:
        for p in procs:
            p.terminate()
        stream.close()
        for p in procs:
            p.wait()

    return stream, close


def chunks(stream, size: int):
    buf = []
    for line in stream:
        buf.append(line)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=30_000_000, help="kept positions to collect")
    parser.add_argument("--input", help=".jsonl.zst or .jsonl file (default: stream from database.lichess.org)")
    parser.add_argument("--shard-size", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob("*.npz")):
        sys.exit(f"{args.out} already contains shards; use an empty directory")

    train = ShardWriter(args.out, "train", args.shard_size)
    val = ShardWriter(args.out, "val", 1 << 62)
    skipped: dict[str, int] = {}
    kept = read = 0
    start = time.perf_counter()
    stream, close = open_source(args.input)
    print(f"reading {args.input or URL} with {args.workers} workers", flush=True)
    try:
        with mp.Pool(args.workers) as pool:
            for feats, stm, score, skip in pool.imap(parse_chunk, chunks(stream, CHUNK_LINES)):
                read += CHUNK_LINES
                for key, v in skip.items():
                    skipped[key] = skipped.get(key, 0) + v
                # Deterministic split: every VAL_EVERY-th kept position is held out.
                pos = np.arange(kept, kept + len(score))
                is_val = pos % VAL_EVERY == VAL_EVERY - 1
                val.add(feats[is_val], stm[is_val], score[is_val])
                train.add(feats[~is_val], stm[~is_val], score[~is_val])
                kept += len(score)
                if read % (CHUNK_LINES * 50) == 0:
                    rate = read / (time.perf_counter() - start)
                    print(f"read {read:,}  kept {kept:,}  ({rate:,.0f} records/s)", flush=True)
                if kept >= args.positions:
                    pool.terminate()
                    break
    finally:
        close()
    train.flush()
    val.flush()

    meta = {
        "source": args.input or URL,
        "records_read_approx": read,
        "positions_kept": kept,
        "skipped": skipped,
        "score_clamp": SCORE_CLAMP,
        "val_every": VAL_EVERY,
        "shards": train.stats,
        "val": val.stats,
        "seconds": round(time.perf_counter() - start),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"done: kept {kept:,} of ~{read:,} records in {meta['seconds']}s; skipped {skipped}")


if __name__ == "__main__":
    main()
