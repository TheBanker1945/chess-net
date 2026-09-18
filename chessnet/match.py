"""Measure playing strength: play a match against Stockfish and estimate Elo.

The opponent is Stockfish with one thread and a limited strength. Only the
UCI_Elo mode (--sf-elo) gives an absolute rating, because Stockfish calibrates
UCI_Elo against the CCRL 40/4 rating list at a 60s+0.6s time control; that is
why 60+0.6 is the default here. --sf-skill / --sf-depth / --sf-nodes opponents
have no published rating, so for those only the Elo *difference* is reported,
which is still useful for comparing two versions of this engine against the
same fixed opponent.

Example:
    python -m chessnet.match --games 100 --sf-elo 1500
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chessnet.elo import EloEstimate, estimate
from chessnet.evaluators import DEFAULT, EVALUATORS
from chessnet.openings import OPENINGS, OPENINGS_SAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_PLIES = 400  # adjudicate as a draw beyond this


def find_stockfish() -> str:
    """System Stockfish if on PATH, else the copy unpacked under tools/."""
    for candidate in (shutil.which("stockfish"), "/usr/games/stockfish"):
        if candidate and Path(candidate).exists():
            return candidate
    return str(PROJECT_ROOT / "tools" / "sf" / "usr" / "games" / "stockfish")


@dataclass
class TimeControl:
    base: float
    inc: float

    @classmethod
    def parse(cls, text: str) -> "TimeControl":
        base, _, inc = text.partition("+")
        return cls(float(base), float(inc or 0))

    def __str__(self) -> str:
        return f"{self.base:g}+{self.inc:g}"


@dataclass
class GameSpec:
    index: int
    opening: int
    ours_white: bool


@dataclass
class GameResult:
    index: int
    ours_white: bool
    points: float  # from our engine's point of view
    result: str  # PGN result string
    termination: str
    plies: int


class Players:
    """One worker's pair of engine processes."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ours: chess.engine.SimpleEngine | None = None
        self.sf: chess.engine.SimpleEngine | None = None
        self.start()

    def start(self) -> None:
        self.close()
        self.ours = chess.engine.SimpleEngine.popen_uci(
            [sys.executable, "-m", "chessnet.uci", "--eval", self.args.eval], cwd=PROJECT_ROOT
        )
        self.sf = chess.engine.SimpleEngine.popen_uci(self.args.stockfish)
        self.sf.configure(stockfish_options(self.args))

    def close(self) -> None:
        for engine in (self.ours, self.sf):
            if engine is not None:
                try:
                    engine.quit()
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError):
                    engine.close()
        self.ours = self.sf = None


def stockfish_options(args: argparse.Namespace) -> dict:
    options: dict = {"Threads": 1, "Hash": 16}
    if args.sf_elo is not None:
        options["UCI_LimitStrength"] = True
        options["UCI_Elo"] = args.sf_elo
    elif args.sf_skill is not None:
        options["Skill Level"] = args.sf_skill
    return options


def opponent_label(args: argparse.Namespace) -> str:
    if args.sf_elo is not None:
        return f"Stockfish UCI_Elo={args.sf_elo}"
    if args.sf_skill is not None:
        return f"Stockfish Skill={args.sf_skill}"
    if args.sf_depth is not None:
        return f"Stockfish depth={args.sf_depth}"
    return f"Stockfish nodes={args.sf_nodes}"


def play_game(players: Players, spec: GameSpec, args: argparse.Namespace, tc: TimeControl):
    board = chess.Board()
    for move in OPENINGS[spec.opening]:
        board.push(move)
    ours_color = chess.WHITE if spec.ours_white else chess.BLACK
    clocks = {chess.WHITE: tc.base, chess.BLACK: tc.base}
    sf_fixed = args.sf_depth is not None or args.sf_nodes is not None
    game_key = object()  # a new object makes python-chess send ucinewgame
    termination = None
    winner: bool | None = None

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            winner = outcome.winner
            termination = outcome.termination.name.lower()
            break
        if board.ply() >= MAX_PLIES:
            termination = "max_plies"
            break

        mover = board.turn
        is_ours = mover == ours_color
        engine = players.ours if is_ours else players.sf
        if not is_ours and sf_fixed:
            limit = chess.engine.Limit(depth=args.sf_depth, nodes=args.sf_nodes)
        else:
            limit = chess.engine.Limit(
                white_clock=clocks[chess.WHITE],
                black_clock=clocks[chess.BLACK],
                white_inc=tc.inc,
                black_inc=tc.inc,
            )
        start = time.perf_counter()
        try:
            result = engine.play(board, limit, game=game_key)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            print(f"  ! engine error ({'ours' if is_ours else 'stockfish'}): {exc!r}", flush=True)
            players.start()
            winner = not mover
            termination = "engine_error"
            break
        elapsed = time.perf_counter() - start
        if is_ours or not sf_fixed:
            clocks[mover] -= elapsed
            if clocks[mover] < 0:
                # A flag only loses if the opponent could still mate.
                winner = None if board.has_insufficient_material(not mover) else (not mover)
                termination = "time_forfeit"
                break
            clocks[mover] += tc.inc
        if result.move is None:
            winner = not mover
            termination = "no_move"
            break
        board.push(result.move)

    result_str = "1/2-1/2" if winner is None else ("1-0" if winner == chess.WHITE else "0-1")
    points = 0.5 if winner is None else (1.0 if winner == ours_color else 0.0)

    game = chess.pgn.Game.from_board(board)
    ours_name = f"chess-net ({args.eval})"
    game.headers["Event"] = f"chess-net vs {opponent_label(args)}"
    game.headers["Round"] = str(spec.index + 1)
    game.headers["White"] = ours_name if spec.ours_white else opponent_label(args)
    game.headers["Black"] = opponent_label(args) if spec.ours_white else ours_name
    game.headers["Result"] = result_str
    game.headers["TimeControl"] = f"{tc.base:g}+{tc.inc:g}"
    game.headers["Opening"] = OPENINGS_SAN[spec.opening]
    game.headers["Termination"] = termination
    return GameResult(spec.index, spec.ours_white, points, result_str, termination, board.ply()), game


def format_estimate(est: EloEstimate, anchor: float | None) -> str:
    text = (
        f"+{est.wins} ={est.draws} -{est.losses}  score {est.score:.3f}  "
        f"Elo diff {est.diff:+.0f} [{est.diff_low:+.0f}, {est.diff_high:+.0f}]"
    )
    if anchor is not None:
        rating, low, high = est.rating(anchor)
        text += f"  => rating {rating:.0f} [{low:.0f}, {high:.0f}]"
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Play chess-net against a strength-limited Stockfish and estimate Elo."
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--tc", default="60+0.6", help="base+increment in seconds (default 60+0.6)")
    parser.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    parser.add_argument("--eval", choices=EVALUATORS, default=DEFAULT)
    parser.add_argument("--stockfish", default=find_stockfish())
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument("--sf-elo", type=int, help="UCI_Elo (calibrated; Stockfish 16 minimum is 1320)")
    strength.add_argument("--sf-skill", type=int, help="Skill Level 0-20 (uncalibrated)")
    strength.add_argument("--sf-depth", type=int, help="fixed search depth (uncalibrated, no clock)")
    strength.add_argument("--sf-nodes", type=int, help="fixed node count (uncalibrated, no clock)")
    parser.add_argument("--out", type=Path, help="output directory (default matches/<timestamp>)")
    args = parser.parse_args(argv)

    if args.sf_elo is None and args.sf_skill is None and args.sf_depth is None and args.sf_nodes is None:
        args.sf_elo = 1500
    if not Path(args.stockfish).exists() and not shutil.which(args.stockfish):
        parser.error(f"Stockfish not found at {args.stockfish!r}; install it or pass --stockfish PATH")

    tc = TimeControl.parse(args.tc)
    out_dir = args.out or PROJECT_ROOT / "matches" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = out_dir / "games.pgn"
    anchor = float(args.sf_elo) if args.sf_elo is not None else None

    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as probe:
        sf_name = probe.id.get("name", "Stockfish")
        if args.sf_elo is not None:
            opt = probe.options["UCI_Elo"]
            if not opt.min <= args.sf_elo <= opt.max:
                parser.error(f"{sf_name} supports UCI_Elo {opt.min}..{opt.max}")

    specs: queue.Queue[GameSpec] = queue.Queue()
    for i in range(args.games):
        specs.put(GameSpec(i, (i // 2) % len(OPENINGS), ours_white=(i % 2 == 0)))

    print(
        f"chess-net ({args.eval}) vs {sf_name} [{opponent_label(args)}], {args.games} games, "
        f"tc {tc}, {args.concurrency} concurrent\nPGN: {pgn_path}",
        flush=True,
    )

    results: list[GameResult] = []
    lock = threading.Lock()
    started = time.perf_counter()

    def worker() -> None:
        players = Players(args)
        try:
            while True:
                try:
                    spec = specs.get_nowait()
                except queue.Empty:
                    return
                res, game = play_game(players, spec, args, tc)
                with lock:
                    results.append(res)
                    with open(pgn_path, "a") as f:
                        print(game, file=f, end="\n\n")
                    w = sum(r.points == 1.0 for r in results)
                    d = sum(r.points == 0.5 for r in results)
                    l = sum(r.points == 0.0 for r in results)
                    colour = "W" if res.ours_white else "B"
                    print(
                        f"[{len(results):3d}/{args.games}] game {res.index + 1:3d} ({colour}) "
                        f"{res.result:7s} {res.termination:22s} | {format_estimate(estimate(w, d, l), anchor)}",
                        flush=True,
                    )
        finally:
            players.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(args.concurrency, args.games))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    w = sum(r.points == 1.0 for r in results)
    d = sum(r.points == 0.5 for r in results)
    l = sum(r.points == 0.0 for r in results)
    est = estimate(w, d, l)
    minutes = (time.perf_counter() - started) / 60
    print(f"\nFinal ({len(results)} games, {minutes:.1f} min): {format_estimate(est, anchor)}")
    if anchor is None:
        print("Opponent is not Elo-calibrated: the difference is relative to this opponent only.")
    else:
        print("Rating is on Stockfish's UCI_Elo scale (anchored to CCRL 40/4), not Lichess/FIDE.")

    terminations: dict[str, int] = {}
    for r in results:
        terminations[r.termination] = terminations.get(r.termination, 0) + 1
    summary = {
        "engine": f"chess-net ({args.eval})",
        "opponent": f"{sf_name} [{opponent_label(args)}]",
        "time_control": str(tc),
        "games": len(results),
        "estimate": asdict(est),
        "rating": est.rating(anchor) if anchor is not None else None,
        "terminations": terminations,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
