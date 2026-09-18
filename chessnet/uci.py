"""UCI front-end, so the engine can run in GUIs and in the match harness.

Usage: python -m chessnet.uci [--eval NAME]
"""

from __future__ import annotations

import argparse
import sys
import threading

import chess

from chessnet.evaluators import DEFAULT, EVALUATORS, get_evaluator
from chessnet.search import SearchInfo, Searcher

ENGINE_NAME = "chess-net"
MOVE_OVERHEAD = 0.05  # seconds kept in reserve for process/IPC latency


def allocate_time(remaining: float, increment: float, moves_to_go: int | None) -> float:
    """Seconds to spend on this move given the clock state."""
    mtg = moves_to_go if moves_to_go else 30
    budget = remaining / mtg + 0.75 * increment
    budget = min(budget, 0.4 * remaining)
    return max(0.02, budget - MOVE_OVERHEAD)


def format_info(info: SearchInfo) -> str:
    if info.mate_in is not None:
        score = f"mate {info.mate_in}"
    else:
        score = f"cp {info.score}"
    pv = " ".join(m.uci() for m in info.pv)
    return (
        f"info depth {info.depth} score {score} nodes {info.nodes} nps {info.nps}"
        f" time {int(info.elapsed * 1000)} pv {pv}"
    )


class UciEngine:
    def __init__(self, eval_name: str):
        self.eval_name = eval_name
        self.searcher = Searcher(get_evaluator(eval_name))
        self.board = chess.Board()
        self.thread: threading.Thread | None = None
        self.out_lock = threading.Lock()

    def send(self, line: str) -> None:
        with self.out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def handle(self, line: str) -> bool:
        """Process one command. Returns False when the engine should exit."""
        tokens = line.split()
        if not tokens:
            return True
        cmd = tokens[0]
        if cmd == "uci":
            self.send(f"id name {ENGINE_NAME} ({self.eval_name})")
            self.send("id author chess-net")
            self.send("uciok")
        elif cmd == "isready":
            self.send("readyok")
        elif cmd == "ucinewgame":
            self.wait()
            self.searcher.new_game()
        elif cmd == "position":
            self.wait()
            self.board = self.parse_position(tokens[1:])
        elif cmd == "go":
            self.wait()
            self.go(tokens[1:])
        elif cmd == "stop":
            self.searcher.stop_requested = True
            self.wait()
        elif cmd == "quit":
            self.searcher.stop_requested = True
            self.wait()
            return False
        return True

    @staticmethod
    def parse_position(args: list[str]) -> chess.Board:
        if not args:
            return chess.Board()
        if args[0] == "startpos":
            board = chess.Board()
            rest = args[1:]
        elif args[0] == "fen":
            fen_parts = []
            rest = args[1:]
            while rest and rest[0] != "moves":
                fen_parts.append(rest.pop(0))
            board = chess.Board(" ".join(fen_parts))
        else:
            return chess.Board()
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                board.push_uci(uci)
        return board

    def go(self, args: list[str]) -> None:
        params: dict[str, int] = {}
        infinite = False
        i = 0
        while i < len(args):
            key = args[i]
            if key == "infinite":
                infinite = True
                i += 1
            elif key in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth", "nodes") and i + 1 < len(args):
                params[key] = int(args[i + 1])
                i += 2
            else:
                i += 1

        time_limit = None
        if "movetime" in params:
            time_limit = max(0.02, params["movetime"] / 1000 - MOVE_OVERHEAD)
        elif not infinite:
            clock_key, inc_key = ("wtime", "winc") if self.board.turn == chess.WHITE else ("btime", "binc")
            if clock_key in params:
                time_limit = allocate_time(
                    params[clock_key] / 1000, params.get(inc_key, 0) / 1000, params.get("movestogo")
                )
        depth = params.get("depth")
        nodes = params.get("nodes")
        if time_limit is None and depth is None and nodes is None and not infinite:
            depth = 6  # bare "go": pick something finite

        board = self.board.copy()

        def run() -> None:
            result = self.searcher.search(
                board,
                time_limit=time_limit,
                max_depth=depth,
                max_nodes=nodes,
                on_iteration=lambda info: self.send(format_info(info)),
            )
            self.send(f"bestmove {result.move.uci() if result.move else '0000'}")

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="chess-net UCI engine")
    parser.add_argument("--eval", choices=EVALUATORS, default=DEFAULT)
    args = parser.parse_args()
    engine = UciEngine(args.eval)
    for line in sys.stdin:
        if not engine.handle(line.strip()):
            break


if __name__ == "__main__":
    main()
