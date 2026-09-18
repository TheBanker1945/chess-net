"""Play against the engine in the terminal.

Usage: python -m chessnet.play [--color white|black|random] [--time SECONDS]
"""

from __future__ import annotations

import argparse
import random
import sys

import chess
import chess.pgn

from chessnet.evaluators import DEFAULT, EVALUATORS, get_evaluator
from chessnet.search import SearchInfo, Searcher

HELP = """\
Enter moves in SAN (Nf3, exd5, O-O, e8=Q) or UCI (g1f3, e7e8q).
Commands:
  undo     take back your last move (and the engine's reply)
  hint     ask the engine for a move suggestion
  fen      print the current position as FEN
  pgn      print the game so far as PGN
  flip     flip the board display
  resign   resign the game
  quit     exit
  help     show this message"""


def render(board: chess.Board, white_bottom: bool, last: chess.Move | None) -> str:
    ranks = range(7, -1, -1) if white_bottom else range(8)
    files = range(8) if white_bottom else range(7, -1, -1)
    highlight = {last.from_square, last.to_square} if last else set()
    lines = []
    for r in ranks:
        row = [f" {r + 1} "]
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            ch = piece.symbol() if piece else "."
            row.append(f"[{ch}]" if sq in highlight else f" {ch} ")
        lines.append("".join(row))
    letters = "abcdefgh" if white_bottom else "hgfedcba"
    lines.append("    " + "  ".join(letters))
    return "\n".join(lines)


def format_score(info: SearchInfo) -> str:
    if info.mate_in is not None:
        return f"mate {info.mate_in}"
    return f"{info.score / 100:+.2f}"


def format_pv(board: chess.Board, pv: list[chess.Move]) -> str:
    b = board.copy(stack=False)
    sans = []
    for move in pv:
        if not b.is_legal(move):
            break
        sans.append(b.san(move))
        b.push(move)
    return " ".join(sans)


def parse_move(board: chess.Board, text: str) -> chess.Move | None:
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play against chess-net.")
    parser.add_argument("--color", choices=("white", "black", "random"), default="white")
    parser.add_argument("--time", type=float, default=3.0, help="engine seconds per move (default 3)")
    parser.add_argument("--depth", type=int, default=None, help="optional maximum search depth")
    parser.add_argument("--eval", choices=EVALUATORS, default=DEFAULT)
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="starting position")
    parser.add_argument("--pgn", help="append the finished game to this PGN file")
    parser.add_argument("--quiet", action="store_true", help="hide engine search output")
    args = parser.parse_args(argv)

    color = args.color if args.color != "random" else random.choice(("white", "black"))
    human = chess.WHITE if color == "white" else chess.BLACK
    board = chess.Board(args.fen)
    searcher = Searcher(get_evaluator(args.eval))
    white_bottom = human == chess.WHITE
    resigned = None

    print(f"You play {color}. Engine thinks {args.time:g}s per move. Type 'help' for commands.\n")

    def engine_report(info: SearchInfo) -> None:
        if not args.quiet:
            print(
                f"  depth {info.depth:2d}  score {format_score(info):>8}  nodes {info.nodes:>8}"
                f"  nps {info.nps:>6}  pv {format_pv(board, info.pv)}",
                flush=True,
            )

    while not board.is_game_over(claim_draw=True):
        last = board.peek() if board.move_stack else None
        if board.turn == human:
            print(render(board, white_bottom, last))
            try:
                text = input(f"\n{'White' if board.turn else 'Black'} to move> ").strip()
            except EOFError:
                print()
                return
            if not text:
                continue
            cmd = text.lower()
            if cmd in ("quit", "exit"):
                return
            if cmd == "help":
                print(HELP)
            elif cmd == "fen":
                print(board.fen())
            elif cmd == "pgn":
                print(chess.pgn.Game.from_board(board))
            elif cmd == "flip":
                white_bottom = not white_bottom
            elif cmd == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                else:
                    print("Nothing to undo.")
            elif cmd == "hint":
                result = searcher.search(board, time_limit=args.time, max_depth=args.depth)
                print(f"Hint: {board.san(result.move)} ({format_score(result.info)})")
            elif cmd == "resign":
                resigned = human
                break
            else:
                move = parse_move(board, text)
                if move is None:
                    print(f"Illegal or unrecognised move: {text!r}. Type 'help' for help.")
                else:
                    board.push(move)
            print()
        else:
            print(render(board, white_bottom, last))
            print("\nEngine thinking...")
            result = searcher.search(
                board, time_limit=args.time, max_depth=args.depth, on_iteration=engine_report
            )
            info = result.info
            print(
                f"Engine plays {board.san(result.move)}  "
                f"(depth {info.depth}, score {format_score(info)}, {info.nodes} nodes, {info.elapsed:.1f}s)\n"
            )
            board.push(result.move)

    print(render(board, white_bottom, board.peek() if board.move_stack else None))
    game = chess.pgn.Game.from_board(board)
    if resigned is not None:
        result = "0-1" if resigned == chess.WHITE else "1-0"
        print(f"\nYou resigned. {result}")
    else:
        outcome = board.outcome(claim_draw=True)
        result = outcome.result()
        reason = outcome.termination.name.replace("_", " ").lower()
        print(f"\nGame over: {result} ({reason})")
    game.headers["Result"] = result
    game.headers["White"] = "You" if human == chess.WHITE else "chess-net"
    game.headers["Black"] = "You" if human == chess.BLACK else "chess-net"
    if args.pgn:
        with open(args.pgn, "a") as f:
            print(game, file=f, end="\n\n")
        print(f"Saved to {args.pgn}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
