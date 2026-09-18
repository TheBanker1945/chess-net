"""Perft: count leaf nodes of the legal move tree to verify move generation.

Usage: python -m chessnet.perft [--divide] DEPTH [FEN]
"""

import argparse
import time

import chess


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.generate_legal_moves():
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def divide(board: chess.Board, depth: int) -> dict[str, int]:
    counts = {}
    for move in board.generate_legal_moves():
        board.push(move)
        counts[move.uci()] = perft(board, depth - 1)
        board.pop()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("depth", type=int)
    parser.add_argument("fen", nargs="?", default=chess.STARTING_FEN)
    parser.add_argument("--divide", action="store_true", help="print the count below each root move")
    args = parser.parse_args()

    board = chess.Board(args.fen)
    start = time.perf_counter()
    if args.divide:
        counts = divide(board, args.depth)
        for move, n in sorted(counts.items()):
            print(f"{move}: {n}")
        total = sum(counts.values())
    else:
        total = perft(board, args.depth)
    elapsed = time.perf_counter() - start
    print(f"nodes {total}  time {elapsed:.2f}s  ({total / elapsed:,.0f} nodes/s)")


if __name__ == "__main__":
    main()
