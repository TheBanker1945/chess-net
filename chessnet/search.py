"""Alpha-beta search with iterative deepening.

The search knows nothing about how positions are scored: it is handed an
evaluation callable ``evaluate(board) -> int`` (centipawns, side to move) and
calls it at quiescence leaves and for pruning decisions. Swapping the
handwritten evaluation for the neural one means passing a different callable.

Features: principal variation search, aspiration windows, transposition table,
quiescence search (captures + queen promotions, delta pruning), check
extension, null-move pruning, reverse futility pruning, futility pruning, late
move reductions, killer moves and history heuristic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import chess

Evaluator = Callable[[chess.Board], int]

INF = 1_000_000
MATE = 100_000
MATE_BOUND = MATE - 1_000  # |score| above this means "mate in N plies"
MAX_PLY = 100

EXACT, LOWER, UPPER = 0, 1, 2

# Rough piece values used only for move ordering and qsearch pruning, never for
# evaluation. Indexed by piece type; index 0 is unused.
ORDER_VALUE = (0, 100, 320, 330, 500, 900, 20_000)

NULL_MOVE = chess.Move.null()
FUTILITY_MARGIN = (0, 150, 300)
DELTA_MARGIN = 200
ASPIRATION_WINDOW = 40


def position_key(board: chess.Board):
    """Exact, hashable identity of a position for the TT and repetition checks.

    python-chess's own zobrist_hash() is ~20x slower because it rehashes from
    scratch in pure Python; _transposition_key() is a tuple of bitboards that
    python-chess itself uses for repetition detection.
    """
    return board._transposition_key()


class SearchTimeout(Exception):
    pass


@dataclass
class SearchInfo:
    depth: int
    score: int
    nodes: int
    elapsed: float
    pv: list[chess.Move] = field(default_factory=list)

    @property
    def nps(self) -> int:
        return int(self.nodes / self.elapsed) if self.elapsed > 0 else 0

    @property
    def mate_in(self) -> int | None:
        """Moves to mate (positive: side to move mates), or None."""
        if abs(self.score) < MATE_BOUND:
            return None
        plies = MATE - abs(self.score)
        moves = (plies + 1) // 2
        return moves if self.score > 0 else -moves


@dataclass
class SearchResult:
    move: chess.Move | None
    info: SearchInfo


def _score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


class Searcher:
    def __init__(self, evaluate: Evaluator, tt_max_entries: int = 2_000_000):
        self.evaluate = evaluate
        self.tt_max_entries = tt_max_entries
        self.tt: dict = {}
        self.history = [[0] * 4096, [0] * 4096]
        self.stop_requested = False

    def new_game(self) -> None:
        self.tt.clear()
        self.history = [[0] * 4096, [0] * 4096]

    # ------------------------------------------------------------------ root

    def search(
        self,
        board: chess.Board,
        *,
        time_limit: float | None = None,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        on_iteration: Callable[[SearchInfo], None] | None = None,
    ) -> SearchResult:
        """Search ``board`` (not modified) and return the best move found.

        Stops at whichever limit is hit first. With no limits it searches to
        MAX_PLY, so callers should always pass at least one.
        """
        self.board = board.copy()
        self.start = time.perf_counter()
        self.deadline = self.start + time_limit if time_limit else None
        self.max_nodes = max_nodes
        self.nodes = 0
        self.stop_requested = False
        self.killers = [[None, None] for _ in range(MAX_PLY + 2)]
        if len(self.tt) > self.tt_max_entries:
            self.tt.clear()
        for side in self.history:
            for i, v in enumerate(side):
                side[i] = v >> 3

        # Positions already on the board's history count as repetitions.
        self.rep: dict = {}
        tmp = board.copy()
        self.rep[position_key(tmp)] = 1
        for _ in range(min(tmp.halfmove_clock, len(tmp.move_stack))):
            tmp.pop()
            k = position_key(tmp)
            self.rep[k] = self.rep.get(k, 0) + 1

        root_moves = list(self.board.legal_moves)
        if not root_moves:
            return SearchResult(None, SearchInfo(0, 0, 0, 0.0))
        root_moves = self._order_moves(self.board, root_moves, self._tt_move(), 0)

        best_move = root_moves[0]
        info = SearchInfo(0, 0, 0, 0.0, [best_move])
        if len(root_moves) == 1:
            return SearchResult(best_move, info)

        max_depth = min(max_depth or MAX_PLY, MAX_PLY)
        prev_score = 0
        for depth in range(1, max_depth + 1):
            self.iter_best = None
            try:
                if depth >= 4:
                    alpha = prev_score - ASPIRATION_WINDOW
                    beta = prev_score + ASPIRATION_WINDOW
                    score = self._root(root_moves, depth, alpha, beta)
                    if score <= alpha or score >= beta:
                        score = self._root(root_moves, depth, -INF, INF)
                else:
                    score = self._root(root_moves, depth, -INF, INF)
            except SearchTimeout:
                # A move that improved alpha in the unfinished iteration was
                # searched with a real window, so it is safe to play.
                if self.iter_best is not None:
                    best_move = self.iter_best[0]
                    if best_move != info.pv[0]:
                        info.pv = [best_move]
                        info.score = self.iter_best[1]
                break

            best_move = self.iter_best[0]
            prev_score = score
            root_moves.remove(best_move)
            root_moves.insert(0, best_move)
            elapsed = time.perf_counter() - self.start
            info = SearchInfo(depth, score, self.nodes, elapsed, self._extract_pv(best_move))
            if on_iteration:
                on_iteration(info)

            if abs(score) >= MATE_BOUND and MATE - abs(score) <= depth:
                break  # found a forced mate that the search horizon fully covers
            if self.deadline and elapsed > 0.5 * (self.deadline - self.start):
                break  # the next iteration would very likely not finish

        info.nodes = self.nodes
        info.elapsed = time.perf_counter() - self.start
        return SearchResult(best_move, info)

    def _root(self, moves: list[chess.Move], depth: int, alpha: int, beta: int) -> int:
        board = self.board
        best_score = -INF
        for i, move in enumerate(moves):
            board.push(move)
            if i == 0:
                score = -self._negamax(depth - 1, -beta, -alpha, 1, True)
            else:
                score = -self._negamax(depth - 1, -alpha - 1, -alpha, 1, True)
                if alpha < score < beta:
                    score = -self._negamax(depth - 1, -beta, -alpha, 1, True)
            board.pop()
            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score
                    self.iter_best = (move, score)
                    if score >= beta:
                        break
        if self.iter_best is None:
            # Everything failed low against the aspiration window; any move
            # is fine as a placeholder because the caller re-searches.
            self.iter_best = (moves[0], best_score)
        return best_score

    # ---------------------------------------------------------------- search

    def _check_limits(self) -> None:
        if self.stop_requested:
            raise SearchTimeout
        if self.deadline and time.perf_counter() >= self.deadline:
            raise SearchTimeout
        if self.max_nodes and self.nodes >= self.max_nodes:
            raise SearchTimeout

    def _negamax(self, depth: int, alpha: int, beta: int, ply: int, allow_null: bool) -> int:
        board = self.board

        if board.halfmove_clock >= 100:
            return 0
        if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
            return 0
        key = position_key(board)
        if self.rep.get(key):
            return 0

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0 or ply >= MAX_PLY:
            return self._qsearch(alpha, beta, ply)

        self.nodes += 1
        if not self.nodes & 255:
            self._check_limits()

        pv_node = beta - alpha > 1

        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            e_depth, e_flag, e_score, tt_move = entry
            if e_depth >= depth and not pv_node:
                s = _score_from_tt(e_score, ply)
                if e_flag == EXACT or (e_flag == LOWER and s >= beta) or (e_flag == UPPER and s <= alpha):
                    return s

        futile = False
        if not in_check and not pv_node:
            static = self.evaluate(board)

            if depth <= 4 and static - 90 * depth >= beta and abs(beta) < MATE_BOUND:
                return static

            if (
                allow_null
                and depth >= 3
                and static >= beta
                and board.occupied_co[board.turn] & ~(board.pawns | board.kings)
            ):
                r = 3 if depth >= 6 else 2
                board.push(NULL_MOVE)
                score = -self._negamax(depth - 1 - r, -beta, -beta + 1, ply + 1, False)
                board.pop()
                if score >= beta:
                    return beta

            if depth <= 2 and static + FUTILITY_MARGIN[depth] <= alpha:
                futile = True

        moves = list(board.generate_legal_moves())
        if not moves:
            return -MATE + ply if in_check else 0
        moves = self._order_moves(board, moves, tt_move, ply)

        self.rep[key] = 1
        best_score = -INF
        best_move = moves[0]
        flag = UPPER
        new_depth = depth - 1
        killers = self.killers[ply]

        for i, move in enumerate(moves):
            quiet = move.promotion is None and not board.is_capture(move)
            board.push(move)
            gives_check = board.is_check()

            if futile and i > 0 and quiet and not gives_check:
                board.pop()
                continue

            if i == 0:
                score = -self._negamax(new_depth, -beta, -alpha, ply + 1, True)
            else:
                r = 0
                if (
                    depth >= 3
                    and i >= 3
                    and quiet
                    and not in_check
                    and not gives_check
                    and move != killers[0]
                    and move != killers[1]
                ):
                    r = 1 if i < 8 else 2
                    if depth >= 6 and i >= 16:
                        r += 1
                    if pv_node:
                        r -= 1
                    r = max(0, min(r, new_depth - 1))
                score = -self._negamax(new_depth - r, -alpha - 1, -alpha, ply + 1, True)
                if score > alpha and r:
                    score = -self._negamax(new_depth, -alpha - 1, -alpha, ply + 1, True)
                if alpha < score < beta:
                    score = -self._negamax(new_depth, -beta, -alpha, ply + 1, True)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    flag = EXACT
                    if score >= beta:
                        flag = LOWER
                        if quiet:
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            hist = self.history[board.turn]
                            idx = move.from_square * 64 + move.to_square
                            # Capped below the killer scores in _order_moves.
                            hist[idx] = min(hist[idx] + depth * depth, 20_000_000)
                        break

        del self.rep[key]
        self.tt[key] = (depth, flag, _score_to_tt(best_score, ply), best_move)
        return best_score

    def _qsearch(self, alpha: int, beta: int, ply: int) -> int:
        board = self.board
        self.nodes += 1
        if not self.nodes & 255:
            self._check_limits()

        if board.is_check():
            moves = list(board.generate_legal_moves())
            if not moves:
                return -MATE + ply
            if ply >= MAX_PLY:
                return self.evaluate(board)
            moves = self._order_moves(board, moves, None, ply)
            best = -INF
            stand = None
        else:
            stand = self.evaluate(board)
            if stand >= beta or ply >= MAX_PLY:
                return stand
            if stand > alpha:
                alpha = stand
            best = stand
            moves = self._qsearch_moves(board, stand, alpha)

        for move in moves:
            board.push(move)
            score = -self._qsearch(-beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        break
        return best

    def _qsearch_moves(self, board: chess.Board, stand: int, alpha: int) -> list[chess.Move]:
        """Captures and queen promotions worth trying, best first."""
        them = not board.turn
        scored = []
        for move in board.generate_legal_moves(board.occupied_co[board.turn], board.occupied_co[them] | chess.BB_BACKRANKS):
            promo = move.promotion
            if promo is not None and promo != chess.QUEEN:
                continue
            victim = board.piece_type_at(move.to_square)
            if victim is None:
                if promo is None:
                    continue  # a quiet move onto the back rank
                victim_value = 0
            else:
                victim_value = ORDER_VALUE[victim]
            gain = victim_value + (800 if promo else 0)
            if stand + gain + DELTA_MARGIN <= alpha:
                continue
            attacker = board.piece_type_at(move.from_square)
            attacker_value = ORDER_VALUE[attacker]
            # Skip obviously losing captures: a more valuable piece takes a
            # defended one.
            if promo is None and attacker_value > victim_value and board.is_attacked_by(them, move.to_square):
                continue
            scored.append((gain * 16 - attacker, move))
        if board.ep_square is not None:
            for move in board.generate_legal_ep():
                scored.append((ORDER_VALUE[chess.PAWN] * 16 - chess.PAWN, move))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    # --------------------------------------------------------------- helpers

    def _tt_move(self) -> chess.Move | None:
        entry = self.tt.get(position_key(self.board))
        return entry[3] if entry else None

    def _order_moves(self, board: chess.Board, moves: list[chess.Move], tt_move, ply: int) -> list[chess.Move]:
        killers = self.killers[ply] if ply <= MAX_PLY else (None, None)
        hist = self.history[board.turn]
        piece_type_at = board.piece_type_at
        scored = []
        for m in moves:
            if m == tt_move:
                s = 100_000_000
            elif board.is_capture(m):
                victim = piece_type_at(m.to_square) or chess.PAWN
                s = 50_000_000 + ORDER_VALUE[victim] * 16 - piece_type_at(m.from_square)
            elif m.promotion:
                s = 40_000_000 + m.promotion
            elif m == killers[0]:
                s = 30_000_000
            elif m == killers[1]:
                s = 29_000_000
            else:
                s = hist[m.from_square * 64 + m.to_square]
            scored.append((s, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    def _extract_pv(self, first: chess.Move, limit: int = 20) -> list[chess.Move]:
        board = self.board.copy(stack=False)
        pv = [first]
        board.push(first)
        seen = {position_key(board)}
        while len(pv) < limit:
            entry = self.tt.get(position_key(board))
            if not entry or entry[1] != EXACT or not board.is_legal(entry[3]):
                break
            board.push(entry[3])
            k = position_key(board)
            if k in seen:
                break
            seen.add(k)
            pv.append(entry[3])
        return pv
