# chess-net

A Python chess engine: alpha-beta search on top of
[python-chess](https://python-chess.readthedocs.io/) (which handles move
generation and the rules), with a pluggable evaluation function.

- **Stage 1** — handwritten evaluation: tapered material + piece-square tables.
- **Stage 2** — learned NNUE-style evaluation (not started yet).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install chess
```

## Play against it

```bash
.venv/bin/python -m chessnet.play                    # you play White, 3 s per engine move
.venv/bin/python -m chessnet.play --color black --time 5
```

Enter moves as SAN (`Nf3`, `O-O`, `e8=Q`) or UCI (`g1f3`). Commands: `undo`,
`hint`, `fen`, `pgn`, `flip`, `resign`, `quit`, `help`. `--pgn games.pgn`
appends the finished game to a file.

The engine also speaks UCI, so any chess GUI (Arena, Cute Chess, En Croissant)
can load it with the command `/path/to/.venv/bin/python -m chessnet.uci` run
from the project directory.

## Layout

| Module | Purpose |
| --- | --- |
| `chessnet/search.py` | Search. Receives an `evaluate(board) -> centipawns` callable; imports no evaluator. |
| `chessnet/eval_pesto.py` | Stage 1 evaluation (PeSTO material + piece-square tables). |
| `chessnet/evaluators.py` | Name → evaluator registry used by every front-end. Stage 2 registers here. |
| `chessnet/play.py` | Terminal game. |
| `chessnet/uci.py` | UCI protocol adapter. |
| `chessnet/perft.py` | Perft / divide for move-generation verification. |
| `chessnet/match.py`, `elo.py`, `openings.py` | Stockfish match runner and Elo estimate. |

### Search

Negamax alpha-beta with principal variation search, iterative deepening,
aspiration windows, and a transposition table keyed on python-chess's exact
position tuple. Move ordering: TT move, then MVV-LVA captures, promotions,
two killer moves per ply, then history heuristic. Pruning and extensions:
null-move pruning, reverse futility pruning, futility pruning, late move
reductions, check extension. Quiescence search covers captures and queen
promotions with delta pruning and skips obviously losing captures. Repetitions
(against game history and the search path), the 50-move rule, and insufficient
material score as draws.

Throughput is roughly 25–35k nodes/s on one core (python-chess move generation
is the ceiling), which is depth 7–9 in a few seconds.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
PERFT_DEEP=1 .venv/bin/python -m unittest tests.test_perft   # adds 1–16M-node perft counts (~1 min)
```

The perft test checks node counts for the six standard
[chessprogramming.org perft positions](https://www.chessprogramming.org/Perft_Results)
(plus mirrored position 4). The search tests check forced mates up to mate in
3 (each verified by an independent brute-force solver), tactics,
stalemate/perpetual handling, time limits, and colour symmetry of the
evaluation.

## Measuring strength

```bash
.venv/bin/python -m chessnet.match --games 100 --sf-elo 1500      # default tc 60+0.6
```

This plays chess-net against Stockfish (1 thread, 16 MB hash) limited with
`UCI_LimitStrength`/`UCI_Elo`, from 30 opening lines with each line played
once per colour. It prints W/D/L, the Elo difference with a 95% confidence
interval, and the implied rating, and writes `games.pgn` + `summary.json` to
`matches/<timestamp>/`.

How to read the number:

- Stockfish calibrates `UCI_Elo` against **CCRL 40/4** at **60s+0.6s**, which is
  why that is the default time control. The rating is on that scale. It is
  not a Lichess or FIDE rating; those scales differ by hundreds of points at
  this level.
- The estimate is most precise when the score is near 50%. If the engine
  scores above ~75% or below ~25%, re-run against a closer `--sf-elo`.
- 100 games still leaves roughly ±60–70 Elo of 95% uncertainty.
- `--sf-skill`, `--sf-depth` and `--sf-nodes` opponents are not calibrated. With
  them, only the difference is meaningful, for example to compare the stage 1
  and stage 2 evaluations against the same fixed opponent.

Stockfish is found on `PATH`, at `/usr/games/stockfish`, or in `tools/sf/`
(Ubuntu's package unpacked locally with `apt-get download stockfish && dpkg -x`).
Override with `--stockfish PATH`.
