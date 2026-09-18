# chess-net

A Python chess engine: alpha-beta search on top of
[python-chess](https://python-chess.readthedocs.io/) (which handles move
generation and the rules), with a pluggable evaluation function.

- **Stage 1** — handwritten evaluation: tapered material + piece-square tables.
- **Stage 2** — learned NNUE-style evaluation trained on Lichess's Stockfish evaluations.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install chess numpy                  # engine + NNUE inference
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # only for training / tests
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
| `chessnet/eval_nnue.py` | Stage 2 evaluation: the trained network, numpy inference on CPU. |
| `chessnet/nnue/prepare.py` | Lichess eval database → training shards. |
| `chessnet/nnue/train.py` | PyTorch training (standalone file, runs on Kaggle). |
| `chessnet/nnue/score_val.py` | Loss of any evaluator on held-out Stockfish evals. |
| `kaggle/train_nnue.ipynb` | Kaggle notebook that runs `train.py`. |

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

## Stage 2: learned evaluation

Everything except the evaluation is shared with stage 1: `--eval nnue` swaps in
`chessnet.eval_nnue`, which loads `models/nnue.npz` (or `$CHESSNET_NNUE`).

```bash
.venv/bin/python -m chessnet.play --eval nnue
.venv/bin/python -m chessnet.match --eval nnue --games 100 --sf-elo 1800
```

### Network

A perspective network, as in NNUE: 768 piece-square inputs per side
(own/opponent × piece type × square, mirrored for Black) → a shared 256-wide
accumulator per side → `[side to move, other side]` → clipped ReLU → 32 → 32 → 1.
The output is centipawns / 400. Training minimises the MSE between
`sigmoid(pred)` and `sigmoid(stockfish_cp / 400)`, so a 50-centipawn miss in a
level position costs more than a 500-centipawn miss in a won one.

Inference recomputes both accumulators on every call (about 30 µs vs 8.5 µs
for PeSTO) and caches results by position. Updating accumulators incrementally
would be faster, but it needs hooks inside the search, and that would break
the "swap the evaluator, nothing else" boundary.

### 1. Prepare data (local)

```bash
.venv/bin/python -m chessnet.nnue.prepare --out data/lichess --positions 30000000
```

This streams `lichess_db_eval.jsonl.zst` (22 GB, CC0) from database.lichess.org
through `curl | zstd -dc`. It stops after 30M kept positions, so the full file
is never stored. For each position it uses the deepest evaluation. It skips
positions that are in check, whose best move is a capture or promotion, or
that are illegal analysis-board setups. Scores are clamped to ±2000 cp, with
mates mapped to ±2000. Every 100th position goes to `val.npz`. `meta.json`
records skip counts and per-shard statistics. `zstd` comes from `PATH` or from
`tools/zstd` (Ubuntu's package unpacked locally).

### 2. Train (Kaggle GPU)

1. `cp chessnet/nnue/train.py data/lichess/`. Then create a Kaggle Dataset from
   `data/lichess/` (Datasets → New Dataset → upload the folder).
2. Import `kaggle/train_nnue.ipynb` (Code → New Notebook → File → Import),
   set Accelerator to GPU, and add the dataset as input.
3. Save Version → Save & Run All.

The script saves `checkpoint.pt` atomically after every epoch and every 15
minutes. `--max-minutes 660` makes it checkpoint and exit before Kaggle's
12-hour limit, so the run's output is always saved. To continue a stopped run,
add that version's output as an input and set `PREV` in the notebook to its
`checkpoint.pt`. Training resumes at the exact batch where it stopped: the
batch order is derived from the seed and epoch number. Each epoch exports
`nnue-epochNNN.npz`, `nnue-latest.npz` and `nnue-best.npz` (lowest validation
loss).

### 3. Use it (local CPU)

Download `nnue-best.npz` into `models/nnue.npz`. Then compare the two evaluations
on held-out positions and in games:

```bash
.venv/bin/python -m chessnet.nnue.score_val data/lichess/val.npz --eval pesto
.venv/bin/python -m chessnet.nnue.score_val data/lichess/val.npz --eval nnue
.venv/bin/python -m chessnet.match --eval nnue --games 100 --sf-elo 1800
```
