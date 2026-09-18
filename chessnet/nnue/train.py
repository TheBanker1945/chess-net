"""Train the NNUE-style evaluation network (PyTorch). Built to run on Kaggle.

Standalone on purpose: needs only torch + numpy and imports nothing from
chessnet, so this single file can be uploaded to a Kaggle notebook.

Architecture (perspective network, as in NNUE):
    768 piece-square features per side -> 256 accumulator (shared weights)
    [acc(side to move), acc(other side)] -> clipped ReLU -> 32 -> 32 -> 1
The output is in units of 400 centipawns, trained with MSE in "win
probability" space: loss = (sigmoid(out) - sigmoid(target_cp / 400))^2.

Checkpointing: <out>/checkpoint.pt is written atomically after every epoch
(and every --checkpoint-minutes within an epoch). It holds model, optimiser,
scheduler, epoch, position inside the epoch, and the best validation loss;
batch order is a pure function of (seed, epoch), so resuming continues at the
exact batch where the previous run stopped. --max-minutes stops cleanly
(checkpoint first) before a hosted session's time limit.

Every epoch also exports <out>/nnue-epochNNN.npz, <out>/nnue-latest.npz and
<out>/nnue-best.npz (lowest validation loss): plain numpy weights that
chessnet.eval_nnue loads for CPU inference without torch.

Usage:
    python train.py --data DATA_DIR --out OUT_DIR --epochs 30
    python train.py --data DATA_DIR --out OUT_DIR --resume-from /kaggle/input/prev/checkpoint.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NUM_FEATURES = 768
PAD = 768
HIDDEN = 256
L2 = 32
L3 = 32
SCALE = 400.0  # centipawns per output unit
ARCH = f"768x2->{HIDDEN}x2->{L2}->{L3}->1 crelu"


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, HIDDEN, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.l1 = nn.Linear(2 * HIDDEN, L2)
        self.l2 = nn.Linear(L2, L3)
        self.out = nn.Linear(L3, 1)
        with torch.no_grad():
            # ~30 active features: keep accumulator sums inside the clipped
            # ReLU's useful range at initialisation.
            self.ft.weight.normal_(0.0, 0.05)
            self.ft.weight[PAD].zero_()

    def forward(self, white_idx: torch.Tensor, black_idx: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        acc_w = self.ft(white_idx) + self.ft_bias
        acc_b = self.ft(black_idx) + self.ft_bias
        white_to_move = stm.unsqueeze(1).bool()
        us = torch.where(white_to_move, acc_w, acc_b)
        them = torch.where(white_to_move, acc_b, acc_w)
        x = torch.clamp(torch.cat([us, them], dim=1), 0.0, 1.0)
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return self.out(x).squeeze(1)


def flip_features(idx: torch.Tensor) -> torch.Tensor:
    """White-perspective feature indices -> Black-perspective (padding kept)."""
    relation = idx // 384
    rest = idx % 384
    flipped = (1 - relation) * 384 + (rest ^ 56)
    return torch.where(idx == PAD, idx, flipped)


def load_split(paths: list[str], device: torch.device):
    feats, stm, score = [], [], []
    for path in paths:
        with np.load(path) as z:
            feats.append(torch.from_numpy(z["feats"]))
            stm.append(torch.from_numpy(z["stm"]))
            score.append(torch.from_numpy(z["score"]))
    return torch.cat(feats).to(device), torch.cat(stm).to(device), torch.cat(score).to(device)


def batch_loss(model: NNUE, feats, stm, score) -> torch.Tensor:
    white_idx = feats.long()
    black_idx = flip_features(white_idx)
    pred = model(white_idx, black_idx, stm)
    target = torch.sigmoid(score.float() / SCALE)
    return F.mse_loss(torch.sigmoid(pred), target)


@torch.no_grad()
def evaluate_loss(model: NNUE, data, batch_size: int) -> float:
    model.eval()
    feats, stm, score = data
    total, n = 0.0, len(score)
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        total += batch_loss(model, feats[sl], stm[sl], score[sl]).item() * len(score[sl])
    model.train()
    return total / n


def export_npz(model: NNUE, path: str, extra: dict) -> None:
    """Write float32 weights for numpy inference (chessnet/eval_nnue.py)."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    tmp = path + ".tmp.npz"
    np.savez(
        tmp,
        ft_w=sd["ft.weight"],  # [769, HIDDEN], row 768 is padding (zeros)
        ft_b=sd["ft_bias"],
        l1_w=sd["l1.weight"],  # [L2, 2*HIDDEN]
        l1_b=sd["l1.bias"],
        l2_w=sd["l2.weight"],
        l2_b=sd["l2.bias"],
        out_w=sd["out.weight"],
        out_b=sd["out.bias"],
        scale=np.float32(SCALE),
        meta=np.array(json.dumps({"arch": ARCH, **extra})),
    )
    os.replace(tmp, path)


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a kill mid-write never corrupts the checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the chess-net NNUE evaluation.")
    parser.add_argument("--data", required=True, help="directory with train-*.npz and val.npz")
    parser.add_argument("--out", required=True, help="directory for checkpoints and exported nets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-positions", type=int, default=0, help="0 = use all")
    parser.add_argument("--resume-from", help="checkpoint to resume from if <out>/checkpoint.pt does not exist")
    parser.add_argument("--checkpoint-minutes", type=float, default=15.0)
    parser.add_argument("--max-minutes", type=float, default=0.0, help="stop cleanly after this long (0 = no limit)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    started = time.time()

    train_paths = sorted(glob.glob(os.path.join(args.data, "**", "train-*.npz"), recursive=True))
    val_paths = sorted(glob.glob(os.path.join(args.data, "**", "val.npz"), recursive=True))
    if not train_paths or not val_paths:
        raise SystemExit(f"no train-*.npz / val.npz under {args.data}")
    train = load_split(train_paths, device)
    if args.max_train_positions:
        train = tuple(t[: args.max_train_positions] for t in train)
    val = load_split(val_paths, device)
    n_train = len(train[2])
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    print(f"device {device}  train {n_train:,}  val {len(val[2]):,}  steps/epoch {steps_per_epoch}", flush=True)

    model = NNUE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=args.min_lr)

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    resume = ckpt_path if os.path.exists(ckpt_path) else args.resume_from
    epoch, step_in_epoch, best_val, history = 0, 0, math.inf, []
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        epoch, step_in_epoch = state["epoch"], state["step_in_epoch"]
        best_val, history = state["best_val"], state["history"]
        if state["n_train"] != n_train or state["batch_size"] != args.batch_size:
            print("warning: dataset size or batch size changed since the checkpoint; batch order will differ")
        print(f"resumed from {resume}: epoch {epoch}, step {step_in_epoch}, best val {best_val:.6f}", flush=True)

    def state_dict() -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "best_val": best_val,
            "history": history,
            "n_train": n_train,
            "batch_size": args.batch_size,
            "args": vars(args),
        }

    last_ckpt = time.time()
    while epoch < args.epochs:
        gen = torch.Generator().manual_seed(args.seed * 100_003 + epoch)
        perm = torch.randperm(n_train, generator=gen).to(device)
        running, count, t0 = 0.0, 0, time.time()
        while step_in_epoch < steps_per_epoch:
            idx = perm[step_in_epoch * args.batch_size : (step_in_epoch + 1) * args.batch_size]
            loss = batch_loss(model, train[0][idx], train[1][idx], train[2][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step_in_epoch += 1
            running += loss.item()
            count += 1
            if step_in_epoch % 200 == 0:
                rate = count * args.batch_size / (time.time() - t0)
                print(
                    f"epoch {epoch + 1} step {step_in_epoch}/{steps_per_epoch} loss {running / count:.6f} "
                    f"lr {sched.get_last_lr()[0]:.2e} ({rate:,.0f} pos/s)",
                    flush=True,
                )
            out_of_time = args.max_minutes and time.time() - started > args.max_minutes * 60
            if out_of_time or time.time() - last_ckpt > args.checkpoint_minutes * 60:
                save_checkpoint(ckpt_path, state_dict())
                last_ckpt = time.time()
                print(f"checkpoint saved (epoch {epoch + 1}, step {step_in_epoch})", flush=True)
                if out_of_time:
                    print("time budget reached; stopping. Re-run with the same --out (or --resume-from) to continue.")
                    return

        val_loss = evaluate_loss(model, val, args.batch_size * 4)
        train_loss = running / max(count, 1)
        epoch += 1
        step_in_epoch = 0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        info = {"epoch": epoch, "val_loss": val_loss, "train_positions": n_train}
        export_npz(model, os.path.join(args.out, f"nnue-epoch{epoch:03d}.npz"), info)
        export_npz(model, os.path.join(args.out, "nnue-latest.npz"), info)
        if val_loss < best_val:
            best_val = val_loss
            export_npz(model, os.path.join(args.out, "nnue-best.npz"), info)
        save_checkpoint(ckpt_path, state_dict())
        last_ckpt = time.time()
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(
            f"== epoch {epoch}/{args.epochs}  train {train_loss:.6f}  val {val_loss:.6f}  "
            f"best {best_val:.6f}  ({time.time() - t0:.0f}s) checkpoint saved",
            flush=True,
        )


if __name__ == "__main__":
    main()
