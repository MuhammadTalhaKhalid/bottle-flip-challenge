"""
train.py — train the bottle-flip landing classifier.

    python src/train.py --epochs 30 --batch 8

Saves best checkpoint (by val macro-F1) to runs/best.pt and a training log to
runs/train_log.csv. Uses mixed precision on GPU.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import FlipClips
from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = (y == p).mean()
    f1 = f1_score(y, p, average="macro")
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(RUNS, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tr = FlipClips("train", augment=True)
    va = FlipClips("val", augment=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # class weights from train distribution (slightly more failed than success)
    ys = np.array([int(r["y"]) for r in tr.rows])
    w = len(ys) / (2 * np.bincount(ys))
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"train={len(tr)} val={len(va)}  class_weights={w.round(3)}")

    model = FlipClassifier().to(device)
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    log_path = os.path.join(RUNS, "train_log.csv")
    log = open(log_path, "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_acc", "val_f1", "lr", "sec"])

    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y, *_ in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        acc, f1 = evaluate(model, vl, device)
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(f"ep {ep:2d}/{args.epochs}  loss={np.mean(losses):.4f}  "
              f"val_acc={acc:.4f}  val_f1={f1:.4f}  lr={lr_now:.2e}  {dt:.0f}s")
        lw.writerow([ep, round(float(np.mean(losses)), 4), round(float(acc), 4),
                     round(float(f1), 4), lr_now, round(dt, 1)])
        log.flush()
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_f1": f1, "val_acc": acc}, os.path.join(RUNS, "best.pt"))
            print(f"   ^ saved best (val_f1={f1:.4f})")
    log.close()
    print(f"\nBest val macro-F1: {best_f1:.4f}  -> runs/best.pt")


if __name__ == "__main__":
    main()
