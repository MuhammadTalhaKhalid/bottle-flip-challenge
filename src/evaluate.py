"""
evaluate.py — full evaluation of runs/best.pt on the held-out TEST split.

Reports accuracy, precision/recall/F1 per class, macro-F1, ROC-AUC, the confusion
matrix, optimal-threshold analysis, and a per-issue-tag breakdown (how the model
does on the hard edge cases: body_contact, bottle_left_frame, ...).

Writes runs/test_report.txt and runs/test_predictions.csv.

    python src/evaluate.py
"""
import csv
import os

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from torch.utils.data import DataLoader

from dataset import FlipClips
from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
NAMES = ["failed_landing", "successful_landing"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FlipClips("test", augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = FlipClassifier().to(device)
    ckpt = torch.load(os.path.join(RUNS, "best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, probs, files, tags = [], [], [], []
    with torch.no_grad():
        for x, y, fn, tag in dl:
            x = x.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logit = model(x)
            p = torch.softmax(logit.float(), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
            files.extend(fn)
            tags.extend(tag)
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(int)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out(f"Checkpoint: best.pt  (trained to epoch {ckpt['epoch']}, "
        f"val_f1={ckpt['val_f1']:.4f})")
    out(f"Test clips: {len(y)}  (success={int(y.sum())}, fail={int((1-y).sum())})")
    out("")
    out("=== Threshold = 0.50 ===")
    out(classification_report(y, pred, target_names=NAMES, digits=4))
    cm = confusion_matrix(y, pred)
    out("Confusion matrix (rows=true, cols=pred) [fail, success]:")
    out(f"        pred_fail  pred_succ")
    out(f"  fail   {cm[0,0]:7d}   {cm[0,1]:8d}")
    out(f"  succ   {cm[1,0]:7d}   {cm[1,1]:8d}")
    out("")
    auc = roc_auc_score(y, p)
    out(f"ROC-AUC: {auc:.4f}")
    acc = (pred == y).mean()
    out(f"Accuracy @0.50: {acc:.4f}")

    # best-threshold sweep (maximize accuracy)
    best_t, best_a = 0.5, acc
    for t in np.linspace(0.2, 0.8, 61):
        a = ((p >= t).astype(int) == y).mean()
        if a > best_a:
            best_a, best_t = a, t
    out(f"Best accuracy {best_a:.4f} at threshold {best_t:.3f}")
    out("")

    # per-issue-tag breakdown
    out("=== Per-issue-tag accuracy (edge cases) ===")
    tags = np.array(tags)
    for tag in sorted(set(tags)):
        m = tags == tag
        name = tag if tag else "(clean / no tag)"
        a = (pred[m] == y[m]).mean()
        out(f"  {name:20s} n={m.sum():3d}  acc={a:.3f}")

    with open(os.path.join(RUNS, "test_report.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(RUNS, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true", "pred", "prob_success", "issue_tag", "correct"])
        for fn, yt, pr, pp, tg in zip(files, y, pred, p, tags):
            w.writerow([fn, NAMES[yt], NAMES[pr], round(float(pp), 4), tg,
                        int(yt == pr)])
    print(f"\nWrote runs/test_report.txt and runs/test_predictions.csv")


if __name__ == "__main__":
    main()
