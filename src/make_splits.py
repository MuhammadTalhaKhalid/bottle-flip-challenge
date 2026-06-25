"""
make_splits.py — stratified train/val/test split for the bottle-flip dataset.

Reads data/labels.csv and writes data/splits.csv with an added `split` column.
Split is stratified on the binary label so each split keeps the ~44% success rate.
Issue-tagged clips (body_contact, bottle_left_frame, ...) are kept in their natural
proportion across splits so the test set reflects real-world edge cases.

Run:
    python src/make_splits.py
"""
import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "data", "labels.csv")
VIDEOS = os.path.join(ROOT, "data", "videos")
OUT = os.path.join(ROOT, "data", "splits.csv")

# binary mapping
LABEL2ID = {"failed_landing": 0, "successful_landing": 1}

VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 1337


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    with open(LABELS, newline="") as f:
        for r in csv.DictReader(f):
            fn = r["filename"]
            if not os.path.exists(os.path.join(VIDEOS, fn)):
                continue  # skip any clip whose video is missing
            rows.append({
                "filename": fn,
                "label": r["label"],
                "y": LABEL2ID[r["label"]],
                "issue_tag": r.get("issue_tag", "") or "",
                "duration_seconds": r.get("duration_seconds", ""),
            })

    # stratify by label so success/fail ratio is preserved in each split
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["y"]].append(r)

    for y, group in by_label.items():
        idx = rng.permutation(len(group))
        n = len(group)
        n_test = int(round(n * TEST_FRAC))
        n_val = int(round(n * VAL_FRAC))
        for rank, i in enumerate(idx):
            if rank < n_test:
                group[i]["split"] = "test"
            elif rank < n_test + n_val:
                group[i]["split"] = "val"
            else:
                group[i]["split"] = "train"

    rows.sort(key=lambda r: r["filename"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "filename", "label", "y", "issue_tag", "duration_seconds", "split"])
        w.writeheader()
        w.writerows(rows)

    # report
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["split"]][r["label"]] += 1
    print(f"Wrote {OUT}  ({len(rows)} clips)")
    for sp in ("train", "val", "test"):
        s = counts[sp]["successful_landing"]
        fl = counts[sp]["failed_landing"]
        tot = s + fl
        print(f"  {sp:5s}: {tot:4d}  success={s:3d}  fail={fl:3d}  "
              f"success_rate={s / max(tot,1):.1%}")


if __name__ == "__main__":
    main()
