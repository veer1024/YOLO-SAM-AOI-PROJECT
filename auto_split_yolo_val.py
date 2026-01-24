#!/usr/bin/env python3
"""
auto_split_yolo_val.py

Auto-create a YOLO val split by moving image/label pairs from train -> val,
then delete Ultralytics *.cache so the next training run rebuilds them.

- Positives = label file has at least one line
- Negatives (not_a_building) = empty label file (0 bytes or only whitespace)

Default behavior:
- Target val fraction ~15% (clamped to sensible min/max)
- Tries to include both positives and negatives in val
- Moves only from train to val (does NOT copy). Re-running is safe-ish:
  it will top-up val if too small; it won't move back automatically.

Folder layout expected:
ml/feedback_yolo/
  images/train/*.png
  labels/train/*.txt
  images/val/*.png
  labels/val/*.txt
  labels/train.cache (optional)
  labels/val.cache   (optional)
"""

from __future__ import annotations
import os
import glob
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------
# Config (edit if needed)
# ----------------------------
BASE = "ml/feedback_yolo"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")

DEFAULT_VAL_FRAC = 0.15  # 15%
MIN_VAL = 5              # at least 5 if possible
MAX_VAL_FRAC = 0.25      # don't exceed 25% unless dataset is tiny
SEED = 1337

DRY_RUN = False  # set True to preview without moving


@dataclass
class Sample:
    stem: str
    img_path: str
    lbl_path: str
    is_pos: bool


def _read_label_is_positive(lbl_path: str) -> bool:
    # empty or whitespace-only => negative
    try:
        txt = open(lbl_path, "r", encoding="utf-8").read().strip()
    except FileNotFoundError:
        # In your pipeline labels should exist; if missing, treat as bad.
        return False
    return len(txt) > 0


def _collect_split(base: str, split: str) -> Dict[str, Sample]:
    img_dir = os.path.join(base, "images", split)
    lbl_dir = os.path.join(base, "labels", split)

    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    out: Dict[str, Sample] = {}

    for img_path in sorted(imgs):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if not os.path.exists(lbl_path):
            # Skip unpaired; user can fix separately
            continue
        is_pos = _read_label_is_positive(lbl_path)
        out[stem] = Sample(stem=stem, img_path=img_path, lbl_path=lbl_path, is_pos=is_pos)
    return out


def _ensure_dirs(base: str) -> None:
    for split in ("train", "val"):
        os.makedirs(os.path.join(base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base, "labels", split), exist_ok=True)


def _delete_ultralytics_caches(base: str) -> None:
    # Ultralytics uses labels/train.cache and labels/val.cache
    lbl_dir = os.path.join(base, "labels")
    caches = glob.glob(os.path.join(lbl_dir, "*.cache"))
    for c in caches:
        if DRY_RUN:
            print("[dry-run] would delete cache:", c)
        else:
            try:
                os.remove(c)
                print("deleted cache:", c)
            except FileNotFoundError:
                pass


def _move_sample(sample: Sample, base: str, src_split: str, dst_split: str) -> None:
    src_img = sample.img_path
    src_lbl = sample.lbl_path

    dst_img_dir = os.path.join(base, "images", dst_split)
    dst_lbl_dir = os.path.join(base, "labels", dst_split)

    dst_img = os.path.join(dst_img_dir, os.path.basename(src_img))
    dst_lbl = os.path.join(dst_lbl_dir, os.path.basename(src_lbl))

    if DRY_RUN:
        print(f"[dry-run] move {src_img} -> {dst_img}")
        print(f"[dry-run] move {src_lbl} -> {dst_lbl}")
        return

    shutil.move(src_img, dst_img)
    shutil.move(src_lbl, dst_lbl)


def main():
    random.seed(SEED)
    _ensure_dirs(BASE)

    train = _collect_split(BASE, "train")
    val = _collect_split(BASE, "val")

    n_train = len(train)
    n_val = len(val)
    n_total = n_train + n_val

    if n_total == 0:
        print("No samples found. Check your folder paths.")
        return

    # Compute target val size dynamically
    # For small datasets, keep val smaller but non-zero; for large, cap fraction
    frac = DEFAULT_VAL_FRAC
    frac = min(frac, MAX_VAL_FRAC)
    target_val = int(round(n_total * frac))

    # Minimum val if possible
    if n_total >= MIN_VAL:
        target_val = max(target_val, MIN_VAL)

    # Don't make val bigger than train
    target_val = min(target_val, max(1, n_total - 1))

    # If val already big enough, just rebuild caches
    if n_val >= target_val:
        print(f"val already has {n_val} samples (target {target_val}). No moves needed.")
        _delete_ultralytics_caches(BASE)
        print("Done.")
        return

    need = target_val - n_val
    print(f"Total samples: {n_total} (train={n_train}, val={n_val})")
    print(f"Target val: {target_val} (~{int(frac*100)}%). Need to move: {need} from train -> val")

    # Split train into pos/neg candidates
    train_pos = [s for s in train.values() if s.is_pos]
    train_neg = [s for s in train.values() if not s.is_pos]

    val_pos = [s for s in val.values() if s.is_pos]
    val_neg = [s for s in val.values() if not s.is_pos]

    # Try to keep both classes represented in val (if available in train)
    moves: List[Sample] = []

    # Guarantee at least 1 positive in val if possible
    if len(val_pos) == 0 and len(train_pos) > 0 and need > 0:
        pick = random.choice(train_pos)
        moves.append(pick)
        train_pos = [s for s in train_pos if s.stem != pick.stem]
        need -= 1

    # Guarantee at least 1 negative in val if possible
    if len(val_neg) == 0 and len(train_neg) > 0 and need > 0:
        pick = random.choice(train_neg)
        moves.append(pick)
        train_neg = [s for s in train_neg if s.stem != pick.stem]
        need -= 1

    # Fill remaining moves to roughly match dataset pos/neg ratio
    remaining = train_pos + train_neg
    random.shuffle(remaining)

    # If we still need, just take random from remaining
    if need > 0:
        moves.extend(remaining[:need])

    # De-duplicate moves (just in case)
    uniq: Dict[str, Sample] = {}
    for m in moves:
        uniq[m.stem] = m
    moves = list(uniq.values())

    # Final: don't move more than available
    moves = moves[: (target_val - n_val)]

    # Report what will be moved
    mp = sum(1 for m in moves if m.is_pos)
    mn = len(moves) - mp
    print(f"Moving {len(moves)} samples: positives={mp}, negatives={mn}")

    for m in moves:
        _move_sample(m, BASE, "train", "val")

    _delete_ultralytics_caches(BASE)

    # Recount after moves
    train2 = _collect_split(BASE, "train")
    val2 = _collect_split(BASE, "val")
    print(f"After split: train={len(train2)}, val={len(val2)}")
    vp = sum(1 for s in val2.values() if s.is_pos)
    vn = sum(1 for s in val2.values() if not s.is_pos)
    print(f"Val composition: positives={vp}, negatives={vn}")
    print("Done. Now run your YOLO training; Ultralytics will rebuild val.cache automatically.")


if __name__ == "__main__":
    main()
