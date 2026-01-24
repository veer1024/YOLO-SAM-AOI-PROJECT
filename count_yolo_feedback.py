#!/usr/bin/env python3
import os
import glob

BASE = "ml/feedback_yolo/labels"
SPLITS = ["train", "val"]

def count_split(split: str):
    lbl_dir = os.path.join(BASE, split)
    lbls = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

    pos = 0   # missing_building (has at least 1 valid YOLO line)
    neg = 0   # not_a_building (empty file)
    bad = 0   # non-empty but not valid format

    for p in lbls:
        txt = open(p, "r", encoding="utf-8", errors="ignore").read().strip()

        if txt == "":
            neg += 1
            continue

        ok_any = False
        for line in txt.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                ok_any = True

        if ok_any:
            pos += 1
        else:
            bad += 1

    return len(lbls), pos, neg, bad

def main():
    total_lbls = total_pos = total_neg = total_bad = 0

    for s in SPLITS:
        n, pos, neg, bad = count_split(s)
        total_lbls += n
        total_pos += pos
        total_neg += neg
        total_bad += bad
        print(f"{s}: labels={n}  missing_building(pos)={pos}  not_a_building(neg)={neg}  bad={bad}")

    print("\nTOTAL:")
    print(f"labels={total_lbls}  missing_building(pos)={total_pos}  not_a_building(neg)={total_neg}  bad={total_bad}")

if __name__ == "__main__":
    main()
