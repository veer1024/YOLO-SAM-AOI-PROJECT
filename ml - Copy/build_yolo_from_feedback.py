import os
import cv2
import json
import numpy as np
from pathlib import Path

BASE = Path("ml/feedback")
OUT = Path("ml/feedback_yolo")

IMG_OUT = OUT / "images"
LBL_OUT = OUT / "labels"

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

INDEX = []

def mask_to_yolo(mask_path, img_shape):
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None

    h, w = img_shape[:2]

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def process_category(cat, has_labels):
    img_dir = BASE / cat / "images"
    mask_dir = BASE / cat / "masks"

    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        prefix = {
            "missing_building": "mb",
            "wrong_geometry": "wg",
            "not_a_building": "nab"
        }[cat]

        out_name = f"{prefix}_{img_file.stem}.png"
        out_img = IMG_OUT / out_name

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        cv2.imwrite(str(out_img), img)

        record = {
            "image": out_name,
            "source": cat,
            "yolo_label": False
        }

        if has_labels:
            mask_path = mask_dir / img_file.name
            yolo_line = mask_to_yolo(mask_path, img.shape)

            if yolo_line:
                label_path = LBL_OUT / out_name.replace(".png", ".txt")
                with open(label_path, "w") as f:
                    f.write(yolo_line + "\n")

                record["yolo_label"] = True

        INDEX.append(record)

# Process datasets
process_category("missing_building", has_labels=True)
process_category("wrong_geometry", has_labels=True)
process_category("not_a_building", has_labels=False)

with open(OUT / "index.json", "w") as f:
    json.dump(INDEX, f, indent=2)

print(f"YOLO dataset built: {len(INDEX)} images")
