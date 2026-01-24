# ml/feedback/build_dataset.py

import os
import json
import logging
import cv2
import numpy as np

FEEDBACK_ROOT = "ml/feedback"
OUTPUT_FILE = os.path.join(FEEDBACK_ROOT, "feedback_dataset_clean.json")

logging.basicConfig(level=logging.INFO)

FEEDBACK_TYPES = [
    "missing_building",
    "wrong_geometry",
    "not_a_building",
]

EMPTY_MASK_SIZE = 1024


def build_dataset():
    dataset = []

    for feedback_type in FEEDBACK_TYPES:
        base = os.path.join(FEEDBACK_ROOT, feedback_type)

        img_dir = os.path.join(base, "images")
        mask_dir = os.path.join(base, "masks")
        meta_dir = os.path.join(base, "metadata")

        if not os.path.isdir(meta_dir):
            continue

        for meta_file in os.listdir(meta_dir):
            if not meta_file.endswith(".json"):
                continue

            meta_path = os.path.join(meta_dir, meta_file)
            with open(meta_path) as f:
                meta = json.load(f)

            sample_id = meta["id"]
            image_path = meta["image"]

            if not os.path.exists(image_path):
                continue

            # ---------------- NOT A BUILDING ----------------
            if feedback_type == "not_a_building":
                mask_path = meta.get("mask")

                if not mask_path or not os.path.exists(mask_path):
                    empty = np.zeros((EMPTY_MASK_SIZE, EMPTY_MASK_SIZE), dtype=np.uint8)
                    mask_path = os.path.join(mask_dir, f"{sample_id}.png")
                    cv2.imwrite(mask_path, empty)

                dataset.append({
                    "id": sample_id,
                    "type": feedback_type,
                    "image": image_path,
                    "mask": mask_path,
                    "metadata": meta_path
                })

            # ---------------- WRONG GEOMETRY ----------------
            elif feedback_type == "wrong_geometry":
                correct = meta.get("correct_mask")
                original = meta.get("original_mask")

                if not correct or not original:
                    logging.warning(
                        "Skipping invalid wrong_geometry sample %s", sample_id
                    )
                    continue

                if not (os.path.exists(correct) and os.path.exists(original)):
                    continue

                dataset.append({
                    "id": sample_id,
                    "type": feedback_type,
                    "image": image_path,
                    "correct_mask": correct,
                    "original_mask": original,
                    "metadata": meta_path
                })

            # ---------------- MISSING BUILDING ----------------
            else:
                mask_path = meta.get("mask")
                if not mask_path or not os.path.exists(mask_path):
                    continue

                dataset.append({
                    "id": sample_id,
                    "type": feedback_type,
                    "image": image_path,
                    "mask": mask_path,
                    "metadata": meta_path
                })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    logging.info("Dataset built with %d samples", len(dataset))
    logging.info("Saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()
