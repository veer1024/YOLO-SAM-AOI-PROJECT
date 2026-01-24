import json
import os
import cv2

DATASET_JSON = "feedback/feedback_dataset.json"
CLEAN_JSON = "feedback/feedback_dataset_clean.json"

with open(DATASET_JSON) as f:
    data = json.load(f)

clean = []

for s in data:
    if os.path.exists(s["image"]) and os.path.exists(s["mask"]):
        img = cv2.imread(s["image"])
        msk = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if img is not None and msk is not None:
            clean.append(s)
    else:
        print("❌ Dropped:", s["image"])

print(f"Kept {len(clean)} / {len(data)} samples")

with open(CLEAN_JSON, "w") as f:
    json.dump(clean, f, indent=2)
