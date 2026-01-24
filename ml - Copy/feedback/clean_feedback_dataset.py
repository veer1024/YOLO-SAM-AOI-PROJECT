import json
import os
import cv2

DATASET_JSON = "ml/feedback/feedback_dataset.json"
CLEAN_JSON = "ml/feedback/feedback_dataset_clean.json"

with open(DATASET_JSON) as f:
    data = json.load(f)

clean = []
dropped = 0

for s in data:
    img_path = s.get("image")
    mask_path = s.get("mask")
    sample_type = s.get("type")

    # --------------------------------------------------
    # IMAGE (ALWAYS REQUIRED)
    # --------------------------------------------------
    if not img_path or not os.path.exists(img_path):
        print("❌ Dropped (missing image):", img_path)
        dropped += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Dropped (image unreadable):", img_path)
        dropped += 1
        continue

    # --------------------------------------------------
    # MASK (ALWAYS REQUIRED — EVEN FOR not_a_building)
    # --------------------------------------------------
    if not mask_path or not os.path.exists(mask_path):
        print("❌ Dropped (missing mask):", mask_path)
        dropped += 1
        continue

    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        print("❌ Dropped (mask unreadable):", mask_path)
        dropped += 1
        continue

    clean.append(s)

print(f"✅ Kept {len(clean)} / {len(data)} samples")
print(f"🗑 Dropped {dropped} samples")

with open(CLEAN_JSON, "w") as f:
    json.dump(clean, f, indent=2)

print("💾 Clean dataset saved to:", CLEAN_JSON)
