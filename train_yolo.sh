#!/bin/bash
set -e
python3 auto_split_yolo_val.py
# Train
yolo detect train \
  model=yolo_to_train/best.pt \
  data=ml/feedback_yolo/data.yaml \
  epochs=80 imgsz=640 batch=4 lr0=0.001 \
  patience=15 close_mosaic=10 freeze=10 device=0

# Find latest run dir (train*, train2, train16 etc.)
LATEST_RUN=$(ls -dt runs/detect/train* | head -n 1)
echo "Latest run: $LATEST_RUN"

# Ensure best.pt exists
BEST="$LATEST_RUN/weights/best.pt"
if [ ! -f "$BEST" ]; then
  echo "❌ best.pt not found at: $BEST"
  echo "Contents of $LATEST_RUN/weights:"
  ls -lah "$LATEST_RUN/weights" || true
  exit 1
fi

# Copy best.pt to current_yolo
mkdir -p current_yolo
cp -f "$BEST" current_yolo/best.pt

echo "✅ Updated current_yolo/best.pt from $LATEST_RUN"
