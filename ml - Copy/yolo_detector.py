# ml/yolo_detector.py

from ultralytics import YOLO
import numpy as np
import logging
import sys
import cv2
from pathlib import Path
import time


BUILDING_CLASS_ID = 0          # change ONLY if your model differs
MAX_BOX_FRAC_TILE = 0.90   # allow big roofs in a tile
MAX_SIDE_FRAC_TILE = 0.98      # reject boxes spanning >80% of tile side
MIN_BOX_PX = 10                # ignore tiny noise

MODEL_PATH = "runs/detect/train11/weights/best.pt"
model = YOLO(MODEL_PATH)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

file_handler = logging.FileHandler("yolo_detector.log", mode="a")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

# Where to dump debug images (optional)
DEBUG_DIR = Path("ml/tmp/yolo_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _draw_boxes(img_rgb, boxes, out_path, title="YOLO"):
    """Draw boxes on a copy of the image and save."""
    if img_rgb is None or img_rgb.size == 0:
        return
    dbg = img_rgb.copy()
    h, w = dbg.shape[:2]
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            dbg,
            f"{i}:{conf:.2f}",
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        dbg,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))


def detect_building_boxes(img_rgb, conf=0.15, imgsz=640, debug=True, debug_tag="aoi"):
    """
    Runs YOLO on the full image (NO manual resize; let Ultralytics letterbox correctly).
    Returns list of (x1, y1, x2, y2, score) in original image pixels.
    """
    assert img_rgb is not None
    assert img_rgb.dtype == np.uint8
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3

    h, w = img_rgb.shape[:2]
    logger.info(f"YOLO_DETECTOR[{debug_tag}]: img shape={img_rgb.shape}, dtype={img_rgb.dtype}")
    logger.info(f"YOLO_DETECTOR[{debug_tag}]: pixel min/max={int(img_rgb.min())}/{int(img_rgb.max())}")

    t0 = time.time()
    res = model(img_rgb, conf=conf, iou=0.4, imgsz=imgsz, verbose=False)[0]
    dt = (time.time() - t0) * 1000.0
    logger.info(f"YOLO_DETECTOR[{debug_tag}]: inference_ms={dt:.1f}, conf={conf}, imgsz={imgsz}")

    boxes = []
    if res.boxes is None or len(res.boxes) == 0:
        logger.info(f"YOLO_DETECTOR[{debug_tag}]: no boxes")
        return boxes

    for b in res.boxes:
        score = float(b.conf.item())
        if score < conf:
            continue

        # if you truly have a single class, you can ignore cls. Otherwise log it.
        cls_id = int(b.cls.item()) if b.cls is not None else -1
        x1, y1, x2, y2 = b.xyxy[0].tolist()

        # clamp
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        logger.info(
            f"YOLO_DETECTOR[{debug_tag}]: box cls={cls_id} "
            f"x1={x1},y1={y1},x2={x2},y2={y2}, w={bw},h={bh}, area={area}, conf={score:.3f}"
        )
        boxes.append((x1, y1, x2, y2, score))

    logger.info(f"YOLO_DETECTOR[{debug_tag}]: final boxes={len(boxes)}")

    if debug:
        out = DEBUG_DIR / f"yolo_{debug_tag}_{int(time.time()*1000)}.png"
        _draw_boxes(img_rgb, boxes, out, title=f"YOLO {debug_tag} ({len(boxes)})")
        logger.info(f"YOLO_DETECTOR[{debug_tag}]: debug image saved: {out}")

    return boxes


def detect_building_boxes_microtiles(
    img_rgb,
    tile_size=256,
    overlap=0.25,
    conf=0.05,
    imgsz=640,
    debug=True,
    debug_tag="tile",
    debug_save_every_tile=False,
):
    """
    Runs YOLO on overlapping tiles to improve small-object recall.
    Returns list of (x1, y1, x2, y2, score) in full-image pixel coords.

    NOTE:
    - We ensure right/bottom edge coverage by adding last tile positions.
    - This does not NMS-merge duplicates; your SAM merge stage can handle it.
      (If you want pre-NMS merging, tell me and I’ll add a lightweight NMS.)
    """
    assert img_rgb is not None
    assert img_rgb.dtype == np.uint8
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3

    h, w = img_rgb.shape[:2]
    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: img shape={img_rgb.shape}, tile_size={tile_size}, overlap={overlap}, conf={conf}, imgsz={imgsz}"
    )

    step = max(1, int(tile_size * (1 - overlap)))

    # Generate tile starts and FORCE last tile to hit right/bottom edges
    xs = list(range(0, max(1, w - tile_size + 1), step))
    ys = list(range(0, max(1, h - tile_size + 1), step))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]
    last_x = max(0, w - tile_size)
    last_y = max(0, h - tile_size)
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)

    logger.info(f"YOLO_MICROTILES[{debug_tag}]: tiles_x={len(xs)}, tiles_y={len(ys)}, step={step}")

    all_boxes = []
    tile_count = 0
    total_infer_ms = 0.0

    for y in ys:
        for x in xs:
            tile_count += 1
            crop = img_rgb[y : y + tile_size, x : x + tile_size]
            if crop.size == 0:
                continue

            t0 = time.time()
            res = model(crop, conf=conf, iou=0.4, imgsz=imgsz, verbose=False)[0]
            dt = (time.time() - t0) * 1000.0
            total_infer_ms += dt

            tile_boxes = 0
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    score = float(b.conf.item())
                    if score < conf:
                        continue
                    cls_id = int(b.cls.item()) if b.cls is not None else -1

                    # If single-class model, you can skip this filter
                    # but keeping it is safe.
                    if cls_id != BUILDING_CLASS_ID:
                        continue

                    x1l, y1l, x2l, y2l = b.xyxy[0].tolist()

                    # local tile coords
                    bw = x2l - x1l
                    bh = y2l - y1l
                    tile_area = tile_size * tile_size

                    # ---- HARD GEOMETRIC FILTERS (VERY IMPORTANT) ----
                    if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                        continue

                    frac = (bw * bh) / tile_area
                    if frac > 0.85 and score < 0.98:
                        logger.warning(
                            f"YOLO_MICROTILES[{debug_tag}]: REJECT big box lowconf "
                            f"frac={frac:.2f} score={score:.2f} at tile ({x},{y})"
                        )
                        continue

                    # if (bw / tile_size) > MAX_SIDE_FRAC_TILE or (bh / tile_size) > MAX_SIDE_FRAC_TILE:
                    #     continue

                    # convert to full-image coords
                    x1 = int(np.clip(x1l + x, 0, w - 1))
                    y1 = int(np.clip(y1l + y, 0, h - 1))
                    x2 = int(np.clip(x2l + x, 0, w - 1))
                    y2 = int(np.clip(y2l + y, 0, h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    all_boxes.append((x1, y1, x2, y2, score))
                    tile_boxes += 1

            logger.info(
                f"YOLO_MICROTILES[{debug_tag}]: tile#{tile_count} origin=({x},{y}) "
                f"infer_ms={dt:.1f} boxes={tile_boxes}"
            )

            if debug and debug_save_every_tile and tile_boxes > 0:
                out = DEBUG_DIR / f"yolo_tile_{debug_tag}_{x}_{y}_{int(time.time()*1000)}.png"
                _draw_boxes(
                    crop,
                    [(bx - x, by - y, bxx - x, byy - y, sc) for (bx, by, bxx, byy, sc) in all_boxes[-tile_boxes:]],
                    out,
                    title=f"tile ({x},{y}) {tile_boxes}",
                )
                logger.info(f"YOLO_MICROTILES[{debug_tag}]: tile debug saved: {out}")

    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: total tiles={tile_count}, total_infer_ms={total_infer_ms:.1f}, total_boxes={len(all_boxes)}"
    )

    if debug:
        out = DEBUG_DIR / f"yolo_microtiles_{debug_tag}_{int(time.time()*1000)}.png"
        _draw_boxes(img_rgb, all_boxes, out, title=f"YOLO microtiles {debug_tag} ({len(all_boxes)})")
        logger.info(f"YOLO_MICROTILES[{debug_tag}]: debug image saved: {out}")

    # -----------------------------
    # GLOBAL NMS ACROSS ALL TILES
    # -----------------------------
    def _iou(a, b):
        ax1, ay1, ax2, ay2, _ = a
        bx1, by1, bx2, by2, _ = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


    # sort by confidence
    all_boxes = sorted(all_boxes, key=lambda b: b[4], reverse=True)

    nms_boxes = []
    for box in all_boxes:
        keep = True
        for kept in nms_boxes:
            if _iou(box, kept) > 0.6:
                keep = False
                break
        if keep:
            nms_boxes.append(box)

    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: NMS reduced boxes "
        f"{len(all_boxes)} -> {len(nms_boxes)}"
    )

    all_boxes = nms_boxes

    return all_boxes
