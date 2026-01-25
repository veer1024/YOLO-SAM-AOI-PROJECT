# ml/yolo_detector.py

from ultralytics import YOLO
import numpy as np
import logging
import sys
import cv2
from pathlib import Path
import time


# -----------------------------
# Tunables / guards
# -----------------------------

# -----------------------------
# Tunables / guards
# -----------------------------
MIN_ASPECT = 0.10
MAX_ASPECT = 10.0

BUILDING_CLASS_ID = 0

# "Mega" thresholds for dropping near-full-tile garbage boxes






MEGA_AREA_THR = 0.80
MEGA_SIDE_THR = 0.95


MEGA_STRIP_SIDE_THR = 0.97     # almost full width OR height
#MEGA_STRIP_OTHER_MIN = 0.55    # must still be thick enough
MEGA_STRIP_OTHER_MIN = 0.80   # was 0.55
# Stricter thresholds ONLY for the full-image probe
PROBE_MEGA_AREA_THR = 0.85
PROBE_MEGA_SIDE_THR = 0.98

# Box geometry filters
MIN_BOX_PX = 14                # ignore tiny noise

MIN_THICK_PX = 6   # kills roads & thin shadows
#MODEL_PATH = "runs/detect/train12/weights/best.pt"
MODEL_PATH = "current_yolo/best.pt"
model = YOLO(MODEL_PATH)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

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


def _round_to(x, base=32):
    x = int(x)
    return max(base, int(base * round(x / base)))


def _clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))


def _box_wh(b):
    x1, y1, x2, y2, _ = b
    return max(0, int(x2) - int(x1)), max(0, int(y2) - int(y1))





def _overlap_1d(a1, a2, b1, b2):
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    return inter

def _cluster_fuse_fragments(
    boxes,
    W,
    H,
    minor_overlap_thr=0.55,
    center_align_thr=0.35,
    gap_factor=0.45,
    max_cluster=40,
    min_cluster=3,
):
    """
    Fuse roof-fragment boxes into long sheds.

    boxes: [(x1,y1,x2,y2,conf), ...] in full-image coords.

    Works best for long warehouses: many adjacent boxes with strong overlap on the short axis.
    """
    if not boxes or len(boxes) < 2:
        return boxes

    # Separate by orientation (horizontal vs vertical-ish)
    horiz = []
    vert = []
    for (x1, y1, x2, y2, s) in boxes:
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        if bw >= bh:
            horiz.append((x1, y1, x2, y2, s, bw, bh))
        else:
            vert.append((x1, y1, x2, y2, s, bw, bh))

    def fuse_group(group, orientation):
        if len(group) < 2:
            return [(g[0], g[1], g[2], g[3], g[4]) for g in group]

        # Typical "minor" size sets gap allowance scale
        #minors = np.array([g[7] if orientation == "h" else g[6] for g in group], dtype=np.float32)
        minors = np.array([g[6] if orientation == "h" else g[5] for g in group], dtype=np.float32)
        minor_med = float(np.median(minors))
        # Allow some gap along major axis (shed roof seams or tile splits)
        gap_thr = max(8.0, gap_factor * minor_med)

        # Union-find
        n = len(group)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Precompute centers/sizes
        cx = np.array([(g[0] + g[2]) * 0.5 for g in group], dtype=np.float32)
        cy = np.array([(g[1] + g[3]) * 0.5 for g in group], dtype=np.float32)

        # Brute connect (n small after NMS); if it ever grows, we can grid-index.
        for i in range(n):
            x1i, y1i, x2i, y2i, si, bwi, bhi = group[i]
            for j in range(i + 1, n):
                x1j, y1j, x2j, y2j, sj, bwj, bhj = group[j]

                # Basic sanity: don't fuse if one is mega vs the other
                # (your mega filter should already have removed those)
                if _is_mega_box_xyxy(x1i, y1i, x2i, y2i, W, H) or _is_mega_box_xyxy(x1j, y1j, x2j, y2j, W, H):
                    continue

                if orientation == "h":
                    # Strong overlap on Y (short axis)
                    oy = _overlap_1d(y1i, y2i, y1j, y2j)
                    min_h = max(1.0, min(bhi, bhj))
                    if (oy / min_h) < minor_overlap_thr:
                        continue

                    # Centers aligned in Y
                    if abs(float(cy[i] - cy[j])) > (center_align_thr * min_h):
                        continue

                    # Along X (major axis): allow overlap OR small gap
                    gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))  # positive if separated
                    if gap > gap_thr:
                        continue

                    if gap > 0.0 and gap > 0.40 * min_h:
                        continue

                    # Similar heights (avoid fusing road strips etc.)
                    if abs(bhi - bhj) > 0.60 * min_h:
                        continue

                else:  # vertical
                    ox = _overlap_1d(x1i, x2i, x1j, x2j)
                    min_w = max(1.0, min(bwi, bwj))
                    if (ox / min_w) < minor_overlap_thr:
                        continue

                    if abs(float(cx[i] - cx[j])) > (center_align_thr * min_w):
                        continue

                    gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))
                    if gap > gap_thr:
                        continue

                    if gap > 0.0 and gap > 0.40 * min_w:
                        continue

                    if abs(bwi - bwj) > 0.60 * min_w:
                        continue

                # Passed all: connect
                union(i, j)

        # Collect clusters
        clusters = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(i)

        fused = []
        for ids in clusters.values():
            # if len(ids) == 1:
            #     g = group[ids[0]]
            #     fused.append((int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[4])))
            #     continue

            if len(ids) < min_cluster:
                for k in ids:
                    g = group[k]
                    fused.append((int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[4])))
                continue

            # Safety: don't let clusters explode
            if len(ids) > max_cluster:
                # keep as-is (no fusion) if suspiciously huge
                for k in ids:
                    g = group[k]
                    fused.append((int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[4])))
                continue

            x1 = float(min(group[k][0] for k in ids))
            y1 = float(min(group[k][1] for k in ids))
            x2 = float(max(group[k][2] for k in ids))
            y2 = float(max(group[k][3] for k in ids))
            best_s = float(max(group[k][4] for k in ids))

            # Slight confidence bump for fused clusters (helps ranking)
            # (small +0.02 per extra fragment, capped)
            best_s = float(min(0.999, best_s + 0.02 * (len(ids) - 1)))

            fused.append((int(x1), int(y1), int(x2), int(y2), best_s))

        return fused

    fused = fuse_group(horiz, "h") + fuse_group(vert, "v")

    # Optional: final dedupe by IoU (light)
    fused.sort(key=lambda b: b[4], reverse=True)
    out = []
    for b in fused:
        keep = True
        for k in out:
            # if very similar, keep the higher-conf one
            ax1, ay1, ax2, ay2, _ = b
            bx1, by1, bx2, by2, _ = k
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            iou = inter / (area_a + area_b - inter)
            if iou >= 0.70:
                keep = False
                break
        if keep:
            out.append(b)

    return out

def _touches_borders(x1, y1, x2, y2, W, H, pad=2.0):
    return (x1 <= pad) or (y1 <= pad) or (x2 >= (W - pad)) or (y2 >= (H - pad))

def _is_mega_box_xyxy(x1, y1, x2, y2, tile_w, tile_h,
                      area_thr=MEGA_AREA_THR, side_thr=MEGA_SIDE_THR):
    bw = max(0.0, float(x2) - float(x1))
    bh = max(0.0, float(y2) - float(y1))
    if tile_w <= 0 or tile_h <= 0:
        return False

    bw_r = bw / float(tile_w)
    bh_r = bh / float(tile_h)
    area_r = (bw * bh) / float(tile_w * tile_h)

    # 1) Truly screen-filling by area (strong signal of "tile-as-building")
    if area_r >= area_thr:
        return True

    # 2) Truly full tile in BOTH dims
    if (bw_r >= side_thr) and (bh_r >= side_thr):
        return True

    # 3) Strip-mega: ONLY if it also touches borders (usually junk edges),
    #    and is extremely close to full tile in one dimension.
    #    This avoids killing long warehouses that are wide but not "screen-filling".
    if _touches_borders(x1, y1, x2, y2, tile_w, tile_h, pad=2.0):
        if (bw_r >= 0.985 and bh_r >= 0.90):
            return True
        if (bh_r >= 0.985 and bw_r >= 0.90):
            return True

    return False


def _is_mega_box(b, tile_w, tile_h,
                 area_thr=MEGA_AREA_THR, side_thr=MEGA_SIDE_THR):
    x1, y1, x2, y2, _ = b
    return _is_mega_box_xyxy(x1, y1, x2, y2, tile_w, tile_h, area_thr, side_thr)


def estimate_building_size_px_from_boxes(boxes, tile_w=None, tile_h=None, fallback=160):
    """
    boxes: [(x1,y1,x2,y2,conf), ...]
    returns robust building size estimate in pixels (max side length).
    IMPORTANT: ignores mega boxes (near full tile), because they poison auto-tiling.
    """
    if not boxes:
        return int(fallback)

    sizes = []
    for x1, y1, x2, y2, c in boxes:
        bw = max(0, int(x2) - int(x1))
        bh = max(0, int(y2) - int(y1))
        s = max(bw, bh)
        if s < MIN_BOX_PX:
            continue

        # ignore mega boxes if we know tile dims
        if tile_w is not None and tile_h is not None:
            if _is_mega_box_xyxy(x1, y1, x2, y2, tile_w, tile_h):
                continue

        sizes.append(s)

    if not sizes:
        return int(fallback)

    sizes = np.array(sizes, dtype=np.float32)
    # 70th percentile biases to "bigger typical" so tiles capture full roof
    return int(np.percentile(sizes, 70))
def _tile_imgsz(tile_size: int) -> int:
    # Ultralytics prefers multiples of 32; keep within sane bounds
    return int(np.clip(_round_to(tile_size, 32), 128, 640))


def _is_squareish(wr, hr):
    # treat near-square mega boxes as junk (common failure under upscaling)
    ar = (wr / max(1e-6, hr)) if hr > 0 else 999.0
    ar = max(ar, 1.0 / max(1e-6, ar))
    return (0.6 <= ar <= 1.7)


def _reject_tile_mega_as_junk(x1, y1, x2, y2, tile_w, tile_h,
                             area_thr=0.60, side_thr=0.88):
    """
    Returns True if box is "mega-ish" AND near-square (junk).
    Keeps strip-like mega boxes (good for long sheds).
    """
    bw = max(0.0, float(x2) - float(x1))
    bh = max(0.0, float(y2) - float(y1))
    if tile_w <= 0 or tile_h <= 0:
        return False

    wr = bw / float(tile_w)
    hr = bh / float(tile_h)
    area_r = (bw * bh) / float(tile_w * tile_h)

    if area_r >= area_thr and (wr >= side_thr or hr >= side_thr) and _is_squareish(wr, hr):
        return True
    return False



def _drop_mega(boxes, W, H):
    if not boxes:
        return boxes
    normal = [b for b in boxes if not _is_mega_box_xyxy(b[0], b[1], b[2], b[3], W, H)]
    return normal if normal else boxes  # <-- IMPORTANT fallback


# def _drop_mega_soft(boxes, W, H):
#     out = []
#     for b in boxes:
#         x1,y1,x2,y2,s = b
#         if _is_mega_box_xyxy(x1,y1,x2,y2,W,H):
#             if s < 0.50:   # only drop weak mega boxes
#                 continue
#         out.append(b)
#     return out if out else boxes

def _drop_mega_soft(boxes, W, H):
    out = []
    for b in boxes:
        x1,y1,x2,y2,s = b
        if _is_mega_box_xyxy(x1,y1,x2,y2,W,H):
            # always drop mega boxes (they are bad prompts), unless absolutely nothing else exists
            continue
        out.append(b)
    return out if out else boxes


# def detect_building_boxes_ensemble(img_rgb, debug=False, debug_tag="ens"):
#     H, W = img_rgb.shape[:2]

#     runs = []
#     r = detect_building_boxes_microtiles(img_rgb, tile_size=None, overlap=None,
#                                          conf=0.14, debug=debug, debug_tag=f"{debug_tag}_auto")
#     #runs.append(_drop_mega(r, W, H))
#     runs.append(_drop_mega_soft(r, W, H))

#     r = detect_building_boxes_microtiles(img_rgb, tile_size=256, overlap=0.60,
#                                          conf=0.14, debug=debug, debug_tag=f"{debug_tag}_256")
#     #runs.append(_drop_mega(r, W, H))
#     runs.append(_drop_mega_soft(r, W, H))

#     r = detect_building_boxes_microtiles(img_rgb, tile_size=416, overlap=0.55,
#                                          conf=0.16, debug=debug, debug_tag=f"{debug_tag}_416")
#     #runs.append(_drop_mega(r, W, H))
#     runs.append(_drop_mega_soft(r, W, H))

#     r = detect_building_boxes(img_rgb, conf=0.18, imgsz=640, debug=debug, debug_tag=f"{debug_tag}_full")
#     #runs.append(_drop_mega(r, W, H))
#     runs.append(_drop_mega_soft(r, W, H))

#     # final_boxes = merge_boxes_ensemble(
#     #     runs, iou_thr=0.45, min_votes=2, conf_strong=0.35, wbf=True
#     # )
#     final_boxes = merge_boxes_ensemble(
#         runs, iou_thr=0.50, min_votes=2, conf_strong=0.22, wbf=True,image_wh=(W, H)
#     )

#     if len(final_boxes) < 6:  # tune this threshold for your AOI size
#         r = detect_building_boxes_microtiles(img_rgb, tile_size=224, overlap=0.80, conf=0.10,
#                                              debug=debug, debug_tag=f"{debug_tag}_recall224")
#         r = _drop_mega_soft(r, W, H)
#         # merge AGAIN but allow singleton rescue to work
#         final_boxes = merge_boxes_ensemble(
#             runs + [r], iou_thr=0.50, min_votes=2, conf_strong=0.22, wbf=True, image_wh=(W, H)
#         )
#     return final_boxes


def detect_building_boxes_ensemble(img_rgb, debug=False, debug_tag="ens"):
    """
    High-recall YOLO ensemble for satellite roofs.
    Designed for ~0.5 m/px imagery and container-like buildings.
    """
    #LARGE_AOI_AREA_PX = 1200 * 1200
    LARGE_AOI_AREA_PX = 450 * 450

    H, W = img_rgb.shape[:2]
    img_area = H * W
    logging.info(f"[YOLO DBG] img_hw=({H},{W}) area={H*W} LARGE_AOI_AREA_PX={LARGE_AOI_AREA_PX}")

    runs = []

    def _dbg_count(tag, boxes):
        logging.info(f"[YOLO DBG] {tag}: {0 if boxes is None else len(boxes)} boxes")

    # =========================================================
    # BASELINE RUNS (always executed)
    # =========================================================

    # 1) Auto microtiles (good general recall)
    r = detect_building_boxes_microtiles(
        img_rgb,
        tile_size=None,
        overlap=None,
        conf=0.14,
        imgsz=640,
        debug=debug,
        debug_tag=f"{debug_tag}_auto",
    )
    _dbg_count("auto_raw", r)
    r = _drop_mega_soft(r, W, H)
    _dbg_count("auto_after_drop", r)
    runs.append(r)

    # 2) 256 tiles (sweet spot for container roofs)
    r = detect_building_boxes_microtiles(
        img_rgb,
        tile_size=256,
        overlap=0.60,
        conf=0.14,
        imgsz=640,
        debug=debug,
        debug_tag=f"{debug_tag}_256",
    )
    _dbg_count("256_raw", r)
    r = _drop_mega_soft(r, W, H)
    _dbg_count("256_after_drop", r)
    runs.append(r)

    # 3) 416 tiles (captures elongated roofs)
    r = detect_building_boxes_microtiles(
        img_rgb,
        tile_size=416,
        overlap=0.55,
        conf=0.16,
        imgsz=640,
        debug=debug,
        debug_tag=f"{debug_tag}_416",
    )
    _dbg_count("416_raw", r)
    r = _drop_mega_soft(r, W, H)
    _dbg_count("416_after_drop", r)
    runs.append(r)

    # 4) Full-frame YOLO (sometimes finds isolated roofs)
    r = detect_building_boxes(
        img_rgb,
        conf=0.18,
        imgsz=640,
        debug=debug,
        debug_tag=f"{debug_tag}_full",
    )
    _dbg_count("full_raw", r)
    r = _drop_mega_soft(r, W, H)
    _dbg_count("full_after_drop", r)
    runs.append(r)

    # =========================================================
    # EXTRA RECALL RUNS (ONLY for large AOIs)
    # =========================================================

    #LARGE_AOI_AREA_PX = 700 * 700
    if img_area >= LARGE_AOI_AREA_PX:

        # 5) Small tiles + heavy overlap (recovers missed seams)
        r = detect_building_boxes_microtiles(
            img_rgb,
            tile_size=224,
            overlap=0.80,
            conf=0.10,
            imgsz=640,
            debug=debug,
            debug_tag=f"{debug_tag}_224_hiov",
        )
        _dbg_count("224_raw", r)
        r = _drop_mega_soft(r, W, H)
        _dbg_count("224_after_drop", r)
        runs.append(r)

        # 6) Medium tiles, high overlap (balanced recall)
        r = detect_building_boxes_microtiles(
            img_rgb,
            tile_size=320,
            overlap=0.70,
            conf=0.12,
            imgsz=640,
            debug=debug,
            debug_tag=f"{debug_tag}_320_hiov",
        )
        _dbg_count("320_raw", r)
        r = _drop_mega_soft(r, W, H)
        _dbg_count("320_after_drop", r)
        runs.append(r)

        # 7) Higher imgsz for small roofs (expensive but effective)
        r = detect_building_boxes_microtiles(
            img_rgb,
            tile_size=256,
            overlap=0.65,
            conf=0.12,
            imgsz=768,
            debug=debug,
            debug_tag=f"{debug_tag}_256_imgsz768",
        )
        _dbg_count("256_768_raw", r)
        r = _drop_mega_soft(r, W, H)
        _dbg_count("256_768_after_drop", r)
        runs.append(r)

        # 8) Full-frame, low confidence (singleton recovery)
        r = detect_building_boxes(
            img_rgb,
            conf=0.12,
            imgsz=768,
            debug=debug,
            debug_tag=f"{debug_tag}_full_low",
        )
        _dbg_count("full_low_raw", r)
        r = _drop_mega_soft(r, W, H)
        _dbg_count("full_low_after_drop", r)
        runs.append(r)

    # =========================================================
    # MERGE
    # =========================================================

    final_boxes = merge_boxes_ensemble(
        runs,
        iou_thr=0.45,        # slightly looser → preserves close roofs
        min_votes=2,         # you already verified this is safe
        conf_strong=0.20,    # gentle confidence floor
        wbf=True,
        image_wh=(W, H),
    )

    _dbg_count("final_merged", final_boxes)

    # =========================================================
    # LAST-RESORT RESCUE (ONLY if recall is terrible)
    # =========================================================

    if len(final_boxes) < 6:
        logging.warning("[YOLO] Low recall detected → rescue pass")

        r = detect_building_boxes_microtiles(
            img_rgb,
            tile_size=192,
            overlap=0.85,
            conf=0.08,
            imgsz=640,
            debug=debug,
            debug_tag=f"{debug_tag}_rescue192",
        )
        _dbg_count("rescue_raw", r)
        r = _drop_mega_soft(r, W, H)
        _dbg_count("rescue_after_drop", r)

        final_boxes = merge_boxes_ensemble(
            runs + [r],
            iou_thr=0.45,
            min_votes=2,
            conf_strong=0.20,
            wbf=True,
            image_wh=(W, H),
        )

        _dbg_count("final_after_rescue", final_boxes)

    return final_boxes


SINGLETON_MIN_AREA_FRAC = 0.0010  # 0.4% of image area
SINGLETON_MAX_AREA_FRAC = 0.60    # reject mega-ish singletons


def merge_boxes_ensemble(
    all_runs,
    iou_thr=0.45,
    min_votes=2,
    conf_strong=0.35,
    wbf=True,
    max_cluster_size=2000,
    image_wh=None,
):
    """
    Merge YOLO box outputs from multiple runs into a single set of boxes.

    Inputs
    ------
    all_runs: list[list[tuple]]
        Each run is a list of (x1,y1,x2,y2,score) in same pixel coords.
    iou_thr: float
        IoU threshold to cluster boxes as the "same object".
    min_votes: int
        Keep a cluster if it is supported by at least this many runs.
    conf_strong: float
        Keep a cluster even with < min_votes if its best score >= conf_strong.
    wbf: bool
        If True, use Weighted Box Fusion (confidence-weighted avg coords).
        If False, just pick the best-score box in the cluster.

    Returns
    -------
    merged: list[tuple]
        List of (x1,y1,x2,y2,score) merged.
        score is max score in the cluster (not the average), so it’s comparable.
    """

    # -----------------------------
    # Flatten  tag with run_id
    # -----------------------------
    items = []
    for run_id, boxes in enumerate(all_runs):
        if not boxes:
            continue
        for (x1, y1, x2, y2, s) in boxes:
            if x2 <= x1 or y2 <= y1:
                continue
            
            # if image_wh is not None:
            #     W, H = image_wh
            #     bw = x2 - x1
            #     bh = y2 - y1
            #     pre = int(np.clip(0.04 * min(bw, bh), 4, 24))  # small pre-grow
            #     x1 = max(0.0, x1 - pre)
            #     y1 = max(0.0, y1 - pre)
            #     x2 = min(float(W - 1), x2 + pre)
            #     y2 = min(float(H - 1), y2 + pre)

            if image_wh is not None:
                W, H = image_wh
                # DO NOT pre-grow here; it causes nearby buildings to overlap and cluster together
                x1 = max(0.0, float(x1))
                y1 = max(0.0, float(y1))
                x2 = min(float(W - 1), float(x2))
                y2 = min(float(H - 1), float(y2))

            items.append((float(x1), float(y1), float(x2), float(y2), float(s), int(run_id)))

    if not items:
        return []

    # Sort by confidence (seed clusters from strongest boxes)
    items.sort(key=lambda t: t[4], reverse=True)

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # -----------------------------
    # Clustering (greedy, fast)
    # -----------------------------
    clusters = []  # each: dict with members  cached representative box
    for (x1, y1, x2, y2, s, run_id) in items:
        box = (x1, y1, x2, y2)
        placed = False

        # Try to match an existing cluster by IoU against its representative
        # (Representative is the current fused box if wbf else best box)
        for c in clusters:
            if _iou(box, c["rep_box"]) >= iou_thr:
                c["members"].append((x1, y1, x2, y2, s, run_id))
                c["runs"].add(run_id)
                c["best_score"] = max(c["best_score"], s)

                # Update representative if using WBF
                if wbf:
                    # coords = np.array([[m[0], m[1], m[2], m[3]] for m in c["members"]], dtype=np.float32)
                    # weights = np.array([max(1e-6, m[4]) for m in c["members"]], dtype=np.float32)
                    # fused = (coords * weights[:, None]).sum(axis=0) / weights.sum()
                    # c["rep_box"] = (float(fused[0]), float(fused[1]), float(fused[2]), float(fused[3]))

                    coords = np.array([[m[0], m[1], m[2], m[3]] for m in c["members"]], dtype=np.float32)
                    weights = np.array([max(1e-6, m[4]) for m in c["members"]], dtype=np.float32)

                    areas = (coords[:,2] - coords[:,0]) * (coords[:,3] - coords[:,1])
                    areas = np.clip(areas, 1.0, None)
                    a_med = np.median(areas)
                    a_max = float(np.max(areas))

                    # If one member is way larger than typical, avoid WBF inflation
                    if a_max > 2.2 * a_med:
                        # choose a "tight but confident" member: among top-2 confidences, pick smaller area
                        members_sorted = sorted(c["members"], key=lambda m: m[4], reverse=True)[:2]
                        best = min(members_sorted, key=lambda m: (m[2]-m[0])*(m[3]-m[1]))
                        c["rep_box"] = (best[0], best[1], best[2], best[3])
                    else:
                        fused = (coords * weights[:, None]).sum(axis=0) / weights.sum()
                        c["rep_box"] = (float(fused[0]), float(fused[1]), float(fused[2]), float(fused[3]))

                else:
                    # rep is just the best-score member
                    best = max(c["members"], key=lambda m: m[4])
                    c["rep_box"] = (best[0], best[1], best[2], best[3])

                placed = True
                break

        if not placed:
            clusters.append({
                "members": [(x1, y1, x2, y2, s, run_id)],
                "runs": {run_id},
                "best_score": s,
                "rep_box": (x1, y1, x2, y2),
            })

        # Safety guard in case something explodes (shouldn’t)
        if len(clusters) > max_cluster_size:
            break

    # -----------------------------
    # Voting / keep rules  output
    # -----------------------------
    merged = []
    for c in clusters:
        votes = len(c["runs"])
        best_s = float(c["best_score"])
        x1, y1, x2, y2 = c["rep_box"]
        bw = max(0.0, float(x2) - float(x1))
        bh = max(0.0, float(y2) - float(y1))

        keep = False

        # 1) Normal rule: require consensus
        if votes >= min_votes:
            keep = True

        elif best_s >= conf_strong:
            if image_wh is not None:
                W, H = image_wh
                if (not _is_mega_box_xyxy(x1, y1, x2, y2, W, H)) and (not _reject_tile_mega_as_junk(x1, y1, x2, y2, W, H)):
                    keep = True
            else:
                keep = True

        # 2) Singleton rescue (recall restore) with strict gates
        elif votes == 1:
            if image_wh is not None:
                W, H = image_wh
                img_area = max(1.0, float(W) * float(H))
                area_frac = (bw * bh) / img_area

                # strict singleton requirements
                if (
                    best_s >= 0.45 and
                    bw >= MIN_BOX_PX and bh >= MIN_BOX_PX and
                    SINGLETON_MIN_AREA_FRAC <= area_frac <= SINGLETON_MAX_AREA_FRAC and
                    (not _is_mega_box_xyxy(x1, y1, x2, y2, W, H)) and
                    (not _reject_tile_mega_as_junk(x1, y1, x2, y2, W, H))
                ):
                    keep = True
            else:
                # If no image size info, be conservative
                keep = (best_s >= 0.55 and bw >= MIN_BOX_PX and bh >= MIN_BOX_PX)

        if keep:
            x1, y1, x2, y2 = c["rep_box"]

            # sanitize ints, but keep geometry stable
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            if x2 > x1 and y2 > y1:
                # Final "inflate" to help roofs be complete
                if image_wh is not None:
                    W, H = image_wh
                    bw = x2 - x1
                    bh = y2 - y1
                    #grow = int(np.clip(0.06 * min(bw, bh), 6, 40))  # 6% of short side
                    #grow = int(np.clip(0.06 * min(bw, bh), 4, 28))
                    grow = int(np.clip(0.03 * min(bw, bh), 2, 18))
                    x1 = max(0, x1 - grow)
                    y1 = max(0, y1 - grow)
                    x2 = min(W - 1, x2 + grow)
                    y2 = min(H - 1, y2 + grow)

                merged.append((x1, y1, x2, y2, best_s))

    # Sort final merged boxes by confidence
    merged.sort(key=lambda b: b[4], reverse=True)
    return merged


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


def _tile_starts(L, tile, step, offset):
    if L <= tile:
        return [0]
    last = max(0, L - tile)
    starts = list(range(int(offset), last + 1, step))
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    # clamp
    return [int(np.clip(s, 0, last)) for s in starts]

def detect_building_boxes(img_rgb, conf=0.15, imgsz=640, debug=True, debug_tag="aoi"):
    """
    Runs YOLO on the full image (NO manual resize; let Ultralytics letterbox correctly).
    Returns list of (x1, y1, x2, y2, score) in original image pixels.
    """
    assert img_rgb is not None
    assert img_rgb.dtype == np.uint8
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3

    h, w = img_rgb.shape[:2]
    logger.info(f"YOLO_DETECTOR[{debug_tag}]: model={MODEL_PATH}")
    logger.info(f"YOLO model.names = {model.names}")
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

        cls_id = int(b.cls.item()) if b.cls is not None else -1
        # If your model is truly single-class, this is safe anyway.
        if cls_id != BUILDING_CLASS_ID:
            continue

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
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue

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
    tile_size=None,      # auto if None
    overlap=None,        # auto if None
    conf=0.15,
    imgsz=640,
    debug=True,
    debug_tag="tile",
    debug_save_every_tile=False,
    target_tile_factor=1.5,      # your requirement
    overlap_cap=0.80,            # cap to avoid absurd tile explosion
):
    """
    Runs YOLO on overlapping tiles to improve recall.

    KEY FIXES vs previous version:
    1) "Mega boxes" (near-full tile) are detected and DO NOT poison auto-tiling.
    2) If probe sees mega boxes, we FORCE smaller micro-tiles (e.g. 256/320) to recover recall.
    3) Containment suppression WILL NOT drop real boxes just because a mega box exists.
    4) We drop mega boxes before NMS/containment unless they are the ONLY boxes left.
    """

    assert img_rgb is not None
    assert img_rgb.dtype == np.uint8
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3
    #tile_conf = max(0.08, conf * 0.6)
    #tile_conf = max(0.03, conf * 0.4)
    tile_conf = max(0.025, conf * 0.40)

    H, W = img_rgb.shape[:2]

    # ---------------------------------------------------------
    # AUTO TILE SIZE  OVERLAP (robust to mega boxes)
    # ---------------------------------------------------------
    if tile_size is None or overlap is None:
        #probe_conf = max(0.05, conf * 0.5)
        probe_conf = max(0.01, conf * 0.25)
        probe_boxes = detect_building_boxes(
            img_rgb, conf=probe_conf, imgsz=imgsz, debug=False, debug_tag=f"{debug_tag}_probe"
        )

        # detect "mega" predictions on the full image probe
        mega_in_probe = any(
            _is_mega_box_xyxy(b[0], b[1], b[2], b[3], W, H,
                              area_thr=PROBE_MEGA_AREA_THR,
                              side_thr=PROBE_MEGA_SIDE_THR)
            for b in probe_boxes
        )

        if mega_in_probe:
            # Mega predictions poison size estimate. Force smaller tiles.
            # For your ~496px tiles this will produce a 3x3 or 4x4 grid.
            forced = 256 if min(W, H) >= 256 else max(128, _round_to(min(W, H), 32))
            T = forced
            ov = 0.50
            logger.warning(
                f"YOLO_MICROTILES[{debug_tag}]: probe saw MEGA boxes => forcing tile_size={T}, overlap={ov:.2f}"
            )
        else:
            B = estimate_building_size_px_from_boxes(probe_boxes, tile_w=W, tile_h=H, fallback=160)
            T = _round_to(int(target_tile_factor * B), base=32)
            T = int(np.clip(T, 224, 640))

            # overlap >= B/T
            ov = (B / max(1, T))
            #ov = _clamp(ov,+ 0.25, overlap_cap)
            #ov = _clamp(ov + 0.15, 0.45, overlap_cap) 
            #ov = _clamp(ov + 0.15, 0.55, overlap_cap) 
            #ov = _clamp(ov + 0.20, 0.60, overlap_cap)
            ov = _clamp(ov + 0.10, 0.35, overlap_cap)

            logger.info(
                f"YOLO_MICROTILES[{debug_tag}]: auto_tile B~{B}px => tile_size={T}, overlap={ov:.2f}"
            )

        if tile_size is None:
            tile_size = T
        if overlap is None:
            overlap = ov

    # Ensure tile_size isn't bigger than the image (still works, but keep sane)
    tile_size = int(min(tile_size, max(1, W), max(1, H)))

    # logger.info(
    #     f"YOLO_MICROTILES[{debug_tag}]: img shape={img_rgb.shape}, tile_size={tile_size}, overlap={overlap}, conf={conf}, imgsz={imgsz}"
    # )
    # logger.info(
    #     f"YOLO_MICROTILES[{debug_tag}]: img shape={img_rgb.shape}, tile_size={tile_size}, "
    #     f"overlap={overlap}, conf={conf}, tile_imgsz=640"
    # )

    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: img shape={img_rgb.shape}, tile_size={tile_size}, "
        f"overlap={overlap}, conf={conf}, tile_imgsz={_tile_imgsz(tile_size)}"
    )

    step = max(1, int(tile_size * (1 - overlap)))

    # Generate tile starts and FORCE last tile to hit right/bottom edges
    xs = list(range(0, max(1, W - tile_size + 1), step))
    ys = list(range(0, max(1, H - tile_size + 1), step))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]
    last_x = max(0, W - tile_size)
    last_y = max(0, H - tile_size)
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)

    logger.info(f"YOLO_MICROTILES[{debug_tag}]: tiles_x={len(xs)}, tiles_y={len(ys)}, step={step}")

    all_boxes = []
    tile_count = 0
    total_infer_ms = 0.0

    offsets = [(0, 0), (step // 2, step // 2)]
    #offsets = [(0,0), (step//2,0), (0,step//2), (step//2,step//2)]
    for ox, oy in offsets:
        xs = _tile_starts(W, tile_size, step, ox)
        ys = _tile_starts(H, tile_size, step, oy)

        for y in ys:
            for x in xs:
                tile_count += 1
                crop = img_rgb[y: y + tile_size, x: x + tile_size]
                if crop.size == 0:
                    continue

                tile_had_mega = False  # <-- FIX 1: initialize per tile
                tile_mega_candidates = []

                tile_imgsz = _tile_imgsz(tile_size)
                #tile_imgsz = 640


                t0 = time.time()
                res = model(crop, conf=tile_conf, iou=0.4, imgsz=tile_imgsz, verbose=False)[0]
                dt = (time.time() - t0) * 1000.0
                total_infer_ms += dt

                tile_boxes = 0
                if res.boxes is not None and len(res.boxes) > 0:
                    for b in res.boxes:
                        score = float(b.conf.item())
                        if score < tile_conf:
                            continue
                        cls_id = int(b.cls.item()) if b.cls is not None else -1
                        if cls_id != BUILDING_CLASS_ID:
                            continue

                        x1l, y1l, x2l, y2l = b.xyxy[0].tolist()
                        bw = float(x2l) - float(x1l)
                        bh = float(y2l) - float(y1l)

                        aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
                        if aspect > MAX_ASPECT:
                            continue
                        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                            continue
                        if min(bw, bh) < MIN_THICK_PX:
                            continue

                        # ratios in tile-local space
                        bw_r = bw / float(tile_size)
                        bh_r = bh / float(tile_size)

                        # Drop very-thin near-full-width/height strips (roads / yard edges)
                        # Keep only if very confident (so real long buildings can survive)
                        if (bw_r >= 0.95 and bh_r <= 0.18 and score < 0.70):
                            continue
                        if (bh_r >= 0.95 and bw_r <= 0.18 and score < 0.70):
                            continue

                        # Mega on tile: track & skip for now
                        if _is_mega_box_xyxy(x1l, y1l, x2l, y2l, tile_size, tile_size):
                            tile_had_mega = True

                            # Drop only if it's the classic junk mega (near-square)
                            # if _reject_tile_mega_as_junk(x1l, y1l, x2l, y2l, tile_size, tile_size):
                            #     logger.warning(
                            #         f"YOLO_MICROTILES[{debug_tag}]: JUNK tile-mega dropped "
                            #         f"w={bw:.0f} h={bh:.0f} score={score:.3f} at tile ({x},{y})"
                            #     )
                            #     continue

                            # Fix A: keep non-junk mega as candidate (don’t drop it)
                            tile_mega_candidates.append((x1l, y1l, x2l, y2l, score))

                            logger.warning(
                                f"YOLO_MICROTILES[{debug_tag}]: NON-JUNK MEGA tile-box kept-as-candidate "
                                f"w={bw:.0f} h={bh:.0f} score={score:.3f} at tile ({x},{y})"
                            )
                            continue  # we "continue" here because we will add it later (Fix B), to avoid mixing logic

                        # convert to full-image coords
                        x1 = int(np.clip(x1l + x, 0, W - 1))
                        y1 = int(np.clip(y1l + y, 0, H - 1))
                        x2 = int(np.clip(x2l + x, 0, W - 1))
                        y2 = int(np.clip(y2l + y, 0, H - 1))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        bw_pix = max(1, x2 - x1)
                        bh_pix = max(1, y2 - y1)
                        #PAD = max(12, int(0.06 * min(bw_pix, bh_pix)))
                        #PAD = int(np.clip(PAD, 12, 40))
                        min_side = min(bw_pix, bh_pix)
                        max_side = max(bw_pix, bh_pix)

                        PAD = max(
                            16,
                            int(0.12 * min_side),   # a bit more around edges
                            int(0.04 * max_side),    # was 0.02
                        )
                        PAD = int(np.clip(PAD, 16, 96))

                        x1 = int(np.clip(x1 - PAD, 0, W - 1))
                        y1 = int(np.clip(y1 - PAD, 0, H - 1))
                        x2 = int(np.clip(x2 + PAD, 0, W - 1))
                        y2 = int(np.clip(y2 + PAD, 0, H - 1))

                        all_boxes.append((x1, y1, x2, y2, score))
                        tile_boxes += 1

                
                # Fix B (better): if tile produced only mega candidates, retry using 2x2 subtiles
                if tile_boxes == 0 and tile_mega_candidates:
                    sub = max(128, tile_size // 2)
                    sub_imgsz = _tile_imgsz(sub)
                    retry_conf = max(tile_conf, 0.12)  # modest bump

                    retry_added = 0
                    for sy in (0, tile_size - sub) if tile_size > sub else (0,):
                        for sx in (0, tile_size - sub) if tile_size > sub else (0,):
                            sub_crop = crop[sy:sy+sub, sx:sx+sub]
                            if sub_crop.size == 0:
                                continue

                            res3 = model(sub_crop, conf=retry_conf, iou=0.4, imgsz=sub_imgsz, verbose=False)[0]
                            if res3.boxes is None or len(res3.boxes) == 0:
                                continue

                            for b3 in res3.boxes:
                                score3 = float(b3.conf.item())
                                if score3 < retry_conf:
                                    continue
                                cls3 = int(b3.cls.item()) if b3.cls is not None else -1
                                if cls3 != BUILDING_CLASS_ID:
                                    continue

                                x1l3, y1l3, x2l3, y2l3 = b3.xyxy[0].tolist()
                                bw3 = float(x2l3) - float(x1l3)
                                bh3 = float(y2l3) - float(y1l3)
                                if bw3 < MIN_BOX_PX or bh3 < MIN_BOX_PX:
                                    continue
                                if min(bw3, bh3) < MIN_THICK_PX:
                                    continue

                                # IMPORTANT: still drop mega boxes at the subtile level
                                if _is_mega_box_xyxy(x1l3, y1l3, x2l3, y2l3, sub, sub):
                                    #continue
                                    shrink = int(np.clip(0.06 * sub, 6, 18))
                                    x1l3 = x1l3 + shrink
                                    y1l3 = y1l3 + shrink
                                    x2l3 = x2l3 - shrink
                                    y2l3 = y2l3 - shrink
                                    if x2l3 <= x1l3 or y2l3 <= y1l3:
                                        continue

                                # convert to full-image coords (tile origin + subtile origin + local box)
                                x1 = int(np.clip(x + sx + x1l3, 0, W - 1))
                                y1 = int(np.clip(y + sy + y1l3, 0, H - 1))
                                x2 = int(np.clip(x + sx + x2l3, 0, W - 1))
                                y2 = int(np.clip(y + sy + y2l3, 0, H - 1))
                                if x2 <= x1 or y2 <= y1:
                                    continue

                                # same padding logic you already use
                                bw_pix = max(1, x2 - x1)
                                bh_pix = max(1, y2 - y1)
                                min_side = min(bw_pix, bh_pix)
                                max_side = max(bw_pix, bh_pix)
                                PAD = max(16, int(0.12 * min_side), int(0.04 * max_side))
                                PAD = int(np.clip(PAD, 16, 96))
                                x1 = int(np.clip(x1 - PAD, 0, W - 1))
                                y1 = int(np.clip(y1 - PAD, 0, H - 1))
                                x2 = int(np.clip(x2 + PAD, 0, W - 1))
                                y2 = int(np.clip(y2 + PAD, 0, H - 1))

                                all_boxes.append((x1, y1, x2, y2, score3))
                                retry_added += 1

                    if retry_added > 0:
                        tile_boxes += retry_added
                        logger.warning(
                            f"YOLO_MICROTILES[{debug_tag}]: mega-only tile ({x},{y}) retried with 2x2 subtiles "
                            f"sub={sub} imgsz={sub_imgsz} recovered={retry_added}"
                        )

                # #Fix B: mega-only tile fallback
                # if tile_boxes == 0 and tile_mega_candidates:
                #     # take best scoring mega candidate
                #     x1m, y1m, x2m, y2m, sm = max(tile_mega_candidates, key=lambda t: t[4])

                #     # convert to full-image coords
                #     x1 = int(np.clip(x1m + x, 0, W - 1))
                #     y1 = int(np.clip(y1m + y, 0, H - 1))
                #     x2 = int(np.clip(x2m + x, 0, W - 1))
                #     y2 = int(np.clip(y2m + y, 0, H - 1))

                #     if x2 > x1 and y2 > y1:
                #         # apply same padding logic (optional but consistent)
                #         bw_pix = max(1, x2 - x1)
                #         bh_pix = max(1, y2 - y1)
                #         min_side = min(bw_pix, bh_pix)
                #         max_side = max(bw_pix, bh_pix)

                #         PAD = max(
                #             16,
                #             int(0.12 * min_side),
                #             int(0.04 * max_side),
                #         )
                #         PAD = int(np.clip(PAD, 16, 96))

                #         x1 = int(np.clip(x1 - PAD, 0, W - 1))
                #         y1 = int(np.clip(y1 - PAD, 0, H - 1))
                #         x2 = int(np.clip(x2 + PAD, 0, W - 1))
                #         y2 = int(np.clip(y2 + PAD, 0, H - 1))

                #         all_boxes.append((x1, y1, x2, y2, sm))
                #         tile_boxes += 1

                #         logger.warning(
                #             f"YOLO_MICROTILES[{debug_tag}]: mega-only tile fallback added 1 box "
                #             f"score={sm:.3f} at tile ({x},{y})"
                #         )



                # ----------------------------
                # FIX 2: Retry mega-only tiles
                # ----------------------------
                # if tile_boxes == 0 and tile_had_mega:
                #     retry_conf = float(np.clip(max(tile_conf, 0.15), 0.12, 0.35))
                #     t1 = time.time()
                #     res2 = model(crop, conf=retry_conf, iou=0.4, imgsz=tile_imgsz, verbose=False)[0]
                #     dt2 = (time.time() - t1) * 1000.0
                #     total_infer_ms += dt2

                #     retry_added = 0
                #     if res2.boxes is not None and len(res2.boxes) > 0:
                #         for b2 in res2.boxes:
                #             score2 = float(b2.conf.item())
                #             if score2 < retry_conf:
                #                 continue
                #             cls2 = int(b2.cls.item()) if b2.cls is not None else -1
                #             if cls2 != BUILDING_CLASS_ID:
                #                 continue

                #             x1l2, y1l2, x2l2, y2l2 = b2.xyxy[0].tolist()
                #             bw2 = float(x2l2) - float(x1l2)
                #             bh2 = float(y2l2) - float(y1l2)

                #             aspect2 = max(bw2 / max(1.0, bh2), bh2 / max(1.0, bw2))
                #             if aspect2 > MAX_ASPECT:
                #                 continue
                #             if bw2 < MIN_BOX_PX or bh2 < MIN_BOX_PX:
                #                 continue
                #             if min(bw2, bh2) < MIN_THICK_PX:
                #                 continue

                #             # reject junky near-square mega-ish boxes
                #             if _reject_tile_mega_as_junk(x1l2, y1l2, x2l2, y2l2, tile_size, tile_size):
                #                 continue

                #             x1 = int(np.clip(x1l2 + x, 0, W - 1))
                #             y1 = int(np.clip(y1l2 + y, 0, H - 1))
                #             x2 = int(np.clip(x2l2 + x, 0, W - 1))
                #             y2 = int(np.clip(y2l2 + y, 0, H - 1))
                #             if x2 <= x1 or y2 <= y1:
                #                 continue

                #             bw_pix = max(1, x2 - x1)
                #             bh_pix = max(1, y2 - y1)
                #             PAD = max(12, int(0.06 * min(bw_pix, bh_pix)))
                #             PAD = int(np.clip(PAD, 12, 40))
                #             x1 = int(np.clip(x1 - PAD, 0, W - 1))
                #             y1 = int(np.clip(y1 - PAD, 0, H - 1))
                #             x2 = int(np.clip(x2 + PAD, 0, W - 1))
                #             y2 = int(np.clip(y2 + PAD, 0, H - 1))

                #             all_boxes.append((x1, y1, x2, y2, score2))
                #             retry_added += 1

                #     if retry_added > 0:
                #         tile_boxes += retry_added
                #         logger.warning(
                #             f"YOLO_MICROTILES[{debug_tag}]: tile ({x},{y}) RETRY recovered {retry_added} boxes "
                #             f"(retry_conf={retry_conf:.2f}, retry_ms={dt2:.1f})"
                #         )

                logger.info(
                    f"YOLO_MICROTILES[{debug_tag}]: tile#{tile_count} origin=({x},{y}) "
                    f"infer_ms={dt:.1f} boxes={tile_boxes}"
                )



    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: total tiles={tile_count}, total_infer_ms={total_infer_ms:.1f}, total_boxes={len(all_boxes)}"
    )
    # if tile_count > 500:
    #     logger.warning("YOLO_MICROTILES: too many tiles, reducing overlap")
    #     overlap = max(0.35, overlap - 0.1)

    if debug:
        out = DEBUG_DIR / f"yolo_microtiles_{debug_tag}_{int(time.time()*1000)}.png"
        _draw_boxes(img_rgb, all_boxes, out, title=f"YOLO microtiles {debug_tag} ({len(all_boxes)})")
        logger.info(f"YOLO_MICROTILES[{debug_tag}]: debug image saved: {out}")

    # -----------------------------
    # GLOBAL NMS  SAFE CONTAINMENT
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
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _containment(inner, outer):
        ix1, iy1, ix2, iy2, _ = inner
        ox1, oy1, ox2, oy2, _ = outer
        inter_x1 = max(ix1, ox1)
        inter_y1 = max(iy1, oy1)
        inter_x2 = min(ix2, ox2)
        inter_y2 = min(iy2, oy2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_inner = max(1, (ix2 - ix1) * (iy2 - iy1))
        return inter / area_inner

    # Sort by confidence
    all_boxes = sorted(all_boxes, key=lambda b: b[4], reverse=True)

    # Drop any mega boxes on the FULL image before NMS/containment.
    # (Just in case something slipped through due to clipping.)
    normal_boxes = [b for b in all_boxes if not _is_mega_box(b, W, H)]
    mega_boxes = [b for b in all_boxes if _is_mega_box(b, W, H)]

    if len(mega_boxes) > 0:
        logger.warning(
            f"YOLO_MICROTILES[{debug_tag}]: found {len(mega_boxes)} MEGA boxes on full image; "
            f"they will NOT be used for containment suppression."
        )

    # If we somehow only have mega boxes, keep the best one (last-resort)
    # but this should be rare now because we dropped tile-mega earlier and forced smaller microtiles.
    working = normal_boxes if len(normal_boxes) > 0 else mega_boxes[:1]

    # CONTAIN_THR = 0.90  # if a box is 90% inside a higher-conf box, drop it
    # IOU_THR = 0.60

    CONTAIN_THR = 0.92  # if a box is 90% inside a higher-conf box, drop it
    IOU_THR = 0.70

    nms_boxes = []
    for box in working:
        keep = True
        for kept in nms_boxes:
            if _iou(box, kept) > IOU_THR:
                keep = False
                break

            # SAFE containment:
            # - Do NOT allow mega "containers" to delete other boxes (the exact bug you hit).
            if _is_mega_box(kept, W, H):
                continue

            #if _containment(box, kept) > CONTAIN_THR:
            # if _containment(box, kept) > CONTAIN_THR and kept[4] >= (box[4]  0.08):
            #     keep = False
            #     break

            # Only suppress if the kept box is clearly better AND much larger
            if _containment(box, kept) > CONTAIN_THR:
                kbw, kbh = _box_wh(kept)
                bbw, bbh = _box_wh(box)

                # do NOT kill roof fragments
                if (kbw * kbh) > 1.6 * (bbw * bbh) and kept[4] >= (box[4] + 0.12):
                    keep = False
                    break
        if keep:
            nms_boxes.append(box)

    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: NMS reduced boxes "
        f"{len(working)} -> {len(nms_boxes)}"
    )

    all_boxes = nms_boxes
    logger.info(
        f"YOLO_MICROTILES[{debug_tag}]: FINAL buildings after NMS+containment = {len(all_boxes)}"
    )


    before = len(all_boxes)

    img_area = float(W * H)
    frag = []
    keep = []
    for b in all_boxes:
        x1, y1, x2, y2, s = b
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if (area / img_area) <= 0.18:
            frag.append(b)
        else:
            keep.append(b)

    before_frag = len(frag)
    # frag = _cluster_fuse_fragments(
    #     frag,
    #     W=W,
    #     H=H,
    #     minor_overlap_thr=0.60,
    #     center_align_thr=0.25,
    #     gap_factor=0.35,
    #     min_cluster=2,
    # )
    frag = _cluster_fuse_fragments(
        frag,
        W=W,
        H=H,
        minor_overlap_thr=0.70,   # stricter
        center_align_thr=0.20,    # stricter alignment
        gap_factor=0.25,          # smaller allowed gap
        min_cluster=3,            # ✅ must have 3+ fragments to fuse
    )
    after = len(frag)
    if after != before_frag:
        logger.warning(f"YOLO_MICROTILES[{debug_tag}]: fragment_fusion merged {before_frag} -> {after}")

    all_boxes = keep + frag
    all_boxes.sort(key=lambda b: b[4], reverse=True)



    # all_boxes = _cluster_fuse_fragments(
    #     all_boxes,
    #     W=W,
    #     H=H,
    #     minor_overlap_thr=0.55,
    #     center_align_thr=0.35,
    #     gap_factor=0.35,
    #     min_cluster=2,
    # )
    after = len(all_boxes)
    if after != before:
        logger.warning(
            f"YOLO_MICROTILES[{debug_tag}]: fragment_fusion merged {before} -> {after}"
        )

    if debug:
        out = DEBUG_DIR / f"yolo_fused_{debug_tag}_{int(time.time()*1000)}.png"
        _draw_boxes(img_rgb, all_boxes, out, title=f"YOLO fused {debug_tag} ({len(all_boxes)})")
        logger.info(f"YOLO_MICROTILES[{debug_tag}]: fused debug image saved: {out}")

    return all_boxes
