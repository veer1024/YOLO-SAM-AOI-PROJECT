# ml/detect_buildings_aoi.py
import os
import logging
import sys
import cv2
import numpy as np
import rasterio
import geopandas as gpd
import torch
import math
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import box
from rasterio.warp import transform_bounds
from datetime import datetime, timezone
from astral.sun import elevation, azimuth
from astral import LocationInfo
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator , SamPredictor

from ml.roof_geometry import flat_roof
from ml.glb_export import export_glb
import hashlib
import json
import uuid
from shapely.ops import transform as shp_transform
#from ml.yolo_detector import detect_building_boxes
from rasterio.transform import Affine
#from ml.yolo_detector import detect_building_boxes_microtiles
from shapely.geometry import box as shp_box
from logging.handlers import RotatingFileHandler
from datetime import datetime


from ml.yolo_detector import (
    detect_building_boxes,
    detect_building_boxes_microtiles,
    detect_building_boxes_ensemble,
)



BOOT_LOG = logging.getLogger("ml.detect_buildings_aoi")
BOOT_LOG.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in BOOT_LOG.handlers):
    BOOT_LOG.addHandler(logging.StreamHandler(sys.stdout))



TMP_DIR = "ml/tmp"
os.makedirs(TMP_DIR, exist_ok=True)


# --- Drop mega / strip boxes on tiles ---
MEGA_AREA_THR_TILE   = 0.45   # drop if padded box >45% of tile area
MEGA_SIDE_THR_TILE   = 0.92   # drop if box spans >92% of tile width or height

STRIP_OTHER_MAX_TILE = 0.45   # ...while other side is still thick enough -> it's a strip, drop

# Confidence escape hatch (keep only if extremely confident)
MEGA_KEEP_CONF = 0.92

### MEAGBOX AOI

MEGA_AREA_THR_AOI  = 0.60
MEGA_SIDE_THR_AOI  = 0.95
STRIP_SIDE_THR_AOI = 0.90
STRIP_OTHER_THR_AOI = 0.20

STRIP_SIDE_THR_TILE = 0.92
STRIP_OTHER_THR_TILE = 0.22
STRIP_OTHER_MAX_AOI = 0.55
MEGA_KEEP_CONF_AOI = 0.95

MIN_MASK_AREA_M2 = 7.0  # tune: 5–20 m² depending on smallest buildings you want
# --------------------------------------------------
# TILING CONFIG
# --------------------------------------------------
#TILE_SIZE_M = 256
TILE_SIZE_M = 256        # fewer tiles
#TILE_OVERLAP = 0.10     # enough continuity
TILE_OVERLAP = 0.25
#MIN_TILING_AREA_M2 = 120 * 120  # below this → no tiling
#MIN_TILING_AREA_M2 = 500 * 500
MIN_TILING_AREA_M2 = 300 * 300


BIG_BOX_FRAC_AOI = 0.65         # if YOLO box covers >65% of AOI -> suspicious
BIG_BOX_FRAC_TILE = 0.50        # for tiles (stricter)
BIG_BOX_MIN_CONF = 0.60         # ignore big-box rejection if confidence is low? (optional)
UPSCALE_FOR_FALLBACK = 3        # upscale tiny AOIs before microtiling
MIN_AOI_FOR_DIRECT_YOLO = 256   # if AOI smaller than this, don't trust single-shot YOLO much


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
#BASE_TIF = "data/processed/30cm_test/pneo4_30cm_3857_cog.tif"
BASE_TIF = "data/data/processed/phr1a_20210227_rgb_3857_cog.tif"

AOI_TIF = "ml/aoi.tif"

OUT_DIR = "ml/output"
OUT_GEOJSON = f"{OUT_DIR}/buildings.geojson"
OUT_GLB = f"{OUT_DIR}/aoi_buildings.glb"

DEFAULT_HEIGHT = 6.0
MIN_AREA_PX = 600        # slightly higher for airports
#MAX_AOI_RATIO = 0.08     # reject giant flat regions

MAX_AOI_RATIO = 0.06   # reject giant flat regions

#IMAGE_ACQ_TIME = "2023-08-01T06:48:46.620"
#IMAGE_ACQ_TIME = "2022-11-24T05:59:54.545+00:00"
IMAGE_ACQ_TIME = "2021-02-27T04:56:00.3+00:00"


os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_LOG.info("Using device: %s", device)

# --------------------------------------------------
# LOAD SAM
# --------------------------------------------------
sam = sam_model_registry["vit_b"](checkpoint="ml/checkpoints/sam_vit_b.pth")
sam.to(device=device, dtype=torch.float32)
sam.eval()

#decoder_ckpt = "ml/sam_training/checkpoints/sam_decoder_finetuned.pth.epoch12.pth"
decoder_ckpt="SAM_TRAINED_MODEL/sam_decoder_finetuned.pth.epoch15.pth"
sam.mask_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
BOOT_LOG.info("Fine-tuned SAM decoder loaded")



LOGS_DIR = os.path.join(OUT_DIR, "logs") if "OUT_DIR" in globals() else "ml/output/logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class ContextAdapter(logging.LoggerAdapter):
    """
    Adds run_id/stage/det_id/tile_id automatically to every log line.
    """
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        merged = dict(self.extra)
        merged.update(extra)
        kwargs["extra"] = merged
        return msg, kwargs

    def with_ctx(self, **updates):
        new_extra = dict(self.extra)
        for k, v in updates.items():
            if v is not None:
                new_extra[k] = v
        return ContextAdapter(self.logger, new_extra)


def setup_logging(run_id: str, level=logging.INFO, also_stdout=True):
    """
    Creates a per-run log file:
      ml/output/logs/<run_id>.log

    Returns:
      base_logger (ContextAdapter) with run_id already set.
    """
    logger_name = f"ml.detect_buildings_aoi.{run_id}"
    base_logger = logging.getLogger(logger_name)
    base_logger.setLevel(level)
    base_logger.propagate = False

    # Prevent duplicate handlers if detect_buildings() called multiple times in same process
    if not getattr(base_logger, "_configured", False):
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s | "
            "run=%(run_id)s stage=%(stage)s det=%(det_id)s tile=%(tile_id)s | %(message)s"
        )
        formatter = logging.Formatter(fmt)

        # Per-run file
        logfile = os.path.join(LOGS_DIR, f"{run_id}.log")
        fh = RotatingFileHandler(logfile, maxBytes=25 * 1024 * 1024, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        base_logger.addHandler(fh)

        if also_stdout:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(level)
            sh.setFormatter(formatter)
            base_logger.addHandler(sh)

        base_logger._configured = True  # type: ignore[attr-defined]
        base_logger._logfile = logfile  # type: ignore[attr-defined]

    # default context fields (never missing in format)
    adapter = ContextAdapter(
        base_logger,
        {
            "run_id": run_id,
            "stage": "-",
            "det_id": "-",
            "tile_id": "-",
        },
    )
    return adapter


def get_logger(run_id: str, stage: str = "-", det_id: str = "-", tile_id: str = "-",
               level=logging.INFO, also_stdout=True) -> ContextAdapter:
    """
    Convenience entry: ensures per-run log exists and returns adapter with ctx set.
    """
    base = setup_logging(run_id, level=level, also_stdout=also_stdout)
    return base.with_ctx(stage=stage, det_id=det_id, tile_id=tile_id)


def safe_json_dump(path: str, obj: dict, log: logging.Logger | None = None):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if log is not None:
            log.exception(f"Failed to write json: {path}")
        else:
            raise


predictor = SamPredictor(sam)

mask_generator_full = SamAutomaticMaskGenerator(
    sam,
    points_per_side=8,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.80,
    min_mask_region_area=600
)

mask_generator_crop = SamAutomaticMaskGenerator(
    sam,
    points_per_side=4,
    pred_iou_thresh=0.55,
    stability_score_thresh=0.65,
    min_mask_region_area=120
)



# --------------------------------------------------
# HELPERS
# --------------------------------------------------
SAM_DEBUG_DIR = os.path.join(OUT_DIR, "sam_debug")
os.makedirs(SAM_DEBUG_DIR, exist_ok=True)


def _save_sam_debug(
    log: logging.Logger | None,
    run_id: str,
    stage_tag: str,
    img_rgb_uint8: np.ndarray | None,
    box_xyxy,
    mask_bool,
    score,
    dbg,
    det_id: str | None = None,
    tile_id: str | None = None,
):
    """
    Saves (per det):
      - overlay PNG (mask + box + optional best_prompt_box)
      - mask PNG
      - meta JSON (score, dbg, box, det_id, stage, tile_id)

    Also logs the exact saved paths (critical for later debugging).
    """
    if img_rgb_uint8 is None:
        if log:
            log.warning("SAM_DEBUG skipped: img is None")
        return None

    det_id = det_id or "det-unknown"
    tile_id = tile_id or "-"

    d = os.path.join(SAM_DEBUG_DIR, run_id, stage_tag)
    os.makedirs(d, exist_ok=True)

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    # stable base name includes stage + det_id
    base = f"{stage_tag}_{det_id}_x{x1}_y{y1}_x{x2}_y{y2}_s{float(score):.3f}" if score is not None \
           else f"{stage_tag}_{det_id}_x{x1}_y{y1}_x{x2}_y{y2}_sNA"

    mask_path = os.path.join(d, f"{base}_mask.png")
    overlay_path = os.path.join(d, f"{base}_overlay.png")
    meta_path = os.path.join(d, f"{base}_meta.json")

    # mask png
    if mask_bool is not None:
        m = (mask_bool.astype(np.uint8) * 255)
        cv2.imwrite(mask_path, m)

    # overlay png
    overlay = img_rgb_uint8.copy()
    if mask_bool is not None:
        overlay[mask_bool] = (0.35 * overlay[mask_bool] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)

    # YOLO/sub-box in blue
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # best_prompt_box in red if present
    if isinstance(dbg, dict) and "best_prompt_box" in dbg:
        bx1, by1, bx2, by2 = dbg["best_prompt_box"]
        cv2.rectangle(overlay, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)

    #cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    meta = {
        "run_id": run_id,
        "stage": stage_tag,
        "det_id": det_id,
        "tile_id": tile_id,
        "box": [x1, y1, x2, y2],
        "score": float(score) if score is not None else None,
        "dbg": dbg if isinstance(dbg, dict) else {},
    }
    safe_json_dump(meta_path, meta, log=log)

    if log:
        log.info(
            "SAM_DEBUG saved base=%s overlay=%s mask=%s meta=%s",
            base, overlay_path, mask_path if mask_bool is not None else "-", meta_path
        )

    return {"base": base, "overlay": overlay_path, "mask": (mask_path if mask_bool is not None else None), "meta": meta_path}

def _box_expand_and_shift_xyxy(box, W, H, scale=1.0, dx=0.0, dy=0.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)

    bw2 = bw * scale
    bh2 = bh * scale

    cx2 = cx + dx
    cy2 = cy + dy

    nx1 = cx2 - 0.5 * bw2
    ny1 = cy2 - 0.5 * bh2
    nx2 = cx2 + 0.5 * bw2
    ny2 = cy2 + 0.5 * bh2

    nx1 = int(np.clip(nx1, 0, W - 1))
    ny1 = int(np.clip(ny1, 0, H - 1))
    nx2 = int(np.clip(nx2, 0, W - 1))
    ny2 = int(np.clip(ny2, 0, H - 1))

    # keep valid
    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        return None
    return (nx1, ny1, nx2, ny2)

def _boundary_touch_ratio(mask_bool):
    # fraction of boundary pixels that are "on" (captures chopped masks)
    if mask_bool is None or mask_bool.size == 0:
        return 1.0
    m = mask_bool.astype(np.uint8)
    top = m[0, :].mean()
    bot = m[-1, :].mean()
    left = m[:, 0].mean()
    right = m[:, -1].mean()
    return float(0.25 * (top + bot + left + right))

def _mask_solidity(mask_bool):
    # area / convex hull area, cheap and robust
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 1.0:
        return 0.0
    hull = cv2.convexHull(c)
    harea = float(cv2.contourArea(hull))
    if harea <= 1.0:
        return 0.0
    return float(area / harea)


def predict_mask_multi_prompt(
    predictor,
    yolo_box_xyxy,               # (x1,y1,x2,y2) in full image coords
    yolo_conf=0.5,               # used only for mild bias
    scales=(1.00, 1.15, 1.28, 1.40),
    shift_frac=0.03,             # shift as fraction of min(box_w,box_h)
    max_shift_px=18,
    min_shift_px=6,
    touch_thr=0.010,             # boundary touch threshold
    max_union=2,                 # union top-k masks when they complement
):
    """
    Returns: best_mask_bool (H,W), best_score (float), debug dict

    Key improvements:
      1) Coverage penalty vs ORIGINAL YOLO box -> stops "half roof" shifted winners
      2) Small shift penalty -> prevents large dx/dy prompts from dominating
      3) Union rule relaxed based on gain -> allows completing long roofs
      4) Rescue mode uses same improved scoring and updates dbg at the end
    """
    H, W = predictor.original_size

    # --- robust box handling (tile branch sometimes passes None/invalid) ---
    if yolo_box_xyxy is None:
        return None, -1.0, {"error": "yolo_box_xyxy is None"}
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in yolo_box_xyxy]
    except Exception as e:
        return None, -1.0, {"error": f"bad yolo_box_xyxy: {e}"}

    bw0 = max(2, x2 - x1)
    bh0 = max(2, y2 - y1)

    # d = int(np.clip(shift_frac * min(bw0, bh0), min_shift_px, max_shift_px))
    # shifts = [(0, 0), (d, 0), (-d, 0), (0, d), (0, -d)]

    ar = bw0 / bh0

    d_small = int(np.clip(shift_frac * min(bw0, bh0), min_shift_px, max_shift_px))

    # new: long-axis shift
    d_long = int(np.clip(0.12 * max(bw0, bh0), 18, 70))

    shifts = [(0,0), (d_small,0), (-d_small,0), (0,d_small), (0,-d_small)]

    if ar >= 2.2:
        shifts += [(d_long,0), (-d_long,0), (2*d_long,0), (-2*d_long,0)]
    elif ar <= 1/2.2:
        shifts += [(0,d_long), (0,-d_long), (0,2*d_long), (0,-2*d_long)]

    shift_ref = max(
        max_shift_px,
        int(0.12 * max(bw0, bh0))  # matches d_long upper bound
    )

    # ---------- scoring helpers ----------
    obw = max(1, x2 - x1)
    obh = max(1, y2 - y1)

    def _coverage_penalty(mask_bool):
        # Penalize masks whose bbox span is much smaller than the ORIGINAL YOLO box span.
        # This kills "shifted half-roof" masks.
        yy, xx = np.where(mask_bool)
        if len(xx) < 10:
            return 0.4  # small masks are generally bad; mild penalty
        mx1 = int(xx.min()); mx2 = int(xx.max())
        my1 = int(yy.min()); my2 = int(yy.max())
        mw = max(1, mx2 - mx1)
        mh = max(1, my2 - my1)

        w_ratio = mw / obw
        h_ratio = mh / obh

        cover_pen = 0.0
        # tune: 0.88 works well for long roofs; lower if you have very loose YOLO boxes
        if w_ratio < 0.88:
            cover_pen += 0.55 * (0.88 - w_ratio) / 0.88
        if h_ratio < 0.88:
            cover_pen += 0.35 * (0.88 - h_ratio) / 0.88

        return float(np.clip(cover_pen, 0.0, 0.9))

    

    shift_ref = float(max(1, min(bw0, bh0)))  # normalize by smaller side of the prompt box

    def _shift_penalty(dx, dy):
        shift_norm = (abs(dx) + abs(dy)) / shift_ref
        return 0.08 * float(min(1.0, shift_norm))
    


    def _composite_score(mask_bool, sam_score, touch, solidity, fill, dx, dy):
        # discourage too tiny or too full (often background)
        fill_pen = 0.0
        if fill < 0.10:
            fill_pen = 0.40
        elif fill > 0.95:
            fill_pen = 0.25

        # boundary penalty (strong for chopped roofs)
        touch_w = 4.5 if fill > 0.35 else 2.5
        touch_pen = touch_w * max(0.0, float(touch) - float(touch_thr))

        # solidity reward (cap)
        sol_rew = 0.35 * float(np.clip(solidity, 0.0, 1.0))

        # mild bias from yolo_conf (don’t dominate)
        yolo_rew = 0.10 * float(np.clip(yolo_conf, 0.0, 1.0))

        cover_pen = _coverage_penalty(mask_bool)
        shift_pen = _shift_penalty(dx, dy)

        return float(sam_score) + sol_rew + yolo_rew - touch_pen - fill_pen - cover_pen - shift_pen

    def _union_top_masks(cands, start_mask, base_area, k):
        if k <= 1:
            return start_mask
        union = start_mask.copy()
        used = 1
        for c in cands[1:]:
            if used >= k:
                break
            m = c["mask"]
            inter = float(np.logical_and(union, m).sum())
            uni = float(np.logical_or(union, m).sum())
            iou = inter / max(1.0, uni)

            new_area = float(np.logical_or(union, m).sum())
            gain = (new_area - float(union.sum())) / max(1.0, base_area)

            # Relax union rule: allow if it adds meaningful area even if IoU is moderate
            if (gain > 0.10 and iou < 0.90) or (gain > 0.06 and iou < 0.80):
                union = np.logical_or(union, m)
                used += 1
        return union

    # ---------- main candidate search ----------
    candidates = []
    tried = 0

    for s in scales:
        for (dx, dy) in shifts:
            b = _box_expand_and_shift_xyxy((x1, y1, x2, y2), W, H, scale=s, dx=dx, dy=dy)
            if b is None:
                continue
            tried += 1

            box_in = np.array(b, dtype=np.float32)[None, :]
            masks, scores, _ = predictor.predict(box=box_in, multimask_output=True)

            if masks is None or len(masks) == 0:
                continue

            box_area = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))

            for mi in range(len(masks)):
                m = masks[mi].astype(bool)
                area = float(m.sum())
                if area <= 25:
                    continue

                touch = _boundary_touch_ratio(m)
                solidity = _mask_solidity(m)
                fill = area / box_area

                comp = _composite_score(m, float(scores[mi]), touch, solidity, fill, dx, dy)

                candidates.append({
                    "comp": comp,
                    "sam": float(scores[mi]),
                    "touch": float(touch),
                    "sol": float(solidity),
                    "fill": float(fill),
                    "box": tuple(int(v) for v in b),
                    "dx": int(dx),
                    "dy": int(dy),
                    "scale": float(s),
                    "mask": m,
                })

    if not candidates:
        return None, -1.0, {"tried": tried, "kept": 0}

    candidates.sort(key=lambda c: c["comp"], reverse=True)
    best = candidates[0]

    best_mask = best["mask"]
    best_score = float(best["comp"])

    # union in main mode
    if max_union > 1:
        base_area = float(best_mask.sum())
        best_mask = _union_top_masks(candidates, best_mask, base_area, max_union)

    # ---------------------------
    # LOW-FILL RESCUE RETRY (long roofs)
    # ---------------------------
    # If best mask under-fills its own prompt box, try a wider prompt set.
    if best["fill"] < 0.45:
        rescue_scales = (1.10, 1.25, 1.40, 1.55, 1.70)
        rescue_max_union = max(3, max_union)

        rescue_candidates = []

        for s in rescue_scales:
            for (dx, dy) in shifts:
                b2 = _box_expand_and_shift_xyxy((x1, y1, x2, y2), W, H, scale=s, dx=dx, dy=dy)
                if b2 is None:
                    continue

                box_in2 = np.array(b2, dtype=np.float32)[None, :]
                masks2, scores2, _ = predictor.predict(box=box_in2, multimask_output=True)
                if masks2 is None or len(masks2) == 0:
                    continue

                box_area2 = float(max(1, (b2[2] - b2[0]) * (b2[3] - b2[1])))

                for mi in range(len(masks2)):
                    m2 = masks2[mi].astype(bool)
                    area2 = float(m2.sum())
                    if area2 <= 25:
                        continue

                    touch2 = _boundary_touch_ratio(m2)
                    sol2 = _mask_solidity(m2)
                    fill2 = area2 / box_area2

                    comp2 = _composite_score(m2, float(scores2[mi]), touch2, sol2, fill2, dx, dy)

                    rescue_candidates.append({
                        "comp": comp2,
                        "sam": float(scores2[mi]),
                        "touch": float(touch2),
                        "sol": float(sol2),
                        "fill": float(fill2),
                        "box": tuple(int(v) for v in b2),
                        "dx": int(dx),
                        "dy": int(dy),
                        "scale": float(s),
                        "mask": m2,
                    })

        if rescue_candidates:
            rescue_candidates.sort(key=lambda c: c["comp"], reverse=True)
            rbest = rescue_candidates[0]

            rmask = rbest["mask"]
            if rescue_max_union > 1:
                base_area = float(rmask.sum())
                rmask = _union_top_masks(rescue_candidates, rmask, base_area, rescue_max_union)

            # accept rescue if it meaningfully increases area and isn't more chopped
            gain_ratio = (float(rmask.sum()) - float(best_mask.sum())) / max(1.0, float(best_mask.sum()))
            if gain_ratio > 0.10 and (_boundary_touch_ratio(rmask) <= _boundary_touch_ratio(best_mask) + 0.01):
                best_mask = rmask
                best_score = float(rbest["comp"])
                best = rbest  # so dbg reflects rescue winner
    
    # ---------------------------
    # WIDE-BOX SPLIT RESCUE
    # ---------------------------
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    ar = float(bw) / float(bh)

    #if best_mask is not None and best[4] < 0.60 and ar > 3.0 and bw >= 220:
    best_sam = float(best.get("sam", 1.0))
    if best_mask is not None and best_sam < 0.60 and ar > 3.0 and bw >= 220:
        pad = int(0.08 * bw)  # overlap
        mid = (x1 + x2) // 2

        bL = (x1, y1, min(W, mid + pad), y2)
        bR = (max(0, mid - pad), y1, x2, y2)

        mL, sL, _ = predict_mask_multi_prompt(
            predictor, bL, yolo_conf=yolo_conf,
            scales=scales, shift_frac=shift_frac,
            max_shift_px=max_shift_px, min_shift_px=min_shift_px,
            touch_thr=touch_thr, max_union=max_union
        )
        mR, sR, _ = predict_mask_multi_prompt(
            predictor, bR, yolo_conf=yolo_conf,
            scales=scales, shift_frac=shift_frac,
            max_shift_px=max_shift_px, min_shift_px=min_shift_px,
            touch_thr=touch_thr, max_union=max_union
        )

        if mL is not None and mR is not None:
            mU = np.logical_or(mL, mR)
            # accept if it adds meaningful area and not more chopped
            gain_ratio = (float(mU.sum()) - float(best_mask.sum())) / max(1.0, float(best_mask.sum()))
            if gain_ratio > 0.08 and (_boundary_touch_ratio(mU) <= _boundary_touch_ratio(best_mask) + 0.02):
                best_mask = mU

    

    dbg = {
        "tried": int(tried),
        "kept": int(len(candidates)),
        "best_raw_sam": float(best.get("sam", 0.0)),
        "best_touch": float(best.get("touch", 0.0)),
        "best_solidity": float(best.get("sol", 0.0)),
        "best_fill": float(best.get("fill", 0.0)),
        "best_prompt_box": tuple(int(v) for v in best.get("box", (x1, y1, x2, y2))),
        "best_score": float(best_score),
        "best_dxdy": (int(best.get("dx", 0)), int(best.get("dy", 0))),
        "best_scale": float(best.get("scale", 1.0)),
    }

    return best_mask, best_score, dbg


def _expand_box_xyxy(x1, y1, x2, y2, W, H, scale=1.25):
    """Expand a box around its center by `scale`, clamp to image bounds."""
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1)
    bh = (y2 - y1)
    nbw = bw * scale
    nbh = bh * scale
    nx1 = int(max(0, round(cx - 0.5 * nbw)))
    ny1 = int(max(0, round(cy - 0.5 * nbh)))
    nx2 = int(min(W - 1, round(cx + 0.5 * nbw)))
    ny2 = int(min(H - 1, round(cy + 0.5 * nbh)))
    # keep valid
    if nx2 <= nx1 + 2: nx2 = min(W - 1, nx1 + 3)
    if ny2 <= ny1 + 2: ny2 = min(H - 1, ny1 + 3)
    return nx1, ny1, nx2, ny2


def _pick_best_mask_for_box(masks, scores, box_area, bw, bh, yconf):
    """Apply your existing min_area/max_fill logic and pick best SAM mask."""
    min_area = max(20, int(0.002 * box_area))
    max_fill = adaptive_max_fill(box_area, bw, bh, yconf)

    best = None
    best_score = -1.0

    for m, s in zip(masks, scores):
        area = int(m.sum())
        if area < min_area:
            continue

        fill_ratio = area / max(1, box_area)
        too_full = fill_ratio > max_fill

        # your current policy: allow near-full masks only if very confident
        if too_full and not (yconf >= 0.80 and float(s) >= 0.85 and fill_ratio <= 0.995):
            continue

        s = float(s)
        if s > best_score:
            best = m
            best_score = s

    return best, float(best_score), min_area


def predict_mask_twopass_union(predictor, x1p, y1p, x2p, y2p, yconf, expand_scale=1.25):
    """
    Two SAM passes:
      pass1: original box
      pass2: expanded box
    Return: union_mask_uint8, combined_score, min_area_for_original_box
    Assumes predictor.set_image(...) already called for current image.
    """
    H, W = predictor.original_size  # (H, W) for current image in SAM predictor
    # NOTE: predictor.input_size exists in SAM predictor; if not, use image shape you already have.
    # If this ever fails, replace with H,W from the current img you have in scope.

    bw1 = x2p - x1p
    bh1 = y2p - y1p
    box1_area = max(1, bw1 * bh1)

    # pass 1
    box1 = np.array([x1p, y1p, x2p, y2p], dtype=np.float32)[None, :]
    masks1, scores1, _ = predictor.predict(box=box1, multimask_output=True)
    m1, s1, min_area = _pick_best_mask_for_box(masks1, scores1, box1_area, bw1, bh1, yconf)

    # pass 2 (expanded)
    ex1, ey1, ex2, ey2 = _expand_box_xyxy(x1p, y1p, x2p, y2p, W, H, scale=expand_scale)
    bw2 = ex2 - ex1
    bh2 = ey2 - ey1
    box2_area = max(1, bw2 * bh2)

    box2 = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)[None, :]
    masks2, scores2, _ = predictor.predict(box=box2, multimask_output=True)
    m2, s2, _ = _pick_best_mask_for_box(masks2, scores2, box2_area, bw2, bh2, yconf)

    if m1 is None and m2 is None:
        return None, None, min_area, (ex1, ey1, ex2, ey2)

    if m1 is None:
        union = m2
        score = s2
    elif m2 is None:
        union = m1
        score = s1
    else:
        union = (m1.astype(np.uint8) | m2.astype(np.uint8))
        score = max(s1, s2)  # conservative: keep best SAM score

    return union.astype(np.uint8), float(score), int(min_area), (ex1, ey1, ex2, ey2)

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2,_ = a
    bx1,by1,bx2,by2,_ = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    aa = max(1,(ax2-ax1)*(ay2-ay1))
    ba = max(1,(bx2-bx1)*(by2-by1))
    return inter / (aa + ba - inter + 1e-9), inter, aa, ba

def center_dist(a,b):
    ax1,ay1,ax2,ay2,_ = a
    bx1,by1,bx2,by2,_ = b
    acx, acy = 0.5*(ax1+ax2), 0.5*(ay1+ay2)
    bcx, bcy = 0.5*(bx1+bx2), 0.5*(by1+by2)
    dx, dy = acx-bcx, acy-bcy
    return (dx*dx + dy*dy) ** 0.5

def fuse_boxes_conservative(boxes, img_w, img_h):
    # boxes: [(x1,y1,x2,y2,conf), ...]
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    fused = []

    diag = (img_w*img_w + img_h*img_h) ** 0.5
    CTHR = 0.06 * diag

    for b in boxes:
        merged = False
        for i, f in enumerate(fused):
            iou, inter, aa, ba = iou_xyxy(b, f)
            min_overlap = inter / (min(aa,ba) + 1e-9)
            cd = center_dist(b,f)

            ok = (
                (iou >= 0.55) or
                (min_overlap >= 0.85) or
                (cd <= CTHR and iou >= 0.25)
            )

            if not ok:
                continue

            # propose merged box
            x1 = min(b[0], f[0]); y1 = min(b[1], f[1])
            x2 = max(b[2], f[2]); y2 = max(b[3], f[3])
            conf = max(b[4], f[4])
            merged_box = (x1,y1,x2,y2,conf)

            # reject if merged becomes mega/strip
            if is_mega_or_strip_box(x1,y1,x2,y2, img_w,img_h, conf):
                continue

            fused[i] = merged_box
            merged = True
            break

        if not merged:
            fused.append(b)

    return fused


def is_mega_or_strip_box(x1p,y1p,x2p,y2p, tile_w,tile_h, yconf):
    bw = max(1, x2p-x1p); bh = max(1, y2p-y1p)
    area_frac = (bw*bh) / max(1, tile_w*tile_h)
    w_frac = bw / max(1, tile_w)
    h_frac = bh / max(1, tile_h)

    mega = (area_frac >= MEGA_AREA_THR_TILE) or (w_frac >= MEGA_SIDE_THR_TILE) or (h_frac >= MEGA_SIDE_THR_TILE)

    strip = (w_frac >= STRIP_SIDE_THR_AOI and h_frac <= STRIP_OTHER_THR_AOI) or (
        h_frac >= STRIP_SIDE_THR_AOI and w_frac <= STRIP_OTHER_THR_AOI
    )

    if (mega or strip) and (yconf < MEGA_KEEP_CONF):
        return True
    return False


def looks_like_container_box(x1, y1, x2, y2, W, H):
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw <= 1 or bh <= 1:
        return False

    area_frac = (bw * bh) / float(W * H + 1e-9)
    w_frac = bw / float(W + 1e-9)
    h_frac = bh / float(H + 1e-9)
    aspect = max(bw / float(bh + 1e-9), bh / float(bw + 1e-9))

    # Heuristics tuned for AOI “strip/yard” boxes:
    # - very large area
    # - very wide strip
    # - very tall strip
    # - extreme aspect with moderate area
    if area_frac >= 0.25:
        return True
    if w_frac >= 0.92 and h_frac >= 0.25:
        return True
    if h_frac >= 0.92 and w_frac >= 0.25:
        return True
    if aspect >= 6.0 and area_frac >= 0.12:
        return True

    return False


def drop_container_boxes(boxes, W, H, conf_margin=0.06):
    """
    Removes 'container' boxes that wrap multiple smaller detections.
    IMPORTANT: run this BEFORE NMS so container boxes don't suppress real boxes.
    """
    if not boxes:
        return boxes

    centers = [((x1 + x2) * 0.5, (y1 + y2) * 0.5) for (x1, y1, x2, y2, c) in boxes]
    keep = [True] * len(boxes)

    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        contained_idxs = []
        for j, (cx, cy) in enumerate(centers):
            if j == i:
                continue
            if (cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2):
                contained_idxs.append(j)

        contained = len(contained_idxs)
        if contained < 1:
            continue

        # Strong rule: if it "looks like a container" and it contains others, drop it.
        if looks_like_container_box(x1, y1, x2, y2, W, H):
            keep[i] = False
            logging.warning(
                f"[AOI] Dropped container-like box i={i} conf={c:.3f} "
                f"xyxy={[int(x1),int(y1),int(x2),int(y2)]} contained={contained}"
            )
            continue

        # Soft rule: classic containment suppression
        best_inside = c
        for j in contained_idxs:
            best_inside = max(best_inside, boxes[j][4])

        if c + conf_margin < best_inside:
            keep[i] = False
            logging.warning(
                f"[AOI] Dropped low-conf container box i={i} conf={c:.3f} "
                f"best_inside={best_inside:.3f} xyxy={[int(x1),int(y1),int(x2),int(y2)]} contained={contained}"
            )

    return [b for k, b in zip(keep, boxes) if k]


def is_mega_or_strip_box_aoi(x1p,y1p,x2p,y2p, W,H, yconf):
    bw = max(1, x2p-x1p); bh = max(1, y2p-y1p)
    area_frac = (bw*bh) / max(1, W*H)
    w_frac = bw / max(1, W)
    h_frac = bh / max(1, H)


    aspect = max(bw / bh, bh / bw)

    # --- NEW: container-box heuristic (your failing case) ---
    # If a box covers a large chunk of AOI, it is almost always a "container".
    if area_frac >= 0.40:
        return True
    if (w_frac >= 0.75 and h_frac >= 0.35 and aspect >= 1.6 and area_frac >= 0.25):
        return True

    mega = (area_frac >= MEGA_AREA_THR_AOI) or (w_frac >= MEGA_SIDE_THR_AOI) or (h_frac >= MEGA_SIDE_THR_AOI)

    # strip = ((w_frac >= STRIP_SIDE_THR_AOI and h_frac >= STRIP_OTHER_MAX_AOI) or
    #          (h_frac >= STRIP_SIDE_THR_AOI and w_frac >= STRIP_OTHER_MAX_AOI))

    strip = ((w_frac >= STRIP_SIDE_THR_AOI and h_frac <= STRIP_OTHER_MAX_AOI) or
         (h_frac >= STRIP_SIDE_THR_AOI and w_frac <= STRIP_OTHER_MAX_AOI))

    if (mega or strip) and (yconf < MEGA_KEEP_CONF_AOI):
        return True
    return False


def poly_stats(poly):
    hull = poly.convex_hull
    solidity = float(poly.area / max(hull.area, 1e-6))
    rect = polygon_rectangularity(poly)
    minx, miny, maxx, maxy = poly.bounds
    ww = maxx - minx
    hh = maxy - miny
    aspect = float(max(ww, hh) / max(1e-6, min(ww, hh)))
    return solidity, rect, aspect

def world_box_polygon(transform, x1, y1, x2, y2):
    x1w, y1w = rasterio.transform.xy(transform, y1, x1, offset="center")
    x2w, y2w = rasterio.transform.xy(transform, y2, x2, offset="center")
    minx = min(x1w, x2w); maxx = max(x1w, x2w)
    miny = min(y1w, y2w); maxy = max(y1w, y2w)
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

def clip_gdf_to_aoi_4326(gdf_4326, bounds_4326):
    """
    gdf_4326: GeoDataFrame in EPSG:4326
    bounds_4326: (west,south,east,north)
    - Filters out features completely outside AOI
    - Clips geometries to AOI boundary
    """
    west, south, east, north = bounds_4326
    aoi_poly = shp_box(west, south, east, north)

    # filter quickly
    gdf_4326 = gdf_4326[gdf_4326.geometry.intersects(aoi_poly)].copy()
    if gdf_4326.empty:
        return gdf_4326

    # strict clip
    gdf_4326["geometry"] = gdf_4326.geometry.intersection(aoi_poly)

    # drop empties produced by intersection
    gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty].copy()
    return gdf_4326

def snap_geom_to_raster_grid(geom, raster_transform):
    px = raster_transform.a
    ox = raster_transform.c
    oy = raster_transform.f

    def _snap(x, y, z=None):
        xs = ox + round((x - ox) / px) * px
        ys = oy + round((y - oy) / px) * px
        return xs, ys

    return shp_transform(_snap, geom)




SHADOW_RETRY_LEVELS = [
    # try 1: current strict-ish
    dict(seed_dilate=2, open_iter=1, close_iter=1, min_area=150,
         perp_margin_min=6,  perp_margin_max=20, min_shadow_rays=25),

    # try 2: relax a bit (more area + wider band)
    dict(seed_dilate=3, open_iter=0, close_iter=1, min_area=90,
         perp_margin_min=10, perp_margin_max=28, min_shadow_rays=18),

    # try 3: most relaxed (still safe)
    dict(seed_dilate=4, open_iter=0, close_iter=1, min_area=60,
         perp_margin_min=14, perp_margin_max=36, min_shadow_rays=12),
]


def directional_halfplane_filter(shadow_mask, building_mask, dx, dy):
    """
    Keep only shadow pixels that lie behind the building along (dx,dy).
    Uses building centroid in pixel coords as reference.
    """
    ys, xs = np.where(building_mask > 0)
    if len(xs) == 0:
        return shadow_mask
    cx = xs.mean()
    cy = ys.mean()

    yy, xx = np.where(shadow_mask > 0)
    if len(xx) == 0:
        return shadow_mask

    # projection along shadow dir
    proj = (xx - cx) * dx + (yy - cy) * dy

    out = np.zeros_like(shadow_mask, dtype=np.uint8)
    out[yy[proj > 0], xx[proj > 0]] = 1
    return out



def morph_cleanup(shadow_mask, open_iter=1, close_iter=1, min_area=200):
    k = np.ones((3,3), np.uint8)
    m = shadow_mask.astype(np.uint8)

    if open_iter and open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=int(open_iter))
    if close_iter and close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=int(close_iter))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


import cv2
import numpy as np

def keep_shadow_connected_to_building(shadow_mask, building_mask, seed_dilate=3):
    """
    Keep only shadow pixels that are connected to a thin band outside building.
    seed_dilate: pixels to dilate the building to make the "seed band".
    """
    k = np.ones((3,3), np.uint8)
    dil = cv2.dilate(building_mask, k, iterations=seed_dilate)

    # thin band outside the building
    seed_band = (dil > 0).astype(np.uint8)
    seed_band[building_mask > 0] = 0

    # seeds = shadow pixels that touch this band
    seeds = (shadow_mask > 0) & (seed_band > 0)
    if not np.any(seeds):
        return shadow_mask  # nothing to guide, return as-is

    # connected components on shadow
    num, labels = cv2.connectedComponents(shadow_mask.astype(np.uint8), connectivity=8)

    keep = np.zeros_like(shadow_mask, dtype=np.uint8)
    seed_labels = np.unique(labels[seeds])
    for lab in seed_labels:
        if lab == 0:
            continue
        keep[labels == lab] = 1

    return keep

def robust_shadow_length_from_mask(shadow_mask, seed_mask, dx, dy, min_bins=10):
    """
    Robust shadow length in pixels:
    For each width-wise bin, compute:
        length_bin = max_t(shadow in bin) - max_t(seed in bin)
    Then take a trimmed mean over bins.
    """

    sy, sx = np.where(seed_mask > 0)
    yy, xx = np.where(shadow_mask > 0)
    if len(sx) < 20 or len(xx) < 50:
        return None

    # Use any consistent origin (cancels out because we subtract)
    cx0 = float(np.mean(sx))
    cy0 = float(np.mean(sy))

    # unit dir + perpendicular
    dnx, dny = float(dx), float(dy)
    pnx, pny = -dny, dnx

    # seed projections
    t_seed = (sx - cx0) * dnx + (sy - cy0) * dny
    p_seed = (sx - cx0) * pnx + (sy - cy0) * pny

    # shadow projections
    t_sh = (xx - cx0) * dnx + (yy - cy0) * dny
    p_sh = (xx - cx0) * pnx + (yy - cy0) * pny

    # bin range based on seed width (stable)
    pmin, pmax = np.percentile(p_seed, 2), np.percentile(p_seed, 98)
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return None

    nbins = 40
    bins = np.linspace(pmin, pmax, nbins + 1)

    idx_seed = np.digitize(p_seed, bins) - 1
    idx_sh   = np.digitize(p_sh,   bins) - 1

    lengths = []
    for b in range(nbins):
        ssel = (idx_seed == b)
        hsel = (idx_sh == b)

        if np.count_nonzero(ssel) < 5 or np.count_nonzero(hsel) < 10:
            continue

        t0 = float(np.max(t_seed[ssel]))   # start = shadow-facing edge
        t1 = float(np.max(t_sh[hsel]))     # end   = farthest shadow
        L = t1 - t0

        if L > 0:
            lengths.append(L)

    if len(lengths) < min_bins:
        return None

    lengths = np.array(lengths, dtype=np.float32)

    # trim outliers (kills tiny spike that goes long)
    lo = np.percentile(lengths, 20)
    hi = np.percentile(lengths, 80)
    lengths = lengths[(lengths >= lo) & (lengths <= hi)]
    if len(lengths) == 0:
        return None

    return float(np.mean(lengths))

def estimate_height_with_retries(poly, img_rgb, transform, raster_crs, conf=None, max_tries=3,log=None):
    """
    Try shadow height estimation multiple times with progressively relaxed cfg.
    Returns shadow_info or None.
    """
    for i in range(min(max_tries, len(SHADOW_RETRY_LEVELS))):
        cfg = SHADOW_RETRY_LEVELS[i]
        if log: log.info("[SHADOW] try=%d cfg=%s", i+1, cfg)
        out = estimate_height_from_shadow(poly, img_rgb, transform, raster_crs, cfg=cfg,log=log)
        if out is not None and np.isfinite(out.get("height", None)):
            if log: log.info(f"[SHADOW] success try={i+1} height={out['height']:.2f} shadow_len={out['shadow_length_m']:.2f}")
            return out
    if log: log.info("[SHADOW] all retries failed")
    return None


def estimate_height_from_shadow(poly, img_rgb, transform, raster_crs, cfg=None,log=None):
    if log: 
        log.info("Shadow height estimation invoked")

    # ---------------- CFG OVERRIDES ----------------
    cfg = cfg or {}

    seed_dilate     = int(cfg.get("seed_dilate", 2))
    open_iter       = int(cfg.get("open_iter", 1))
    close_iter      = int(cfg.get("close_iter", 1))
    min_area        = int(cfg.get("min_area", 150))
    min_shadow_rays = int(cfg.get("min_shadow_rays", 25))

    perp_margin_min = int(cfg.get("perp_margin_min", 6))
    perp_margin_max = int(cfg.get("perp_margin_max", 20))
    # ----------------------------------------------

    # -------------------------------------------------------------------------------

    try:
        # 1) centroid -> lat/lon for astral
        gdf = gpd.GeoSeries([poly], crs=raster_crs).to_crs(epsg=4326)
        lon, lat = gdf.iloc[0].centroid.x, gdf.iloc[0].centroid.y

        ts = datetime.fromisoformat(IMAGE_ACQ_TIME)
        loc = LocationInfo(latitude=lat, longitude=lon)

        sun_elev = elevation(loc.observer, ts)
        sun_az   = azimuth(loc.observer, ts)

        if not np.isfinite(sun_elev) or not np.isfinite(sun_az):
            return None
        if sun_elev < 12:
            return None

        # 2) shadow direction in image coords
        theta = math.radians((sun_az + 180) % 360)
        dx = math.sin(theta)
        dy = -math.cos(theta)

        h, w = img_rgb.shape[:2]

        # 3) rasterize building footprint -> building_mask
        from rasterio.features import rasterize
        building_mask = rasterize(
            [(poly, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        if int(building_mask.sum()) < 50:
            return None

        # 4) boundary pixels
        edges = cv2.Canny(building_mask * 255, 50, 150)
        ys, xs = np.where(edges > 0)
        if len(xs) < 20:
            return None

        # 5) outward normals using distance transform on OUTSIDE
        outside = (building_mask == 0).astype(np.uint8)
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)

        # 6) grayscale and threshold reference
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        #thr = np.percentile(gray, 50)

        # Sun-facing reference (bright side)
        ring = cv2.dilate(building_mask, np.ones((7,7),np.uint8), 2)
        ring[building_mask == 1] = 0

        ys_r, xs_r = np.where(ring > 0)
        cx = xs.mean(); cy = ys.mean()
        proj = (xs_r - cx) * dx + (ys_r - cy) * dy

        sun_side = proj < 0
        if np.sum(sun_side) > 30:
            ref = np.median(gray[ys_r[sun_side], xs_r[sun_side]])
            thr = ref * 0.85
        else:
            thr = np.percentile(gray, 50)

        pixel_size_m = abs(transform.a)
        if pixel_size_m <= 0:
            return None
        max_steps = int(40 / pixel_size_m)

        # shadow_mask = np.zeros((h, w), dtype=np.uint8)
        # max_shadow_px = 0

        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        shadow_lengths = []   # <-- NEW

        for x0, y0 in zip(xs, ys):

            nx = float(gx[y0, x0])
            ny = float(gy[y0, x0])
            nrm = (nx * nx + ny * ny) ** 0.5
            if nrm < 1e-6:
                continue
            nx /= nrm
            ny /= nrm

            if (nx * dx + ny * dy) < 0.55:
                continue

            local_len = 0

            for step in range(1, max_steps):
                x = int(round(x0 + dx * step))
                y = int(round(y0 + dy * step))
                if x < 0 or y < 0 or x >= w or y >= h:
                    break

                if building_mask[y, x] == 1:
                    continue

                if gray[y, x] > thr:
                    break

                shadow_mask[y, x] = 1
                local_len = step


            if local_len > 0:
                shadow_lengths.append(local_len)

        
        # if len(shadow_lengths) < 25:
        #     return None

        if len(shadow_lengths) < min_shadow_rays:
            return None

        max_shadow_px = int(np.percentile(shadow_lengths, 80))
        # Remove any accidental overlap
        shadow_mask[building_mask == 1] = 0

        if max_shadow_px < 6 or int(shadow_mask.sum()) < 10:
            return None

        # ===================== NEW: CLEANUP TO REMOVE EXTRA NOISE =====================
        # A) Keep only shadow blobs connected to a band just outside building
        #shadow_mask = keep_shadow_connected_to_building(shadow_mask, building_mask, seed_dilate=2)
        shadow_mask = keep_shadow_connected_to_building(shadow_mask, building_mask, seed_dilate=seed_dilate)
        if int(shadow_mask.sum()) < 10:
            return None

        # B) Keep only pixels "behind" building along shadow direction
        shadow_mask = directional_halfplane_filter(shadow_mask, building_mask, dx, dy)
        if int(shadow_mask.sum()) < 10:
            return None

        # C) Morphological cleanup (spikes / little islands)
        #shadow_mask = morph_cleanup(shadow_mask, open_iter=1, close_iter=1, min_area=150)
        shadow_mask = morph_cleanup(shadow_mask, open_iter=open_iter, close_iter=close_iter, min_area=min_area)
        shadow_mask[building_mask == 1] = 0
        if int(shadow_mask.sum()) < 10:
            return None



        # ================= SHADOW CORRIDOR (WIDTH LIMIT) =================

        # Seed = only shadow-facing edge pixels
        seed = np.zeros((h, w), np.uint8)

        for x0, y0 in zip(xs, ys):
            nx = float(gx[y0, x0])
            ny = float(gy[y0, x0])
            nrm = (nx*nx + ny*ny) ** 0.5
            if nrm < 1e-6:
                continue
            nx /= nrm
            ny /= nrm

            # STRONGER shadow-facing condition
            if (nx * dx + ny * dy) < 0.55:
                continue

            seed[y0, x0] = 1

        # Thin edge band only
        seed = cv2.dilate(seed, np.ones((3,3), np.uint8), 1)

        # ---- NEW: Perpendicular band constraint (prevents wedge/slanted sides) ----
        # unit perpendicular to shadow direction
        px, py = -dy, dx

        ys_s, xs_s = np.where(seed > 0)
        if len(xs_s) == 0:
            return None

        # reference center from building edge centroid (you already have xs,ys from edges)
        cx0 = float(xs.mean())
        cy0 = float(ys.mean())

        # perpendicular coordinates of seed pixels
        p_seed = (xs_s - cx0) * px + (ys_s - cy0) * py
        pmin = float(np.min(p_seed))
        pmax = float(np.max(p_seed))

        # allow a small margin (in pixels) so we don't chop valid shadow
        #PERP_MARGIN_PX = max(4, int(0.02 * max(h, w)))  # or based on building bbox width
        #PERP_MARGIN_PX = min(PERP_MARGIN_PX, 20)
        PERP_MARGIN_PX = max(perp_margin_min, int(0.02 * max(h, w)))
        PERP_MARGIN_PX = min(PERP_MARGIN_PX, perp_margin_max)
        pmin -= PERP_MARGIN_PX
        pmax += PERP_MARGIN_PX

        # build perp-band mask for whole image
        YY, XX = np.indices((h, w))
        p_all = (XX - cx0) * px + (YY - cy0) * py
        perp_band = ((p_all >= pmin) & (p_all <= pmax)).astype(np.uint8)


        # Sweep seed forward along shadow direction
        corridor = np.zeros((h, w), np.uint8)

        for step in range(1, max_shadow_px + 1):
            M = np.float32([[1, 0, dx * step],
                            [0, 1, dy * step]])
            shifted = cv2.warpAffine(
                seed,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0
            )
            corridor |= shifted

        # # Corridor must not include building
        # corridor[building_mask == 1] = 0

        # # Apply width constraint
        # shadow_mask &= corridor

        corridor[building_mask == 1] = 0

        # apply perp band to kill wedge sides
        corridor &= perp_band

        shadow_mask &= corridor

        shadow_mask = morph_cleanup(shadow_mask, open_iter=0, close_iter=1, min_area=120)
        shadow_mask[building_mask == 1] = 0
        if int(shadow_mask.sum()) < 10:
            return None


        #t_len_px = robust_shadow_length_from_mask(shadow_mask, building_mask, dx, dy)
        t_len_px = robust_shadow_length_from_mask(shadow_mask, seed, dx, dy)
        if t_len_px is None:
            return None

        shadow_len_m = t_len_px * pixel_size_m
        height = shadow_len_m * math.tan(math.radians(sun_elev))

        # ================================================================

        # ============================================================================

        # 7) polygonize shadow_mask
        cnts, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        # pick largest shadow blob
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            return None

        cnt = cnt.reshape(-1, 2)
        coords = []
        for (cx, cy) in cnt:
            wx, wy = rasterio.transform.xy(transform, int(cy), int(cx), offset="center")
            coords.append((wx, wy))

        shadow_poly = Polygon(coords)
        shadow_poly = shadow_poly.simplify(pixel_size_m * 0.8, preserve_topology=True)
        if not shadow_poly.is_valid:
            shadow_poly = shadow_poly.buffer(0)

        if shadow_poly.is_empty or shadow_poly.area < 1.0:
            return None

        # 8) HARD GUARANTEE: shadow does not include building (and doesn't penetrate)
        shadow_poly = shadow_poly.difference(poly)
        if shadow_poly.is_empty:
            return None
        if shadow_poly.geom_type == "MultiPolygon":
            shadow_poly = max(shadow_poly.geoms, key=lambda g: g.area)

        # 9) height math
        #shadow_len_m = max_shadow_px * pixel_size_m
        #height = shadow_len_m * math.tan(math.radians(sun_elev))
        if not np.isfinite(height):
            return None

        return {
            "height": float(np.clip(height, 3.0, 80.0)),
            "shadow_length_m": float(shadow_len_m),
            "sun_azimuth": float(sun_az),
            "shadow_polygon": shadow_poly
        }

    except Exception as e:
        if log:
            log.debug("Shadow height estimation failed: %s", e)
        return None


def is_box_too_big(x1p, y1p, x2p, y2p, img_w, img_h, frac_thresh):
    box_area = max(0, x2p - x1p) * max(0, y2p - y1p)
    img_area = max(1, img_w * img_h)
    return (box_area / img_area) > frac_thresh


def compute_confidence(mask, tex_var, area_px, aoi_area_px):
    """
    Returns confidence in range [0,1]
    """

    iou = mask.get("predicted_iou", 0.0)
    stability = mask.get("stability_score", 0.0)
    fill = float(mask["segmentation"].mean())

    # normalize texture variance (empirical)
    tex_score = np.clip((tex_var - 4.0) / 6.0, 0.0, 1.0)

    # penalize huge blobs
    area_ratio = area_px / aoi_area_px
    area_score = 1.0 - np.clip(area_ratio / MAX_AOI_RATIO, 0.0, 1.0)

    # weighted sum
    confidence = (
        0.45 * iou +
        0.20 * stability +
        0.15 * fill +
        0.10 * tex_score +
        0.10 * area_score
    )

    return float(np.clip(confidence, 0.0, 1.0))


def split_bounds_into_tiles(bounds, tile_size_m, overlap):
    """
    bounds: (west, south, east, north) in EPSG:4326
    returns list of bounds in EPSG:4326
    """
    west, south, east, north = bounds

    # Project AOI to meters (EPSG:3857)
    aoi_gdf = gpd.GeoDataFrame(
        geometry=[Polygon([
            (west, south),
            (east, south),
            (east, north),
            (west, north)
        ])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)



    minx, miny, maxx, maxy = aoi_gdf.total_bounds

    step = tile_size_m * (1 - overlap)
    tiles = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = Polygon([
                (x, y),
                (x + tile_size_m, y),
                (x + tile_size_m, y + tile_size_m),
                (x, y + tile_size_m)
            ])
            tiles.append(tile)
            y += step
        x += step

    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs="EPSG:3857").to_crs(epsg=4326)

    return [tuple(t.bounds) for t in tiles_gdf.geometry]


def run_sam_on_image(img_rgb, transform, max_aoi_ratio=None):
    AOI_AREA = img_rgb.shape[0] * img_rgb.shape[1]
    max_ratio = MAX_AOI_RATIO if max_aoi_ratio is None else float(max_aoi_ratio)
    buildings = []

    small = min(img_rgb.shape[:2]) < 256
    mg = mask_generator_crop if small else mask_generator_full

    masks = mg.generate(img_rgb)

    for m in masks:
        if m["area"] < MIN_AREA_PX:
            continue
        if m["area"] > max_ratio  * AOI_AREA:
            continue
        if m.get("predicted_iou", 0) < 0.6:
            continue
        if m["segmentation"].mean() < 0.015:
            continue

        poly = mask_to_polygon(m, transform)
        if poly is None:
            continue

        poly = snap_polygon_to_pixel(poly,transform)

        if not is_airport_building(poly,relaxed=True):
            continue

        tex_var = texture_variance(img_rgb, poly, transform)
        #tex_var = 5.0

        area_m2 = poly.area
        confidence = compute_confidence(
            m,
            tex_var,
            m["area"],
            AOI_AREA
        )

        if area_m2 > 2000 and tex_var < 4.5:
            confidence *= 0.6

        buildings.append((poly, DEFAULT_HEIGHT, confidence))


        #buildings.append((poly, DEFAULT_HEIGHT))

    return buildings


def resolve_overlaps_by_subtraction(buildings, min_frac=0.02, buffer_m=0.0):
    """
    buildings: [(poly, h, conf), ...]
    For each polygon (low conf), subtract union of higher-conf polygons if overlap is meaningful.
    min_frac: overlap area fraction vs min(areaA, areaB) to trigger subtraction
    buffer_m: optional small buffer applied to the higher-conf union before subtraction (0.0..0.2)
    """
    buildings = sorted(buildings, key=lambda x: float(x[2]), reverse=True)
    kept = []

    for poly, h, conf in buildings:
        if poly is None or poly.is_empty or poly.area <= 0:
            continue

        cutter = None
        for kpoly, _, kconf in kept:
            inter = poly.intersection(kpoly).area
            denom = min(poly.area, kpoly.area) + 1e-9
            if (inter / denom) >= min_frac:
                cutter = kpoly if cutter is None else cutter.union(kpoly)

        if cutter is not None:
            if buffer_m > 0:
                cutter = cutter.buffer(buffer_m)
            new_poly = poly.difference(cutter)

            if new_poly.is_empty:
                continue
            if new_poly.geom_type == "MultiPolygon":
                new_poly = max(new_poly.geoms, key=lambda g: g.area)
            poly = new_poly

        kept.append((poly, h, conf))

    return kept



def split_box_long_axis(x1, y1, x2, y2, max_splits=3):
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return []

    aspect = max(bw / max(1, bh), bh / max(1, bw))
    if aspect < 2.0:
        return [(x1, y1, x2, y2)]

    n = 3 if aspect >= 4.0 else 2
    n = min(max_splits, max(2, n))

    boxes = []
    if bw >= bh:
        step = bw / n
        for i in range(n):
            xa = int(round(x1 + i * step))
            xb = int(round(x1 + (i + 1) * step))
            boxes.append((xa, y1, xb, y2))
    else:
        step = bh / n
        for i in range(n):
            ya = int(round(y1 + i * step))
            yb = int(round(y1 + (i + 1) * step))
            boxes.append((x1, ya, x2, yb))

    return boxes

def suppress_duplicates_by_overlap(buildings, overlap_min=0.60):
    """
    buildings: list of (poly, height, confidence)
    Removes duplicates when intersection area / min(areaA, areaB) is high.
    Keeps higher-confidence polygon.
    """
    # sort high confidence first
    buildings = sorted(buildings, key=lambda x: float(x[2]), reverse=True)
    kept = []

    for poly, h, conf in buildings:
        drop = False
        for kpoly, kh, kconf in kept:
            inter = poly.intersection(kpoly).area
            denom = min(poly.area, kpoly.area) + 1e-9
            if (inter / denom) >= overlap_min:
                drop = True
                break
        if not drop:
            kept.append((poly, h, conf))
    return kept

def merge_buildings(buildings, iou_thresh=0.45):
    """
    Merge polygons using IoU (intersection / union).
    This avoids "min-area overlap" which is too aggressive and causes blob unions.
    """
    merged = []

    for poly, h, conf in buildings:
        keep = True

        for i, (mp, mh, mconf) in enumerate(merged):
            inter = poly.intersection(mp).area
            union = poly.union(mp).area
            iou = (inter / union) if union > 0 else 0.0

            if iou > iou_thresh and poly.distance(mp) < 1.2:
                # keep higher-confidence poly
                if conf > mconf:
                    merged[i] = (poly, h, conf)
                keep = False
                break

        if keep:
            merged.append((poly, h, conf))

    return merged


def snap_polygon_to_pixel(poly, transform):
    snapped = []
    for x, y in poly.exterior.coords:
        row, col = rasterio.transform.rowcol(transform, x, y)
        sx, sy = rasterio.transform.xy(transform, row, col, offset="center")
        snapped.append((sx, sy))
    return Polygon(snapped)



def extract_aoi(bounds,out_tif):
    with rasterio.open(BASE_TIF) as src:
        bounds_3857 = transform_bounds("EPSG:4326", src.crs, *bounds)
        pad_m = 6.0  # 6 meters padding (tune 4..10)
        minx, miny, maxx, maxy = bounds_3857
        bounds_3857 = (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)
        #window = rasterio.windows.from_bounds(*bounds_3857, transform=src.transform)
        window = rasterio.windows.from_bounds(
                    *bounds_3857,
                    transform=src.transform
                ).round_offsets().round_lengths()

        data = src.read([1, 2, 3], window=window)
        transform = src.window_transform(window)

        meta = src.meta.copy()
        #meta.update({"height": data.shape[1], "width": data.shape[2],"transform": transform})
        meta.update({
            "height": data.shape[1],
            "width":  data.shape[2],
            "transform": transform,
            "count": data.shape[0],   # ✅ 3, matches what we write
        })

        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(data)

    return transform, src.crs

def texture_variance(img_rgb, poly, transform):
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    pts = [
        rasterio.transform.rowcol(transform, x, y)
        for x, y in poly.exterior.coords
    ]

    cv2.fillPoly(mask, [np.array(pts)], 255)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    values = gray[mask == 255]

    if len(values) < 50:
        return 0

    return np.std(values)


def overlaps_existing(poly, existing, thresh=0.5):
    for p, _ in existing:
        if poly.intersection(p).area / poly.area > thresh:
            return True
    return False



def mask_to_polygon(mask, transform, offx=0, offy=0, min_contour_area_px=30):
    seg = (mask["segmentation"] > 0).astype(np.uint8)
    #seg = cv2.erode(seg, np.ones((3, 3), np.uint8), iterations=2)

    # Clean mask a bit (fills tiny gaps, removes speckles)
    #seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    #seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)

    #seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    seg = (mask["segmentation"] > 0).astype(np.uint8)

    # Fill small gaps first, then remove speckles
    #k = np.ones((3, 3), np.uint8)
    #seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, k, iterations=1)
    #seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN,  k, iterations=1)

    # Light erosion only (optional) - prevents “mask grows into land”
    #seg = cv2.erode(seg, k, iterations=1)
    k = np.ones((3, 3), np.uint8)

    # Keep close (fills tiny gaps), but make open/erode conditional:
    # Containers are often thin; open/erode can delete them.
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, k, iterations=1)

    area_px = int(seg.sum())
    if area_px >= 1500:
        seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, k, iterations=1)

    # Only erode large blobs (background bleed). Never erode small roofs.
    if area_px >= 4000:
        seg = cv2.erode(seg, k, iterations=1)

    cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    polys = []
    for cnt in cnts:
        if cnt is None or len(cnt) < 3:
            continue
        area = cv2.contourArea(cnt)
        if area < min_contour_area_px:
            continue

        cnt = cnt.reshape(-1, 2)  # <- critical (avoids squeeze bugs)

        coords = []
        for (cx, cy) in cnt:
            r = int(cy) + offy
            c = int(cx) + offx
            x, y = rasterio.transform.xy(transform, r, c, offset="center")
            coords.append((x, y))

        if len(coords) >= 3:
            p = Polygon(coords)
            if p.is_valid and not p.is_empty and p.area > 0:
                polys.append(p)

    if not polys:
        return None

    merged = unary_union(polys)
    if merged.is_empty:
        return None

    if merged.geom_type == "Polygon":
        return merged

    if merged.geom_type == "MultiPolygon":
        return max(list(merged.geoms), key=lambda g: g.area)

    return None


def is_airport_building(poly, relaxed=False):
    hull = poly.convex_hull
    solidity = poly.area / max(hull.area, 1e-6)

    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    aspect = max(w, h) / max(1e-6, min(w, h))

    # Runway / apron rejection
    if not relaxed:
        if solidity < 0.25:
            return False
        if aspect > 20.0:
            return False
    else:
        if solidity < 0.12:
            return False
        if aspect > 35.0:
            return False

    return True

# NOTE: deprecated (kept for reference); main pipeline uses predictor.predict(box=...) on full tile/AOI image.
# def run_sam_on_crop(crop_rgb, full_transform, offset_x, offset_y, yolo_conf):
#     h, w = crop_rgb.shape[:2]
#     predictor.set_image(crop_rgb)

#     # Box over the whole crop (since crop is already YOLO box)
#     box = np.array([0, 0, w, h], dtype=np.float32)

#     masks, scores, _ = predictor.predict(
#         box=box,
#         multimask_output=True
#     )

#     results = []
#     min_area = max(20, int(0.003 * h * w))

#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         area = int(mask.sum())
#         if area < min_area:
#             BOOT_LOG.info(f"SAM mask {i} rejected: area {area} < {min_area}")
#             continue

#         poly = mask_to_polygon({"segmentation": mask.astype(np.uint8)}, full_transform, offset_x, offset_y)
#         if poly is None:
#             BOOT_LOG.info(f"SAM mask {i} rejected: polygon None")
#             continue

#         hull = poly.convex_hull
#         solidity = poly.area / max(hull.area, 1e-6)
#         minx, miny, maxx, maxy = poly.bounds
#         w0 = maxx - minx; h0 = maxy - miny
#         aspect = max(w0, h0) / max(1e-6, min(w0, h0))

#         if not is_airport_building(poly, relaxed=True):
#             logging.info(f"SAM mask {i} rejected: airport_filter solidity={solidity:.3f} aspect={aspect:.2f}")
#             continue

#         conf = float(0.6 * float(score) + 0.4 * float(yolo_conf))
#         results.append((poly, conf))

#     return results


# def overlaps_any(poly, buildings, thresh=0.3):
#     for item in buildings:
#         if not isinstance(item, (tuple, list)) or len(item) < 1:
#             continue
#         p = item[0]
#         if poly.intersection(p).area / poly.area > thresh:
#             logging.info("OVERLAP DETECTION IN overlaps_any")
#             return True
#     return False


def split_box_grid(x1, y1, x2, y2, nx=2, ny=2):
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return []
    xs = np.linspace(x1, x2, nx + 1)
    ys = np.linspace(y1, y2, ny + 1)
    out = []
    for i in range(nx):
        for j in range(ny):
            xa = int(round(xs[i])); xb = int(round(xs[i+1]))
            ya = int(round(ys[j])); yb = int(round(ys[j+1]))
            if xb > xa + 2 and yb > ya + 2:
                out.append((xa, ya, xb, yb))
    return out


def crop_transform(full_transform, x_off, y_off):
    return full_transform * Affine.translation(x_off, y_off)

# def nms_boxes(boxes, iou_thr=0.5):
#     """
#     boxes: [(x1,y1,x2,y2,conf), ...]
#     returns filtered boxes (same format)
#     """
#     if not boxes:
#         return []

#     boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

#     def iou(a, b):
#         ax1, ay1, ax2, ay2 = a
#         bx1, by1, bx2, by2 = b
#         inter_x1 = max(ax1, bx1)
#         inter_y1 = max(ay1, by1)
#         inter_x2 = min(ax2, bx2)
#         inter_y2 = min(ay2, by2)
#         iw = max(0, inter_x2 - inter_x1)
#         ih = max(0, inter_y2 - inter_y1)
#         inter = iw * ih
#         area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
#         area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
#         union = area_a + area_b - inter
#         return inter / union if union > 0 else 0.0

#     keep = []
#     for b in boxes:
#         bx1, by1, bx2, by2, bc = b
#         ok = True
#         for k in keep:
#             kx1, ky1, kx2, ky2, kc = k
#             if iou((bx1, by1, bx2, by2), (kx1, ky1, kx2, ky2)) > iou_thr:
#                 ok = False
#                 break
#         if ok:
#             keep.append(b)
#     return keep


def nms_boxes(boxes, iou_thr=0.55):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def inter_area(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    def iou(a, b):
        inter = inter_area(a, b)
        if inter <= 0:
            return 0.0
        ua = area(a) + area(b) - inter
        return inter / max(1.0, ua)

    # NEW: treat “small mostly inside big” as NOT a duplicate
    def is_nested_keep(b_big, b_small):
        a_big = float(area(b_big))
        a_small = float(area(b_small))
        if a_small <= 0 or a_big <= 0:
            return False
        inter = float(inter_area(b_big, b_small))
        # small is mostly covered by big, and is substantially smaller
        cover_small = inter / a_small
        size_ratio = a_small / a_big
        return (cover_small >= 0.88) and (size_ratio <= 0.72)

    keep = []
    suppressed = set()

    for i in range(len(boxes)):
        if i in suppressed:
            continue
        keep.append(boxes[i])
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            ov = iou(boxes[i], boxes[j])
            if ov > iou_thr:
                # NEW: if j is a smaller roof box inside i, KEEP it
                if is_nested_keep(boxes[i], boxes[j]):
                    continue
                suppressed.add(j)

    return keep



def polygon_rectangularity(poly):
    mrr = poly.minimum_rotated_rectangle
    if mrr is None or mrr.area <= 0:
        return 0.0
    return float(poly.area / mrr.area)

def pick_best_building(polys_with_scores, aoi_bounds=None):
    """
    polys_with_scores: [(poly, conf), ...]  poly in SAME CRS/units as aoi_bounds
    aoi_bounds: (minx, miny, maxx, maxy) in same CRS as poly
    """
    best = None
    best_s = -1.0

    # optional edge-touch reject config
    eps = 1.5  # meters (since you're in EPSG:3857 here)
    aoi_minx = aoi_miny = aoi_maxx = aoi_maxy = None
    aoi_area_m2 = None

    if aoi_bounds is not None:
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi_bounds
        aoi_area_m2 = max(1e-6, (aoi_maxx - aoi_minx) * (aoi_maxy - aoi_miny))

    for poly, conf in polys_with_scores:
        if poly is None or poly.is_empty:
            continue

        hull = poly.convex_hull
        solidity = poly.area / max(hull.area, 1e-6)

        minx, miny, maxx, maxy = poly.bounds

        # Reject big blobs that touch AOI edge (only if AOI bounds provided)
        if aoi_bounds is not None:
            touches_edge = (
                abs(minx - aoi_minx) < eps or abs(maxx - aoi_maxx) < eps or
                abs(miny - aoi_miny) < eps or abs(maxy - aoi_maxy) < eps
            )
            if touches_edge and (poly.area / aoi_area_m2) > 0.25:
                continue

        w = maxx - minx
        h = maxy - miny
        aspect = max(w, h) / max(1e-6, min(w, h))

        rect = polygon_rectangularity(poly)

        # hard guards
        if solidity < 0.25:
            continue
        if aspect > 6.0:
            continue
        if rect < 0.55:
            continue

        score = 0.55 * float(conf) + 0.25 * rect + 0.20 * solidity
        if score > best_s:
            best_s = score
            best = poly

    return best

def stitch_touching(polys, buffer_m=1.0):
    # polys: [(poly, h, conf), ...] in meters CRS
    if not polys:
        return polys

    buffered = [p.buffer(buffer_m) for (p, _, _) in polys]
    u = unary_union(buffered)

    # turn union back into polygons
    geoms = [u] if u.geom_type == "Polygon" else list(u.geoms)

    out = []
    for g in geoms:
        g2 = g.buffer(-buffer_m)
        if g2.is_empty:
            continue
        if g2.geom_type == "MultiPolygon":
            g2 = max(g2.geoms, key=lambda x: x.area)
        # keep best conf among originals that intersect this merged blob
        best_conf = 0.0
        for p, _, c in polys:
            if p.intersects(g2):
                best_conf = max(best_conf, c)
        out.append((g2, DEFAULT_HEIGHT, best_conf))
    return out
def adaptive_max_fill(box_area, bw, bh, yconf, base_small=0.985, base_large=0.975):
    """
    Containers often produce masks that fill ~0.95-0.99 of a tight YOLO box.
    Raise max_fill for elongated boxes / confident YOLO.
    """
    aspect = max(bw / max(1e-6, bh), bh / max(1e-6, bw))
    max_fill = base_small if box_area < 8000 else base_large

    # long roofs: allow higher fill
    if aspect >= 3.0:
        max_fill += 0.02
    if aspect >= 6.0:
        max_fill += 0.01

    # confident YOLO: tolerate tight boxes
    if yconf >= 0.70:
        max_fill += 0.01
    if yconf >= 0.85:
        max_fill += 0.01

    return float(min(0.999, max_fill))

def aoi_bounds_from_transform(transform, H, W):
    x0 = transform.c
    x1 = transform.c + (W * transform.a)
    y0 = transform.f
    y1 = transform.f + (H * transform.e)
    minx, maxx = (min(x0, x1), max(x0, x1))
    miny, maxy = (min(y0, y1), max(y0, y1))
    return (minx, miny, maxx, maxy)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def _ensure_bounds_4326(bounds):
    west, south, east, north = bounds

    # if numbers look like meters (EPSG:3857), convert to degrees
    if (abs(west) > 180 or abs(east) > 180 or abs(south) > 90 or abs(north) > 90):
        BOOT_LOG.warning("Bounds look non-4326; converting from EPSG:3857 -> EPSG:4326")
        west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", west, south, east, north)

    # ensure order is correct
    west, east = (min(west, east), max(west, east))
    south, north = (min(south, north), max(south, north))
    return (west, south, east, north)

def detect_buildings(bounds):
    yolo_boxes_total = 0          # total boxes before NMS (raw)
    yolo_boxes_total_nms = 0      # total boxes after NMS
    bounds = _ensure_bounds_4326(bounds)


    # Stabilize float jitter (important if frontend sends slightly different bounds)
    bounds = tuple(round(float(b), 7) for b in bounds)

    run_id = uuid.uuid4().hex
    setup_logging(run_id)
    log = get_logger(run_id, stage="AOI", det_id="START")

    try:

        RECALL = {"yolo": 0, "sam_ok": 0, "sam_drop": 0}
        out_geojson_rel = f"output/buildings_{run_id}.geojson"
        out_glb_rel     = f"output/aoi_buildings_{run_id}.glb"
        out_geojson_abs = os.path.join(OUT_DIR, f"buildings_{run_id}.geojson")
        out_glb_abs     = os.path.join(OUT_DIR, f"aoi_buildings_{run_id}.glb")

        log.info(f"AOI bounds (EPSG:4326): {bounds} run_id={run_id}")
        all_buildings = []
        sam_buildings = []
        final_buildings = []
        aoi_path = f"ml/tmp/aoi_{uuid.uuid4().hex}.tif"
        transform, raster_crs = extract_aoi(bounds, aoi_path)
        pixel_size_m = abs(transform.a)
        log.info(f"AOI transform origin: transform.c =  {transform.c} , transform.f = {transform.f} ")

        with rasterio.open(aoi_path) as src:
            img = src.read().transpose(1, 2, 0)

        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img

        h, w = img_rgb.shape[:2]
        pixel_area_m2 = abs(transform.a * transform.e)
        aoi_area_m2 = h * w * pixel_area_m2

        log = log.with_ctx(stage="aoi")  # stage update
        log.info("AOI image size=%sx%s pixel_size_m=%.4f crs=%s", w, h, abs(transform.a), str(raster_crs))
        log.info("AOI tif=%s", aoi_path)

        tiles = []

        if aoi_area_m2 > MIN_TILING_AREA_M2:
            tiles = split_bounds_into_tiles(bounds, TILE_SIZE_M, TILE_OVERLAP)
            for tile_idx, tb in enumerate(tiles):
                tile_id = f"T{tile_idx:03d}"
                tlog = get_logger(run_id, stage="TILE", det_id=tile_id)
                t_path = f"ml/tmp/aoi_tile_{uuid.uuid4().hex}.tif"
                t_transform, t_crs = extract_aoi(tb, t_path)

                tlog.info("Tile bounds=%s", tb)
                tlog.info("Tile tif=%s crs=%s origin=(%.3f,%.3f) px=%.3f",t_path, str(t_crs), t_transform.c, t_transform.f, abs(t_transform.a))


                with rasterio.open(t_path) as src:
                    tile_img = src.read().transpose(1, 2, 0)

                #tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                tile_rgb = tile_img
                tile_rgb = tile_rgb.astype(np.uint8)
                predictor.set_image(tile_rgb)   # ONCE per tile


                yolo_boxes = detect_building_boxes_ensemble(tile_rgb, debug=True, debug_tag="tile_ens",log=tlog,run_id=run_id)
                tlog.info("YOLO boxes=%d (pre-nms)", len(yolo_boxes))

                # 2) gate: if too few boxes, run ensemble as recall booster
                # if len(yolo_boxes) < 3:
                #     yolo_boxes = detect_building_boxes_ensemble(tile_rgb, debug=True, debug_tag="tile_ens")

                tlog.info(f"YOLO Detected {len(yolo_boxes)} boxes (micro-tiles)")
                RECALL["yolo"] += len(yolo_boxes)
                yolo_boxes_total += len(yolo_boxes)

                before = len(yolo_boxes)
                #yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.50)
                yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.35)
                yolo_boxes_total_nms += len(yolo_boxes)

                if len(yolo_boxes) != before:
                    tlog.info(f"[TILE] NMS reduced boxes {before} -> {len(yolo_boxes)}")
                

                DROP = {"min_area": 0, "too_full": 0, "none": 0, "poly_none": 0}
                PAD = None   # ✅ always defined
                for box_idx, (x1, y1, x2, y2, yconf) in enumerate(yolo_boxes):
                    det_id = f"{tile_id}_Y{box_idx:04d}"
                    dlog = get_logger(run_id, stage="TILE_DET", det_id=det_id)

                    bw = x2 - x1
                    bh = y2 - y1
                    tile_h, tile_w = tile_rgb.shape[:2]

                    dlog.info(
                        f"[TILE] YOLO raw: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                        f"w={bw}, h={bh}, conf={yconf:.3f}, tile_w={tile_w}, tile_h={tile_h}"
                    )



                    #PAD = int(np.clip(0.08 * min(bw, bh), 2, 14))  # 8% of min dim, clamp 2..12
                    PAD = int(np.clip(0.10 * min(bw, bh), 6, 32))  # was max 14
                    x1p = max(0, x1 - PAD); y1p = max(0, y1 - PAD)
                    x2p = min(tile_rgb.shape[1] - 1, x2 + PAD)
                    y2p = min(tile_rgb.shape[0] - 1, y2 + PAD)


                    if is_mega_or_strip_box(x1p, y1p, x2p, y2p, tile_w, tile_h, yconf):
                        dlog.debug(f"[TILE] Dropped mega/strip box conf={yconf:.3f}")
                        continue


                    pbw = x2p - x1p
                    pbh = y2p - y1p

                    dlog.debug(
                        f"[TILE] YOLO padded: x1p={x1p}, y1p={y1p}, x2p={x2p}, y2p={y2p}, "
                        f"pw={pbw}, ph={pbh}, pad={PAD}"
                    )


                    # If YOLO box is huge, don't drop it (kills recall on container grids).
                    # Split into sub-boxes and run SAM per sub-box.
                    huge = is_box_too_big(x1p, y1p, x2p, y2p, tile_w, tile_h, BIG_BOX_FRAC_TILE)
                    sub_boxes = [(x1p, y1p, x2p, y2p)]

                    if huge:
                        bw2 = x2p - x1p
                        bh2 = y2p - y1p
                        aspect2 = max(bw2 / max(1, bh2), bh2 / max(1, bw2))

                        dlog.warning("[TILE] Huge YOLO box -> splitting")
                        if aspect2 < 2.0:
                            # square-ish mega → split as grid (2x2 or 3x3)
                            # 2x2 is safer; 3x3 increases recall but adds compute
                            sub_boxes = split_box_grid(x1p, y1p, x2p, y2p, nx=2, ny=2)
                        else:
                            # long mega → split along long axis
                            sub_boxes = split_box_long_axis(x1p, y1p, x2p, y2p, max_splits=3)


                    raw_area = max(0, bw) * max(0, bh)
                    pad_area = max(0, pbw) * max(0, pbh)
                    tile_area = tile_w * tile_h

                    if pad_area / max(tile_area, 1) > 0.35:
                        dlog.warning(f"[TILE] Very large padded box vs tile: {pad_area/tile_area:.2f}")

                    dlog.info(
                        f"[TILE] box_area raw={raw_area}, padded={pad_area}, "
                        f"padded/tile={pad_area/max(tile_area,1):.3f}"
                    )

                    

                    for sub_idx, (sx1, sy1, sx2, sy2) in enumerate(sub_boxes):
                        sub_det_id = f"{det_id}_S{sub_idx:02d}"
                        slog = get_logger(run_id, stage="SAM", det_id=sub_det_id)

                        sbw = sx2 - sx1
                        sbh = sy2 - sy1
                        sbox_area = max(1, sbw * sbh)

                        sbox = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

                        mask, mscore, dbg = predict_mask_multi_prompt(
                            predictor,
                            yolo_box_xyxy=(sx1, sy1, sx2, sy2),
                            yolo_conf=yconf,
                            scales=(1.00, 1.15, 1.28, 1.40),
                            shift_frac=0.03,
                            max_union=2
                        )

                        _save_sam_debug(
                            run_id=run_id,
                            det_id=sub_det_id,
                            stage_tag="tile_raw",
                            img_rgb_uint8=tile_rgb,
                            box_xyxy=(sx1, sy1, sx2, sy2),
                            mask_bool=mask,
                            score=mscore,
                            dbg=dbg,
                             log=slog,
                        )

                        if mask is None:
                            DROP["none"] += 1
                            RECALL["sam_drop"] += 1
                            continue

                        RECALL["sam_ok"] += 1

                        best = mask.astype(np.uint8)
                        best_score = float(mscore)

                        # For clipping, use dbg["best_prompt_box"] as reference
                        #ex1, ey1, ex2, ey2 = dbg["best_prompt_box"]

                        if not isinstance(dbg, dict) or "best_prompt_box" not in dbg:
                            ex1, ey1, ex2, ey2 = sx1, sy1, sx2, sy2  # (or x1p,y1p,x2p,y2p in AOI branch)
                        else:
                            ex1, ey1, ex2, ey2 = dbg["best_prompt_box"]

                        # ✅ FIX 1: remove union_mask (undefined)
                        best_preclip = best.copy()
                        best_area = int(best.sum())

                        # ✅ FIX 2: use sub-box area (not x2p/x1p from outer box)
                        box_area = max(1, (sx2 - sx1) * (sy2 - sy1))
                        min_area = max(20, int(0.002 * box_area))

                        # --- loose clip around EXPANDED box ---
                        MIN_PAD_PX = 16
                        MAX_PAD_PX = 96
                        CLIP_EXTRA_PX = 16

                        sbw = sx2 - sx1
                        sbh = sy2 - sy1
                        pad = int(0.10 * min(sbw, sbh))
                        pad = max(MIN_PAD_PX, min(MAX_PAD_PX, pad))

                        tile_h, tile_w = tile_rgb.shape[:2]
                        x1c = max(0, int(ex1 - pad - CLIP_EXTRA_PX))
                        y1c = max(0, int(ey1 - pad - CLIP_EXTRA_PX))
                        x2c = min(tile_w - 1, int(ex2 + pad + CLIP_EXTRA_PX))
                        y2c = min(tile_h - 1, int(ey2 + pad + CLIP_EXTRA_PX))

                        mask_clip = np.zeros_like(best, dtype=np.uint8)
                        mask_clip[y1c:y2c+1, x1c:x2c+1] = 1

                        best = best & mask_clip
                        best_clipped_area = int(best.sum())

                        _save_sam_debug(
                            run_id=run_id,
                            det_id=sub_det_id,
                            stage_tag="tile_final",
                            img_rgb_uint8=tile_rgb,
                            box_xyxy=(sx1, sy1, sx2, sy2),
                            mask_bool=(best.astype(bool) if best is not None else None),
                            score=best_score,
                            dbg={"after_clip": True, **(dbg or {})},
                            log=slog
                        )

                        # 🔒 area-loss guard
                        if best_clipped_area < 0.85 * best_area:
                            best = best_preclip
                            best_clipped_area = best_area

                        if best_clipped_area < min_area:
                            # Fallback: keep pre-clip mask (prevents missing buildings if clipping/cleanup over-prunes).
                            if best_preclip is not None and best_area >= min_area:
                                poly_fb = mask_to_polygon({"segmentation": best_preclip}, t_transform)
                                if poly_fb is not None:
                                    final_conf = float(np.clip(0.85 * yconf + 0.15 * max(0.0, min(1.0, best_score)), 0.0, 1.0))
                                    all_buildings.append((poly_fb, DEFAULT_HEIGHT, final_conf))
                                    continue
                            DROP["min_area"] += 1
                            continue

                        poly = mask_to_polygon({"segmentation": best}, t_transform)
                        if poly is None:
                            # Fallback: polygonize pre-clip (often contains full roof even if postprocessing failed).
                            if best_preclip is not None:
                                poly_fb = mask_to_polygon({"segmentation": best_preclip}, t_transform)
                                if poly_fb is not None:
                                    final_conf = float(np.clip(0.85 * yconf + 0.15 * max(0.0, min(1.0, best_score)), 0.0, 1.0))
                                    all_buildings.append((poly_fb, DEFAULT_HEIGHT, final_conf))
                                    continue
                            DROP["poly_none"] += 1
                            continue


                        #final_conf = 0.6 * best_score + 0.4 * yconf
                        # use YOLO conf as primary; SAM score as a mild modifier
                        final_conf = float(np.clip(0.85 * yconf + 0.15 * max(0.0, min(1.0, best_score)), 0.0, 1.0))
                        all_buildings.append((poly, DEFAULT_HEIGHT, final_conf))


                        # if not is_airport_building(poly, relaxed=True):
                        #     continue

                # ✅ put this here: end of ONE TILE
                tlog.info(
                    f"[TILE DROP] tb={tb} boxes={len(yolo_boxes)} "
                    f"min_area={DROP['min_area']} too_full={DROP['too_full']} "
                    f"none={DROP['none']} poly_none={DROP['poly_none']}"
                )




        else:
            img_rgb = img_rgb.astype(np.uint8)
            predictor.set_image(img_rgb)
            #yolo_boxes = detect_building_boxes(img_rgb)
            yolo_boxes = detect_building_boxes_ensemble(img_rgb, debug=True, debug_tag="aoi_ens",log=log,run_id=run_id)
            orig_yolo_boxes = list(yolo_boxes)
            yolo_boxes_total += len(yolo_boxes)

            # --- BIG BOX / TINY AOI fallback trigger
            H, W = img_rgb.shape[:2]
            q = [0, 0, 0, 0]
            if yolo_boxes:
                # compute largest box fraction
                max_frac = 0.0
                for (x1, y1, x2, y2, yconf) in yolo_boxes:
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    idx = (0 if cy < H / 2 else 2) + (0 if cx < W / 2 else 1)
                    q[idx] += 1

                log.info(f"[AOI] YOLO quadrant counts TL/TR/BL/BR={q}")

                if min(q) == 0:
                    log.info("[AOI] Sparse quadrant -> running ensemble recall booster")
                    yolo_boxes = detect_building_boxes_ensemble(
                        img_rgb,
                        debug=True,
                        debug_tag="aoi_ens_boost",
                        log=log,
                        run_id=run_id,
                    )


                max_conf = max([c for *_, c in yolo_boxes]) if yolo_boxes else 0.0
                # --- BIG BOX / TINY AOI fallback trigger (FIXED)
                H, W = img_rgb.shape[:2]
                suspicious = False

                if not yolo_boxes:
                    suspicious = True
                else:
                    max_frac = 0.0
                    max_conf = 0.0
                    for (x1, y1, x2, y2, yconf) in yolo_boxes:
                        
                        frac = ((x2 - x1) * (y2 - y1)) / max(1, W * H)
                        max_frac = max(max_frac, frac)
                        max_conf = max(max_conf, float(yconf))

                    # --- Special case: single giant YOLO box => box prompt is useless for SAM
                    handled_giant = False
                    if len(yolo_boxes) == 1:
                        x1,y1,x2,y2,yconf = yolo_boxes[0]
                        box_frac = ((x2-x1)*(y2-y1)) / max(1, W*H)

                        # If AOI is tiny, DON'T trust this shortcut; force microtile fallback instead
                        if min(H, W) < MIN_AOI_FOR_DIRECT_YOLO:
                            suspicious = True
                        else:
                            if box_frac > 0.85 and yconf > 0.95:
                                sam_cands = run_sam_on_image(img_rgb, transform, max_aoi_ratio=0.60)
                                cand = [(p, c) for (p, _, c) in sam_cands]
                                best_poly = pick_best_building(cand, aoi_bounds=aoi_bounds_from_transform(transform, H, W))
                                if best_poly is not None:
                                    all_buildings.append((best_poly, DEFAULT_HEIGHT, 0.90))
                                    handled_giant = True


                    if handled_giant:
                        yolo_boxes = []   # ensures the later for-loop is skipped

                    else:

                        # Case A: multiple boxes + one huge box + not extremely confident
                        if (len(yolo_boxes) >= 2) and (max_frac > BIG_BOX_FRAC_AOI) and (max_conf < 0.95):
                            suspicious = True

                        # Case B: tiny AOI — fallback only if YOLO is not clearly confident/correct
                        if (min(H, W) < MIN_AOI_FOR_DIRECT_YOLO):
                            if len(yolo_boxes) >= 2:
                                suspicious = True
                            elif (max_frac > 0.90) and (max_conf < 0.98):
                                suspicious = True

                        if suspicious:
                            log.warning(
                                f"[AOI] YOLO suspicious (boxes={len(yolo_boxes)}, max_box_frac={max_frac:.3f}, "
                                f"max_conf={max_conf:.3f}, size={W}x{H}) -> using microtile fallback"
                            )

                            # ---- Fallback: upscale AOI then microtile YOLO on it
                            scale = UPSCALE_FOR_FALLBACK
                            up = cv2.resize(img_rgb, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)

                            fb_boxes = detect_building_boxes_microtiles(
                                up,
                                tile_size=None,
                                overlap=None,
                                conf=0.05,
                                imgsz=640,
                                run_id=run_id,
                                log=log,
                                debug=True,
                                debug_tag="aoi_microtiles_fallback"
                            )

                            # scale fallback boxes back down to original AOI coords
                            yolo_boxes = []
                            for (x1, y1, x2, y2, yconf) in fb_boxes:
                                x1 = int(x1 / scale); y1 = int(y1 / scale)
                                x2 = int(x2 / scale); y2 = int(y2 / scale)
                                x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
                                y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
                                if x2 > x1 and y2 > y1:
                                    yolo_boxes.append((x1, y1, x2, y2, yconf))

                            log.info(f"[AOI] fallback microtile boxes={len(yolo_boxes)}")
                            if not yolo_boxes:
                                log.warning("[AOI] fallback microtiles returned 0 → keeping original YOLO boxes")
                                yolo_boxes = orig_yolo_boxes

            
            H, W = img_rgb.shape[:2]
            log.info(f"YOLO Detected {len(yolo_boxes)} boxes")

            before0 = len(yolo_boxes)

            # 1) Drop container-like boxes BEFORE NMS (critical)
            yolo_boxes = drop_container_boxes(yolo_boxes, W, H, conf_margin=0.06)

            if len(yolo_boxes) != before0:
                log.info(f"[AOI] Dropped container boxes {before0} -> {len(yolo_boxes)}")

            before1 = len(yolo_boxes)
            yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.45)

            if len(yolo_boxes) != before1:
                log.info(f"[AOI] NMS reduced boxes {before1} -> {len(yolo_boxes)}")

            RECALL["yolo"] += len(yolo_boxes)

            yolo_boxes_total_nms += len(yolo_boxes)

            for box_idx, (x1, y1, x2, y2, yconf) in enumerate(yolo_boxes):

                det_id = f"AOI_Y{box_idx:04d}"
                dlog = get_logger(run_id, stage="AOI_DET", det_id=det_id)

                bw = x2 - x1
                bh = y2 - y1
                H, W = img_rgb.shape[:2]

                dlog.debug(
                    f"[AOI] YOLO raw: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                    f"w={bw}, h={bh}, conf={yconf:.3f}, aoi_w={W}, aoi_h={H}"
                )


                box_frac = (bw * bh) / max(1, W * H)
                # if box_frac > 0.5:
                #     PAD = 0
                # else:
                #     #PAD = int(np.clip(0.08 * min(bw, bh), 2, 12))
                #     #PAD = int(np.clip(0.10 * min(bw, bh), 6, 28))
                #     PAD = int(np.clip(0.18 * min(bw, bh), 8, 40))

                pad_short = int(np.clip(0.18 * min(bw, bh), 8, 40))
                pad_long  = int(np.clip(0.08 * max(bw, bh), 8, 70))
                PAD = None   # ✅ always defined

                if bw >= bh:
                    padx, pady = pad_long, pad_short   # long horizontal roof → expand more on X
                else:
                    padx, pady = pad_short, pad_long   # long vertical roof → expand more on Y

                x1p = max(0, x1 - padx)
                x2p = min(W, x2 + padx)
                y1p = max(0, y1 - pady)
                y2p = min(H, y2 + pady)
                # x1p = max(0, x1 - PAD)
                # y1p = max(0, y1 - PAD)
                # x2p = min(img_rgb.shape[1] - 1, x2 + PAD)
                # y2p = min(img_rgb.shape[0] - 1, y2 + PAD)

                # ✅ Mega/strip drop for AOI
                if is_mega_or_strip_box_aoi(x1p, y1p, x2p, y2p, W, H, yconf):
                    dlog.warning(f"[AOI] Dropped mega/strip box conf={yconf:.3f}")
                    continue

                if len(yolo_boxes) > 1 and is_box_too_big(x1p, y1p, x2p, y2p, W, H, BIG_BOX_FRAC_AOI):
                    dlog.warning(f"[AOI] Rejecting huge YOLO padded box frac>{BIG_BOX_FRAC_AOI}")
                    continue



                pbw = x2p - x1p
                pbh = y2p - y1p

                dlog.info(
                    f"[AOI] YOLO padded: x1p={x1p}, y1p={y1p}, x2p={x2p}, y2p={y2p}, "
                    f"pw={pbw}, ph={pbh}, pad={PAD}"
                )


                raw_area = max(0, bw) * max(0, bh)
                pad_area = max(0, pbw) * max(0, pbh)
                aoi_area = W * H

                dlog.info(
                    f"[AOI] box_area raw={raw_area}, padded={pad_area}, "
                    f"padded/aoi={pad_area/max(aoi_area,1):.3f}"
                )

                # also ensure valid
                if x2p <= x1p + 2 or y2p <= y1p + 2:
                    dlog.warning(f"Skipping invalid padded box [NON TILED ELSE BRANCH]: ({x1p},{y1p})-({x2p},{y2p})")
                    continue

                box = np.array([x1p, y1p, x2p, y2p], dtype=np.float32)

                mask, mscore, dbg = predict_mask_multi_prompt(
                    predictor,
                    yolo_box_xyxy=(x1p, y1p, x2p, y2p),
                    yolo_conf=yconf,
                    scales=(1.00, 1.15, 1.28, 1.40),
                    shift_frac=0.03,
                    max_union=2
                )

                # --- Compute min area in pixels from m² threshold ---
                px_w = abs(transform.a)
                px_h = abs(transform.e)   # usually negative in north-up rasters
                px_area_m2 = max(1e-9, px_w * px_h)

                min_area_px_from_m2 = int(MIN_MASK_AREA_M2 / px_area_m2)
                min_area_px_from_m2 = max(20, min_area_px_from_m2)  # hard floor

                # --- Early reject / retry ---
                if mask is None:
                    RECALL["sam_drop"] += 1
                else:
                    # If low fill, retry with a larger prompt (helps long roofs)
                    if isinstance(dbg, dict) and dbg.get("best_fill", 1.0) < 0.45:
                        rx1, ry1, rx2, ry2 = x1p, y1p, x2p, y2p  # keep padded base
                        mask2, mscore2, dbg2 = predict_mask_multi_prompt(
                            predictor,
                            yolo_box_xyxy=(rx1, ry1, rx2, ry2),
                            yolo_conf=yconf,
                            scales=(1.15, 1.30, 1.45, 1.60, 1.75),
                            shift_frac=0.03,
                            max_union=3
                        )
                        if mask2 is not None and int(mask2.sum()) > int(mask.sum()) * 1.08:
                            mask, mscore, dbg = mask2, mscore2, dbg2



                _save_sam_debug(
                    run_id=run_id,
                    det_id=det_id,
                    stage_tag="aoi_raw",
                    img_rgb_uint8=img_rgb,
                    box_xyxy=(x1p, y1p, x2p, y2p),
                    mask_bool=mask,
                    score=mscore,
                    dbg=dbg,
                    log=dlog
                )

                if mask is None:
                    dlog.info(
                        f"[AOI] SAM: multi-prompt returned None (box_area={(x2p-x1p)*(y2p-y1p)}, yconf={yconf:.3f})"
                    )
                    sam_polys = []
                    RECALL["sam_drop"] += 1
                else:
                    RECALL["sam_ok"] += 1

                    best = mask.astype(np.uint8)
                    best_score = float(mscore)

                    best_preclip = best.copy()
                    best_area = int(best.sum())

                    # Use the winning prompt box from dbg for clipping reference
                    #ex1, ey1, ex2, ey2 = dbg["best_prompt_box"]

                    if not isinstance(dbg, dict) or "best_prompt_box" not in dbg:
                        ex1, ey1, ex2, ey2 = x1p, y1p, x2p, y2p  # (or x1p,y1p,x2p,y2p in AOI branch)
                    else:
                        ex1, ey1, ex2, ey2 = dbg["best_prompt_box"]

                    box_area = max(1, (x2p - x1p) * (y2p - y1p))
                    min_area = max(20, int(0.002 * box_area))

                    # --- Clip around "best prompt box" (loose), with area-loss guard ---
                    MIN_PAD_PX = 16
                    MAX_PAD_PX = 96
                    CLIP_EXTRA_PX = 16

                    bw = x2p - x1p
                    bh = y2p - y1p
                    pad = int(0.10 * min(bw, bh))
                    pad = max(MIN_PAD_PX, min(MAX_PAD_PX, pad))

                    H, W = img_rgb.shape[:2]
                    x1c = max(0, int(ex1 - pad - CLIP_EXTRA_PX))
                    y1c = max(0, int(ey1 - pad - CLIP_EXTRA_PX))
                    x2c = min(W - 1, int(ex2 + pad + CLIP_EXTRA_PX))
                    y2c = min(H - 1, int(ey2 + pad + CLIP_EXTRA_PX))

                    box_mask = np.zeros(best.shape, dtype=np.uint8)
                    box_mask[y1c:y2c+1, x1c:x2c+1] = 1

                    best = best & box_mask
                    best_clipped_area = int(best.sum())

                    _save_sam_debug(
                        run_id=run_id,
                        det_id=det_id,
                        stage_tag="aoi_final",
                        img_rgb_uint8=img_rgb,
                        box_xyxy=(x1p, y1p, x2p, y2p),
                        mask_bool=(best.astype(bool) if best is not None else None),
                        score=best_score,
                        dbg={"after_clip": True, **(dbg or {})},
                        log=dlog,
                    )

                    # 🔒 area-loss guard
                    if best_clipped_area < 0.85 * best_area:
                        best = best_preclip
                        best_clipped_area = best_area

                    if int(best.sum()) < min_area:
                        sam_polys = []
                    else:
                        poly = mask_to_polygon({"segmentation": best}, transform)
                        if poly is None:
                            sam_polys = []
                        else:

                            # Guard against background-bleed for giant boxes (ONLY reject near-full AOI blobs)
                            # Guard against background-bleed for giant boxes
                            box_frac = (bw * bh) / max(1, W * H)
                            pass_fallback = False
                            if (box_frac > 0.85) and (x1p <= 1 or y1p <= 1 or x2p >= W-2 or y2p >= H-2):
                                #aoi_area_m2 = (W * H) * (abs(transform.a) ** 2)
                                aoi_area_m2 = (W * H) * abs(transform.a * transform.e)
                                poly_frac = poly.area / max(aoi_area_m2, 1e-6)

                                if poly_frac > 0.85:
                                    dlog.info(f"[AOI] Giant-edge poly (poly_frac={poly_frac:.3f}, box_frac={box_frac:.3f}) -> fallback to SAM auto")

                                    # Fallback: SAM automatic masks + pick best rectangular/solid building
                                    sam_cands = run_sam_on_image(img_rgb, transform, max_aoi_ratio=0.60)
                                    cand = [(p, c) for (p, _, c) in sam_cands]
                                    best_poly = pick_best_building(cand, aoi_bounds=aoi_bounds_from_transform(transform, H, W))

                                    if best_poly is not None:
                                        sam_polys = [(best_poly, 0.90)]
                                        poly = best_poly
                                        best_score = 0.90
                                        pass_fallback = True
                                    else:
                                        sam_polys = []
                                        pass_fallback = True


                            minx, miny, maxx, maxy = poly.bounds
                            ww = maxx - minx
                            hh = maxy - miny
                            aspect = max(ww, hh) / max(1e-6, min(ww, hh))
                            dlog.info(
                                f"POLY stats: area_m2={poly.area:.1f}, aspect={aspect:.2f}, "
                                f"bounds_w={ww:.1f}, bounds_h={hh:.1f}"
                            )

                            if not pass_fallback:

                                fill_ratio = poly.area / (box_area * (abs(transform.a) ** 2))
                                if fill_ratio > 0.80:
                                    sam_polys = []
                                elif not is_airport_building(poly, relaxed=True):
                                    sam_polys = []
                                elif poly.area > 20000:
                                    sam_polys = []
                                else:
                                    minx, miny, maxx, maxy = poly.bounds
                                    ww = (maxx - minx)
                                    hh = (maxy - miny)
                                    aspect = max(ww, hh) / max(1e-6, min(ww, hh))
                                    min_dim = min(ww, hh)

                                    # Drop seam/road "strip" artifacts (thin, long, small area)
                                    # (Your bad case: aspect~8.7, min_dim~1.6m, area~9m2)
                                    if (min_dim < 3.0 and aspect > 4.0):
                                        sam_polys = []
                                    elif aspect > 12.0:
                                        sam_polys = []
                                    else:
                                        sam_polys = [(poly, best_score)]


                # Build YOLO bbox polygon in world coords
                x1w, y1w = rasterio.transform.xy(transform, y1p, x1p, offset="center")
                x2w, y2w = rasterio.transform.xy(transform, y2p, x2p, offset="center")

                minx = min(x1w, x2w); maxx = max(x1w, x2w)
                miny = min(y1w, y2w); maxy = max(y1w, y2w)
                yolo_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


                for poly, conf in sam_polys:
                    #final_conf = min(1.0, 0.6 * conf + 0.4 * yconf)
                    if yconf > 0.9:
                        final_conf = max(conf, yconf) * 0.9
                    else:
                        final_conf = 0.6 * conf + 0.4 * yconf

                    all_buildings.append((poly, DEFAULT_HEIGHT, final_conf))

                       
        

        
        log.info("[AOI YOLO SUMMARY] raw_boxes=%d after_nms=%d", yolo_boxes_total, yolo_boxes_total_nms)

        print(f"[AOI YOLO SUMMARY] raw_boxes={yolo_boxes_total}, after_nms={yolo_boxes_total_nms}")

        # --- Normalize/clean all_buildings so it's ALWAYS [(poly, height, conf), ...]
        normalized = []
        for item in all_buildings:
            if not isinstance(item, (tuple, list)):
                continue
            if len(item) < 3:
                continue

            poly = item[0]
            h = float(item[1])
            conf = float(item[2])

            #normalized.append((snap_polygon_to_pixel(poly, transform), h, conf))
            normalized.append((poly, h, conf))

        all_buildings = normalized

        sam_buildings = merge_buildings(all_buildings, iou_thresh=0.55)

        stitch_buf = max(0.2, 0.6 * pixel_size_m)
        sam_buildings = stitch_touching(sam_buildings, buffer_m=stitch_buf)

        sam_buildings = suppress_duplicates_by_overlap(sam_buildings, overlap_min=0.65)


        cleaned = []
        for poly, h, conf in sam_buildings:
            if poly is None or poly.is_empty:
                continue

            # Basic size guard
            if poly.area < 8.0:         # too tiny in m^2
                continue
            if poly.area > 15000.0:     # too huge (land patches)
                continue

            hull = poly.convex_hull
            solidity = poly.area / max(hull.area, 1e-6)

            rect = polygon_rectangularity(poly)

            minx, miny, maxx, maxy = poly.bounds
            ww = maxx - minx
            hh = maxy - miny
            aspect = max(ww, hh) / max(1e-6, min(ww, hh))

            # Strong building-ish constraints (tune if needed)
            if solidity < 0.18:
                continue
            if rect < 0.42:
                continue
            if aspect > 18.0:
                continue

            cleaned.append((poly, h, conf))

        sam_buildings = cleaned



        log.info(f"final total merged buildings : {len(sam_buildings)}")

        if not sam_buildings and all_buildings:
            log.warning("Merge removed all buildings → using raw SAM outputs")
            sam_buildings = all_buildings

        else:
            log.info(f"SAM detected {len(sam_buildings)} buildings")

        buildings = list(sam_buildings)  # ALWAYS keep SAM results

        for poly, _, conf in buildings:
            height = None

            shadow_info = None

            if height is None and conf >= 0.75:
                shadow_info = estimate_height_with_retries(poly, img_rgb, transform, raster_crs, conf=conf, max_tries=3,log=log)
                if shadow_info:
                    height = shadow_info["height"]



            # 3️⃣ Fallback
            if height is None:
                log.debug("FALLING BACK TO DEAFULT HEIGHT")
                height = DEFAULT_HEIGHT
                #conf = min(conf, 0.5)


            log.info("CONF=%.2f -> HEIGHT=%.2f", conf, height)

            #final_buildings.append((poly, height, conf,shadow_info))
            final_buildings.append((poly, height, conf,shadow_info))

        #polys, heights, confidences = zip(*final_buildings)
        polys = [b[0] for b in final_buildings]
        heights = [b[1] for b in final_buildings]
        confidences = [b[2] for b in final_buildings]
        shadows = [b[3] for b in final_buildings]  # may be None

        #logging.info(f"First Ploy Centroid: polys[0].centroid.x = {polys[0].centroid.x} , polys[0].centroid.y = {polys[0].centroid.y}")
        if polys:
            log.info(
                f"First Poly Centroid: x={polys[0].centroid.x:.2f}, "
                f"y={polys[0].centroid.y:.2f}"
            )
        else:
            log.warning("No polygons available for centroid logging")



        features = []
        b_id = 1
        s_id = 1

        for poly, height, conf, shadow in final_buildings:
            bid = f"B{b_id}"
            b_id += 1

            # Building feature
            features.append({
                "type": "Feature",
                "geometry": poly,
                "properties": {
                    "feature_type": "building",
                    "id": bid,
                    "height": height,
                    "confidence": conf
                }
            })

            # Shadow feature
            if shadow:
                sid = f"S{s_id}"
                s_id += 1

                features.append({
                    "type": "Feature",
                    "geometry": shadow["shadow_polygon"],
                    "properties": {
                        "feature_type": "shadow",
                        "id": sid,
                        "building_id": bid,
                        "shadow_len": shadow["shadow_length_m"],
                        "sun_azimuth": shadow["sun_azimuth"]
                    }
                })

        # --------------------------------------------------
        # FINAL OUTPUT GUARD
        # --------------------------------------------------

        if not features:
            log.warning("No features generated → returning empty AOI result")

            return {
                "geojson": None,
                "glb": None,
                "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else [],
                "reason": "no_buildings_detected"
            }

        gdf = gpd.GeoDataFrame.from_features(features, crs=raster_crs).to_crs(epsg=4326)

        # 🔒 STRICT AOI GUARD (fix outside-AOI buildings)
        gdf = clip_gdf_to_aoi_4326(gdf, bounds)

        if gdf.empty:
            log.warning("All features clipped out by AOI guard -> returning empty")
            return {
                "geojson": None,
                "glb": None,
                "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else [],
                "reason": "all_features_outside_aoi_after_clip"
            }

        gdf.to_file(out_geojson_abs, driver="GeoJSON")

        # Export GLB using only building polys that survived clip
        # (Keep your original final_buildings list, but filter by AOI now)
        from shapely.geometry import box as shp_box
        aoi_poly_4326 = shp_box(*bounds)
        aoi_poly_raster = gpd.GeoSeries([aoi_poly_4326], crs="EPSG:4326").to_crs(raster_crs).iloc[0]

        kept_buildings = []
        for poly, height, conf, shadow in final_buildings:
            if poly is None or poly.is_empty:
                continue
            clipped = poly.intersection(aoi_poly_raster)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "MultiPolygon":
                clipped = max(clipped.geoms, key=lambda g: g.area)
            kept_buildings.append((clipped, height, conf, shadow))

        export_glb([flat_roof(p, h) for p, h, _, _ in kept_buildings], out_glb_abs)

        log.warning(
            f"[RECALL] YOLO={RECALL['yolo']} | SAM_OK={RECALL['sam_ok']} | SAM_DROP={RECALL['sam_drop']} | LOSS={(RECALL['sam_drop']/max(1,RECALL['yolo'])):.2%}"
        )


        log.info("RUN_DONE run_id=%s geojson=%s glb=%s", run_id, out_geojson_abs, out_glb_abs)
        log.info("SAM_DEBUG_DIR %s", os.path.join(SAM_DEBUG_DIR, run_id))
        log.info("YOLO_DEBUG_DIR %s", os.path.join(OUT_DIR, "yolo_debug", run_id))


        return {
            "geojson": out_geojson_rel,
            "glb": out_glb_rel,
            "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else []
        }
    except Exception:
        log.exception("detect_buildings crashed")
        return {
            "geojson": None,
            "glb": None,
            "tiles": [],
            "reason": "exception",
            "run_id": run_id,
            "log_file": f"{LOGS_DIR}/{run_id}.log",
        }