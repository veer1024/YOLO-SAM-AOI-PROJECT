# ml/detect_buildings_aoi.py
import os
import logging
import sys
import cv2
import numpy as np
import rasterio
import geopandas as gpd
import torch
import osmnx as ox
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

from ml.yolo_detector import (
    detect_building_boxes,
    detect_building_boxes_microtiles,
    detect_building_boxes_ensemble,
)

TMP_DIR = "ml/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

### OSM CAHCING CONFIG

OSM_CACHE_DIR = "ml/osm_cache_local"
OSM_CACHE_INDEX = os.path.join(OSM_CACHE_DIR, "osm_cache_index.geojson")
os.makedirs(OSM_CACHE_DIR, exist_ok=True)

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
#logging.basicConfig(level=logging.INFO)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ---- File handler ----
file_handler = logging.FileHandler("detect_buildings_aoi.log", mode="a")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# ---- STDOUT handler ----
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)

# Avoid duplicate logs
# if not logger.handlers:
#     logger.addHandler(file_handler)
#     logger.addHandler(stdout_handler)

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(stdout_handler)




# --- Drop mega / strip boxes on tiles ---
MEGA_AREA_THR_TILE   = 0.45   # drop if padded box >45% of tile area
MEGA_SIDE_THR_TILE   = 0.92   # drop if box spans >92% of tile width or height

STRIP_SIDE_THR_TILE  = 0.94   # near-full width or height
STRIP_OTHER_MAX_TILE = 0.45   # ...while other side is still thick enough -> it's a strip, drop

# Confidence escape hatch (keep only if extremely confident)
MEGA_KEEP_CONF = 0.92

### MEAGBOX AOI

MEGA_AREA_THR_AOI  = 0.60
MEGA_SIDE_THR_AOI  = 0.95
STRIP_SIDE_THR_AOI = 0.96
STRIP_OTHER_MAX_AOI = 0.55
MEGA_KEEP_CONF_AOI = 0.95


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
logging.info("Using device: %s", device)

# --------------------------------------------------
# LOAD SAM
# --------------------------------------------------
sam = sam_model_registry["vit_b"](checkpoint="ml/checkpoints/sam_vit_b.pth")
sam.to(device=device, dtype=torch.float32)
sam.eval()

#decoder_ckpt = "ml/sam_training/checkpoints/sam_decoder_finetuned.pth.epoch12.pth"
decoder_ckpt="SAM_TRAINED_MODEL/sam_decoder_finetuned.pth.epoch15.pth"
sam.mask_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
logging.info("Fine-tuned SAM decoder loaded")


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
# OSM SETTINGS
# --------------------------------------------------
ox.settings.use_cache = True
ox.settings.cache_folder = "osm_cache"
ox.settings.log_console = False

# --------------------------------------------------
# HELPERS
# --------------------------------------------------


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
    box1 = np.array([x1p, y1p, x2p, y2p], dtype=np.float32)
    masks1, scores1, _ = predictor.predict(box=box1, multimask_output=True)
    m1, s1, min_area = _pick_best_mask_for_box(masks1, scores1, box1_area, bw1, bh1, yconf)

    # pass 2 (expanded)
    ex1, ey1, ex2, ey2 = _expand_box_xyxy(x1p, y1p, x2p, y2p, W, H, scale=expand_scale)
    bw2 = ex2 - ex1
    bh2 = ey2 - ey1
    box2_area = max(1, bw2 * bh2)

    box2 = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)
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

    strip = ((w_frac >= STRIP_SIDE_THR_TILE and h_frac >= STRIP_OTHER_MAX_TILE) or
             (h_frac >= STRIP_SIDE_THR_TILE and w_frac >= STRIP_OTHER_MAX_TILE))

    if (mega or strip) and (yconf < MEGA_KEEP_CONF):
        return True
    return False

def is_mega_or_strip_box_aoi(x1p,y1p,x2p,y2p, W,H, yconf):
    bw = max(1, x2p-x1p); bh = max(1, y2p-y1p)
    area_frac = (bw*bh) / max(1, W*H)
    w_frac = bw / max(1, W)
    h_frac = bh / max(1, H)

    mega = (area_frac >= MEGA_AREA_THR_AOI) or (w_frac >= MEGA_SIDE_THR_AOI) or (h_frac >= MEGA_SIDE_THR_AOI)

    strip = ((w_frac >= STRIP_SIDE_THR_AOI and h_frac >= STRIP_OTHER_MAX_AOI) or
             (h_frac >= STRIP_SIDE_THR_AOI and w_frac >= STRIP_OTHER_MAX_AOI))

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


def load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857"):
    """
    Fallback loader:
    - Finds cached AOI that overlaps requested bounds most.
    - If none overlap, pick nearest by centroid distance.
    - Returns cached buildings CLIPPED to requested bounds.
    """
    if not os.path.exists(OSM_CACHE_INDEX):
        return None

    try:
        idx = gpd.read_file(OSM_CACHE_INDEX)
        if idx.empty:
            return None
    except Exception as e:
        logging.warning("Failed to read OSM cache index: %s", e)
        return None

    west, south, east, north = bounds
    req_geom = box(west, south, east, north)
    req_gdf = gpd.GeoDataFrame(geometry=[req_geom], crs="EPSG:4326")

    # Prefer metric for overlap/distance scoring
    req_m = req_gdf.to_crs(raster_crs).geometry.iloc[0]
    idx_m = idx.to_crs(raster_crs)

    # Compute overlap area (in m^2)
    overlaps = idx_m.geometry.intersection(req_m)
    idx_m["overlap_area"] = overlaps.area

    # 1) pick best overlapping
    cand = idx_m[idx_m["overlap_area"] > 0].copy()
    if not cand.empty:
        best = cand.sort_values("overlap_area", ascending=False).iloc[0]
    else:
        # 2) else pick nearest centroid
        idx_m["dist"] = idx_m.geometry.centroid.distance(req_m.centroid)
        best = idx_m.sort_values("dist", ascending=True).iloc[0]

    best_path = best.get("path", None)
    if not best_path or not os.path.exists(best_path):
        return None

    try:
        gdf = gpd.read_file(best_path)
        if gdf is None or gdf.empty:
            return None
    except Exception as e:
        logging.warning("Failed to load fallback cached OSM file: %s", e)
        return None

    # Clip cached buildings to requested AOI bounds
    try:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        clipped = gpd.clip(gdf, req_gdf)
        if clipped.empty:
            return None
        logging.warning(
            "Using fallback OSM cache from %s (overlap_area=%.1f)",
            os.path.basename(best_path),
            float(best.get("overlap_area", 0.0)),
        )
        return clipped
    except Exception as e:
        logging.warning("Fallback clip failed: %s", e)
        return None


def osm_cache_key(bounds):
    """
    bounds = (west, south, east, north)
    """
    s = json.dumps([round(b, 6) for b in bounds])
    return hashlib.md5(s.encode()).hexdigest()

def save_osm_cache(gdf, bounds):
    key = osm_cache_key(bounds)
    path = os.path.join(OSM_CACHE_DIR, f"osm_{key}.geojson")
    
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    
    gdf.to_file(path, driver="GeoJSON")
    logging.info("OSM cache saved: %s", path)

    # ---- Update cache index ----
    west, south, east, north = bounds
    geom = box(west, south, east, north)

    rec = gpd.GeoDataFrame(
        [{"key": key, "path": path}],
        geometry=[geom],
        crs="EPSG:4326",
    )

    if os.path.exists(OSM_CACHE_INDEX):
        try:
            idx = gpd.read_file(OSM_CACHE_INDEX)
            # remove old record for same key if exists
            if "key" in idx.columns:
                idx = idx[idx["key"] != key]
            idx = pd.concat([idx, rec], ignore_index=True)
        except Exception as e:
            logging.warning("Failed reading OSM cache index, recreating: %s", e)
            idx = rec
    else:
        idx = rec

    idx.to_file(OSM_CACHE_INDEX, driver="GeoJSON")

def load_osm_cache(bounds):
    key = osm_cache_key(bounds)
    path = os.path.join(OSM_CACHE_DIR, f"osm_{key}.geojson")

    if os.path.exists(path):
        try:
            gdf = gpd.read_file(path)
            if not gdf.empty:
                logging.info("Loaded OSM from cache")
                return gdf
        except Exception as e:
            logging.warning("Failed to read OSM cache: %s", e)

    return None



def osm_height_from_tags(osm_row):
    """
    Extract height from OSM row safely.
    """
    if "height" in osm_row and osm_row["height"] is not None:
        try:
            return float(str(osm_row["height"]).replace("m", ""))
        except:
            pass

    if "building:levels" in osm_row and osm_row["building:levels"] is not None:
        try:
            return float(osm_row["building:levels"]) * 3.0
        except:
            pass

    return None







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
# def morph_cleanup(shadow_mask, open_iter=1, close_iter=1, min_area=200):
#     k = np.ones((3,3), np.uint8)
#     m = shadow_mask.astype(np.uint8)

#     # remove thin spikes/noise
#     m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=open_iter)
#     # fill small gaps
#     m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iter)

#     # remove tiny blobs
#     num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#     out = np.zeros_like(m)
#     for i in range(1, num):
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             out[labels == i] = 1
#     return out

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



def estimate_height_from_shadow(poly, img_rgb, transform, raster_crs):
    logging.info("Shadow height estimation invoked")

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


            # min_gray = int(gray_s[y0, x0])
            # bright_count = 0

            # # tuned defaults (good starting point)
            # DELTA = 8            # was too small; try 6–12
            # BRIGHT_LIMIT = 6     # allow more bumps before stopping
            # HARD_THR_PAD = 5     # allow a little above thr before stopping

            # for step in range(1, max_steps):
            #     x = int(round(x0 + dx * step))
            #     y = int(round(y0 + dy * step))
            #     if x < 0 or y < 0 or x >= w or y >= h:
            #         break

            #     if building_mask[y, x] == 1:
            #         continue

            #     g = int(gray_s[y, x])

            #     # 1) absolute stop (but allow a small pad)
            #     # if g > (thr + HARD_THR_PAD):
            #     #     break

            #     # 2) soft monotonic check (tolerant)
            #     if g > (min_gray + DELTA):
            #         bright_count += 1
            #         if bright_count >= BRIGHT_LIMIT:
            #             break
            #     else:
            #         bright_count = 0
            #         if g < min_gray:
            #             min_gray = g

            #     shadow_mask[y, x] = 1
            #     local_len = step


            if local_len > 0:
                shadow_lengths.append(local_len)

        
        if len(shadow_lengths) < 25:
            return None

        max_shadow_px = int(np.percentile(shadow_lengths, 80))
        # Remove any accidental overlap
        shadow_mask[building_mask == 1] = 0

        if max_shadow_px < 6 or int(shadow_mask.sum()) < 10:
            return None

        # ===================== NEW: CLEANUP TO REMOVE EXTRA NOISE =====================
        # A) Keep only shadow blobs connected to a band just outside building
        shadow_mask = keep_shadow_connected_to_building(shadow_mask, building_mask, seed_dilate=2)
        if int(shadow_mask.sum()) < 10:
            return None

        # B) Keep only pixels "behind" building along shadow direction
        shadow_mask = directional_halfplane_filter(shadow_mask, building_mask, dx, dy)
        if int(shadow_mask.sum()) < 10:
            return None

        # C) Morphological cleanup (spikes / little islands)
        shadow_mask = morph_cleanup(shadow_mask, open_iter=1, close_iter=1, min_area=150)
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
        PERP_MARGIN_PX = max(4, int(0.02 * max(h, w)))  # or based on building bbox width
        PERP_MARGIN_PX = min(PERP_MARGIN_PX, 20)
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
        logging.debug("Shadow height estimation failed: %s", e)
        return None


def is_box_too_big(x1p, y1p, x2p, y2p, img_w, img_h, frac_thresh):
    box_area = max(0, x2p - x1p) * max(0, y2p - y1p)
    img_area = max(1, img_w * img_h)
    return (box_area / img_area) > frac_thresh



def find_matching_osm_height(poly, osm_gdf, raster_crs):
    """
    Returns OSM height if overlap is strong.
    """
    if osm_gdf is None or osm_gdf.empty:
        return None

    for _, row in osm_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)

        geom = (
            gpd.GeoSeries([geom], crs="EPSG:4326")
            .to_crs(raster_crs)
            .iloc[0]
        )

        inter = poly.intersection(geom).area
        union = poly.union(geom).area

        if union > 0 and inter / union > 0.5:
            return osm_height_from_tags(row)

    return None


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

# def resolve_overlaps_by_subtraction(buildings, min_frac=0.25, buffer_m=0.0, min_keep_ratio=0.85):
#     buildings = sorted(buildings, key=lambda x: float(x[2]), reverse=True)
#     kept = []

#     for poly, h, conf in buildings:
#         if poly is None or poly.is_empty or poly.area <= 0:
#             continue

#         cutter = None
#         for kpoly, _, kconf in kept:
#             inter = poly.intersection(kpoly).area
#             denom = min(poly.area, kpoly.area) + 1e-9
#             if (inter / denom) >= min_frac:
#                 cutter = kpoly if cutter is None else cutter.union(kpoly)

#         if cutter is not None:
#             if buffer_m > 0:
#                 cutter = cutter.buffer(buffer_m)

#             new_poly = poly.difference(cutter)
#             if new_poly.is_empty:
#                 continue

#             # keep biggest piece if multipoly
#             if new_poly.geom_type == "MultiPolygon":
#                 new_poly = max(new_poly.geoms, key=lambda g: g.area)

#             # 🔒 area-loss guard (prevents half roofs)
#             if (new_poly.area / (poly.area + 1e-9)) < min_keep_ratio:
#                 # skip subtraction; keep original
#                 kept.append((poly, h, conf))
#                 continue

#             poly = new_poly

#         kept.append((poly, h, conf))

#     return kept

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




# def get_osm_buildings(bounds):
#     """
#     Fetch OSM buildings with local cache fallback.
#     """
#     # 1️⃣ Try local cache first
#     return None
#     logging.info(f"osmnx version: {ox.__version__}")
#     logging.info(f"max_query_area_size: {ox.settings.max_query_area_size}")
#     logging.info(f"use_cache: {ox.settings.use_cache}, cache_folder: {ox.settings.cache_folder}")
#     cached = load_osm_cache(bounds)
#     if cached is not None:
#         return cached

#     west, south, east, north = bounds

#     fallback = load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857")
#     if fallback is not None and not fallback.empty:
#         return fallback
#     else:
#         logging.info(f"NO CACHE RESULT FOUND, OSM API WILL BE CALLED ")

#     try:
#         # OSMnx 2.x expects bbox=(west, south, east, north)
#         gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
#     except Exception as e:
#         logging.warning("OSM API failed: %s", e)
#         fallback = load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857")
#         if fallback is not None and not fallback.empty:
#             return fallback
#         return None

#     if gdf is None or gdf.empty:
#         logging.warning("OSM returned empty result")
#         fallback = load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857")
#         if fallback is not None and not fallback.empty:
#             return fallback
#         return None

#     gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

#     # 2️⃣ Save to cache
#     save_osm_cache(gdf, bounds)

#     return gdf



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
def run_sam_on_crop(crop_rgb, full_transform, offset_x, offset_y, yolo_conf):
    h, w = crop_rgb.shape[:2]
    predictor.set_image(crop_rgb)

    # Box over the whole crop (since crop is already YOLO box)
    box = np.array([0, 0, w, h], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    results = []
    min_area = max(20, int(0.003 * h * w))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        area = int(mask.sum())
        if area < min_area:
            logging.info(f"SAM mask {i} rejected: area {area} < {min_area}")
            continue

        poly = mask_to_polygon({"segmentation": mask.astype(np.uint8)}, full_transform, offset_x, offset_y)
        if poly is None:
            logging.info(f"SAM mask {i} rejected: polygon None")
            continue

        hull = poly.convex_hull
        solidity = poly.area / max(hull.area, 1e-6)
        minx, miny, maxx, maxy = poly.bounds
        w0 = maxx - minx; h0 = maxy - miny
        aspect = max(w0, h0) / max(1e-6, min(w0, h0))

        if not is_airport_building(poly, relaxed=True):
            logging.info(f"SAM mask {i} rejected: airport_filter solidity={solidity:.3f} aspect={aspect:.2f}")
            continue

        conf = float(0.6 * float(score) + 0.4 * float(yolo_conf))
        results.append((poly, conf))

    return results


def overlaps_any(poly, buildings, thresh=0.3):
    for item in buildings:
        if not isinstance(item, (tuple, list)) or len(item) < 1:
            continue
        p = item[0]
        if poly.intersection(p).area / poly.area > thresh:
            logging.info("OVERLAP DETECTION IN overlaps_any")
            return True
    return False


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

def nms_boxes(boxes, iou_thr=0.5):
    """
    boxes: [(x1,y1,x2,y2,conf), ...]
    returns filtered boxes (same format)
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    keep = []
    for b in boxes:
        bx1, by1, bx2, by2, bc = b
        ok = True
        for k in keep:
            kx1, ky1, kx2, ky2, kc = k
            if iou((bx1, by1, bx2, by2), (kx1, ky1, kx2, ky2)) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
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
        logging.warning("Bounds look non-4326; converting from EPSG:3857 -> EPSG:4326")
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
    RECALL = {"yolo": 0, "sam_ok": 0, "sam_drop": 0}
    out_geojson_rel = f"output/buildings_{run_id}.geojson"
    out_glb_rel     = f"output/aoi_buildings_{run_id}.glb"
    out_geojson_abs = os.path.join(OUT_DIR, f"buildings_{run_id}.geojson")
    out_glb_abs     = os.path.join(OUT_DIR, f"aoi_buildings_{run_id}.glb")

    logging.info(f"AOI bounds (EPSG:4326): {bounds} run_id={run_id}")


    logging.info(f"AOI bounds (EPSG:4326): {bounds}")
    all_buildings = []
    sam_buildings = []
    final_buildings = []
    aoi_path = f"ml/tmp/aoi_{uuid.uuid4().hex}.tif"
    transform, raster_crs = extract_aoi(bounds, aoi_path)
    pixel_size_m = abs(transform.a)
    logging.info(f"AOI transform origin: transform.c =  {transform.c} , transform.f = {transform.f} ")

    with rasterio.open(aoi_path) as src:
        img = src.read().transpose(1, 2, 0)

    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img

    h, w = img_rgb.shape[:2]
    pixel_area_m2 = abs(transform.a * transform.e)
    aoi_area_m2 = h * w * pixel_area_m2

    #osm = get_osm_buildings(bounds)
    osm = None
    tiles = []

    if aoi_area_m2 > MIN_TILING_AREA_M2:
        tiles = split_bounds_into_tiles(bounds, TILE_SIZE_M, TILE_OVERLAP)
        for tb in tiles:
            t_path = f"ml/tmp/aoi_tile_{uuid.uuid4().hex}.tif"
            t_transform, t_crs = extract_aoi(tb, t_path)

            with rasterio.open(t_path) as src:
                tile_img = src.read().transpose(1, 2, 0)

            #tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
            tile_rgb = tile_img
            tile_rgb = tile_rgb.astype(np.uint8)
            predictor.set_image(tile_rgb)   # ONCE per tile


            # 1) default pass (fast)
            # yolo_boxes = detect_building_boxes_microtiles(
            #     tile_rgb,
            #     tile_size=None,
            #     overlap=None,
            #     conf=0.05,
            #     imgsz=640,
            #     debug=True,
            #     debug_tag="tile",
            #     target_tile_factor=1.5,
            #     overlap_cap=0.75
            # )

            yolo_boxes = detect_building_boxes_ensemble(tile_rgb, debug=True, debug_tag="tile_ens")

            # 2) gate: if too few boxes, run ensemble as recall booster
            if len(yolo_boxes) < 3:
                yolo_boxes = detect_building_boxes_ensemble(tile_rgb, debug=True, debug_tag="tile_ens")

            logging.info(f"YOLO Detected {len(yolo_boxes)} boxes (micro-tiles)")
            RECALL["yolo"] += len(yolo_boxes)
            yolo_boxes_total += len(yolo_boxes)

            before = len(yolo_boxes)
            #yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.50)
            yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.35)
            yolo_boxes_total_nms += len(yolo_boxes)

            if len(yolo_boxes) != before:
                logging.info(f"[TILE] NMS reduced boxes {before} -> {len(yolo_boxes)}")
            

            DROP = {"min_area": 0, "too_full": 0, "none": 0, "poly_none": 0}
            for (x1, y1, x2, y2, yconf) in yolo_boxes:

                bw = x2 - x1
                bh = y2 - y1
                tile_h, tile_w = tile_rgb.shape[:2]

                logging.info(
                    f"[TILE] YOLO raw: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                    f"w={bw}, h={bh}, conf={yconf:.3f}, tile_w={tile_w}, tile_h={tile_h}"
                )



                #PAD = int(np.clip(0.08 * min(bw, bh), 2, 14))  # 8% of min dim, clamp 2..12
                PAD = int(np.clip(0.10 * min(bw, bh), 6, 32))  # was max 14
                x1p = max(0, x1 - PAD); y1p = max(0, y1 - PAD)
                x2p = min(tile_rgb.shape[1] - 1, x2 + PAD)
                y2p = min(tile_rgb.shape[0] - 1, y2 + PAD)


                if is_mega_or_strip_box(x1p, y1p, x2p, y2p, tile_w, tile_h, yconf):
                    logging.warning(f"[TILE] Dropped mega/strip box conf={yconf:.3f}")
                    continue


                pbw = x2p - x1p
                pbh = y2p - y1p

                logging.info(
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

                    logging.warning("[TILE] Huge YOLO box -> splitting")
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
                    logging.warning(f"[TILE] Very large padded box vs tile: {pad_area/tile_area:.2f}")

                logging.info(
                    f"[TILE] box_area raw={raw_area}, padded={pad_area}, "
                    f"padded/tile={pad_area/max(tile_area,1):.3f}"
                )

                

                for (sx1, sy1, sx2, sy2) in sub_boxes:

                    sbw = sx2 - sx1
                    sbh = sy2 - sy1
                    sbox_area = max(1, sbw * sbh)

                    sbox = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

                    # masks, scores, _ = predictor.predict(
                    #     box=sbox,
                    #     multimask_output=True
                    # )

                    # min_area = max(20, int(0.002 * sbox_area))
                    # max_fill = adaptive_max_fill(sbox_area, sbw, sbh, yconf)

                    # best = None
                    # best_score = -1.0

                    # for m, s in zip(masks, scores):
                    #     area = int(m.sum())
                    #     fill_ratio = area / sbox_area

                    #     if area < min_area:
                    #         DROP["min_area"] += 1
                    #         continue

                    #     # if fill_ratio > max_fill and not (yconf >= 0.80 and float(s) >= 0.85):
                    #     #     continue

                    #     if fill_ratio > max_fill + 0.03 and not (yconf >= 0.75):
                    #         DROP["too_full"] += 1
                    #         continue

                    #     if s > best_score:
                    #         best = m
                    #         best_score = float(s)

                    # if best is None:
                    #     DROP["none"] += 1
                    #     RECALL["sam_drop"] += 1
                    #     continue

                    # RECALL["sam_ok"] += 1


                    # # --- clip SAM mask to sub-box (LOOSER, with minimum context)
                    # best = best.astype(np.uint8)

                    # MIN_PAD_PX = 16          # <-- key: never allow tiny padding
                    # MAX_PAD_PX = 64
                    # CLIP_EXTRA_PX = 12       # <-- extra margin only for clipping (prevents chopping corners)

                    # pad = int(0.08 * min(sbw, sbh))
                    # pad = max(MIN_PAD_PX, min(MAX_PAD_PX, pad))

                    # # prompt box is (sx1..sx2); clip box is slightly larger than prompt+pad
                    # x1c = max(0, int(sx1 - pad - CLIP_EXTRA_PX))
                    # y1c = max(0, int(sy1 - pad - CLIP_EXTRA_PX))
                    # x2c = min(best.shape[1]-1, int(sx2 + pad + CLIP_EXTRA_PX))
                    # y2c = min(best.shape[0]-1, int(sy2 + pad + CLIP_EXTRA_PX))

                    # mask_clip = np.zeros_like(best, dtype=np.uint8)
                    # mask_clip[y1c:y2c+1, x1c:x2c+1] = 1
                    

                    # best_preclip = best.copy()
                    # best_area = int(best.sum())

                    # best = best & mask_clip
                    # best_clipped_area = int(best.sum())

                    # # 🔒 one-line area-loss guard (prevents chopping container roofs)
                    # if best_clipped_area < 0.85 * best_area:
                    #     best = best_preclip
                    #     best_clipped_area = best_area

                    # if best_clipped_area < min_area:
                    #     continue

                    # poly = mask_to_polygon(
                    #     {"segmentation": best},
                    #     t_transform
                    # )

                    union_mask, best_score, min_area, (ex1, ey1, ex2, ey2) = predict_mask_twopass_union(
                        predictor,
                        sx1, sy1, sx2, sy2,
                        yconf=yconf,
                        expand_scale=1.28   # 1.20–1.35 is typical; 1.28 is a good start for roofs
                    )

                    if union_mask is None:
                        DROP["none"] += 1
                        RECALL["sam_drop"] += 1
                        continue

                    RECALL["sam_ok"] += 1

                    best_preclip = union_mask.copy()
                    best_area = int(union_mask.sum())

                    # --- loose clip around EXPANDED box (not the tight one)
                    MIN_PAD_PX = 16
                    MAX_PAD_PX = 96          # allow a bit more now
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

                    mask_clip = np.zeros_like(union_mask, dtype=np.uint8)
                    mask_clip[y1c:y2c+1, x1c:x2c+1] = 1

                    best = union_mask & mask_clip
                    best_clipped_area = int(best.sum())

                    # 🔒 area-loss guard
                    if best_clipped_area < 0.85 * best_area:
                        best = best_preclip
                        best_clipped_area = best_area


                    box_area = max(1, (x2p - x1p) * (y2p - y1p))

                    if best_clipped_area < min_area:
                        DROP["min_area"] += 1
                        continue

                    poly = mask_to_polygon({"segmentation": best}, t_transform)
                    if poly is None:
                        DROP["poly_none"] += 1
                        continue

                    final_conf = 0.6 * best_score + 0.4 * yconf
                    all_buildings.append((poly, DEFAULT_HEIGHT, final_conf))


                    # if not is_airport_building(poly, relaxed=True):
                    #     continue

            # ✅ put this here: end of ONE TILE
            logging.info(
                f"[TILE DROP] tb={tb} boxes={len(yolo_boxes)} "
                f"min_area={DROP['min_area']} too_full={DROP['too_full']} "
                f"none={DROP['none']} poly_none={DROP['poly_none']}"
            )




    else:
        img_rgb = img_rgb.astype(np.uint8)
        predictor.set_image(img_rgb)
        #yolo_boxes = detect_building_boxes(img_rgb)
        yolo_boxes = detect_building_boxes_ensemble(img_rgb, debug=True, debug_tag="aoi_ens")
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

            logging.info(f"[AOI] YOLO quadrant counts TL/TR/BL/BR={q}")

            if min(q) == 0:
                logging.info("[AOI] Sparse quadrant -> running ensemble recall booster")
                yolo_boxes = detect_building_boxes_ensemble(img_rgb, debug=True, debug_tag="aoi_ens_boost")


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
                        logging.warning(
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
                            imgsz=640
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

                        logging.info(f"[AOI] fallback microtile boxes={len(yolo_boxes)}")
                        if not yolo_boxes:
                            logging.warning("[AOI] fallback microtiles returned 0 → keeping original YOLO boxes")
                            yolo_boxes = orig_yolo_boxes

        logging.info(f"YOLO Detected {len(yolo_boxes)} boxes")

        before = len(yolo_boxes)
        #yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.50)
        yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.45)

        RECALL["yolo"] += len(yolo_boxes)

        yolo_boxes_total_nms += len(yolo_boxes)
        if len(yolo_boxes) != before:
            logging.info(f"[AOI] NMS reduced boxes {before} -> {len(yolo_boxes)}")

        for (x1, y1, x2, y2, yconf) in yolo_boxes:

            bw = x2 - x1
            bh = y2 - y1
            H, W = img_rgb.shape[:2]

            logging.info(
                f"[AOI] YOLO raw: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                f"w={bw}, h={bh}, conf={yconf:.3f}, aoi_w={W}, aoi_h={H}"
            )


            box_frac = (bw * bh) / max(1, W * H)
            if box_frac > 0.5:
                PAD = 0
            else:
                #PAD = int(np.clip(0.08 * min(bw, bh), 2, 12))
                PAD = int(np.clip(0.10 * min(bw, bh), 6, 28))
            x1p = max(0, x1 - PAD)
            y1p = max(0, y1 - PAD)
            x2p = min(img_rgb.shape[1] - 1, x2 + PAD)
            y2p = min(img_rgb.shape[0] - 1, y2 + PAD)

            # ✅ Mega/strip drop for AOI
            if is_mega_or_strip_box_aoi(x1p, y1p, x2p, y2p, W, H, yconf):
                logging.warning(f"[AOI] Dropped mega/strip box conf={yconf:.3f}")
                continue

            if len(yolo_boxes) > 1 and is_box_too_big(x1p, y1p, x2p, y2p, W, H, BIG_BOX_FRAC_AOI):
                logging.warning(f"[AOI] Rejecting huge YOLO padded box frac>{BIG_BOX_FRAC_AOI}")
                continue



            pbw = x2p - x1p
            pbh = y2p - y1p

            logging.info(
                f"[AOI] YOLO padded: x1p={x1p}, y1p={y1p}, x2p={x2p}, y2p={y2p}, "
                f"pw={pbw}, ph={pbh}, pad={PAD}"
            )


            raw_area = max(0, bw) * max(0, bh)
            pad_area = max(0, pbw) * max(0, pbh)
            aoi_area = W * H

            logging.info(
                f"[AOI] box_area raw={raw_area}, padded={pad_area}, "
                f"padded/aoi={pad_area/max(aoi_area,1):.3f}"
            )

            # also ensure valid
            if x2p <= x1p + 2 or y2p <= y1p + 2:
                logging.warning(f"Skipping invalid padded box [NON TILED ELSE BRANCH]: ({x1p},{y1p})-({x2p},{y2p})")
                continue

            box = np.array([x1p, y1p, x2p, y2p], dtype=np.float32)

            # masks, scores, _ = predictor.predict(
            #     box=box,
            #     multimask_output=True
            # )

            # box_area = (x2p - x1p) * (y2p - y1p)

            # min_area = max(20, int(0.002 * box_area))     # instead of hard 80
            # #max_fill = 0.95 if box_area < 5000 else 0.90   # allow fuller masks on small boxes
            # max_fill = adaptive_max_fill(box_area, bw, bh, yconf)

            # best = None
            # best_score = -1.0

            # for m, s in zip(masks, scores):
            #     area = int(m.sum())

            #     logging.info(
            #         f"SAM mask: area={area}, box_area={box_area}, "
            #         f"min_area={min_area}, max_allowed={int(max_fill*box_area)}"
            #     )
            #     if area < min_area:
            #         continue
                
            #     fill_ratio = area / max(1, box_area)
            #     too_full = fill_ratio > max_fill


            #     # allow very full masks if SAM+YOLO confident
            #     if too_full and not (yconf >= 0.80 and float(s) >= 0.85 and fill_ratio <= 0.995):
            #         continue

            #     s = float(s)
            #     if s > best_score:
            #         best = m
            #         best_score = s

            # if best is None:
            #     logging.info(
            #         f"SAM: no mask passed filters (min_area={min_area}, max_fill={max_fill}, "
            #         f"box_area={box_area}, yconf={yconf:.3f})"
            #     )
            #     sam_polys = []
            #     RECALL["sam_drop"] += 1
            # else:
            #     RECALL["sam_ok"] += 1
            #     #poly = mask_to_polygon({"segmentation": best.astype(np.uint8)}, transform)
            #     #sam_polys = [(poly, best_score)] if poly is not None else []


            #     best = best.astype(np.uint8)
            #     best_preclip = best.copy()   # <-- add this
            #     best_area = int(best.sum())
            #     fill = best_area / max(box_area, 1)
            #     logging.info(f"SAM best(pre-clip): area={best_area}, box_area={box_area}, fill={fill:.3f}, score={best_score:.3f}")


            #     box_mask = np.zeros(best.shape, dtype=np.uint8)

            #     MIN_PAD_PX = 16
            #     MAX_PAD_PX = 64
            #     CLIP_EXTRA_PX = 12

            #     pad = int(0.08 * min(bw, bh))
            #     pad = max(MIN_PAD_PX, min(MAX_PAD_PX, pad))

            #     y1c = max(0, int(y1p - pad - CLIP_EXTRA_PX))
            #     x1c = max(0, int(x1p - pad - CLIP_EXTRA_PX))
            #     y2c = min(best.shape[0] - 1, int(y2p + pad + CLIP_EXTRA_PX))
            #     x2c = min(best.shape[1] - 1, int(x2p + pad + CLIP_EXTRA_PX))

            #     box_mask[y1c:y2c+1, x1c:x2c+1] = 1
            #     best = (best & box_mask)
            #     best_clipped_area = int(best.sum())

            #     # 🔒 one-line area-loss guard (prevents half roof loss)
            #     if best_clipped_area < 0.85 * best_area:
            #         best = best_preclip
            #         best_clipped_area = best_area  # keep stats consistent

            #     kept_ratio = best_clipped_area / max(best_area, 1)
            #     logging.info(f"SAM clipped: area={best_clipped_area}, kept_ratio={kept_ratio:.3f}")

            #     if kept_ratio < 0.85:
            #         logging.info(f"SAM clip too aggressive (kept_ratio={kept_ratio:.3f}) -> relaxing clip")
            #         # relax only the clip box (not YOLO)
            #         relax = 24  # px
            #         y1c = max(0, y1c - relax); x1c = max(0, x1c - relax)
            #         y2c = min(best.shape[0]-1, y2c + relax); x2c = min(best.shape[1]-1, x2c + relax)

            #         box_mask = np.zeros(best.shape, dtype=np.uint8)
            #         box_mask[y1c:y2c+1, x1c:x2c+1] = 1
            #         best = best & box_mask

            # --- Two-pass SAM (tight + expanded) to reduce half-roof clipping ---
            union_mask, best_score, min_area, (ex1, ey1, ex2, ey2) = predict_mask_twopass_union(
                predictor,
                x1p, y1p, x2p, y2p,
                yconf=yconf,
                expand_scale=1.28
            )

            if union_mask is None:
                logging.info(
                    f"[AOI] SAM: no mask passed filters (two-pass) "
                    f"(box_area={(x2p-x1p)*(y2p-y1p)}, yconf={yconf:.3f})"
                )
                sam_polys = []
                RECALL["sam_drop"] += 1
            else:
                RECALL["sam_ok"] += 1

                best = union_mask.astype(np.uint8)
                best_preclip = best.copy()
                best_area = int(best.sum())

                # --- Clip around EXPANDED box (loose), with area-loss guard ---
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

                # 🔒 area-loss guard
                if best_clipped_area < 0.85 * best_area:
                    best = best_preclip
                    best_clipped_area = best_area
                box_area = max(1, (x2p - x1p) * (y2p - y1p))

                # IMPORTANT: keep your existing downstream indentation unchanged

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
                            aoi_area_m2 = (W * H) * (abs(transform.a) ** 2)
                            poly_frac = poly.area / max(aoi_area_m2, 1e-6)

                            if poly_frac > 0.85:
                                logging.info(f"[AOI] Giant-edge poly (poly_frac={poly_frac:.3f}, box_frac={box_frac:.3f}) -> fallback to SAM auto")

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
                        logging.info(
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
                                ww = maxx - minx
                                hh = maxy - miny
                                aspect = max(ww, hh) / max(1e-6, min(ww, hh))
                                if aspect > 12.0:
                                    sam_polys = []
                                else:
                                    sam_polys = [(poly, best_score)]


            # Build YOLO bbox polygon in world coords
            x1w, y1w = rasterio.transform.xy(transform, y1p, x1p, offset="center")
            x2w, y2w = rasterio.transform.xy(transform, y2p, x2p, offset="center")

            minx = min(x1w, x2w); maxx = max(x1w, x2w)
            miny = min(y1w, y2w); maxy = max(y1w, y2w)
            yolo_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


            # --- OSM fallback only if SAM failed for this YOLO box
            if not sam_polys and osm is not None:
                added = False
                for _, row in osm.iterrows():
                    osm_poly = (
                        gpd.GeoSeries([row.geometry], crs=osm.crs)
                        .to_crs(raster_crs)
                        .iloc[0]
                    )

                    # Only accept OSM that intersects this YOLO bbox
                    if osm_poly.intersects(yolo_poly) and not overlaps_any(osm_poly, all_buildings):
                        all_buildings.append((osm_poly, DEFAULT_HEIGHT, 0.85))
                        added = True
                        break

                if not added:
                    logging.info("YOLO hit but no SAM/OSM geometry → skipping footprint")


            for poly, conf in sam_polys:
                #final_conf = min(1.0, 0.6 * conf + 0.4 * yconf)
                if yconf > 0.9:
                    final_conf = max(conf, yconf) * 0.9
                else:
                    final_conf = 0.6 * conf + 0.4 * yconf

                all_buildings.append((poly, DEFAULT_HEIGHT, final_conf))

            # If microtile fallback produced many overlapping candidates,
            # pick the single best building footprint to avoid unions bleeding into land.
            # if len(all_buildings) >= 3:
            #     H, W = img_rgb.shape[:2]
            #     aoi_b = aoi_bounds_from_transform(transform, H, W)

            #     cands = [(p, c) for (p, _, c) in all_buildings]
            #     best_poly = pick_best_building(cands, aoi_bounds=aoi_b)

            #     if best_poly is not None:
            #         all_buildings = [(best_poly, DEFAULT_HEIGHT, 0.85)]


    
    

    
    logging.info(
        f"[AOI YOLO SUMMARY] raw_boxes={yolo_boxes_total}, after_nms={yolo_boxes_total_nms}"
    )
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

    #sam_buildings = merge_buildings(all_buildings, iou_thresh=0.40)

    #sam_buildings = merge_buildings(all_buildings, iou_thresh=0.60)

    # ✅ remove tile-duplicate buildings that survive IoU merge
    #sam_buildings = suppress_duplicates_by_overlap(sam_buildings, overlap_min=0.60)
    #sam_buildings = resolve_overlaps_by_subtraction(sam_buildings, min_frac=0.03, buffer_m=0.0)

    # 1) First: stitch tile seam fragments (touching / nearly-touching)
    # buffer ~ 1-2 pixels in meters (avoid merging different buildings)
    #stitch_buf = max(0.6, 2.0 * pixel_size_m)

    sam_buildings = merge_buildings(all_buildings, iou_thresh=0.55)

    stitch_buf = max(0.2, 0.6 * pixel_size_m)
    sam_buildings = stitch_touching(sam_buildings, buffer_m=stitch_buf)

    sam_buildings = suppress_duplicates_by_overlap(sam_buildings, overlap_min=0.65)


    # 4) Overlap subtraction LAST, and less aggressive (prevents “half roof disappears”)
    #sam_buildings = resolve_overlaps_by_subtraction(sam_buildings, min_frac=0.06, buffer_m=0.0)
    #sam_buildings = resolve_overlaps_by_subtraction(sam_buildings, min_frac=0.25, min_keep_ratio=0.85)


    refined = []

    for poly, h, conf in sam_buildings:
        replaced = False

        if osm is not None and conf < 0.65:
            for _, row in osm.iterrows():
                osm_poly = (
                    gpd.GeoSeries([row.geometry], crs=osm.crs)
                    .to_crs(raster_crs)
                    .iloc[0]
                )
                if poly.intersection(osm_poly).area / poly.area > 0.4:
                    refined.append((osm_poly, h, max(conf, 0.8)))
                    replaced = True
                    break

        if not replaced:
            refined.append((poly, h, conf))

    sam_buildings = refined

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



    logging.info(f"final total merged buildings : {len(sam_buildings)}")

    if not sam_buildings and all_buildings:
        logging.warning("Merge removed all buildings → using raw SAM outputs")
        sam_buildings = all_buildings

    else:
        logging.info(f"SAM detected {len(sam_buildings)} buildings")

    buildings = list(sam_buildings)  # ALWAYS keep SAM results

    if osm is not None:
        for geom in osm.geometry:
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)

            poly = (
                gpd.GeoSeries([geom], crs=osm.crs)
                .to_crs(raster_crs)
                .iloc[0]
            )

            if not is_airport_building(poly):
                #confidence *= 0.5
                continue

             # only skip if overlaps SAM
            if any(poly.intersects(p) for p,_,_ in sam_buildings):
                continue

            buildings.append((poly, DEFAULT_HEIGHT, 0.95))

    if not buildings:
        logging.warning("No buildings detected")
        #return None

    
    for poly, _, conf in buildings:
        height = None

        # 1️⃣ OSM height if available
        height = find_matching_osm_height(poly, osm, raster_crs)

        # 2️⃣ Shadow height if SAM confident
        shadow_info = None

        ## disabling height calculation for now so that we can focus on footprinting first, but after footprint height is next priority
        
        if height is None and conf >= 0.75:
            shadow_info = estimate_height_from_shadow(poly, img_rgb, transform, raster_crs)
            if shadow_info:
                height = shadow_info["height"]


        # 3️⃣ Fallback
        if height is None:
            logging.debug("FALLING BACK TO DEAFULT HEIGHT")
            height = DEFAULT_HEIGHT
            #conf = min(conf, 0.5)


        print(f"CONF={conf:.2f} → HEIGHT={height}")

        #final_buildings.append((poly, height, conf,shadow_info))
        final_buildings.append((poly, height, conf,shadow_info))

    #polys, heights, confidences = zip(*final_buildings)
    polys = [b[0] for b in final_buildings]
    heights = [b[1] for b in final_buildings]
    confidences = [b[2] for b in final_buildings]
    shadows = [b[3] for b in final_buildings]  # may be None

    #logging.info(f"First Ploy Centroid: polys[0].centroid.x = {polys[0].centroid.x} , polys[0].centroid.y = {polys[0].centroid.y}")
    if polys:
        logging.info(
            f"First Poly Centroid: x={polys[0].centroid.x:.2f}, "
            f"y={polys[0].centroid.y:.2f}"
        )
    else:
        logging.warning("No polygons available for centroid logging")



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
        logging.warning("No features generated → returning empty AOI result")

        return {
            "geojson": None,
            "glb": None,
            "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else [],
            "reason": "no_buildings_detected"
        }

    # gdf = gpd.GeoDataFrame.from_features(features, crs=raster_crs).to_crs(epsg=4326)

    
    # gdf["geometry"] = gdf.geometry.translate(xoff=0.0, yoff=0.0)
    # gdf.to_file(OUT_GEOJSON, driver="GeoJSON")

    # export_glb(
    #     [flat_roof(poly, height) for poly, height, _, _ in final_buildings],
    #     OUT_GLB
    # )




    # return {
    #     "geojson": "output/buildings.geojson",
    #     "glb": "output/aoi_buildings.glb",
    #     "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else []
    # }

    gdf = gpd.GeoDataFrame.from_features(features, crs=raster_crs).to_crs(epsg=4326)

    # 🔒 STRICT AOI GUARD (fix outside-AOI buildings)
    gdf = clip_gdf_to_aoi_4326(gdf, bounds)

    if gdf.empty:
        logging.warning("All features clipped out by AOI guard -> returning empty")
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

    # kept_buildings = []
    # for poly, height, conf, shadow in final_buildings:
    #     poly_4326 = gpd.GeoSeries([poly], crs=raster_crs).to_crs(epsg=4326).iloc[0]
    #     if not poly_4326.intersects(aoi_poly_4326):
    #         continue
    #     kept_buildings.append((poly, height, conf, shadow))

    # export_glb(
    #     [flat_roof(poly, height) for poly, height, _, _ in kept_buildings],
    #     out_glb_abs
    # )

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

    logging.warning(
        f"[RECALL] YOLO={RECALL['yolo']} | SAM_OK={RECALL['sam_ok']} | SAM_DROP={RECALL['sam_drop']} | LOSS={(RECALL['sam_drop']/max(1,RECALL['yolo'])):.2%}"
    )

    return {
        "geojson": out_geojson_rel,
        "glb": out_glb_rel,
        "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else []
    }