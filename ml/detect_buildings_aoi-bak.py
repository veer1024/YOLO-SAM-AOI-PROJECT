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
from ml.yolo_detector import detect_building_boxes
from rasterio.transform import Affine
from ml.yolo_detector import detect_building_boxes_microtiles
from shapely.geometry import box as shp_box

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
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

# --------------------------------------------------
# TILING CONFIG
# --------------------------------------------------
TILE_SIZE_M = 256        # fewer tiles
#TILE_OVERLAP = 0.10     # enough continuity
TILE_OVERLAP = 0.35
#MIN_TILING_AREA_M2 = 120 * 120  # below this → no tiling
MIN_TILING_AREA_M2 = 500 * 500  # below this → no tiling



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

DEFAULT_HEIGHT = 12.0
MIN_AREA_PX = 600        # slightly higher for airports
MAX_AOI_RATIO = 0.08     # reject giant flat regions

#IMAGE_ACQ_TIME = "2023-08-01T06:48:46.620"
IMAGE_ACQ_TIME = "2022-11-24T05:59:54.545+00:00"



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
decoder_ckpt="SAM_TRAINED_MODEL/sam_decoder_finetuned.pth.epoch40.pth"
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




def estimate_height_from_shadow(poly, img_rgb,transform, raster_crs):
    """
    Estimate building height from shadow length using solar geometry.
    poly        : building polygon in raster CRS
    img_rgb     : RGB numpy image (H,W,3)
    raster_crs  : CRS of raster (used only for reprojection safety)

    Returns height in meters or None.
    """

    #logging.info("ESTIMATE HEIGHT FROM SHADOW CALLED")
    logging.info("Shadow height estimation invoked")

    try:
        # --------------------------------------------------
        # 1️⃣ Centroid → lat/lon
        # --------------------------------------------------
        gdf = gpd.GeoSeries([poly], crs=raster_crs).to_crs(epsg=4326)
        lon, lat = gdf.iloc[0].centroid.xy[0][0], gdf.iloc[0].centroid.xy[1][0]

        ts = datetime.fromisoformat(IMAGE_ACQ_TIME)
        loc = LocationInfo(latitude=lat, longitude=lon)

        print("Solar elevation:", elevation(loc.observer, ts))

        sun_elev = elevation(loc.observer, ts)
        sun_az = azimuth(loc.observer, ts)

        # Too low sun = unreliable
        if sun_elev < 12:
            return None

        # Shadow direction (opposite sun)
        theta = math.radians((sun_az + 180) % 360)
        #dx, dy = math.cos(theta), math.sin(theta)
        dx = math.sin(theta)
        dy = -math.cos(theta)


        # --------------------------------------------------
        # 2️⃣ Rasterize building mask
        # --------------------------------------------------
        h, w = img_rgb.shape[:2]

        from rasterio.features import rasterize
        #from rasterio.transform import from_bounds

        minx, miny, maxx, maxy = poly.bounds
        #transform = from_bounds(minx, miny, maxx, maxy, w, h)

        building_mask = rasterize(
            [(poly, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        if building_mask.sum() < 50:
            return None

        # --------------------------------------------------
        # 3️⃣ Extract building edge pixels
        # --------------------------------------------------
        edges = cv2.Canny(building_mask * 255, 50, 150)
        ys, xs = np.where(edges > 0)

        if len(xs) < 20:
            return None

        # --------------------------------------------------
        # 4️⃣ Shadow probing along sun direction
        # --------------------------------------------------
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        pixel_size_m = abs(transform.a)

        max_shadow_px = 0
        shadow_pixels = []

        for x0, y0 in zip(xs, ys):
            max_steps = int(40 / pixel_size_m)  # max 40 meters
            for step in range(1, max_steps):
                x = int(x0 + dx * step)
                y = int(y0 + dy * step)

                if x < 0 or y < 0 or x >= w or y >= h:
                    break

                # Stop when shadow ends (bright ground)
                if gray[y, x] > np.percentile(gray, 50):
                    break

                #max_shadow_px = max(max_shadow_px, step)
                max_shadow_px = max(max_shadow_px, step)

                # store shadow pixel world coords
                wx, wy = rasterio.transform.xy(transform, y, x, offset="center")
                shadow_pixels.append((wx, wy))

        if max_shadow_px < 6:
            return None

        if len(shadow_pixels) < 10:
            return None

        shadow_poly = Polygon(shadow_pixels).convex_hull
        if not shadow_poly.is_valid or shadow_poly.area < 1.0:
            return None

        # --------------------------------------------------
        # 5️⃣ Height calculation
        # --------------------------------------------------
        shadow_len_m = max_shadow_px * pixel_size_m
        height = shadow_len_m * math.tan(math.radians(sun_elev))

        if not np.isfinite(height):
            return None

        return {
            "height": float(np.clip(height, 3.0, 80.0)),
            "shadow_length_m": shadow_len_m,
            "sun_azimuth": sun_az,
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




def get_osm_buildings(bounds):
    """
    Fetch OSM buildings with local cache fallback.
    """
    # 1️⃣ Try local cache first
    return None
    logging.info(f"osmnx version: {ox.__version__}")
    logging.info(f"max_query_area_size: {ox.settings.max_query_area_size}")
    logging.info(f"use_cache: {ox.settings.use_cache}, cache_folder: {ox.settings.cache_folder}")
    cached = load_osm_cache(bounds)
    if cached is not None:
        return cached

    west, south, east, north = bounds

    fallback = load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857")
    if fallback is not None and not fallback.empty:
        return fallback
    else:
        logging.info(f"NO CACHE RESULT FOUND, OSM API WILL BE CALLED ")

    try:
        # OSMnx 2.x expects bbox=(west, south, east, north)
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
    except Exception as e:
        logging.warning("OSM API failed: %s", e)
        fallback = load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857")
        if fallback is not None and not fallback.empty:
            return fallback
        return None

    if gdf is None or gdf.empty:
        logging.warning("OSM returned empty result")
        fallback = load_osm_cache_best_effort(bounds, raster_crs="EPSG:3857")
        if fallback is not None and not fallback.empty:
            return fallback
        return None

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    # 2️⃣ Save to cache
    save_osm_cache(gdf, bounds)

    return gdf



def mask_to_polygon(mask, transform, offx=0, offy=0, min_contour_area_px=30):
    seg = (mask["segmentation"] > 0).astype(np.uint8)

    # Clean mask a bit (fills tiny gaps, removes speckles)
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)

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
    bounds = _ensure_bounds_4326(bounds)


    # Stabilize float jitter (important if frontend sends slightly different bounds)
    bounds = tuple(round(float(b), 7) for b in bounds)

    run_id = uuid.uuid4().hex
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
    logging.info(f"AOI transform origin: transform.c =  {transform.c} , transform.f = {transform.f} ")

    with rasterio.open(aoi_path) as src:
        img = src.read().transpose(1, 2, 0)

    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img

    h, w = img_rgb.shape[:2]
    pixel_area_m2 = abs(transform.a * transform.e)
    aoi_area_m2 = h * w * pixel_area_m2

    osm = get_osm_buildings(bounds)
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


            yolo_boxes = detect_building_boxes_microtiles(
                tile_rgb,
                tile_size=256,
                overlap=0.25,
                conf=0.05,
                imgsz=640,
                debug=True,
                debug_tag="tile"
            )
            logging.info(f"YOLO Detected {len(yolo_boxes)} boxes (micro-tiles)")

            before = len(yolo_boxes)
            yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.45)
            if len(yolo_boxes) != before:
                logging.info(f"[TILE] NMS reduced boxes {before} -> {len(yolo_boxes)}")
            

            for (x1, y1, x2, y2, yconf) in yolo_boxes:

                bw = x2 - x1
                bh = y2 - y1
                tile_h, tile_w = tile_rgb.shape[:2]

                logging.info(
                    f"[TILE] YOLO raw: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                    f"w={bw}, h={bh}, conf={yconf:.3f}, tile_w={tile_w}, tile_h={tile_h}"
                )



                PAD = int(np.clip(0.08 * min(bw, bh), 2, 12))  # 8% of min dim, clamp 2..12
                x1p = max(0, x1 - PAD); y1p = max(0, y1 - PAD)
                x2p = min(tile_rgb.shape[1] - 1, x2 + PAD)
                y2p = min(tile_rgb.shape[0] - 1, y2 + PAD)


                pbw = x2p - x1p
                pbh = y2p - y1p

                logging.info(
                    f"[TILE] YOLO padded: x1p={x1p}, y1p={y1p}, x2p={x2p}, y2p={y2p}, "
                    f"pw={pbw}, ph={pbh}, pad={PAD}"
                )

                if len(yolo_boxes) > 1 and is_box_too_big(x1p, y1p, x2p, y2p, tile_w, tile_h, BIG_BOX_FRAC_TILE):
                    logging.warning(f"[TILE] Rejecting huge YOLO padded box frac>{BIG_BOX_FRAC_TILE}")
                    continue


                raw_area = max(0, bw) * max(0, bh)
                pad_area = max(0, pbw) * max(0, pbh)
                tile_area = tile_w * tile_h

                if pad_area / max(tile_area, 1) > 0.35:
                    logging.warning(f"[TILE] Very large padded box vs tile: {pad_area/tile_area:.2f}")

                logging.info(
                    f"[TILE] box_area raw={raw_area}, padded={pad_area}, "
                    f"padded/tile={pad_area/max(tile_area,1):.3f}"
                )

                box = np.array([x1p, y1p, x2p, y2p], dtype=np.float32)

                if x2p <= x1p + 2 or y2p <= y1p + 2:
                    logging.warning(f"Skipping invalid padded box [TILED IF BRANCH]: ({x1p},{y1p})-({x2p},{y2p})")
                    continue

                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True
                )

                box_area = (x2p - x1p) * (y2p - y1p)

                min_area = max(20, int(0.002 * box_area))     # instead of hard 80
                max_fill = 0.95 if box_area < 5000 else 0.90   # allow fuller masks on small boxes
                best = None
                best_score = -1.0



                for m, s in zip(masks, scores):
                    area = int(m.sum())

                    logging.info(
                        f"SAM mask: area={area}, box_area={box_area}, "
                        f"min_area={min_area}, max_allowed={int(max_fill*box_area)}"
                    )
                    if area < min_area:
                        continue
                    if area > max_fill * box_area:
                        continue

                    s = float(s)
                    if s > best_score:
                        best = m
                        best_score = s

                if best is None:
                    logging.info(
                        f"SAM: no mask passed filters (min_area={min_area}, max_fill={max_fill}, "
                        f"box_area={box_area}, yconf={yconf:.3f})"
                    )
                    if osm is not None and yconf >= 0.6:
                        x1w, y1w = rasterio.transform.xy(t_transform, y1p, x1p, offset="center")
                        x2w, y2w = rasterio.transform.xy(t_transform, y2p, x2p, offset="center")

                        minx = min(x1w, x2w); maxx = max(x1w, x2w)
                        miny = min(y1w, y2w); maxy = max(y1w, y2w)
                        yolo_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

                        for _, row in osm.iterrows():
                            osm_poly = (
                                gpd.GeoSeries([row.geometry], crs=osm.crs)
                                .to_crs(t_crs)
                                .iloc[0]
                            )
                            if osm_poly.intersects(yolo_poly) and not overlaps_any(osm_poly, all_buildings):
                                all_buildings.append((osm_poly, DEFAULT_HEIGHT, 0.85))
                                break
                    continue

                # --- HARD CLIP mask to YOLO box
                best = best.astype(np.uint8)

                best_area = int(best.sum())
                fill = best_area / max(box_area, 1)
                logging.info(f"SAM best(pre-clip): area={best_area}, box_area={box_area}, fill={fill:.3f}, score={best_score:.3f}")

                box_mask = np.zeros(best.shape, dtype=np.uint8)
                box_mask[y1p:min(y2p+1, box_mask.shape[0]), x1p:min(x2p+1, box_mask.shape[1])] = 1
                best = (best & box_mask)

                best_clipped_area = int(best.sum())
                logging.info(f"SAM clipped: area={best_clipped_area}, kept_ratio={best_clipped_area/max(best_area,1):.3f}")

                if int(best.sum()) < min_area:
                    continue

                poly = mask_to_polygon({"segmentation": best}, t_transform)
                if poly is None:
                    continue


                tile_h, tile_w = tile_rgb.shape[:2]
                box_frac = (bw * bh) / max(1, tile_w * tile_h)

                if ((x1p <= 1 or y1p <= 1 or x2p >= tile_w-2 or y2p >= tile_h-2) and box_frac > 0.85):
                    if poly.area > 0.5 * (tile_w * tile_h) * (abs(t_transform.a) ** 2):
                        logging.info("[TILE] Rejecting giant-edge poly (background bleed)")
                        continue


                minx, miny, maxx, maxy = poly.bounds
                ww = maxx - minx
                hh = maxy - miny
                aspect = max(ww, hh) / max(1e-6, min(ww, hh))
                logging.info(
                    f"POLY stats: area_m2={poly.area:.1f}, aspect={aspect:.2f}, "
                    f"bounds_w={ww:.1f}, bounds_h={hh:.1f}"
                )

                # --- reject runway/apron blobs
                if not is_airport_building(poly, relaxed=True):
                    logging.info("REJECT: airport_filter")
                    continue

                # optional but very effective guards (meters, since t_transform is 3857)
                if poly.area > 20000:
                    logging.info(f"REJECT: poly.area {poly.area:.1f} > 20000")
                    continue

                minx, miny, maxx, maxy = poly.bounds
                ww = maxx - minx
                hh = maxy - miny
                aspect = max(ww, hh) / max(1e-6, min(ww, hh))


                if aspect > 12.0:
                    logging.info(f"REJECT: aspect {aspect:.2f} > 12")
                    continue
                if (poly.area / (box_area * (abs(t_transform.a) ** 2))) > 0.65:
                    continue

                final_conf = 0.6 * best_score + 0.4 * yconf
                all_buildings.append((poly, DEFAULT_HEIGHT, final_conf))


    else:
        img_rgb = img_rgb.astype(np.uint8)
        predictor.set_image(img_rgb)
        yolo_boxes = detect_building_boxes(img_rgb)
        orig_yolo_boxes = list(yolo_boxes)

        # --- BIG BOX / TINY AOI fallback trigger
        H, W = img_rgb.shape[:2]
        if yolo_boxes:
            # compute largest box fraction
            max_frac = 0.0
            for (x1, y1, x2, y2, yconf) in yolo_boxes:
                frac = ((x2 - x1) * (y2 - y1)) / max(1, W * H)
                max_frac = max(max_frac, frac)

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
                            tile_size=320,
                            overlap=0.50,
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
        yolo_boxes = nms_boxes(yolo_boxes, iou_thr=0.50)
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
                PAD = int(np.clip(0.08 * min(bw, bh), 2, 12))
            x1p = max(0, x1 - PAD)
            y1p = max(0, y1 - PAD)
            x2p = min(img_rgb.shape[1] - 1, x2 + PAD)
            y2p = min(img_rgb.shape[0] - 1, y2 + PAD)

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

            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True
            )

            box_area = (x2p - x1p) * (y2p - y1p)

            min_area = max(20, int(0.002 * box_area))     # instead of hard 80
            max_fill = 0.95 if box_area < 5000 else 0.90   # allow fuller masks on small boxes

            best = None
            best_score = -1.0

            for m, s in zip(masks, scores):
                area = int(m.sum())

                logging.info(
                    f"SAM mask: area={area}, box_area={box_area}, "
                    f"min_area={min_area}, max_allowed={int(max_fill*box_area)}"
                )
                if area < min_area:
                    continue
                if area > max_fill * box_area:
                    continue

                s = float(s)
                if s > best_score:
                    best = m
                    best_score = s

            if best is None:
                logging.info(
                    f"SAM: no mask passed filters (min_area={min_area}, max_fill={max_fill}, "
                    f"box_area={box_area}, yconf={yconf:.3f})"
                )
                sam_polys = []
            else:
                #poly = mask_to_polygon({"segmentation": best.astype(np.uint8)}, transform)
                #sam_polys = [(poly, best_score)] if poly is not None else []


                best = best.astype(np.uint8)

                best_area = int(best.sum())
                fill = best_area / max(box_area, 1)
                logging.info(f"SAM best(pre-clip): area={best_area}, box_area={box_area}, fill={fill:.3f}, score={best_score:.3f}")

                box_mask = np.zeros(best.shape, dtype=np.uint8)
                box_mask[y1p:min(y2p+1, box_mask.shape[0]), x1p:min(x2p+1, box_mask.shape[1])] = 1
                best = (best & box_mask)

                best_clipped_area = int(best.sum())
                logging.info(f"SAM clipped: area={best_clipped_area}, kept_ratio={best_clipped_area/max(best_area,1):.3f}")

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
                            if fill_ratio > 0.55:
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
            if len(all_buildings) >= 3:
                H, W = img_rgb.shape[:2]
                aoi_b = aoi_bounds_from_transform(transform, H, W)

                cands = [(p, c) for (p, _, c) in all_buildings]
                best_poly = pick_best_building(cands, aoi_bounds=aoi_b)

                if best_poly is not None:
                    all_buildings = [(best_poly, DEFAULT_HEIGHT, 0.85)]


    
    

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

    sam_buildings = merge_buildings(all_buildings, iou_thresh=0.40)

    # Only stitch if there are clearly multiple buildings and they are high-quality
    # if len(sam_buildings) >= 3:
    #     sam_buildings = stitch_touching(sam_buildings, buffer_m=0.4)

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
        
        # if height is None and conf >= 0.75:
        #     shadow_info = estimate_height_from_shadow(poly, img_rgb, transform, raster_crs)
        #     if shadow_info:
        #         height = shadow_info["height"]


        # 3️⃣ Fallback
        if height is None:
            logging.debug("FALLING BACK TO DEAFULT HEIGHT")
            height = DEFAULT_HEIGHT
            conf = min(conf, 0.5)


        print(f"CONF={conf:.2f} → HEIGHT={height}")

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

    kept_buildings = []
    for poly, height, conf, shadow in final_buildings:
        poly_4326 = gpd.GeoSeries([poly], crs=raster_crs).to_crs(epsg=4326).iloc[0]
        if not poly_4326.intersects(aoi_poly_4326):
            continue
        kept_buildings.append((poly, height, conf, shadow))

    export_glb(
        [flat_roof(poly, height) for poly, height, _, _ in kept_buildings],
        out_glb_abs
    )

    return {
        "geojson": out_geojson_rel,
        "glb": out_glb_rel,
        "tiles": tiles if aoi_area_m2 > MIN_TILING_AREA_M2 else []
    }
