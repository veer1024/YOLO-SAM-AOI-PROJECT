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

TARGET_SIZE = 1024

# If a negative sample has no bbox info, you have 2 choices:
# - True  => keep sample, use full-image bbox (may be noisy for "not_a_building")
# - False => skip sample if bbox can't be determined (recommended)
NEGATIVE_FALLBACK_FULL_IMAGE_BBOX = False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def required_files_exist(*paths):
    """Return True only if all provided paths are non-empty and exist on disk."""
    for p in paths:
        if not p or not isinstance(p, str) or not os.path.exists(p):
            return False
    return True

def mask_to_bbox_xyxy(mask_u8: np.ndarray):
    """
    mask_u8: HxW uint8, nonzero pixels define region.
    returns [x0,y0,x1,y1] in pixel coords (0..W-1/H-1) or None if empty.
    """
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def ensure_resized_mask(mask_path: str, out_size: int = TARGET_SIZE):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[0] != out_size or m.shape[1] != out_size:
        m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_path, m)
    return m


def geo_bbox_to_px_bbox(geo_bbox, aoi_bounds, out_size: int = TARGET_SIZE):
    """
    geo_bbox: [minLng,minLat,maxLng,maxLat]
    aoi_bounds: [west,south,east,north]
    returns pixel bbox [x0,y0,x1,y1] in out_size image space
    """
    if not geo_bbox or not aoi_bounds:
        return None

    min_lng, min_lat, max_lng, max_lat = geo_bbox
    west, south, east, north = aoi_bounds

    # avoid div by zero
    if east <= west or north <= south:
        return None

    # Map lon->x, lat->y (note y inverted: north at y=0)
    def lng_to_x(lng):
        return int(round(((lng - west) / (east - west)) * (out_size - 1)))

    def lat_to_y(lat):
        return int(round(((north - lat) / (north - south)) * (out_size - 1)))

    x0 = lng_to_x(min_lng)
    x1 = lng_to_x(max_lng)
    y0 = lat_to_y(max_lat)  # max_lat is closer to north => smaller y
    y1 = lat_to_y(min_lat)  # min_lat is closer to south => larger y

    # normalize ordering
    x0, x1 = (min(x0, x1), max(x0, x1))
    y0, y1 = (min(y0, y1), max(y0, y1))

    x0 = clamp(x0, 0, out_size - 1)
    x1 = clamp(x1, 0, out_size - 1)
    y0 = clamp(y0, 0, out_size - 1)
    y1 = clamp(y1, 0, out_size - 1)

    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def geometry_to_geo_bbox(geom):
    """
    Supports Polygon / MultiPolygon. Returns [minLng,minLat,maxLng,maxLat] or None.
    """
    if not geom or "type" not in geom:
        return None

    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")

    def consume_ring(ring):
        nonlocal min_lng, min_lat, max_lng, max_lat
        for lng, lat in ring:
            min_lng = min(min_lng, lng)
            min_lat = min(min_lat, lat)
            max_lng = max(max_lng, lng)
            max_lat = max(max_lat, lat)

    t = geom["type"]
    if t == "Polygon":
        if geom.get("coordinates") and geom["coordinates"][0]:
            consume_ring(geom["coordinates"][0])
    elif t == "MultiPolygon":
        for poly in geom.get("coordinates", []):
            if poly and poly[0]:
                consume_ring(poly[0])
    else:
        return None

    if not np.isfinite([min_lng, min_lat, max_lng, max_lat]).all():
        return None
    if max_lng <= min_lng or max_lat <= min_lat:
        return None
    return [min_lng, min_lat, max_lng, max_lat]
def geometry_to_bbox_lnglat(geom):
    """
    Compute [minLng, minLat, maxLng, maxLat] from GeoJSON geometry
    """
    coords = []

    def collect(g):
        if g["type"] == "Polygon":
            for ring in g["coordinates"]:
                coords.extend(ring)
        elif g["type"] == "MultiPolygon":
            for poly in g["coordinates"]:
                for ring in poly:
                    coords.extend(ring)

    collect(geom)

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    return [min(lons), min(lats), max(lons), max(lats)]


def lnglat_bbox_to_pixel(bbox, aoi_bounds, size=1024):
    """
    Convert [minLng, minLat, maxLng, maxLat] → pixel bbox [x0,y0,x1,y1]
    """
    west, south, east, north = aoi_bounds
    min_lng, min_lat, max_lng, max_lat = bbox

    x0 = (min_lng - west) / (east - west) * (size - 1)
    x1 = (max_lng - west) / (east - west) * (size - 1)

    y0 = (north - max_lat) / (north - south) * (size - 1)
    y1 = (north - min_lat) / (north - south) * (size - 1)

    return [
        int(max(0, min(size - 1, x0))),
        int(max(0, min(size - 1, y0))),
        int(max(0, min(size - 1, x1))),
        int(max(0, min(size - 1, y1))),
    ]

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

            sample_id = meta.get("id")
            image_path = meta.get("image")

            if not sample_id or not image_path or not os.path.exists(image_path):
                continue

            # common fields possibly present in new metadata
            aoi_bounds = meta.get("aoi_bounds")  # [west,south,east,north]
            bbox_geo = meta.get("bbox")          # geo bbox from drawn polygon (new frontend)
            orig_bbox_geo = meta.get("original_bbox")

            
            # ---------------- NOT A BUILDING (NEGATIVE) ----------------
            if feedback_type == "not_a_building":
                mask_path = meta.get("mask")

                # Ensure we have a path to write to
                if not mask_path:
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_path = os.path.join(mask_dir, f"{sample_id}.png")

                # ALWAYS overwrite with empty mask for negatives (important!)
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                empty = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                cv2.imwrite(mask_path, empty)

                # Ensure correct size (should already be)
                _ = ensure_resized_mask(mask_path)

                # Compute bbox from geo info (NOT from mask)
                bbox_px = None

                if orig_bbox_geo and aoi_bounds:
                    bbox_px = geo_bbox_to_px_bbox(orig_bbox_geo, aoi_bounds)

                if bbox_px is None:
                    og = meta.get("original_geometry")
                    if og and aoi_bounds:
                        og_bbox_geo = geometry_to_geo_bbox(og)
                        if og_bbox_geo:
                            bbox_px = geo_bbox_to_px_bbox(og_bbox_geo, aoi_bounds)

                if bbox_px is None and NEGATIVE_FALLBACK_FULL_IMAGE_BBOX:
                    bbox_px = [0, 0, TARGET_SIZE - 1, TARGET_SIZE - 1]

                dataset.append({
                    "id": sample_id,
                    "type": feedback_type,
                    "image": image_path,
                    "mask": mask_path,      # ✅ empty target
                    "bbox": bbox_px,        # may be None
                    "metadata": meta_path
                })
                continue

            # ---------------- WRONG GEOMETRY ----------------
            if feedback_type == "wrong_geometry":
                # NEW format: meta["mask"] is the combined corrected mask
                mask_path = meta.get("mask")
                if not mask_path or not os.path.exists(mask_path):
                    logging.warning("Skipping invalid wrong_geometry sample %s (no mask)", sample_id)
                    continue

                m = ensure_resized_mask(mask_path)
                if m is None:
                    continue

                # Prompt bbox should come from ORIGINAL geometry (the wrong predicted one)
                bbox_px = None

                og = meta.get("original_geometry")
                if og and aoi_bounds:
                    try:
                        og_bbox_geo = geometry_to_geo_bbox(og)
                        if og_bbox_geo:
                            bbox_px = geo_bbox_to_px_bbox(og_bbox_geo, aoi_bounds)
                    except Exception:
                        bbox_px = None

                # fallback: if backend wrote bbox_xyxy already
                if bbox_px is None:
                    bbox_px = meta.get("bbox_xyxy")

                # last resort: bbox from corrected mask (not ideal but better than skipping)
                if bbox_px is None:
                    bbox_px = mask_to_bbox_xyxy(m)

                if bbox_px is None:
                    logging.warning("Skipping wrong_geometry %s: bbox not determinable", sample_id)
                    continue

                dataset.append({
                    "id": sample_id,
                    "type": feedback_type,
                    "image": image_path,
                    "mask": mask_path,      # ✅ target mask
                    "bbox": bbox_px,        # ✅ prompt bbox (original wrong geometry)
                    "metadata": meta_path
                })
                continue
            # ---------------- MISSING BUILDING ----------------
            # mask_path = meta.get("mask")
            # if not mask_path or not os.path.exists(mask_path):
            #     continue

            # m = ensure_resized_mask(mask_path)
            # if m is None:
            #     continue

            # bbox_px = mask_to_bbox_xyxy(m)

            # # fallback: use geo bbox if present (new frontend)
            # if bbox_px is None and bbox_geo and aoi_bounds:
            #     bbox_px = geo_bbox_to_px_bbox(bbox_geo, aoi_bounds)

            # # fallback: derive from geometry if present
            # if bbox_px is None:
            #     geom = meta.get("geometry")
            #     if geom and aoi_bounds:
            #         gb = geometry_to_geo_bbox(geom)
            #         bbox_px = geo_bbox_to_px_bbox(gb, aoi_bounds) if gb else None

            # if bbox_px is None:
            #     logging.warning("Skipping missing_building %s: bbox not determinable", sample_id)
            #     continue

            # dataset.append({
            #     "id": sample_id,
            #     "type": feedback_type,
            #     "image": image_path,
            #     "mask": mask_path,
            #     "bbox": bbox_px,          # ✅ prompt bbox for SAM
            #     "metadata": meta_path
            # })

            # ---------------- MISSING BUILDING ----------------
            mask_path = meta.get("mask")

            # Strict requirement: metadata must point to image+mask and both must exist
            if not required_files_exist(image_path, mask_path, meta_path):
                logging.warning(
                    "Skipping missing_building %s: missing image/mask/meta file(s)",
                    sample_id
                )
                continue

            m = ensure_resized_mask(mask_path)
            if m is None:
                logging.warning("Skipping missing_building %s: mask read failed", sample_id)
                continue

            # Optional: skip empty masks (shouldn't happen for missing_building)
            if cv2.countNonZero(m) == 0:
                logging.warning("Skipping missing_building %s: empty mask", sample_id)
                continue

            bbox_px = mask_to_bbox_xyxy(m)

            # fallback: use geo bbox if present (new frontend)
            if bbox_px is None and bbox_geo and aoi_bounds:
                bbox_px = geo_bbox_to_px_bbox(bbox_geo, aoi_bounds)

            # fallback: derive from geometry if present
            if bbox_px is None:
                geom = meta.get("geometry")
                if geom and aoi_bounds:
                    gb = geometry_to_geo_bbox(geom)
                    bbox_px = geo_bbox_to_px_bbox(gb, aoi_bounds) if gb else None

            if bbox_px is None:
                logging.warning("Skipping missing_building %s: bbox not determinable", sample_id)
                continue

            # Optional: bbox sanity checks (prevents thin-strip samples)
            x0, y0, x1, y1 = bbox_px
            w = x1 - x0
            h = y1 - y0
            if w <= 0 or h <= 0:
                logging.warning("Skipping missing_building %s: invalid bbox", sample_id)
                continue

            aspect = max(w / (h + 1e-6), h / (w + 1e-6))
            if aspect > 8:  # tune threshold if you want
                logging.warning("Skipping missing_building %s: extreme bbox aspect %.2f", sample_id, aspect)
                continue

            dataset.append({
                "id": sample_id,
                "type": feedback_type,
                "image": image_path,
                "mask": mask_path,
                "bbox": bbox_px,
                "metadata": meta_path
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    logging.info("Dataset built with %d samples", len(dataset))
    logging.info("Saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()
