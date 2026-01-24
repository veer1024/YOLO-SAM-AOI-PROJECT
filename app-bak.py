from flask import Flask, request, jsonify, render_template, send_file
import logging
import sys
import os
import uuid
import json
import cv2
import numpy as np
from datetime import datetime, timezone
from shapely.geometry import shape

from rio_tiler.errors import TileOutsideBounds

from ml.detect_buildings_aoi import detect_buildings
from ml.feedback.utils import refine_mask, ensure_feedback_dirs
from ml.feedback.extract import crop_image_and_mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
# ----------------------------
# LOGGING
# ----------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

file_handler = logging.FileHandler("app.log", mode="a")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("rio_tiler").setLevel(logging.WARNING)

# ----------------------------
# APP + CONFIG
# ----------------------------
app = Flask(__name__)

FEEDBACK_TIF = "data/data/processed/phr1a_20210227_rgb_3857_cog.tif"
FEEDBACK_DIR = "ml/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

VALID_FEEDBACK_TYPES = ["missing_building", "wrong_geometry", "not_a_building"]
ensure_feedback_dirs()

# ----------------------------
# HELPERS
# ----------------------------
def normalize_geometry(geom):
    g = shape(geom)
    if not g.is_valid:
        g = g.buffer(0)
    if g.is_empty:
        raise ValueError("Empty geometry after fix")
    return g

def _mask_stats(mask_u8: np.ndarray):
    mask_u8 = mask_u8.astype(np.uint8)
    unique = np.unique(mask_u8)
    cov = float((mask_u8 > 0).mean())
    s = int(mask_u8.sum())
    return unique.tolist(), cov, s

def _touches_border(mask_u8: np.ndarray) -> bool:
    m = (mask_u8 > 0)
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())

def _largest_bbox_from_mask(mask_u8: np.ndarray, margin: int = 5):
    """
    Returns (x1,y1,x2,y2) in pixel coords.
    mask_u8 must be uint8 0/255.
    """
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W = mask_u8.shape[:2]

    x1 = min(max(x + margin, 0), W - 1)
    y1 = min(max(y + margin, 0), H - 1)
    x2 = min(max(x + w - 1 - margin, 0), W - 1)
    y2 = min(max(y + h - 1 - margin, 0), H - 1)

    if x2 <= x1 or y2 <= y1:
        return None
    return (int(x1), int(y1), int(x2), int(y2))

def _to_contiguous_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be np.ndarray, got {type(image)}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    return np.ascontiguousarray(image).copy()

def _save_debug_images(base_dir, feedback_id, image_rgb, mask_u8, bbox=None):
    os.makedirs(base_dir, exist_ok=True)

    mask_dbg = os.path.join(base_dir, f"{feedback_id}_mask_dbg.png")
    overlay = os.path.join(base_dir, f"{feedback_id}_overlay.png")
    bbox_img = os.path.join(base_dir, f"{feedback_id}_bbox.png")

    cv2.imwrite(mask_dbg, mask_u8)

    # overlay
    ov = image_rgb.copy()
    sel = mask_u8 > 0
    ov[sel] = (ov[sel] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(overlay, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    # bbox image
    bb = image_rgb.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        try:
            cv2.rectangle(bb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            logging.exception("debug cv2.rectangle failed")
    cv2.imwrite(bbox_img, cv2.cvtColor(bb, cv2.COLOR_RGB2BGR))

    return mask_dbg, overlay, bbox_img

# ----------------------------
# FEEDBACK ENDPOINT
# ----------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    ftype = data.get("type")
    geom = data.get("geometry")
    original_geom = data.get("original_geometry")
    correct_geoms = data.get("correct_geometries")
    aoi_bounds = data.get("aoi_bounds")

    logging.info("FEEDBACK request type=%s has_geom=%s has_original=%s num_correct=%s has_aoi=%s",
                 ftype, bool(geom), bool(original_geom),
                 (len(correct_geoms) if isinstance(correct_geoms, list) else None),
                 bool(aoi_bounds))

    if aoi_bounds is not None:
        if not (isinstance(aoi_bounds, list) and len(aoi_bounds) == 4):
            return jsonify({"error": "aoi_bounds must be [west,south,east,north]"}), 400
        west, south, east, north = aoi_bounds
        if east <= west or north <= south:
            return jsonify({"error": "Invalid aoi_bounds ordering"}), 400

    if ftype not in VALID_FEEDBACK_TYPES:
        return jsonify({"error": "Invalid feedback type", "allowed": VALID_FEEDBACK_TYPES}), 400

    def _is_polygon(g):
        return isinstance(g, dict) and g.get("type") == "Polygon" and "coordinates" in g

    # Pick geometry for crop
    if ftype == "not_a_building":
        if original_geom:
            if not _is_polygon(original_geom):
                return jsonify({"error": "original_geometry must be GeoJSON Polygon"}), 400
            geom_for_crop = original_geom
        elif geom:
            if not _is_polygon(geom):
                return jsonify({"error": "geometry must be GeoJSON Polygon"}), 400
            geom_for_crop = geom
        else:
            return jsonify({"error": "Either geometry or original_geometry required for not_a_building"}), 400

    elif ftype == "missing_building":
        if not geom or not _is_polygon(geom):
            return jsonify({"error": "Valid GeoJSON Polygon required in geometry"}), 400
        geom_for_crop = geom

    # elif ftype == "wrong_geometry":
    #     if not original_geom or not _is_polygon(original_geom):
    #         return jsonify({"error": "original_geometry Polygon required for wrong_geometry"}), 400
    #     if not correct_geoms or not isinstance(correct_geoms, list) or len(correct_geoms) == 0:
    #         return jsonify({"error": "correct_geometries[] required for wrong_geometry"}), 400
    #     for g in correct_geoms:
    #         if not _is_polygon(g):
    #             return jsonify({"error": "Each item in correct_geometries must be a GeoJSON Polygon"}), 400
    #     geom_for_crop = correct_geoms[0]

    elif ftype == "wrong_geometry":
        if not original_geom or not _is_polygon(original_geom):
            return jsonify({"error": "original_geometry Polygon required for wrong_geometry"}), 400
        if not correct_geoms or not isinstance(correct_geoms, list) or len(correct_geoms) == 0:
            return jsonify({"error": "correct_geometries[] required for wrong_geometry"}), 400
        for g in correct_geoms:
            if not _is_polygon(g):
                return jsonify({"error": "Each item in correct_geometries must be a GeoJSON Polygon"}), 400

        # ✅ union all corrected polygons so mask includes ALL of them
        try:
            union_geom = unary_union([shape(g) for g in correct_geoms])
            geom_for_crop = mapping(union_geom)
        except Exception as e:
            logging.exception("Failed to union correct_geometries; falling back to first polygon")
            geom_for_crop = correct_geoms[0]
    else:
        return jsonify({"error": "Unhandled feedback type"}), 400

    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Paths
    base = os.path.join(FEEDBACK_DIR, ftype)
    img_dir = os.path.join(base, "images")
    msk_dir = os.path.join(base, "masks")
    meta_dir = os.path.join(base, "metadata")
    dbg_dir = os.path.join(base, "debug")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(dbg_dir, exist_ok=True)

    img_path = os.path.join(img_dir, f"{feedback_id}.png")
    mask_path = os.path.join(msk_dir, f"{feedback_id}.png")
    meta_path = os.path.join(meta_dir, f"{feedback_id}.json")

    # Crop
    try:
        geom_fixed = normalize_geometry(geom_for_crop)
        image, base_mask = crop_image_and_mask(
            FEEDBACK_TIF,
            geom_fixed,
            aoi_bounds=aoi_bounds,
            out_size=1024,
            debug_dir=dbg_dir,
            debug_prefix=f"{feedback_id}_crop"
        )
    except ValueError as e:
        logging.warning("Feedback outside raster/invalid: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception:
        logging.exception("Feedback crop failed")
        return jsonify({"error": "Internal feedback crop error"}), 500

    # Make image safe
    try:
        image = _to_contiguous_uint8_rgb(image)
    except Exception as e:
        logging.exception("Image conversion failed")
        return jsonify({"error": f"Image conversion failed: {e}"}), 500

    # Base mask stats
    base_mask = np.where(base_mask > 0, 255, 0).astype(np.uint8)
    u0, cov0, s0 = _mask_stats(base_mask)
    logging.info("Base mask stats dtype=%s shape=%s unique=%s cov=%.4f sum=%d touches_border=%s",
                 base_mask.dtype, base_mask.shape, u0, cov0, s0, _touches_border(base_mask))

    # Refine mask
    try:

        # if ftype == "wrong_geometry":
        #     #refined = _refine_mask_keep_all(base_mask)
        #     refined = base_mask.copy()
        # else:
        #     refined = refine_mask(base_mask, debug_save_path=os.path.join(dbg_dir, f"{feedback_id}_mask_dbg.png"))
        #     refined = np.where(refined > 0, 255, 0).astype(np.uint8)
        refined = refine_mask(base_mask, debug_save_path=os.path.join(dbg_dir, f"{feedback_id}_mask_dbg.png"))
        refined = np.where(refined > 0, 255, 0).astype(np.uint8)
    except Exception as e:
        logging.exception("Mask refinement failed")
        return jsonify({"error": f"Mask refinement failed: {e}"}), 500

    u1, cov1, s1 = _mask_stats(refined)
    touches1 = _touches_border(refined)
    logging.info("Refined mask stats dtype=%s shape=%s unique=%s cov=%.4f sum=%d touches_border=%s",
                 refined.dtype, refined.shape, u1, cov1, s1, touches1)

    # Safety: warn if mask is basically full-frame
    if cov1 > 0.90:
        logging.warning("WARNING: refined mask coverage > 0.90 (%.4f). Likely CRS/rasterize/border artifact issue.", cov1)

    # Compute bbox
    bbox = _largest_bbox_from_mask(refined, margin=5)
    logging.info("bbox_xyxy=%s", bbox)

    # Save debug images
    mask_dbg_path, overlay_path, bbox_path = _save_debug_images(dbg_dir, feedback_id, image, refined, bbox=bbox)

    # Save files
    ok_img = cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ok_msk = cv2.imwrite(mask_path, refined)

    logging.info("write image ok=%s path=%s | write mask ok=%s path=%s", ok_img, img_path, ok_msk, mask_path)
    if not ok_img or not ok_msk:
        return jsonify({"error": "Failed to write image/mask to disk"}), 500

    metadata = {
        "id": feedback_id,
        "type": ftype,
        "timestamp": timestamp,
        "image": img_path,
        "mask": mask_path,
        "debug_mask": mask_dbg_path,
        "debug_overlay": overlay_path,
        "debug_bbox": bbox_path,
        "bbox_xyxy": list(bbox) if bbox else None,
        "mask_coverage": cov1,
        "mask_touches_border": touches1,
        "base_mask_coverage": cov0,
    }
    if aoi_bounds is not None:
        metadata["aoi_bounds"] = aoi_bounds
    if geom:
        metadata["geometry"] = geom
    if original_geom:
        metadata["original_geometry"] = original_geom
    if correct_geoms:
        metadata["correct_geometries"] = correct_geoms

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Saved feedback meta=%s", meta_path)

    # Update index.json
    index_path = os.path.join(FEEDBACK_DIR, "index.json")
    index = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    index.append({
        "id": feedback_id,
        "type": ftype,
        "timestamp": timestamp,
        "image": img_path,
        "mask": mask_path,
        "bbox_xyxy": list(bbox) if bbox else None,
        "mask_coverage": cov1,
        "mask_touches_border": touches1,
        "aoi_bounds": aoi_bounds,
    })

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return jsonify({"status": "saved", "id": feedback_id, "type": ftype}), 200

# ----------------------------
# FRONTEND
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------
# TILE SERVER
# ----------------------------
@app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def tiles(z, x, y):
    from rio_tiler.io import COGReader
    from rio_tiler.utils import render
    from io import BytesIO

    try:
        with COGReader(FEEDBACK_TIF) as cog:
            img = cog.tile(x, y, z)
            content = render(img.data)
        return send_file(BytesIO(content), mimetype="image/png")
    except TileOutsideBounds:
        empty = np.zeros((256, 256, 4), dtype=np.uint8)
        content = render(empty.transpose(2, 0, 1))
        return send_file(BytesIO(content), mimetype="image/png")

# ----------------------------
# AOI -> PIPELINE
# ----------------------------
@app.route("/run_aoi_ml", methods=["POST"])
def run_aoi_ml():
    data = request.get_json(force=True, silent=True)
    if not data or "bounds" not in data:
        return jsonify({"error": "bounds missing"}), 400

    bounds = data["bounds"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        return jsonify({"error": "bounds must be [west, south, east, north]"}), 400

    try:
        result = detect_buildings(bounds)
    except Exception as e:
        logging.exception("ML pipeline failed")
        return jsonify({"error": str(e)}), 500

    if not result:
        return jsonify({"status": "ok", "buildings": 0, "reason": "No valid buildings after filtering"}), 200

    return jsonify(result)

# ----------------------------
# ML FILE SERVER
# ----------------------------
@app.route("/ml/<path:filename>")
def serve_ml(filename):
    path = os.path.join("ml", filename)
    if filename.endswith(".glb"):
        return send_file(path, mimetype="model/gltf-binary")
    if filename.endswith(".geojson"):
        return send_file(path, mimetype="application/json")
    return send_file(path)

if __name__ == "__main__":
    logging.info("Starting Flask server")
    app.run(host="localhost", port=5000, debug=True)
