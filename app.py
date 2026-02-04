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
import random
from rio_tiler.errors import TileOutsideBounds

from ml.detect_buildings_aoi import detect_buildings
from ml.feedback.utils import refine_mask, ensure_feedback_dirs
from ml.feedback.extract import crop_image_and_mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import rasterio
from rasterio.windows import Window

import pyproj
from shapely.ops import transform
from shapely.geometry import shape, mapping
from ml.glb_export import export_glb
import geopandas as gpd
import math
from shapely.ops import transform
import pyproj

from datetime import datetime, timezone
from shapely.geometry import shape
from ml.sun_utils import estimate_height_from_shadow, get_sun_angles,height_from_shadow_trigonometry,compute_shadow_length_m, shadow_metrics_from_polygons

import trimesh
from shapely.ops import transform as shp_transform
import pyproj

IMAGE_ACQ_TIME = "2021-02-27T04:56:00.3+00:00"

# ----------------------------

# PRE BUILD HELPERS 

# ---------------------------

def merge_all_prebuild_geojson(prebuild_dir, out_path):
    features = []

    for fname in os.listdir(prebuild_dir):
        if not fname.endswith(".geojson"):
            continue
        if fname.startswith("_all"):
            continue

        fpath = os.path.join(prebuild_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            feats = data.get("features", [])
            features.extend(feats)
        except Exception:
            logging.exception("Failed to read %s", fpath)

    merged = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    return out_path



def merge_all_prebuild_glb(prebuild_dir, out_path):
    meshes = []

    for fname in os.listdir(prebuild_dir):
        if not fname.endswith(".geojson"):
            continue
        if fname.startswith("_all"):
            continue

        geojson_path = os.path.join(prebuild_dir, fname)

        try:
            with open(geojson_path) as f:
                data = json.load(f)
        except Exception:
            continue

        for feat in data.get("features", []):
            if feat.get("properties", {}).get("kind") != "building":
                continue

            geom = shape(feat["geometry"])
            h = float(feat["properties"].get("height", 6.0))

            try:
                mesh = shapely_to_trimesh_extruded(geom, h)
                mesh.metadata["height"] = h
                mesh.metadata["kind"] = "building"
                meshes.append(mesh)
            except Exception:
                logging.exception("Failed to build mesh from %s", fname)

    if not meshes:
        raise RuntimeError("No building meshes found to merge")

    export_glb(meshes, out_path)
    return out_path



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

from werkzeug.utils import secure_filename

IMPORT_DIR = "ml/imports"
os.makedirs(IMPORT_DIR, exist_ok=True)

#app = Flask(__name__)
app = Flask(__name__, static_folder="static")
FEEDBACK_TIF = "data/data/processed/phr1a_20210227_rgb_3857_cog.tif"
FEEDBACK_DIR = "ml/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

VALID_FEEDBACK_TYPES = ["missing_building", "wrong_geometry", "not_a_building"]
ensure_feedback_dirs()


@app.route("/ml/<path:path>")
def serve_ml_file(path):
    return send_file(os.path.join("ml", path))


def shapely_to_trimesh_extruded(poly_wgs84, height_m):
    """
    Convert a shapely Polygon (EPSG:4326) to an extruded Trimesh mesh
    """

    # Project polygon to meters
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform

    poly_m = shp_transform(project, poly_wgs84)

    # IMPORTANT: pass SHAPELY POLYGON directly
    mesh = trimesh.creation.extrude_polygon(
        poly_m,
        height=height_m
    )

    return mesh


def build_manual_geojson_to_path(
    path,
    buildings,
    sun_azimuth,
    sun_elevation
):
    """
    PREBUILD GeoJSON with guaranteed numeric height for MapLibre extrusion
    """

    features = []

    for idx, b in enumerate(buildings):
        building_geom = shape(b["building"])
        shadow_geom   = shape(b["shadow"])

        # ---- height estimation ----
        # height_m = estimate_height_from_shadow(
        #     building_geom,
        #     shadow_geom,
        #     sun_azimuth,
        #     sun_elevation
        # )

        shadow_len, height_m, confidence = shadow_metrics_from_polygons(
            building_geom,
            shadow_geom,
            sun_azimuth,
            sun_elevation
        )

        if height_m is None:
            height_m = 6.0
            shadow_len = 6.0 / math.tan(math.radians(sun_elevation))
            confidence = 0.5

        height_m = float(height_m)

        # ---------------- BUILDING ----------------
        features.append({
            "type": "Feature",
            "geometry": b["building"],
            "properties": {
                "id": b.get("id", f"building_{idx}"),
                "kind": "building",
                "source": "prebuild",

                # 🔑 REQUIRED
                "height": height_m,

                # Optional but useful
                "sun_azimuth": float(sun_azimuth),
                "sun_elevation": float(sun_elevation),
                "confidence": round(min(1.0, height_m / 20.0), 2)
            }
        })

        # ---------------- SHADOW ---------------

        # ---------------- SHADOW ----------------
        features.append({
            "type": "Feature",
            "geometry": b["shadow"],
            "properties": {
                "id": f"{b.get('id', f'building_{idx}')}_shadow",
                "kind": "shadow",
                "source": "prebuild",

                # 🔑 shadow-specific fields
                "parent_building": b.get("id", f"building_{idx}"),
                "shadow_length_m": shadow_len,
                "sun_azimuth": float(sun_azimuth),
                "sun_elevation": float(sun_elevation),
                "confidence": confidence
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)




def build_manual_glb_to_path(path, buildings, sun_azimuth, sun_elevation):
    meshes = []

    for b in buildings:
        building_geom = shape(b["building"])
        shadow_geom   = shape(b["shadow"])

        shadow_len, h, conf = shadow_metrics_from_polygons(
            building_geom,
            shadow_geom,
            sun_azimuth,
            sun_elevation
        )
        h = h or 6.0

        mesh = shapely_to_trimesh_extruded(building_geom, h)
        mesh.metadata["height"] = h
        mesh.metadata["kind"] = "building"
        meshes.append(mesh)

    export_glb(meshes, path)

def build_manual_geojson(run_id, buildings):
    features = []

    for b in buildings:
        features.append({
            "type": "Feature",
            "geometry": b["building"],
            "properties": {
                "id": b["id"],
                "source": "manual",
                "kind": "building"
            }
        })

        features.append({
            "type": "Feature",
            "geometry": b["shadow"],
            "properties": {
                "id": b["id"],
                "source": "manual",
                "kind": "shadow"
            }
        })

    fc = {
        "type": "FeatureCollection",
        "features": features
    }

    out_dir = "ml/output"
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{run_id}.geojson")
    with open(path, "w") as f:
        json.dump(fc, f)

    return os.path.relpath(path, "ml")



def build_manual_glb(run_id, buildings, sun_azimuth, sun_elevation):
    meshes = []

    for i, b in enumerate(buildings):
        building_geom = shape(b["building"])
        shadow_geom = shape(b["shadow"])

        h = height_from_shadow_trigonometry(
            building_geom,
            shadow_geom,
            sun_azimuth,
            sun_elevation
        )

        if h is None:
            h = 6.0

        mesh = shapely_to_trimesh_extruded(building_geom, h)
        mesh.metadata["height_m"] = h
        mesh.metadata["id"] = b["id"]

        meshes.append(mesh)

    out_dir = "ml/output"
    os.makedirs(out_dir, exist_ok=True)

    glb_path = os.path.join(out_dir, f"{run_id}.glb")

    # THIS matches your export_glb expectation
    export_glb(meshes, glb_path)

    return os.path.relpath(glb_path, "ml")



@app.route("/manual_build/save", methods=["POST"])
def manual_build_save():
    data = request.get_json(force=True)
    buildings = data.get("buildings")

    if not buildings or not isinstance(buildings, list):
        return jsonify({"error": "No buildings provided"}), 400

    run_id = f"manual_{uuid.uuid4().hex[:8]}"

    raw_dir = "ml/manual_build/raw"
    os.makedirs(raw_dir, exist_ok=True)

    with open(os.path.join(raw_dir, f"{run_id}.json"), "w") as f:
        json.dump(data, f, indent=2)

    PREBUILD_DIR = "ml/manual_build/prebuild"
    os.makedirs(PREBUILD_DIR, exist_ok=True)

    #geojson_rel = build_manual_geojson(run_id, buildings)
    # ----------------------------
    # Sun angle computation (FIXED acquisition time)
    # ----------------------------

    # Parse acquisition time (UTC)
    acq_time = datetime.fromisoformat(IMAGE_ACQ_TIME.replace("Z", "+00:00"))

    # Use centroid of first building as scene reference
    first_geom = shape(buildings[0]["building"])
    centroid = first_geom.centroid
    lat, lon = centroid.y, centroid.x

    sun_azimuth, sun_elevation = get_sun_angles(
        lat=lat,
        lon=lon,
        when_utc=acq_time
    )


    geojson_path = os.path.join(PREBUILD_DIR, f"{run_id}.geojson")
    build_manual_geojson_to_path(geojson_path,buildings,sun_azimuth,sun_elevation)
    glb_path = os.path.join(PREBUILD_DIR, f"{run_id}.glb")
    build_manual_glb_to_path(glb_path, buildings, sun_azimuth, sun_elevation)

    geojson_rel = os.path.relpath(geojson_path, "ml")
    glb_rel     = os.path.relpath(glb_path, "ml")

    return jsonify({
        "status": "ok",
        "geojson": geojson_rel,
        "glb": glb_rel
    })


@app.route("/manual_build/prebuild/latest")
def get_latest_prebuild():
    base = "ml/manual_build/prebuild"
    if not os.path.isdir(base):
        return jsonify({"exists": False})

    files = [f for f in os.listdir(base) if f.endswith(".geojson")]
    if not files:
        return jsonify({"exists": False})

    picked = random.choice(files)
    name = os.path.splitext(picked)[0]

    return jsonify({
        "exists": True,
        "geojson": f"manual_build/prebuild/{name}.geojson",
        "glb": f"manual_build/prebuild/{name}.glb"
    })


@app.route("/manual_build/prebuild/export/<fmt>")
def export_prebuild(fmt):
    PREBUILD_DIR = "ml/manual_build/prebuild"

    geojson_path = os.path.join(PREBUILD_DIR, "_all.geojson")
    glb_path = os.path.join(PREBUILD_DIR, "_all.glb")

    if not os.path.exists(geojson_path) or not os.path.exists(glb_path):
        return jsonify({
            "error": "Prebuild not available. Load prebuild first."
        }), 404

    if fmt == "geojson":
        return send_file(
            geojson_path,
            as_attachment=True,
            download_name="prebuild_map.geojson",
            mimetype="application/json"
        )

    if fmt == "glb":
        return send_file(
            glb_path,
            as_attachment=True,
            download_name="prebuild_map.glb",
            mimetype="model/gltf-binary"
        )

    return jsonify({"error": "Invalid format"}), 400


@app.route("/manual_build/prebuild/all")
def get_all_prebuild():
    PREBUILD_DIR = "ml/manual_build/prebuild"
    if not os.path.isdir(PREBUILD_DIR):
        return jsonify({"exists": False})

    geojson_files = [
        f for f in os.listdir(PREBUILD_DIR)
        if f.endswith(".geojson") and not f.startswith("_all")
    ]

    if not geojson_files:
        return jsonify({"exists": False})

    merged_geojson = os.path.join(PREBUILD_DIR, "_all.geojson")
    merged_glb     = os.path.join(PREBUILD_DIR, "_all.glb")

    try:
        merge_all_prebuild_geojson(PREBUILD_DIR, merged_geojson)
        merge_all_prebuild_glb(PREBUILD_DIR, merged_glb)
    except Exception as e:
        logging.exception("Failed to merge prebuild runs")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "exists": True,
        "geojson": "manual_build/prebuild/_all.geojson",
        "glb": "manual_build/prebuild/_all.glb",
        "runs": len(geojson_files)
    })


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



def _bbox_from_mask_all(mask_u8: np.ndarray, margin: int = 5):
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    mask_u8 = mask_u8.astype(np.uint8)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None

    H, W = mask_u8.shape[:2]
    x1 = max(int(xs.min()) + margin, 0)
    y1 = max(int(ys.min()) + margin, 0)
    x2 = min(int(xs.max()) - margin, W - 1)
    y2 = min(int(ys.max()) - margin, H - 1)

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


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

# ---------------------------
# EXPORT TIFF
# ---------------------------

@app.route("/export_aoi_tif", methods=["POST"])
def export_aoi_tif():
    data = request.get_json(force=True, silent=True) or {}
    bounds = data.get("bounds")
    if not bounds or len(bounds) != 4:
        return jsonify({"error": "bounds must be [west,south,east,north]"}), 400

    out_path = f"ml/tmp/export_aoi_{uuid.uuid4().hex}.tif"
    try:
        # reuse your existing function
        from ml.detect_buildings_aoi import extract_aoi
        extract_aoi(bounds, out_path)
        return send_file(out_path, as_attachment=True, download_name="aoi.tif")
    except Exception as e:
        logging.exception("export_aoi_tif failed")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# IMPORT RUN
# ----------------------------

@app.route("/import_upload", methods=["POST"])
def import_upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided (field name must be 'files')"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Empty upload"}), 400

    run_id = uuid.uuid4().hex
    out_dir = os.path.join(IMPORT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    geojson_path = None
    glb_path = None

    allowed = {".geojson", ".glb"}

    for f in files:
        if not f or not f.filename:
            continue
        name = secure_filename(f.filename)
        ext = os.path.splitext(name.lower())[1]

        if ext not in allowed:
            continue

        save_path = os.path.join(out_dir, name)
        f.save(save_path)

        # return path relative to ml/ for your /ml/<path> server
        rel = os.path.relpath(save_path, "ml")

        if ext == ".geojson":
            geojson_path = rel
        elif ext == ".glb":
            glb_path = rel

    if not geojson_path and not glb_path:
        return jsonify({"error": "No valid .geojson or .glb uploaded"}), 400

    return jsonify({
        "status": "ok",
        "geojson": geojson_path,
        "glb": glb_path
    }), 200



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

        if ftype == "wrong_geometry":
            #refined = _refine_mask_keep_all(base_mask)
            refined = base_mask.copy()
        else:
            refined = refine_mask(base_mask, debug_save_path=os.path.join(dbg_dir, f"{feedback_id}_mask_dbg.png"))
            refined = np.where(refined > 0, 255, 0).astype(np.uint8)
        #refined = refine_mask(base_mask, debug_save_path=os.path.join(dbg_dir, f"{feedback_id}_mask_dbg.png"))
        #refined = np.where(refined > 0, 255, 0).astype(np.uint8)
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
    #bbox = _largest_bbox_from_mask(refined, margin=5)
    bbox = _bbox_from_mask_all(refined, margin=5)
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



#### YOLO FEEDBACK ENDPOINT

 
def polygon_to_pixel_bbox(
    polygon_geojson,
    src,
    padding_ratio=0.15
):
    poly = shape(polygon_geojson)

    project = pyproj.Transformer.from_crs(
        "EPSG:4326", src.crs, always_xy=True
    ).transform

    poly_proj = transform(project, poly)

    def world_to_pixel(x, y):
        col, row = ~src.transform * (x, y)
        return col, row

    poly_px = transform(world_to_pixel, poly_proj)

    minx, miny, maxx, maxy = poly_px.bounds

    bw = maxx - minx
    bh = maxy - miny
    pad = padding_ratio * max(bw, bh)

    return (
        int(minx - pad),
        int(miny - pad),
        int(maxx + pad),
        int(maxy + pad)
    )

def crop_from_pixel_bbox(src, minx, miny, maxx, maxy):
    minx = max(0, minx)
    miny = max(0, miny)
    maxx = min(src.width - 1, maxx)
    maxy = min(src.height - 1, maxy)

    window = Window(
        col_off=minx,
        row_off=miny,
        width=maxx - minx,
        height=maxy - miny
    )

    img = src.read([1, 2, 3], window=window).transpose(1, 2, 0)

    return img.astype(np.uint8), window

def bbox_to_crop_space(full_bbox, window):
    minx, miny, maxx, maxy = full_bbox

    return (
        minx - window.col_off,
        miny - window.row_off,
        maxx - window.col_off,
        maxy - window.row_off
    )


def to_yolo_bbox(bbox_px, img_shape):
    x1, y1, x2, y2 = bbox_px
    H, W = img_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    if bw < 4 or bh < 4:
        raise ValueError("Box too small")

    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w  = bw / W
    h  = bh / H

    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)

def is_valid_yolo_box(
    bbox_px,
    img_shape,
    min_px=16,
    min_area_px=256,
    min_ratio=0.0015,
    max_aspect=8.0
):
    """
    Decide if bbox should be kept for YOLO training.

    bbox_px = (x1, y1, x2, y2)
    img_shape = (H, W, C)
    """

    x1, y1, x2, y2 = bbox_px
    H, W = img_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    if bw < min_px or bh < min_px:
        return False

    area = bw * bh
    if area < min_area_px:
        return False

    img_area = H * W
    if area / img_area < min_ratio:
        return False

    aspect = max(bw, bh) / max(1, min(bw, bh))
    if aspect > max_aspect:
        return False

    return True

def polygon_to_pixel_bounds(polygon_geojson, src):
    poly = shape(polygon_geojson)

    project = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
    poly_proj = transform(project, poly)

    def world_to_pixel(x, y):
        col, row = ~src.transform * (x, y)
        return col, row

    poly_px = transform(world_to_pixel, poly_proj)
    return poly_px.bounds  # (minx, miny, maxx, maxy) in pixel coords (float)


def pad_bounds(bounds, pad_ratio=0.10):
    minx, miny, maxx, maxy = bounds
    bw = maxx - minx
    bh = maxy - miny
    pad = pad_ratio * max(bw, bh)
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


@app.route("/feedback_yolo", methods=["POST"])
def feedback_yolo():
    data = request.get_json(force=True)
    fb_type = data.get("type")
    polygon = data.get("geometry")
    if not polygon:
        return {"error": "Polygon geometry required"}, 400

    img_id = uuid.uuid4().hex

    with rasterio.open(FEEDBACK_TIF) as src:
        tight = polygon_to_pixel_bounds(polygon, src)        # label bbox
        padded = pad_bounds(tight, pad_ratio=0.10)           # crop bbox

        # crop using padded bbox
        img, window = crop_from_pixel_bbox(
            src,
            int(padded[0]), int(padded[1]), int(padded[2]), int(padded[3])
        )

        # label bbox in crop coords using TIGHT bbox
        bbox_px = bbox_to_crop_space(
            (int(tight[0]), int(tight[1]), int(tight[2]), int(tight[3])),
            window
        )

        # clamp bbox to crop
        x1, y1, x2, y2 = bbox_px
        H, W = img.shape[:2]
        bbox_px = (max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2))

        if not is_valid_yolo_box(bbox_px, img.shape):
            return {"status": "rejected"}

        yolo_bbox = to_yolo_bbox(bbox_px, img.shape)

    # save image (RGB->BGR)
    split = "train"  # always append new feedback to train

    os.makedirs(f"ml/feedback_yolo/images/{split}", exist_ok=True)
    os.makedirs(f"ml/feedback_yolo/labels/{split}", exist_ok=True)

    img_path = f"ml/feedback_yolo/images/{split}/{img_id}.png"
    label_path = f"ml/feedback_yolo/labels/{split}/{img_id}.txt"


    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if fb_type == "missing_building":
        with open(label_path, "w") as f:
            f.write(f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")
    else:
        open(label_path, "w").close()

    return {"status": "ok", "id": img_id}

# ----------------------------
# FRONTEND
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/aoi")
def aoi_page():
    return render_template("aoi.html")

@app.route("/build")
def build_page():
    return render_template("build.html")

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