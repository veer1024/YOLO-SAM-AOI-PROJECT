import os
import logging
import cv2
import numpy as np
import rasterio
import geopandas as gpd
import requests
from shapely.geometry import shape
import json
from shapely.geometry import Polygon
from rasterio.warp import transform_bounds
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

### shadow height calcuation
import math
from datetime import datetime, timezone
from astral.sun import elevation
from astral import LocationInfo
import osmnx as ox

from ml.roof_geometry import (
    flat_roof,
    gabled_roof,
    barrel_roof,
    infer_roof_type
)
from ml.glb_export import export_glb

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_TIF = "data/processed/merged_rgb_3857.tif"
AOI_TIF = "ml/aoi.tif"

OUT_DIR = "ml/output"
OUT_GLB = f"{OUT_DIR}/aoi_buildings.glb"
OUT_GEOJSON = f"{OUT_DIR}/buildings.geojson"

DEFAULT_HEIGHT = 12.0          # meters
MIN_AREA_PX = 400              # segmentation noise filter

os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# LOAD SAM (ONCE)
# --------------------------------------------------
sam = sam_model_registry["vit_b"](
    checkpoint="ml/checkpoints/sam_vit_b.pth"
)
sam.to(device="cpu")

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1
)



### FOOTPRINTING FROM OSM



def get_osm_buildings(bounds, timeout=20):
    west, south, east, north = bounds

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out geom;
    """

    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            timeout=timeout
        )
        r.raise_for_status()
    except Exception as e:
        logging.warning("OSM Overpass failed: %s", e)
        return None   # 🚨 DO NOT RAISE

    data = r.json()
    features = []

    for el in data.get("elements", []):
        if "geometry" not in el:
            continue

        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 4:
            continue

        
        props = el.get("tags", {})

        features.append({
            "type": "Feature",
            "properties": {
                "height": props.get("height"),
                "building:levels": props.get("building:levels"),
                "base": 0
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        })


    if not features:
        return None

    return {
        "type": "FeatureCollection",
        "features": features
    }



### FOOTPRINTING FORM OSM
# --------------------------------------------------
# AOI EXTRACTION
# --------------------------------------------------
def extract_aoi(bounds):
    with rasterio.open(BASE_TIF) as src:
        bounds_3857 = transform_bounds(
            "EPSG:4326", src.crs, *bounds
        )

        window = rasterio.windows.from_bounds(
            *bounds_3857, transform=src.transform
        )

        data = src.read([1, 2, 3], window=window)
        transform = src.window_transform(window)
        crs = src.crs

        meta = src.meta.copy()
        meta.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "count": 3
        })

        with rasterio.open(AOI_TIF, "w", **meta) as dst:
            dst.write(data)

    return transform, crs


# --------------------------------------------------
# MAIN PIPELINE (LOD 2.5)
# --------------------------------------------------

######### shadow helper function


def get_solar_elevation(lat, lon, timestamp):
    loc = LocationInfo(latitude=lat, longitude=lon)
    dt = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
    return elevation(loc.observer, dt)


def extract_shadow_mask(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    shadow = hsv[:, :, 2] < 60  # low brightness
    shadow = shadow.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
    return shadow


def estimate_shadow_length(poly, shadow_mask, transform,percentile):
    mask = np.zeros(shadow_mask.shape, dtype=np.uint8)

    pts = [
        rasterio.transform.rowcol(transform, x, y)
        for x, y in poly.exterior.coords
    ]

    cv2.fillPoly(mask, [np.array(pts)], 255)

    ys, xs = np.where((shadow_mask == 255) & (mask == 0))
    if len(xs) == 0:
        return None

    cx = np.mean(xs)
    cy = np.mean(ys)
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)

    max_px = np.percentile(dists, percentile)
    meters_per_px = abs(transform.a)
    return max_px * meters_per_px


def estimate_height_from_shadow(shadow_len_m, solar_elev_deg):
    if solar_elev_deg <= 0 or shadow_len_m is None:
        return None
    return shadow_len_m * math.tan(math.radians(solar_elev_deg))

######## shadow helper function


# --------------------------------------------------
# MAIN PIPELINE (LOD 1 + LOD 2.5) — OSM BASED
# --------------------------------------------------


def detect_buildings(bounds):
    logging.info("AOI → OSM + Shadow based LOD 2.5 pipeline started")
    logging.info("AOI bounds: %s", bounds)

    # --------------------------------------------------
    # 1. EXTRACT AOI IMAGE (FOR SHADOW ONLY)
    # --------------------------------------------------
    transform, raster_crs = extract_aoi(bounds)

    with rasterio.open(AOI_TIF) as src:
        img = src.read().transpose(1, 2, 0)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------
    # 2. SOLAR + SHADOW SETUP
    # --------------------------------------------------
    lat = (bounds[1] + bounds[3]) / 2
    lon = (bounds[0] + bounds[2]) / 2
    timestamp = datetime.utcnow().isoformat()

    solar_elev = get_solar_elevation(lat, lon, timestamp)
    logging.info("Solar elevation: %.2f°", solar_elev)

    shadow_mask = extract_shadow_mask(img_rgb)

    # --------------------------------------------------
    # 3. GET OSM BUILDING FOOTPRINTS (GROUND TRUTH)
    # --------------------------------------------------
    osm_geojson = get_osm_buildings(bounds)

    if osm_geojson is None or not osm_geojson.get("features"):
        logging.error("SAM fallback not implemented yet")
        osm_geojson = None

    if osm_geojson:
        features = osm_geojson["features"]
    else:
        logging.error("OSM fallback to SAM not yet implemented")
        return None   # or call SAM fallback here



    polygons_3857 = []
    heights = []
    meshes = []

    # --------------------------------------------------
    # 4. PROCESS EACH OSM BUILDING
    # --------------------------------------------------
    if not osm_geojson:
        logging.error("No footprints available (OSM + SAM both unavailable)")
        return None

    for feat in osm_geojson["features"]:
        poly_4326 = shape(feat["geometry"])

        if not poly_4326.is_valid or poly_4326.area == 0:
            continue

        # ---- project OSM polygon → raster CRS (EPSG:3857)
        gdf_tmp = gpd.GeoDataFrame(
            geometry=[poly_4326], crs="EPSG:4326"
        ).to_crs(raster_crs)

        poly = gdf_tmp.geometry.iloc[0]

        # --------------------------------------------------
        # HEIGHT FROM SHADOW (NOT FROM OSM)
        # --------------------------------------------------
        shadow_len = estimate_shadow_length(
            poly,
            shadow_mask,
            transform,
            percentile=90
        )
        shadow_height = estimate_height_from_shadow(shadow_len, solar_elev)

        osm_height = feat["properties"].get("height")
        osm_levels = feat["properties"].get("building:levels")

        if shadow_height is not None and shadow_height > 3:
            height = shadow_height
            source = "shadow"
        elif osm_height:
            height = float(osm_height)
            source = "osm_height"
        elif osm_levels:
            height = float(osm_levels) * 3.2
            source = "osm_levels"
        else:
            height = DEFAULT_HEIGHT
            source = "fallback"

        height = float(np.clip(height, 3, 120))


        # --------------------------------------------------
        # ROOF GEOMETRY (LOD 2.5)
        # --------------------------------------------------
        roof = infer_roof_type(poly)

        if roof == "flat":
            mesh = flat_roof(poly, height)
        elif roof == "gabled":
            mesh = gabled_roof(poly, height)
        else:
            mesh = barrel_roof(poly, height)

        polygons_3857.append(poly)
        heights.append(height)
        meshes.append(mesh)

    if not polygons_3857:
        logging.error("No valid buildings after processing")
        return None

    # --------------------------------------------------
    # 5. EXPORT LOD1 (MAPBOX SAFE)
    # --------------------------------------------------
    gdf = gpd.GeoDataFrame(
        {
            "height": heights,
            "base": [0] * len(heights)
        },
        geometry=polygons_3857,
        crs=raster_crs          # EPSG:3857
    )

    # 🚨 THIS LINE PREVENTS ALL POSITION BUGS 🚨
    gdf = gdf.to_crs(epsg=4326)

    gdf.to_file(OUT_GEOJSON, driver="GeoJSON")

    # --------------------------------------------------
    # 6. EXPORT LOD 2.5 (GLB)
    # --------------------------------------------------
    export_glb(meshes, OUT_GLB)

    logging.info("LOD1 + LOD2.5 + Shadow height export complete")

    return {
        "geojson": "output/buildings.geojson",
        "glb": "output/aoi_buildings.glb"
    }




