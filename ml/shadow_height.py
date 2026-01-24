# ml/shadow_height.py

import math
import logging
import numpy as np
import rasterio
from shapely.geometry import Polygon, Point
from rasterio.features import geometry_mask

DEFAULT_HEIGHT = 25.0  # fallback height (meters)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --------------------------------------------------
# Compute building height from shadow
# --------------------------------------------------

def compute_building_height(
    polygon: Polygon,
    raster: rasterio.io.DatasetReader,
    solar_elevation_deg: float,
    solar_azimuth_deg: float
) -> float:
    """
    Estimate building height using shadow length.
    Returns height in meters.
    """

    # ----------------------------------------------
    # Sun too low → unreliable shadows
    # ----------------------------------------------
    if solar_elevation_deg < 15:
        logging.warning("Sun too low — unreliable shadow, using default height")
        return DEFAULT_HEIGHT

    try:
        shadow_len = shadow_length_for_building(
            polygon,
            raster,
            solar_azimuth_deg
        )

        if shadow_len is None or shadow_len <= 0:
            logging.warning("Invalid shadow length — using default height")
            return DEFAULT_HEIGHT

        height = shadow_len * math.tan(math.radians(solar_elevation_deg))

        if not math.isfinite(height) or height <= 0:
            logging.warning("Computed invalid height — using default")
            return DEFAULT_HEIGHT

        return round(height, 2)

    except Exception as e:
        logging.exception("Shadow height computation failed")
        return DEFAULT_HEIGHT


# --------------------------------------------------
# Shadow length estimation
# --------------------------------------------------

def shadow_length_for_building(
    polygon: Polygon,
    raster: rasterio.io.DatasetReader,
    solar_azimuth_deg: float,
    max_distance_m: float = 100.0
) -> float | None:
    """
    Measure shadow length in meters along sun azimuth.
    """

    # ----------------------------------------------
    # Convert polygon to mask
    # ----------------------------------------------
    try:
        mask = geometry_mask(
            [polygon],
            out_shape=(raster.height, raster.width),
            transform=raster.transform,
            invert=True
        )
    except Exception:
        logging.exception("Failed to rasterize building polygon")
        return None

    # ----------------------------------------------
    # Start from centroid
    # ----------------------------------------------
    centroid = polygon.centroid
    if centroid.is_empty:
        return None

    # Convert centroid to pixel
    try:
        px, py = raster.index(centroid.x, centroid.y)
    except Exception:
        return None

    if not np.isfinite(px) or not np.isfinite(py):
        return None

    px, py = int(px), int(py)

    # ----------------------------------------------
    # Shadow direction (opposite sun)
    # ----------------------------------------------
    angle = math.radians((solar_azimuth_deg + 180) % 360)
    dx = math.cos(angle)
    dy = math.sin(angle)

    step = raster.res[0]  # pixel resolution (meters)
    distance = 0.0

    # ----------------------------------------------
    # Walk until leaving shadow or max distance
    # ----------------------------------------------
    while distance < max_distance_m:
        px += int(dx)
        py += int(dy)

        if px < 0 or py < 0 or px >= raster.width or py >= raster.height:
            break

        if not mask[py, px]:
            break

        distance += step

    if distance <= 0:
        return None

    return distance
