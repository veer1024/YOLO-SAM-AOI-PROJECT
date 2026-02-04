import math
from shapely.ops import transform
import pyproj
from astral import LocationInfo
from astral.sun import azimuth, elevation
from datetime import timezone
from shapely.ops import transform as shp_transform






def shadow_metrics_from_polygons(
    building_geom,
    shadow_geom,
    sun_azimuth_deg,
    sun_elevation_deg,
):
    """
    Physically correct shadow length & height computation.

    Shadow length is measured along sun direction as:
      max(shadow_proj) - max(building_proj)

    This avoids diagonal/corner overestimation.
    """

    if sun_elevation_deg <= 1.0:
        return None, None, 0.0

    # Project to meters
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform

    b = shp_transform(project, building_geom)
    s = shp_transform(project, shadow_geom)

    # Sun direction (shadow goes opposite sun)
    az = math.radians(sun_azimuth_deg)
    sun_vec = (-math.sin(az), -math.cos(az))

    # ---- project building onto sun axis ----
    building_proj = []
    for x, y in b.exterior.coords:
        building_proj.append(x * sun_vec[0] + y * sun_vec[1])

    # This is the sun-facing building edge position
    b_max = max(building_proj)

    # ---- project shadow onto sun axis ----
    shadow_proj = []
    for x, y in s.exterior.coords:
        shadow_proj.append(x * sun_vec[0] + y * sun_vec[1])

    s_max = max(shadow_proj)

    # ---- physically correct shadow length ----
    shadow_len = s_max - b_max

    # Reject nonsense
    if shadow_len <= 0 or shadow_len > 200:
        return None, None, 0.0

    # ---- trigonometry ----
    height = shadow_len * math.tan(math.radians(sun_elevation_deg))

    # ---- confidence (manual input = high) ----
    confidence = 0.95 if height < 30 else 0.8

    return round(shadow_len, 2), round(height, 2), confidence






def compute_shadow_length_m(building_geom, shadow_geom):
    """
    Returns max shadow length in meters from building centroid
    """

    project = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform

    b = shp_transform(project, building_geom)
    s = shp_transform(project, shadow_geom)

    origin = b.centroid

    max_dist = 0.0
    for x, y in s.exterior.coords:
        d = math.hypot(x - origin.x, y - origin.y)
        max_dist = max(max_dist, d)

    return round(max_dist, 2)


def get_sun_angles(lat, lon, when_utc):
    loc = LocationInfo(latitude=lat, longitude=lon)
    observer = loc.observer

    sun_az = azimuth(observer, when_utc)
    sun_el = elevation(observer, when_utc)

    return float(sun_az), float(sun_el)

def height_from_shadow_trigonometry(
    building_geom,
    shadow_geom,
    sun_azimuth_deg,
    sun_elevation_deg
):
    """
    Physically correct height computation using:
    height = shadow_length * tan(sun_elevation)

    building_geom, shadow_geom : shapely Polygon (EPSG:4326)
    returns height in meters (float)
    """

    if sun_elevation_deg <= 2.0:
        return None  # unstable geometry

    # ---- project to meters ----
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform

    b = shp_transform(project, building_geom)
    s = shp_transform(project, shadow_geom)

    origin = b.centroid

    # ---- sun direction vector (unit) ----
    az = math.radians(sun_azimuth_deg)
    #sun_vec = (math.sin(az), math.cos(az))  # azimuth from north
    sun_vec = (-math.sin(az), -math.cos(az))

    # ---- compute shadow length by projection ----
    max_proj = 0.0
    for x, y in s.exterior.coords:
        vx = x - origin.x
        vy = y - origin.y
        proj = vx * sun_vec[0] + vy * sun_vec[1]
        if proj > max_proj:
            max_proj = proj

    if max_proj <= 0:
        return None

    # ---- trigonometry ----
    elev = math.radians(sun_elevation_deg)
    height = max_proj * math.tan(elev)

    return float(height)


def estimate_height_from_shadow(building_geom, shadow_geom, sun_azimuth_deg, sun_elevation_deg):
    """
    building_geom, shadow_geom: shapely geometries (EPSG:4326)
    returns height in meters
    """

    if sun_elevation_deg <= 3.0:
        return None  # too low sun → unstable

    # Project to meters
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform

    b = transform(project, building_geom)
    s = transform(project, shadow_geom)

    # Sun direction vector
    az = math.radians(sun_azimuth_deg)
    sun_vec = (math.sin(az), math.cos(az))

    # Building reference point
    origin = b.centroid

    max_proj = 0.0
    for x, y in s.exterior.coords:
        vx = x - origin.x
        vy = y - origin.y
        proj = vx * sun_vec[0] + vy * sun_vec[1]
        if proj > max_proj:
            max_proj = proj

    if max_proj <= 0:
        return None

    elev = math.radians(sun_elevation_deg)
    height = max_proj * math.tan(elev)

    # Sanity clamp (defense / urban scale)
    return float(min(max(height, 1.5), 120.0))