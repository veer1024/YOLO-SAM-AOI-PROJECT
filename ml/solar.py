# ml/solar.py

import math
import datetime

# --------------------------------------------------
# Solar position (NO external libs)
# --------------------------------------------------

def solar_position(lat, lon, dt=None):
    """
    Returns:
        solar_elevation_deg, solar_azimuth_deg

    lat, lon : degrees
    dt       : UTC datetime (optional)
    """

    if dt is None:
        dt = datetime.datetime.utcnow()

    # Julian day
    epoch = datetime.datetime(2000, 1, 1, 12)
    delta = dt - epoch
    days = delta.total_seconds() / 86400.0

    # Mean longitude
    L = (280.46 + 0.9856474 * days) % 360

    # Mean anomaly
    g = math.radians((357.528 + 0.9856003 * days) % 360)

    # Ecliptic longitude
    lam = math.radians(
        L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)
    )

    # Obliquity
    eps = math.radians(23.439)

    # Declination
    delta_sun = math.asin(math.sin(eps) * math.sin(lam))

    # Time correction
    time_utc = dt.hour + dt.minute / 60 + dt.second / 3600
    solar_time = time_utc + lon / 15.0
    hour_angle = math.radians(15 * (solar_time - 12))

    lat_rad = math.radians(lat)

    # Elevation
    elevation = math.asin(
        math.sin(lat_rad) * math.sin(delta_sun)
        + math.cos(lat_rad) * math.cos(delta_sun) * math.cos(hour_angle)
    )

    # Azimuth
    azimuth = math.atan2(
        -math.sin(hour_angle),
        math.tan(delta_sun) * math.cos(lat_rad)
        - math.sin(lat_rad) * math.cos(hour_angle)
    )

    azimuth_deg = (math.degrees(azimuth) + 360) % 360
    elevation_deg = math.degrees(elevation)

    return elevation_deg, azimuth_deg
