import geopandas as gpd
import shapely.geometry as geom
import logging
import os

MS_BUILDINGS = "data/ms_buildings_india.geojson"  # pre-downloaded
OUT_GEOJSON = "ml/buildings.geojson"

logging.basicConfig(level=logging.INFO)

def get_buildings_from_ms(bounds_wgs84):
    """
    bounds_wgs84 = [west, south, east, north]
    """

    west, south, east, north = bounds_wgs84
    aoi = geom.box(west, south, east, north)

    logging.info("Loading Microsoft building footprints")

    gdf = gpd.read_file(MS_BUILDINGS)

    # Ensure CRS
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    # Spatial filter
    gdf = gdf[gdf.intersects(aoi)]

    if gdf.empty:
        logging.error("NO BUILDINGS FOUND IN AOI")
        return None

    # Assign height (temporary constant, later shadow-based)
    gdf["height"] = 25
    gdf["base"] = 0

    os.makedirs("ml", exist_ok=True)
    gdf.to_file(OUT_GEOJSON, driver="GeoJSON")

    logging.info("Buildings saved: %d", len(gdf))
    return OUT_GEOJSON
