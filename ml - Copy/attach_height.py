# ml/attach_height.py

import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO)

DEFAULT_HEIGHT = 25.0  # meters

def attach_height(input_geojson, output_geojson):
    logging.info("Attaching height to buildings")

    gdf = gpd.read_file(input_geojson)

    if "height" not in gdf.columns:
        gdf["height"] = DEFAULT_HEIGHT

    if "base" not in gdf.columns:
        gdf["base"] = 0.0

    gdf["height"] = gdf["height"].astype(float)
    gdf["base"] = gdf["base"].astype(float)

    gdf.to_file(output_geojson, driver="GeoJSON")

    logging.info("Height attached. Saved: %s", output_geojson)

    return output_geojson
