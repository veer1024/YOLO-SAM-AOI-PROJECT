# ml/feedback/extract.py

import rasterio
import numpy as np
import cv2
from shapely.geometry import shape, mapping
import geopandas as gpd
from rasterio.mask import mask


def crop_image_and_mask(tif_path, geojson_geom):
    """
    geojson_geom: geometry in EPSG:4326 (from frontend)
    """

    # 1. Load geometry (4326)
    geom_4326 = shape(geojson_geom)

    # 2. Open raster
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs

        # 3. Reproject geometry → raster CRS (3857)
        gdf = gpd.GeoDataFrame(
            geometry=[geom_4326],
            crs="EPSG:4326"
        ).to_crs(raster_crs)

        geom_proj = gdf.geometry.iloc[0]

        # 4. Safety check
        if not geom_proj.is_valid or geom_proj.is_empty:
            raise ValueError("Invalid geometry after reprojection")

        # 5. Crop raster
        img, transform = mask(
            src,
            [mapping(geom_proj)],
            crop=True
        )

        # RGB only
        img = img[:3].transpose(1, 2, 0)

        # 6. Create mask
        mask_img = np.zeros(img.shape[:2], dtype=np.uint8)

        coords = [
            rasterio.transform.rowcol(transform, x, y)
            for x, y in geom_proj.exterior.coords
        ]

        cv2.fillPoly(mask_img, [np.array(coords)], 255)

    return img, mask_img
