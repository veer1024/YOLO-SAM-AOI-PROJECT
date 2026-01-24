import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Polygon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tile_reader import iter_tiles
import cv2

TIF = "../merged_rgb_3857.tif"
OUT = "../static/buildings.geojson"
CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.9,
    crop_n_layers=0
)

features = []

for img, transform in iter_tiles(TIF, tile_size=1024):

    # Convert CHW → HWC
    image = np.moveaxis(img, 0, -1)

    masks = mask_generator.generate(image)

    for m in masks:
        if m["area"] < 500:   # remove noise
            continue

        mask = m["segmentation"].astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if len(c) < 4:
                continue

            coords = []
            for p in c:
                px, py = p[0]
                gx, gy = transform * (px, py)
                coords.append((gx, gy))

            poly = Polygon(coords)
            if poly.area < 10:
                continue

            features.append({
                "geometry": poly,
                "height": 0   # filled later
            })

gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:3857")
gdf.to_file(OUT, driver="GeoJSON")

print(f"Saved {len(gdf)} buildings")
