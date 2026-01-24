# ml/detect_buildings.py
import cv2
import torch
import rasterio
import json
from shapely.geometry import Polygon
from rasterio.transform import xy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pyproj import Transformer

IMAGE = "../data/ml/image.png"
#TIF = "../data/processed/30cm_test/pneo4_30cm_3857_cog.tif"
TIF = "../data/data/processed/phr1a_20210227_rgb_3857_cog.tif"
OUT = "../data/ml/buildings.geojson"

sam = sam_model_registry["vit_h"](
    checkpoint="checkpoints/sam_vit_h_4b8939.pth"
)
sam.to("cuda" if torch.cuda.is_available() else "cpu")

# mask_generator = SamAutomaticMaskGenerator(
#     sam,
#     points_per_side=32,
#     pred_iou_thresh=0.9,
#     stability_score_thresh=0.9,
#     min_mask_region_area=800
# )

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=0,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.70,
    min_mask_region_area=500
)

image = cv2.imread(IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

features = []

# with rasterio.open(TIF) as src:
#     transform = src.transform

#     for m in masks:
#         mask = m["segmentation"].astype("uint8")
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             if len(cnt) < 4:
#                 continue

#             poly = Polygon(cnt.squeeze())
#             if poly.area < 1000:
#                 continue

#             coords = []
#             for x, y in poly.exterior.coords:
#                 lon, lat = xy(transform, y, x)
#                 coords.append([lon, lat])

#             features.append({
#                 "type": "Feature",
#                 "geometry": {
#                     "type": "Polygon",
#                     "coordinates": [coords]
#                 },
#                 "properties": {
#                     "height": 0
#                 }
#             })


with rasterio.open(TIF) as src:
    transform = src.transform
    src_crs = src.crs

    to4326 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    for m in masks:
        mask = m["segmentation"].astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 4:
                continue

            poly = Polygon(cnt.squeeze())
            if poly.area < 1000:
                continue

            coords = []
            for x, y in poly.exterior.coords:
                X, Y = xy(transform, y, x)          # X,Y in src CRS (3857 meters)
                lon, lat = to4326.transform(X, Y)   # convert to EPSG:4326 degrees
                coords.append([lon, lat])

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"height": 0}
            })


geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(OUT, "w") as f:
    json.dump(geojson, f)

