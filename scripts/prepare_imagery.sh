#!/bin/bash

gdal_merge.py -o merged.tif data/raw/IMG_*.TIF

gdalwarp -t_srs EPSG:3857 merged.tif merged_3857.tif

gdal_translate \
  -b 1 -b 2 -b 3 \
  -co TILED=YES \
  -co COMPRESS=LZW \
  merged_3857.tif data/processed/merged_rgb_3857.tif

