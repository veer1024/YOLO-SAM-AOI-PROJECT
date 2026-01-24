import logging
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import shape, mapping, box

TARGET_SIZE = 1024
log = logging.getLogger(__name__)

def crop_image_and_mask(tif_path, geom_4326, aoi_bounds=None, out_size=TARGET_SIZE, debug_dir=None, debug_prefix="dbg"):
    """
    Returns:
      img_rgb: (H,W,3) uint8
      mask_u8: (H,W) uint8  (255 inside geometry else 0)

    geom_4326: GeoJSON geometry (dict) OR shapely geometry, assumed EPSG:4326
    aoi_bounds: [west,south,east,north] in EPSG:4326 (recommended)
    debug_dir: if provided, writes debug PNGs (mask + overlay)
    """

    # Accept either dict or shapely
    if isinstance(geom_4326, dict):
        geom_shp_4326 = shape(geom_4326)
    else:
        geom_shp_4326 = geom_4326

    if geom_shp_4326.is_empty:
        raise ValueError("Empty geometry")

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("Raster CRS is None")

        log.info("crop_image_and_mask: tif=%s crs=%s size=(%d,%d)", tif_path, src.crs, src.width, src.height)

        # ------------------------------------
        # 1) Compute crop window in raster CRS
        # ------------------------------------
        if aoi_bounds is not None:
            if not (isinstance(aoi_bounds, list) and len(aoi_bounds) == 4):
                raise ValueError("aoi_bounds must be [west,south,east,north]")

            west, south, east, north = aoi_bounds
            if east <= west or north <= south:
                raise ValueError("Invalid aoi_bounds ordering")

            bounds_proj = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
            win = from_bounds(*bounds_proj, transform=src.transform)

            # AOI polygon in raster CRS
            aoi_poly_proj = box(*bounds_proj)

            # Project geometry to raster CRS and clip to AOI
            geom_proj_geojson = transform_geom("EPSG:4326", src.crs, mapping(geom_shp_4326), precision=6)
            geom_shp_proj = shape(geom_proj_geojson)

            geom_shp_proj = geom_shp_proj.intersection(aoi_poly_proj)

            if geom_shp_proj.is_empty:
                raise ValueError("Geometry is empty after clipping to AOI")
        else:
            # Fallback crop around geometry bounds, BUT WITH PADDING
            minx, miny, maxx, maxy = geom_shp_4326.bounds

            # Transform geometry bounds to raster CRS
            bounds_proj = transform_bounds(
                "EPSG:4326", src.crs, minx, miny, maxx, maxy, densify_pts=21
            )

            bx0, by0, bx1, by1 = bounds_proj
            bw = max(1e-9, bx1 - bx0)
            bh = max(1e-9, by1 - by0)

            # Pad by 50% of bbox size (tune: 0.25–1.0)
            pad_frac = 0.5
            pad_x = bw * pad_frac
            pad_y = bh * pad_frac

            padded = (bx0 - pad_x, by0 - pad_y, bx1 + pad_x, by1 + pad_y)

            win = from_bounds(*padded, transform=src.transform)

            geom_proj_geojson = transform_geom(
                "EPSG:4326", src.crs, mapping(geom_shp_4326), precision=6
            )
            geom_shp_proj = shape(geom_proj_geojson)

            log.info(
                "fallback padded bounds (proj): (%.2f,%.2f,%.2f,%.2f) pad_frac=%.2f",
                padded[0], padded[1], padded[2], padded[3], pad_frac
            )

        # Clamp window to raster
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width <= 1 or win.height <= 1:
            raise ValueError("Crop window too small / outside raster")

        win_transform = rasterio.windows.transform(win, src.transform)

        log.info(
            "crop window: col_off=%.2f row_off=%.2f w=%.2f h=%.2f out_size=%d",
            win.col_off, win.row_off, win.width, win.height, out_size
        )

        # ------------------------------------
        # 2) Read RGB and resample to out_size
        # ------------------------------------
        img = src.read(
            [1, 2, 3],
            window=win,
            out_shape=(3, out_size, out_size),
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)

        # Ensure uint8 for web/UI/debug
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Transform for output grid
        scale_x = win.width / float(out_size)
        scale_y = win.height / float(out_size)
        out_transform = win_transform * rasterio.Affine.scale(scale_x, scale_y)

        # ------------------------------------
        # 3) Rasterize geometry in SAME grid
        # ------------------------------------
        geom_proj = mapping(geom_shp_proj)

        mask = rasterize(
            [(geom_proj, 255)],
            out_shape=(out_size, out_size),
            transform=out_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )

        # Debug save
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            import cv2

            mpath = os.path.join(debug_dir, f"{debug_prefix}_mask.png")
            opath = os.path.join(debug_dir, f"{debug_prefix}_overlay.png")

            cv2.imwrite(mpath, mask)

            overlay = img.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            cv2.imwrite(opath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            log.info("debug wrote: %s and %s", mpath, opath)

        return img, mask
