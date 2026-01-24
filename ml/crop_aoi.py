import rasterio
from rasterio.windows import from_bounds

def crop_aoi(src_tif, dst_tif, minx, miny, maxx, maxy):
    with rasterio.open(src_tif) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        profile = src.profile
        profile.update(
            height=window.height,
            width=window.width,
            transform=src.window_transform(window)
        )
        with rasterio.open(dst_tif, "w", **profile) as dst:
            dst.write(src.read(window=window))
