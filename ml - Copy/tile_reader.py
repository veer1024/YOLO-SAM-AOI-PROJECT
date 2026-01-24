import rasterio
from rasterio.windows import Window

def iter_tiles(tif_path, tile_size=1024):
    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)

                window = Window(x, y, w, h)
                transform = src.window_transform(window)
                img = src.read([1,2,3], window=window)

                yield img, transform
