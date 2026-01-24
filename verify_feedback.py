import json
import os
import sys

import numpy as np
import cv2

def load_gray(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read: {path}")
    return m

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1, y1]

def touches_border(bbox, H, W):
    x0,y0,x1,y1 = bbox
    return (x0 == 0) or (y0 == 0) or (x1 == W-1) or (y1 == H-1)

def main(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print("---- META ----")
    print("id:", meta.get("id"))
    print("type:", meta.get("type"))
    print("timestamp:", meta.get("timestamp"))
    print("image:", meta.get("image"))
    print("mask:", meta.get("mask"))
    print("debug_mask:", meta.get("debug_mask"))
    print("bbox_xyxy(meta):", meta.get("bbox_xyxy"))
    print("has aoi_bounds?:", "aoi_bounds" in meta)

    # 1) existence
    for k in ["image", "mask", "debug_mask"]:
        p = meta.get(k)
        if p:
            print(f"exists({k}):", os.path.exists(p), p)

    # 2) load mask
    mask_path = meta["mask"]
    m = load_gray(mask_path)
    H, W = m.shape

    uniq = np.unique(m)
    cov = float((m > 0).mean())
    s = int(m.sum())

    print("\n---- MASK STATS ----")
    print("shape:", (H, W), "dtype:", m.dtype, "unique:", uniq.tolist(), "sum:", s)
    print("coverage (m>0):", cov, f"({int((m>0).sum())}/{m.size})")

    # 3) check if “inverted” likely
    inv = (m == 0).astype(np.uint8) * 255
    inv_cov = float((inv > 0).mean())
    print("coverage if inverted:", inv_cov)

    # 4) bbox checks
    bbox = bbox_from_mask(m)
    inv_bbox = bbox_from_mask(inv)
    print("\n---- BBOX ----")
    print("bbox from mask:", bbox)
    print("touches border:", touches_border(bbox, H, W) if bbox else None)

    print("bbox from inverted:", inv_bbox)
    print("touches border (inverted):", touches_border(inv_bbox, H, W) if inv_bbox else None)

    meta_bbox = meta.get("bbox_xyxy")
    if meta_bbox and bbox:
        # meta stored as xyxy but may be inclusive; we just compare roughly
        print("bbox delta (meta - computed):", [meta_bbox[i] - bbox[i] for i in range(4)])

    # 5) quick warning heuristics
    print("\n---- WARNINGS ----")
    if cov > 0.90:
        print("WARNING: mask coverage > 0.90 (mask is almost full). Likely CRS/rasterize/inversion issue.")
    if bbox and touches_border(bbox, H, W):
        print("WARNING: bbox touches border -> mask probably bleeding to crop edges.")
    if len(uniq) > 2 or not set(uniq.tolist()).issubset({0, 255}):
        print("WARNING: mask not binary {0,255}.")

    # 6) optional debug outputs (write overlay pngs)
    img_path = meta["image"]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is not None:
        # draw bbox
        if bbox:
            x0,y0,x1,y1 = bbox
            img2 = img.copy()
            img2 = np.ascontiguousarray(img2)  # avoid OpenCV layout errors
            cv2.rectangle(img2, (x0,y0), (x1,y1), (0,255,0), 2)
            out = os.path.splitext(img_path)[0] + "_bbox.png"
            cv2.imwrite(out, img2)
            print("\nWrote:", out)

        # write side-by-side mask previews
        m_vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        side = np.hstack([img, m_vis])
        out2 = os.path.splitext(img_path)[0] + "_img_mask.png"
        cv2.imwrite(out2, side)
        print("Wrote:", out2)
    else:
        print("\nNOTE: could not read image for overlay:", img_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 verify_feedback.py <path/to/metadata.json>")
        sys.exit(2)
    main(sys.argv[1])
