import os
import cv2
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)

FEEDBACK_ROOT = "ml/feedback"

def ensure_feedback_dirs():
    for t in ["missing_building", "wrong_geometry", "not_a_building"]:
        for sub in ["images", "masks", "metadata", "debug"]:
            os.makedirs(f"{FEEDBACK_ROOT}/{t}/{sub}", exist_ok=True)

def _largest_component_not_touching_border(bin_mask: np.ndarray) -> np.ndarray:
    """
    Keep the largest connected component that does NOT touch the image border.
    If all components touch border, fall back to original mask (no component filtering).
    """
    H, W = bin_mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num <= 1:
        return bin_mask  # nothing to filter

    best_idx = -1
    best_area = 0

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        touches = (x == 0) or (y == 0) or (x + w >= W) or (y + h >= H)
        if touches:
            continue
        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx == -1:
        # Everything touches border -> don't force a "largest contour" fill
        return bin_mask

    out = np.zeros_like(bin_mask)
    out[labels == best_idx] = 255
    return out

def refine_mask(mask: np.ndarray, debug_save_path: str | None = None) -> np.ndarray:
    """
    Input: mask as numpy array (H,W), ideally uint8 {0,255}.
    Output: cleaned binary mask uint8 {0,255}.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(mask)}")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    # Ensure uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Force binary (IMPORTANT: your mask is already {0,255}; this guarantees it)
    bin_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Morph cleanup (small kernel so we don't grow everything)
    kernel = np.ones((5, 5), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)   # remove tiny specks
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)  # close small holes

    # Critical: remove border-touching junk so we don't select a full-frame contour
    cleaned = _largest_component_not_touching_border(bin_mask)

    # If still too huge, DO NOT try to "fill largest contour" (that causes full-frame fill)
    cov = float((cleaned > 0).mean())
    if cov > 0.90:
        logger.warning("refine_mask: coverage still huge (%.3f). Leaving mask as-is (likely upstream issue).", cov)

    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, cleaned)

    return cleaned
