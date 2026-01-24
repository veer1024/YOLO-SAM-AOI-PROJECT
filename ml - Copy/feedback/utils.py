# ml/feedback/utils.py
import os

FEEDBACK_ROOT = "ml/feedback"

def ensure_feedback_dirs():
    for t in ["missing_building", "wrong_geometry"]:
        for sub in ["images", "masks", "metadata"]:
            os.makedirs(f"{FEEDBACK_ROOT}/{t}/{sub}", exist_ok=True)
