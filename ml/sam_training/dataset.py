# ml/sam_training/dataset.py
import cv2
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class FeedbackDataset(Dataset):
    def __init__(self, dataset_json, sam, device):
        with open(dataset_json) as f:
            self.data = json.load(f)

        self.sam = sam
        self.device = device
        self.TARGET_SIZE = 1024

        self.embedding_cache = {}
        self.sam.image_encoder.eval()

    def __len__(self):
        return len(self.data)

    def _get_image_embedding(self, image_path: str):
        if image_path in self.embedding_cache:
            return self.embedding_cache[image_path]

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        img = cv2.resize(img, (self.TARGET_SIZE, self.TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            img = self.sam.preprocess(img)
            embedding = self.sam.image_encoder(img).squeeze(0).cpu()  # (C,H,W)

        self.embedding_cache[image_path] = embedding
        return embedding

    def _load_mask(self, path: str):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to load mask: {path}")

        m = cv2.resize(m, (self.TARGET_SIZE, self.TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.uint8)
        return torch.from_numpy(m).unsqueeze(0).float()  # (1,H,W)

    # def __getitem__(self, idx):
    #     item = self.data[idx]

    #     image_path = item["image"]
    #     image_embedding = self._get_image_embedding(image_path)

    #     ftype = item["type"]

    #     # --- bbox optional ---
    #     has_box = (item.get("bbox") is not None)
    #     if has_box:
    #         bbox = torch.tensor(item["bbox"], dtype=torch.float32)  # (4,)
    #     else:
    #         bbox = torch.zeros((4,), dtype=torch.float32)

    #     # 0 = not_a_building, 1 = missing_building, 2 = wrong_geometry
    #     if ftype == "not_a_building":
    #         mask = torch.zeros((1, self.TARGET_SIZE, self.TARGET_SIZE), dtype=torch.float32)
    #         sample_type = torch.tensor(0, dtype=torch.long)

    #     elif ftype == "wrong_geometry":
    #         # ✅ support both formats
    #         if "correct_mask" in item:
    #             mask_path = item["correct_mask"]
    #         elif "mask" in item:
    #             mask_path = item["mask"]
    #         else:
    #             raise RuntimeError(f"Invalid wrong_geometry sample (no mask): {item}")

    #         mask = self._load_mask(mask_path).float()
    #         sample_type = torch.tensor(2, dtype=torch.long)

    #     elif ftype == "missing_building":
    #         mask = self._load_mask(item["mask"]).float()
    #         sample_type = torch.tensor(1, dtype=torch.long)

    #     else:
    #         raise RuntimeError(f"Unknown feedback type: {ftype}")

    #     return (
    #         image_embedding.contiguous(),   # (C,H,W)
    #         mask.contiguous(),              # (1,1024,1024)
    #         sample_type,                    # scalar long
    #         bbox.contiguous(),              # (4,)
    #         torch.tensor(1 if has_box else 0, dtype=torch.long),
    #         torch.tensor(idx, dtype=torch.long),  # ✅ dataset index for debug
    #     )

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        image_embedding = self._get_image_embedding(image_path)

        ftype = item["type"]
        has_box = (item.get("bbox") is not None)
        bbox = torch.tensor(item["bbox"], dtype=torch.float32) if has_box else torch.zeros((4,), dtype=torch.float32)

        if ftype == "not_a_building":
            mask = torch.zeros((1, self.TARGET_SIZE, self.TARGET_SIZE), dtype=torch.float32)
            sample_type = torch.tensor(0, dtype=torch.long)
        # elif ftype == "wrong_geometry":
        #     mask = self._load_mask(item["mask"]).float()
        #     sample_type = torch.tensor(2, dtype=torch.long)
        elif ftype == "wrong_geometry":
            if "correct_mask" in item:
                mask_path = item["correct_mask"]
            else:
                mask_path = item["mask"]
            mask = self._load_mask(mask_path).float()
            sample_type = torch.tensor(2, dtype=torch.long)
        elif ftype == "missing_building":
            mask = self._load_mask(item["mask"]).float()
            sample_type = torch.tensor(1, dtype=torch.long)
        else:
            raise RuntimeError(f"Unknown feedback type: {ftype}")

        return (
            image_embedding.contiguous(),
            mask.contiguous(),
            sample_type,
            bbox.contiguous(),
            torch.tensor(1 if has_box else 0, dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),   # ✅ add dataset index
            image_path,                            # ✅ add image path
        )
