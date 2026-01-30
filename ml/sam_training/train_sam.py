import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from segment_anything import sam_model_registry
from datetime import datetime

from dataset import FeedbackDataset
from losses import sam_segmentation_loss
import config

import cv2
import numpy as np
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def debug_save_predictions(
    epoch: int,
    gt_masks: torch.Tensor,        # (B,1,H,W)
    pred_logits: torch.Tensor,     # (B,1,H,W)
    sample_types: torch.Tensor,    # (B,)
    ds_idxs: torch.Tensor,         # (B,) dataset indices
    image_paths,                   # list[str] length B
    save_root: str = "ml/sam_training/debug",
    max_samples: int = 3
):
    """
    Save debug images for visual inspection.
    Uses image_paths from the current batch (robust with samplers).
    """
    Path(save_root).mkdir(parents=True, exist_ok=True)
    epoch_dir = Path(save_root) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    B = pred_logits.shape[0]
    pick = torch.randperm(B)[: min(max_samples, B)].tolist()

    for bi in pick:
        ds_idx = int(ds_idxs[bi].item())
        st = int(sample_types[bi].item())
        label = "not_a_building" if st == 0 else ("missing_building" if st == 1 else "wrong_geometry")

        img_path = image_paths[bi]
        img = cv2.imread(img_path)
        if img is None:
            # fallback: skip if path broken
            continue

        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = gt_masks[bi, 0].detach().cpu().numpy()
        gt = (gt > 0.5).astype(np.uint8) * 255

        pred = torch.sigmoid(pred_logits[bi, 0]).detach().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8) * 255

        overlay_gt = img.copy()
        overlay_gt[gt > 0] = [0, 255, 0]      # green GT

        overlay_pred = img.copy()
        overlay_pred[pred > 0] = [255, 0, 0]  # red Pred

        base = epoch_dir / f"{label}_ds{ds_idx}_b{bi}"
        cv2.imwrite(str(base) + "_img.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(base) + "_gt.png", gt)
        cv2.imwrite(str(base) + "_pred.png", pred)
        cv2.imwrite(str(base) + "_overlay_gt.png", cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(base) + "_overlay_pred.png", cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))


def build_sampler(dataset: FeedbackDataset):
    # Your dataset counts are not extremely imbalanced now,
    # so 10x on wrong_geometry is often too aggressive.
    # Use a milder boost (e.g. 3x) so missing/neg still show up.
    weights = []
    for item in dataset.data:
        # if item["type"] == "wrong_geometry":
        #     weights.append(3.0)
        # elif item["type"] == "missing_building":
        #     weights.append(1.2)
        # else:
        #     weights.append(1.0)
        if item["type"] == "wrong_geometry":
            weights.append(1.5)
        elif item["type"] == "missing_building":
            weights.append(1.3)
        else:
            weights.append(1.1)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def freeze_sam(sam):
    for name, param in sam.named_parameters():
        if "mask_decoder" not in name:
            param.requires_grad = False


def _repeat_to(t: torch.Tensor, target_b: int):
    if t.shape[0] == target_b:
        return t
    if t.shape[0] == 1:
        return t.repeat(target_b, *([1] * (t.ndim - 1)))
    if target_b % t.shape[0] == 0:
        reps = target_b // t.shape[0]
        return t.repeat(reps, *([1] * (t.ndim - 1)))
    raise RuntimeError(f"Cannot repeat tensor of batch {t.shape[0]} to {target_b}")


def run_decoder(sam, image_embeddings, boxes_or_none):
    B = image_embeddings.shape[0]

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=boxes_or_none, masks=None)
    n_sparse = sparse_embeddings.shape[1] if sparse_embeddings.ndim == 3 else 0
    image_pe = sam.prompt_encoder.get_dense_pe()

    target_b = B if (boxes_or_none is None or n_sparse == 0) else B * n_sparse
    dense_embeddings = _repeat_to(dense_embeddings, target_b)
    image_pe = _repeat_to(image_pe, target_b)

    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    if low_res_masks.shape[0] != B:
        k = low_res_masks.shape[0] // B
        low_res_masks = low_res_masks.view(B, k, 1, 256, 256)[:, 0]

    return low_res_masks


def train():
    print("🚀 Loading SAM model")
    sam = sam_model_registry["vit_b"](checkpoint=config.CHECKPOINT).to(DEVICE)
    freeze_sam(sam)

    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.train()

    print("📦 Loading dataset")
    dataset = FeedbackDataset(config.DATASET_JSON, sam=sam, device=DEVICE)

    sampler = build_sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": sam.mask_decoder.transformer.parameters(), "lr": config.LR},
            {"params": sam.mask_decoder.output_hypernetworks_mlps.parameters(), "lr": config.LR * 2},
            {"params": sam.mask_decoder.iou_prediction_head.parameters(), "lr": config.LR * 2},
        ],
        weight_decay=1e-4
    )

    print("🧠 Training started")

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        steps = 0
        count = {"missing_building": 0, "wrong_geometry": 0, "not_a_building": 0}

        # ✅ UNPACK 7 VALUES (matches your dataset.py)
        for image_embeddings, masks, sample_types, boxes, has_box, ds_idxs, image_paths in loader:
            image_embeddings = image_embeddings.to(DEVICE)
            masks = masks.to(DEVICE)
            sample_types = sample_types.to(DEVICE)
            boxes = boxes.to(DEVICE).float()
            has_box = has_box.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            B = image_embeddings.shape[0]
            pred_masks_full = torch.zeros((B, 1, 256, 256), device=DEVICE)

            # BOX DROPOUT
            effective_has_box = has_box.clone()
            if config.BOX_DROPOUT_PROB > 0:
                drop_mask = ((has_box == 1) & (torch.rand_like(has_box.float()) < config.BOX_DROPOUT_PROB))
                effective_has_box[drop_mask] = 0

            idx_box = torch.where(effective_has_box == 1)[0]
            idx_nobox = torch.where(effective_has_box == 0)[0]

            # box prompts B=1
            for i in idx_box.tolist():
                emb_i = image_embeddings[i:i+1]
                box_i = torch.clamp(boxes[i:i+1], 0, 1023)
                pred_masks_full[i:i+1] = run_decoder(sam, emb_i, box_i)

            # no-box prompts batched
            if idx_nobox.numel() > 0:
                emb_nb = image_embeddings[idx_nobox]
                pred_masks_full[idx_nobox] = run_decoder(sam, emb_nb, None)

            pred_masks = torch.nn.functional.interpolate(
                pred_masks_full,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            pred_masks = torch.clamp(pred_masks, -20, 20)

            # DEBUG (first batch every 5 epochs)
            if epoch % 5 == 0 and steps == 0:
                debug_save_predictions(
                    epoch=epoch,
                    gt_masks=masks,
                    pred_logits=pred_masks,
                    sample_types=sample_types,
                    ds_idxs=ds_idxs,
                    image_paths=image_paths,
                )

            loss = sam_segmentation_loss(
                logits=pred_masks,
                targets=masks,
                sample_types=sample_types
            )

            if not torch.isfinite(loss):
                print("⚠️ NaN/Inf loss — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sam.mask_decoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            for t in sample_types.tolist():
                if t == 0:
                    count["not_a_building"] += 1
                elif t == 1:
                    count["missing_building"] += 1
                elif t == 2:
                    count["wrong_geometry"] += 1

        avg_loss = total_loss / max(1, steps)
        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Missing: {count['missing_building']} | "
            f"Wrong: {count['wrong_geometry']} | "
            f"Neg: {count['not_a_building']}"
        )

    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
    save_path = f"{config.SAVE_PATH}.epoch{epoch+1}.pth"
    torch.save(sam.mask_decoder.state_dict(), save_path)
    print("✅ Training complete")
    print(f"💾 Decoder saved at: {save_path}")


if __name__ == "__main__":
    print("TRAINING START:", datetime.now())
    train()
    print("TRAINING END:", datetime.now())
