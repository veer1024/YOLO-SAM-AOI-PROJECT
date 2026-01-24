import os
import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry

from dataset import FeedbackDataset
from losses import combined_loss
import config
from datetime import datetime


# --------------------------------------------------
# DEVICE
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔒 IMPORTANT: Disable AMP for stability (small dataset)
USE_AMP = False




# --------------------------------------------------
# FREEZE SAM (ONLY TRAIN MASK DECODER)
# --------------------------------------------------
def freeze_sam(sam):
    for name, param in sam.named_parameters():
        if "mask_decoder" not in name:
            param.requires_grad = False


# --------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------
def train():
    print("🚀 Loading SAM model")

    sam = sam_model_registry["vit_b"](
        checkpoint=config.CHECKPOINT
    ).to(DEVICE)

    freeze_sam(sam)

    # Explicitly freeze prompt encoder
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # Set correct modes
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.train()

    print("📦 Loading dataset")
    #dataset = FeedbackDataset(config.DATASET_JSON)
    dataset = FeedbackDataset(
        config.DATASET_JSON,
        sam=sam,
        device=DEVICE
    )


    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
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
        valid_batches = 0
        pos_count = 0
        neg_count = 0

        for image_embeddings, masks, sample_type in loader:
            image_embeddings = image_embeddings.to(DEVICE)
            masks = masks.to(DEVICE)
            sample_type = sample_type.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )

            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_masks = torch.nn.functional.interpolate(
                low_res_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

            pred_masks = torch.clamp(pred_masks, -20, 20)

            loss = combined_loss(pred_masks, masks, sample_type)


            # ---------------- NaN GUARD ----------------
            if not torch.isfinite(loss):
                print("⚠️ NaN loss detected — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            # ---------------- BACKWARD ----------------
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                sam.mask_decoder.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # ---------------- STATS ----------------
            total_loss += loss.item()
            valid_batches += 1
            pos_count += int((sample_type > 0).sum().item())
            neg_count += int((sample_type == 0).sum().item())

        avg_loss = total_loss / max(1, valid_batches)

        print("mask min/max:", masks.min().item(), masks.max().item(), "dtype:", masks.dtype)

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Valid batches: {valid_batches} | "
            f"Pos: {pos_count} | Neg: {neg_count}"
        )

    # --------------------------------------------------
    # SAVE TRAINED DECODER
    # --------------------------------------------------
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    save_path = f"{config.SAVE_PATH}.epoch{epoch+1}.pth"
    torch.save(sam.mask_decoder.state_dict(), save_path)

    print("✅ Training complete")
    print(f"💾 Decoder saved at: {save_path}")


# --------------------------------------------------
# ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    print("TRAINING START: " +str(datetime.now()))
    train()
    print("TRAINING END: " +str(datetime.now()))
