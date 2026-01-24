CHECKPOINT = "ml/checkpoints/sam_vit_b.pth"
DATASET_JSON = "ml/feedback/feedback_dataset_clean.json"
#BATCH_SIZE = 4
BATCH_SIZE = 2
LR = 3e-5
#LR = 1e-4
#EPOCHS = 15
EPOCHS = 12
DEVICE = "cuda"  # or "cpu"
SAVE_PATH = "ml/sam_training/checkpoints/sam_decoder_finetuned.pth"


# ----------------------------
# LOSS WEIGHTS (if configurable)
# ----------------------------
POS_WEIGHT = 1.0
NEG_WEIGHT = 0.7   # 🔑 critical: < 1.0

