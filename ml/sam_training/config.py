# CHECKPOINT = "ml/checkpoints/sam_vit_b.pth"
# DATASET_JSON = "ml/feedback/feedback_dataset_clean.json"

# BATCH_SIZE = 8          # ideal for 168 samples
# LR = 8e-5               # decoder-only fine-tuning sweet spot
# EPOCHS = 10             # not 60 yet – stop earlier & inspect

# DEVICE = "cuda"
# SAVE_PATH = "ml/sam_training/checkpoints/sam_decoder_finetuned.pth"

# # Box robustness
# BOX_DROPOUT_PROB = 0.20   # ✅ keep this

# Loss balance
POS_WEIGHT = 1.2
NEG_WEIGHT = 0.6          # strong suppression of false positives



CHECKPOINT = "ml/checkpoints/sam_vit_b.pth"
DATASET_JSON = "ml/feedback/feedback_dataset_clean.json"

BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 16

DEVICE = "cuda"
SAVE_PATH = "ml/sam_training/checkpoints/sam_decoder_finetuned.pth"

# Teach some robustness, but don't overdo it since your pipeline mostly uses box prompts
BOX_DROPOUT_PROB = 0.12
