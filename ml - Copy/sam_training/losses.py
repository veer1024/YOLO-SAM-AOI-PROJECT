import torch
import torch.nn.functional as F


# --------------------------------------------------
# DICE LOSS (ONLY FOR POSITIVE SAMPLES)
# --------------------------------------------------
def dice_loss(pred, target, eps=1e-6):
    """
    pred   : logits (B,1,H,W)
    target : binary mask (B,1,H,W)
    """
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice


# --------------------------------------------------
# COMBINED LOSS WITH POS / NEG WEIGHTING
# --------------------------------------------------


def combined_loss(
    pred,
    target,
    sample_type,
    dice_weight=0.5,
    min_target_area=50.0
):
    """
    pred        : (B,1,H,W) logits
    target      : (B,1,H,W) {0,1}
    sample_type : (B,) int
                  0 = not_a_building
                  1 = missing_building
                  2 = wrong_geometry
    """

    device = pred.device
    total_loss = torch.tensor(0.0, device=device)

    # class weights
    w_not = 0.3
    w_missing = 1.2
    w_wrong = 1.8

    # --------------------------------------------------
    # POSITIVE SAMPLES
    # --------------------------------------------------
    pos_idx = sample_type > 0
    if pos_idx.any():
        pred_pos = pred[pos_idx]
        target_pos = target[pos_idx]
        stype = sample_type[pos_idx]

        target_area = target_pos.sum(dim=(1, 2, 3))
        valid = target_area > min_target_area

        if valid.any():
            # 🔑 reduce BCE to per-sample
            bce = F.binary_cross_entropy_with_logits(
                pred_pos[valid],
                target_pos[valid],
                reduction="none"
            )
            bce = bce.mean(dim=(1, 2, 3))  # (N,)

            dice = dice_loss(
                pred_pos[valid],
                target_pos[valid]
            )  # (N,)

            # per-sample weights
            weights = torch.where(
                stype[valid] == 2,
                torch.full_like(stype[valid], w_wrong, dtype=torch.float),
                torch.full_like(stype[valid], w_missing, dtype=torch.float)
            )

            pos_loss = (bce + dice_weight * dice) * weights
            total_loss += pos_loss.mean()

    # --------------------------------------------------
    # NEGATIVE SAMPLES
    # --------------------------------------------------
    neg_idx = sample_type == 0
    if neg_idx.any():
        pred_neg = torch.sigmoid(pred[neg_idx])
        neg_activation = pred_neg.mean(dim=(1, 2, 3))
        total_loss += w_not * neg_activation.mean()

    return total_loss


