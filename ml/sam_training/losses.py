# ml/losses.py
import torch
import torch.nn.functional as F

EPS = 1e-6


def dice_loss_per_sample(logits, targets):
    """
    logits:  (B,1,H,W)
    targets: (B,1,H,W) in {0,1}
    returns: (B,)
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + EPS
    return 1.0 - (num + EPS) / (den + EPS)


def bce_loss_per_sample(logits, targets):
    """
    returns (B,)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return bce.mean(dim=(1, 2, 3))


def edge_loss_per_sample(probs, targets):
    """
    probs:   (B,1,H,W) in [0,1]
    targets: (B,1,H,W) in {0,1}
    returns: (B,)
    """
    device = probs.device
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    pred_edge = torch.abs(F.conv2d(probs, sobel_x, padding=1)) + torch.abs(F.conv2d(probs, sobel_y, padding=1))
    gt_edge   = torch.abs(F.conv2d(targets, sobel_x, padding=1)) + torch.abs(F.conv2d(targets, sobel_y, padding=1))

    return F.l1_loss(pred_edge, gt_edge, reduction="none").mean(dim=(1, 2, 3))


#TYPE_WEIGHTS_ID = {0: 0.5, 1: 1.0, 2: 1.2}

TYPE_WEIGHTS_ID = {0: 1.4, 1: 1.0, 2: 1.1}

def sam_segmentation_loss(
    logits, targets, sample_types,
    lambda_bce=1.0, lambda_dice=1.0, lambda_edge=0.3
):
    """
    sample_types: (B,) tensor int: 0=not,1=missing,2=wrong
    """
    B = logits.shape[0]
    sample_types = sample_types.view(-1).to(logits.device)

    bce  = bce_loss_per_sample(logits, targets)          # (B,)
    dice = dice_loss_per_sample(logits, targets)         # (B,)
    edge = edge_loss_per_sample(torch.sigmoid(logits), targets)  # (B,)

    # only apply dice/edge when GT has positives (missing/wrong)
    is_pos = sample_types > 0
    dice = torch.where(is_pos, dice, torch.zeros_like(dice))
    edge = torch.where(is_pos, edge, torch.zeros_like(edge))

    weights = torch.ones((B,), device=logits.device, dtype=torch.float32)
    weights = torch.where(sample_types == 0, torch.tensor(TYPE_WEIGHTS_ID[0], device=logits.device), weights)
    weights = torch.where(sample_types == 1, torch.tensor(TYPE_WEIGHTS_ID[1], device=logits.device), weights)
    weights = torch.where(sample_types == 2, torch.tensor(TYPE_WEIGHTS_ID[2], device=logits.device), weights)


    bce = torch.where(sample_types == 0, 1.5 * bce, bce)

    loss_per = (lambda_bce * bce) + (lambda_dice * dice) + (lambda_edge * edge)
    return (loss_per * weights).mean()
