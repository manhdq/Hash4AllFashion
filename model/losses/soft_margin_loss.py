import torch
import torch.nn.functional as F


def soft_margin_loss(x):
    target = torch.ones_like(x)
    return F.soft_margin_loss(x, target, reduction="none")