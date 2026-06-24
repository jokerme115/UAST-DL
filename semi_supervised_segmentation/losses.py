import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for highly imbalanced segmentation tasks.
    Reduces the weight of well-classified samples and focuses on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(inputs, dim=1)
        log_probs = F.log_softmax(inputs, dim=1)
        pt = torch.sum(targets_one_hot * probs, dim=1)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * focal_weight * torch.sum(targets_one_hot * log_probs, dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalDiceLoss(nn.Module):
    """Combined Focal Loss + Weighted Dice Loss for imbalanced segmentation."""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, dice_weight_c0=1.0, dice_weight_c1=5.0):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight_c0 = dice_weight_c0
        self.dice_weight_c1 = dice_weight_c1

    def _dice_loss(self, pred, target, smooth=1e-5, weight_c0=1.0, weight_c1=1.0):
        pred_soft = F.softmax(pred, dim=1)
        pred_c0 = pred_soft[:, 0, :, :]
        target_c0 = (target == 0).float()
        pred_c1 = pred_soft[:, 1, :, :]
        target_c1 = (target == 1).float()
        intersection_c0 = (pred_c0 * target_c0).sum(dim=(1, 2))
        union_c0 = pred_c0.sum(dim=(1, 2)) + target_c0.sum(dim=(1, 2))
        dice_c0 = (2.0 * intersection_c0 + smooth) / (union_c0 + smooth)
        intersection_c1 = (pred_c1 * target_c1).sum(dim=(1, 2))
        union_c1 = pred_c1.sum(dim=(1, 2)) + target_c1.sum(dim=(1, 2))
        dice_c1 = (2.0 * intersection_c1 + smooth) / (union_c1 + smooth)
        loss_dice = weight_c0 * (1 - dice_c0.mean()) + weight_c1 * (1 - dice_c1.mean())
        return loss_dice

    def forward(self, inputs, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        focal_loss_val = self.focal_loss(inputs, targets)
        dice_loss_val = self._dice_loss(
            inputs, targets,
            weight_c0=self.dice_weight_c0,
            weight_c1=self.dice_weight_c1
        )
        total_loss = focal_loss_val + dice_loss_val
        return total_loss, focal_loss_val, dice_loss_val


class BoundaryLoss(nn.Module):
    """Boundary loss for reinforcing edge learning. Optional component."""
    def __init__(self, smooth=1e-5):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth
        self.laplacian_kernel = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

    def forward(self, inputs, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        if inputs.is_cuda:
            self.laplacian_kernel = self.laplacian_kernel.cuda()
        probs = F.softmax(inputs, dim=1)
        pred_boundary = F.conv2d(probs[:, 1:2], self.laplacian_kernel, padding=1)
        target_boundary = F.conv2d(targets.float().unsqueeze(1), self.laplacian_kernel, padding=1)
        target_boundary = torch.abs(target_boundary) > 0.1
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred_boundary,
            target_boundary.float(),
            reduction='mean'
        )
        return boundary_loss
