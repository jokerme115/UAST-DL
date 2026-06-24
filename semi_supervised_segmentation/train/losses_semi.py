import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        log_probs_flat = log_probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() == 0:
            return inputs.sum() * 0.0

        targets_valid = targets_flat[valid_mask]
        probs_valid = probs_flat[valid_mask, :]
        log_probs_valid = log_probs_flat[valid_mask, :]

        indices = torch.arange(targets_valid.numel(), device=inputs.device)
        pt = probs_valid[indices, targets_valid]
        log_pt = log_probs_valid[indices, targets_valid]

        if isinstance(self.alpha, (list, tuple)):
            alpha_t = torch.as_tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)[targets_valid]
        elif torch.is_tensor(self.alpha) and self.alpha.numel() > 1:
            alpha_t = self.alpha.to(device=inputs.device, dtype=inputs.dtype)[targets_valid]
        else:
            alpha_t = torch.as_tensor(float(self.alpha), device=inputs.device, dtype=inputs.dtype)

        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * log_pt

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] logits
        # targets: [B, H, W] labels
        
        num_classes = inputs.size(1)
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Tversky Index for each class
        # TP = inputs * targets
        # FP = inputs * (1 - targets)
        # FN = (1 - inputs) * targets
        
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs * (1 - targets)).sum(dim=(0, 2, 3))
        fn = ((1 - inputs) * targets).sum(dim=(0, 2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal Tversky
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky.mean()

class DiceLossWithMask(nn.Module):
    def __init__(self, n_classes=2, weight=None, smooth=1e-5):
        super(DiceLossWithMask, self).__init__()
        self.n_classes = n_classes
        self.weight = weight # list or tensor of weights per class
        self.smooth = smooth

    def forward(self, inputs, targets, mask=None):
        # inputs: [B, C, H, W] logits or probs
        # targets: [B, H, W]
        # mask: [B, 1, H, W] or None, valid region mask
        
        if inputs.dim() > 3 and inputs.size(1) == self.n_classes:
             inputs = F.softmax(inputs, dim=1)
        
        targets_one_hot = F.one_hot(targets, self.n_classes).permute(0, 3, 1, 2).float()
        
        if mask is not None:
            # Apply mask to both inputs and targets
            # mask shape should be [B, 1, H, W]
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask
            
        # Compute Dice for each class
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_score
        
        if self.weight is not None:
            if isinstance(self.weight, list):
                w = torch.tensor(self.weight).to(inputs.device)
            else:
                w = self.weight
            
            # Normalize weights
            # w = w / w.sum() * self.n_classes 
            # Or just weighted average
            dice_loss = (dice_loss * w).sum() / w.sum()
        else:
            dice_loss = dice_loss.mean()
            
        return dice_loss

class CE_Dice_FocalTverskyLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.02, beta=0.98, gamma=4.0, dice_weights=[1.0, 15.0]):
        super(CE_Dice_FocalTverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = DiceLossWithMask(n_classes=num_classes, weight=dice_weights)
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        loss_ft = self.ft(inputs, targets)
        
        return 0.4 * loss_ce + 0.3 * loss_dice + 0.3 * loss_ft
