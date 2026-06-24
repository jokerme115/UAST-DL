import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from semi_supervised_segmentation.config import MEAN, STD

def update_ema_variables(model, ema_model, alpha, global_step, copy_bn_stats=False):
    """
    Update EMA model parameters
    Args:
        model: Student model
        ema_model: Teacher model (EMA)
        alpha: Decay coefficient
        global_step: Current iteration step
    """
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    model_buffers = dict(model.named_buffers())
    for name, ema_buf in ema_model.named_buffers():
        buf = model_buffers.get(name, None)
        if buf is None:
            continue
        if copy_bn_stats and (name.endswith('running_mean') or name.endswith('running_var')):
            ema_buf.data.copy_(buf.data)
            continue
        if not torch.is_floating_point(ema_buf):
            ema_buf.data.copy_(buf.data)
        else:
            ema_buf.data.mul_(alpha).add_(buf.data, alpha=1 - alpha)

def sigmoid_rampup(current, rampup_length):
    """Sigmoid ramp-up curve"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    """Get the current consistency weight"""
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    """
    Generate CutMix mask
    Args:
        img_size: Image size (int)
        p: Trigger probability
        size_min: Minimum area ratio
        size_max: Maximum area ratio
    """
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask

# --- TP-RAM (Frequency Domain Mixing) Utilities ---

def extract_amp_spectrum(img_np):
    """Extract amplitude spectrum"""
    # img_np: [C, H, W]
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, pha_np = np.abs(fft), np.angle(fft)
    return amp_np

def low_freq_mutate_np(amp_src, amp_trg, L=0.1, degree=1):
    """Replace low-frequency components"""
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h,w))*L)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = random.uniform(0, degree)

    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * (1-ratio) + a_trg[:,h1:h2,w1:w2] * ratio
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def source_to_target_freq(src_img, amp_trg, L=0.1, degree=1):
    """Replace the low-frequency amplitude of the source image with that of the target image"""
    # src_img: [C, H, W] numpy array
    src_img_np = src_img
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L, degree=degree)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

class TPRAM_Mixer:
    """TP-RAM mixer wrapper class"""
    def __init__(self, L=0.005, device='cuda'):
        self.L = L
        self.device = device
    
    def __call__(self, source_imgs, target_imgs, current_iter, max_iters):
        """
        Apply frequency-domain mixing to images in the batch
        Args:
            source_imgs: [B, C, H, W] Tensor (Normalized)
            target_imgs: [B, C, H, W] Tensor (Normalized)
        Returns:
            mixed_imgs: [B, C, H, W] Tensor (Normalized)
        """
        # Denormalize to [0, 1] for better frequency mixing
        # Assuming source_imgs are on self.device or compatible
        device = source_imgs.device
        mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(device)
        std = torch.tensor(STD).view(1, 3, 1, 1).to(device)
        
        src_denorm = source_imgs * std + mean
        trg_denorm = target_imgs * std + mean
        
        # Convert to numpy for processing
        src_np = src_denorm.cpu().numpy()
        trg_np = trg_denorm.cpu().numpy()
        
        mixed_np = []
        for i in range(len(src_np)):
            amp_trg = extract_amp_spectrum(trg_np[i])
            img_freq = source_to_target_freq(src_np[i], amp_trg, L=self.L, degree=current_iter/max_iters)
            mixed_np.append(img_freq)
            
        mixed_tensor = torch.from_numpy(np.array(mixed_np)).float().to(device)
        
        # Renormalize
        mixed_tensor = (mixed_tensor - mean) / std
        
        return mixed_tensor
