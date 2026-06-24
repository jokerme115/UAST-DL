import numpy as np
import random


def extract_amp_spectrum(img_np):
    """
    Extract the amplitude spectrum of the image
    
    Args:
        img_np (numpy.ndarray): Input image, shape (B, C, H, W) or (C, H, W)
        
    Returns:
        numpy.ndarray: Amplitude spectrum, same shape as input
    """
    # Ensure input is a 4D array (B, C, H, W)
    if img_np.ndim == 3:
        img_np = img_np[np.newaxis, :, :, :]
    
    batch_size, channels, h, w = img_np.shape
    amp_list = []
    
    for b in range(batch_size):
        amp = []
        for c in range(channels):
            # Perform FFT on each channel individually
            fft = np.fft.fft2(img_np[b, c])
            amp.append(np.abs(fft))
        amp_list.append(np.array(amp))
    
    return np.array(amp_list)


def low_freq_mutate_np(amp_src, amp_trg, L=0.01, degree=1.0):
    """
    Low-frequency mutation mixing - based on reference implementation
    
    Args:
        amp_src (numpy.ndarray): Source amplitude spectrum, shape (B, C, H, W)
        amp_trg (numpy.ndarray): Target amplitude spectrum, shape (B, C, H, W)
        L (float, optional): Low-frequency region size, defaults to 0.01 consistent with reference implementation. Defaults to 0.01.
        degree (float, optional): Mixing degree. Defaults to 1.0.
        
    Returns:
        numpy.ndarray: Mixed amplitude spectrum, same shape as input
    """
    # Ensure input is a 4D array (B, C, H, W)
    assert amp_src.ndim == 4 and amp_trg.ndim == 4, f"Expected 4D input, got {amp_src.ndim}D and {amp_trg.ndim}D"
    
    batch_size = amp_src.shape[0]
    result = []
    
    for b in range(batch_size):
        # Apply fftshift to source amplitude spectrum
        a_src = np.fft.fftshift(amp_src[b], axes=(-2, -1))
        # Randomly select a target sample
        target_idx = random.randint(0, amp_trg.shape[0] - 1)
        a_trg = np.fft.fftshift(amp_trg[target_idx], axes=(-2, -1))
        
        # Ensure channel count matches
        assert a_src.shape[0] == a_trg.shape[0], f"Channel count mismatch: {a_src.shape[0]} vs {a_trg.shape[0]}"
        
        _, h, w = a_src.shape
        # Calculate low-frequency region size, reference code uses 2*β*min(H,W)
        b_size = int(np.floor(np.amin((h, w)) * L))
        c_h = int(np.floor(h / 2.0))
        c_w = int(np.floor(w / 2.0))
        
        # Calculate low-frequency region boundaries, ensuring they stay within image bounds
        h1 = max(0, c_h - b_size)
        h2 = min(h, c_h + b_size + 1)
        w1 = max(0, c_w - b_size)
        w2 = min(w, c_w + b_size + 1)
        
        # Low-frequency amplitude mixing - process channel by channel to avoid shape mismatch issues
        for c in range(a_src.shape[0]):
            a_src[c, h1:h2, w1:w2] = a_src[c, h1:h2, w1:w2] * (1 - degree) + a_trg[c, h1:h2, w1:w2] * degree
        
        # Apply inverse fftshift
        a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
        result.append(a_src)
    
    return np.array(result)


def source_to_target_freq(src_img, amp_trg, L=0.01, degree=1.0):
    """
    Source to target frequency transformation - based on reference implementation
    
    Args:
        src_img (numpy.ndarray): Source image, shape (B, C, H, W) or (C, H, W)
        amp_trg (numpy.ndarray): Target amplitude spectrum, shape (B, C, H, W)
        L (float, optional): Low-frequency region size, defaults to 0.01 consistent with reference implementation. Defaults to 0.01.
        degree (float, optional): Mixing degree. Defaults to 1.0.
        
    Returns:
        numpy.ndarray: Transformed image, same shape as input
    """
    # Ensure input is a 4D array (B, C, H, W)
    if src_img.ndim == 3:
        src_img = src_img[np.newaxis, :, :, :]
    
    batch_size, channels, height, width = src_img.shape
    result = np.zeros_like(src_img, dtype=np.float64)
    
    for b in range(batch_size):
        for c in range(channels):
            # Perform FFT on a single channel
            fft_src = np.fft.fft2(src_img[b, c].astype(np.float64))
            amp_src = np.abs(fft_src)
            pha_src = np.angle(fft_src)
            
            # Randomly select a target sample
            target_idx = np.random.randint(0, amp_trg.shape[0])
            
            # Process target amplitude spectrum, ensuring correct dimensions and fftshift
            a_trg = amp_trg[target_idx, c]  # Directly take the amplitude spectrum of the corresponding channel
            a_trg_shifted = np.fft.fftshift(a_trg)
            
            # Apply fftshift to source amplitude spectrum
            a_src_shifted = np.fft.fftshift(amp_src)
            
            # Calculate low-frequency region size - ensure consistency with reference implementation
            b_size = int(min(height, width) * L)
            c_h, c_w = height // 2, width // 2
            h1, h2 = max(0, c_h - b_size), min(height, c_h + b_size + 1)
            w1, w2 = max(0, c_w - b_size), min(width, c_w + b_size + 1)
            
            # Low-frequency amplitude mixing - ensure correct region calculation
            a_src_mixed_shifted = a_src_shifted.copy()
            # Ensure the mixing region does not go out of bounds
            a_src_mixed_shifted[h1:h2, w1:w2] = (a_src_shifted[h1:h2, w1:w2] * (1 - degree) + 
                                                a_trg_shifted[h1:h2, w1:w2] * degree)
            
            # Apply inverse fftshift
            a_src_mixed = np.fft.ifftshift(a_src_mixed_shifted)
            
            # Reconstruct image - keep phase, only change amplitude
            fft_src_ = a_src_mixed * np.exp(1j * pha_src)
            src_in_trg = np.fft.ifft2(fft_src_)
            
            # Take the real part and normalize pixel values to the [0, 255] range
            real_part = np.real(src_in_trg)
            # Normalize to [0, 255]
            min_val = np.min(real_part)
            max_val = np.max(real_part)
            if max_val > min_val:
                normalized = (real_part - min_val) / (max_val - min_val) * 255
            else:
                normalized = real_part
            
            result[b, c] = normalized.astype(np.uint8)
    
    # Keep output shape consistent with input
    # If input is 3D, return 3D
    if src_img.ndim == 3:
        result = result[0]
    
    return result


def obtain_cutmix_box(img_size, p=0.5, size_min=0.15, size_max=0.2, ratio_1=0.3, ratio_2=1/0.3):
    """
    Generate CutMix mask
    
    Args:
        img_size (int): Image size
        p (float, optional): Probability of generating a mask. Defaults to 0.5.
        size_min (float, optional): Minimum mask size ratio. Defaults to 0.15.
        size_max (float, optional): Maximum mask size ratio. Defaults to 0.2.
        ratio_1 (float, optional): Minimum aspect ratio. Defaults to 0.3.
        ratio_2 (float, optional): Maximum aspect ratio. Defaults to 1/0.3.
        
    Returns:
        numpy.ndarray: CutMix mask, shape (img_size, img_size)
    """
    mask = np.zeros((img_size, img_size))
    if random.random() > p:
        return mask
    
    # Ensure mask area ratio is within the specified range
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
