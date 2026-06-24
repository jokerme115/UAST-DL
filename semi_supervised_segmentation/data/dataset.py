import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import semi_supervised_segmentation.config as config
from semi_supervised_segmentation.config import MEAN, STD


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


class CassavaDataset(Dataset):
    """
    Cassava dataset class
    
    Args:
        image_paths (list): List of image paths
        label_paths (list, optional): List of label paths
        transform (callable, optional): Image transform function (Albumentations pipeline)
        is_supervised (bool, optional): Whether this is supervised data
    """
    def __init__(self, image_paths, label_paths=None, transform=None, is_supervised=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.is_supervised = is_supervised
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a data item
        
        Args:
            idx (int): Data index
            
        Returns:
            tuple: (image, mask) or image (unsupervised)
        """
        # Read image
        image_path = self.image_paths[idx]
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = None
        if self.is_supervised and self.label_paths is not None:
            label_path = self.label_paths[idx]
            
            # Determine label file type based on file extension
            _, ext = os.path.splitext(label_path)
            ext = ext.lower()
            
            if ext == '.png':
                # Read PNG image directly as mask
                try:
                    mask = imread_unicode(label_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"Failed to read PNG mask: {label_path}")
                    
                    # Ensure mask dimensions are correct (Albumentations will handle Resize, but this is a safety check)
                    # Must ensure image and mask dimensions are consistent before passing to Albumentations
                    if mask.shape[:2] != image.shape[:2]:
                         h, w = image.shape[:2]
                         mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Ensure correct data type
                    mask = mask.astype(np.uint8)
                    
                    # Ensure label values are within valid range [0, 1]
                    # If mask is 0-255, convert it to 0-1
                    if mask.max() > 1:
                        mask = (mask > 0).astype(np.uint8)
                    
                except Exception as e:
                    print(f"Error processing PNG mask {label_path}: {str(e)}")
                    raise
                    
            elif ext == '.txt':
                # Process text format labels (polygon annotations)
                try:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    
                    with open(label_path, 'r', encoding='utf-8') as f:  # Use UTF-8 encoding
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Parse one line of data: class ID + coordinate point pairs
                            data = list(map(float, line.split()))
                            if len(data) < 3:
                                continue  # At least class ID and one coordinate point (2 values) needed
                            
                            class_id = int(data[0])
                            # Extract coordinate point pairs (x1, y1, x2, y2, ...)
                            coords = np.array(data[1:]).reshape(-1, 2)
                            # Convert to pixel coordinates (if normalized coordinates)
                            # Assume txt contains normalized coordinates [0, 1]
                            h, w = image.shape[:2]
                            pixel_coords = coords * np.array([w, h])
                            pixel_coords = pixel_coords.astype(np.int32)
                            
                            # Draw polygon
                            # Note: In OpenCV, class_id is typically used as pixel value
                            # Assume class_id starts from 0, 0 is background?
                            # In practice, annotation files usually only contain foreground.
                            # Here we assume class_id is the foreground class ID.
                            cv2.fillPoly(mask, [pixel_coords], color=class_id)
                            
                except Exception as e:
                    print(f"Error processing TXT label {label_path}: {str(e)}")
                    # Return all-black mask on error
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                raise ValueError(f"Unsupported label file format: {ext} for file {label_path}")
            
        # Apply transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
        
        if self.is_supervised:
            # Ensure mask is LongTensor and has no channel dimension (H, W)
            if mask is not None:
                # Albumentations output mask is [H, W] or [H, W, 1] depending on setup
                # ToTensorV2 usually keeps it as tensor.
                # Check shape
                if mask.ndim == 3:
                    mask = mask.squeeze(-1) # Remove channel dim if present
                mask = mask.long()
            return image, mask
        else:
            return image


def get_transforms():
    """
    Get data transforms (Albumentations)
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Training set transforms: augmentation strategy (crop to fixed size first, then augment, no scaling)
    train_transform = A.Compose([
        # Ensure image dimensions are at least the crop size, pad with 0 for insufficient areas
        A.PadIfNeeded(min_height=config.IMAGE_SIZE[0], min_width=config.IMAGE_SIZE[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        
        # Elastic deformation & grid distortion
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
        ], p=0.3),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.8),
            A.RGBShift(p=0.8),
        ], p=0.5),
        
        # Noise & blur
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.3),
        
        # CoarseDropout (similar to Cutout)
        A.CoarseDropout(max_holes=8, max_height=config.IMAGE_SIZE[0]//8, max_width=config.IMAGE_SIZE[1]//8, 
                        min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.3),
        
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=config.IMAGE_SIZE[0], min_width=config.IMAGE_SIZE[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    return train_transform, val_transform

def get_semi_transforms():
    """
    Get semi-supervised data transforms
    Returns:
        dict: Contains 'geo', 'weak', 'strong', 'val' transforms
    """
    # 1. Geometric transforms (shared)
    geo_transform = A.Compose([
        A.PadIfNeeded(min_height=config.IMAGE_SIZE[0], min_width=config.IMAGE_SIZE[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    ])

    # 2. Weak augmentation (normalization only)
    weak_transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    # 3. Strong augmentation (color, noise, occlusion + normalization)
    strong_transform = A.Compose([
        # Color transforms - reduced intensity to preserve disease color features
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5), # Limit range, reduce probability
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        ], p=0.8),
        
        # Noise & blur
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.5),
        
        # CoarseDropout (Cutout)
        A.CoarseDropout(max_holes=8, max_height=config.IMAGE_SIZE[0]//8, max_width=config.IMAGE_SIZE[1]//8, 
                        min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.5),
        
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    # 4. Validation transforms (standard, with Crop)
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=config.IMAGE_SIZE[0], min_width=config.IMAGE_SIZE[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    # 5. Sliding window validation transforms (no Crop, keep full image)
    val_sliding_transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    return {
        'geo': geo_transform,
        'weak': weak_transform,
        'strong': strong_transform,
        'val': val_transform,
        'val_sliding': val_sliding_transform
    }

class CassavaSemiDataset(Dataset):
    """
    Cassava semi-supervised dataset class
    Supports returning both strongly and weakly augmented images simultaneously
    """
    def __init__(self, image_paths, label_paths=None, transforms=None, mode='labeled'):
        """
        Args:
            image_paths: List of image paths
            label_paths: List of label paths (for labeled data)
            transforms: Dictionary containing 'geo', 'weak', 'strong' (from get_semi_transforms)
            mode: 'labeled' or 'unlabeled' or 'val'
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.mode = mode
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'labeled':
            mask = None
            if self.label_paths:
                label_path = self.label_paths[idx]
                # ... (reuse the previous mask reading logic) ...
                # For brevity, the mask reading logic is copied and simplified here, or we should refactor.
                # Given that only replace is possible, I will fully implement the logic.
                
                _, ext = os.path.splitext(label_path)
                ext = ext.lower()
                try:
                    if ext == '.png':
                        mask = imread_unicode(label_path, cv2.IMREAD_GRAYSCALE)
                        if mask.shape[:2] != image.shape[:2]:
                             h, w = image.shape[:2]
                             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        if mask.max() > 1:
                            mask = (mask > 0).astype(np.uint8)
                    elif ext == '.txt':
                        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                        with open(label_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line: continue
                                data = list(map(float, line.split()))
                                if len(data) < 3: continue
                                class_id = int(data[0])
                                coords = np.array(data[1:]).reshape(-1, 2)
                                h, w = image.shape[:2]
                                pixel_coords = (coords * np.array([w, h])).astype(np.int32)
                                cv2.fillPoly(mask, [pixel_coords], color=class_id)
                except Exception as e:
                    print(f"Error: {e}")
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            # Labeled: Geo -> Weak
            if self.transforms:
                # 1. Geo
                if 'geo' in self.transforms:
                    aug = self.transforms['geo'](image=image, mask=mask)
                    image = aug['image']
                    mask = aug['mask']
                
                # 2. Weak (Intensity)
                if 'weak' in self.transforms:
                    aug = self.transforms['weak'](image=image)
                    image = aug['image'] # Tensor
                
                if mask is not None:
                    # If mask is a numpy array, convert to tensor
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask).long()
                    else:
                        mask = mask.long()
            
            return image, mask

        elif self.mode == 'unlabeled':
            # Unlabeled: Geo -> (Weak, Strong)
            if self.transforms:
                # 1. Geo (Shared)
                if 'geo' in self.transforms:
                    aug = self.transforms['geo'](image=image)
                    image_geo = aug['image']
                else:
                    image_geo = image

                # 2. Weak
                image_weak = self.transforms['weak'](image=image_geo)['image']
                
                # 3. Strong
                image_strong = self.transforms['strong'](image=image_geo)['image']
                
                return image_weak, image_strong
            else:
                return image

        elif self.mode == 'val':
            mask = None
            if self.label_paths:
                label_path = self.label_paths[idx]
                _, ext = os.path.splitext(label_path)
                try:
                    if ext.lower() == '.png':
                        mask = imread_unicode(label_path, cv2.IMREAD_GRAYSCALE)
                        if mask.shape[:2] != image.shape[:2]:
                             h, w = image.shape[:2]
                             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        if mask.max() > 1:
                            mask = (mask > 0).astype(np.uint8)
                    elif ext.lower() == '.txt':
                        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                        with open(label_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line: continue
                                data = list(map(float, line.split()))
                                if len(data) < 3: continue
                                class_id = int(data[0])
                                coords = np.array(data[1:]).reshape(-1, 2)
                                h, w = image.shape[:2]
                                pixel_coords = (coords * np.array([w, h])).astype(np.int32)
                                cv2.fillPoly(mask, [pixel_coords], color=class_id)
                except Exception as e:
                    # Create dummy mask on failure
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # If no label paths or loading failed completely (though try-except handles it), ensure mask exists for transforms
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if self.transforms and 'val' in self.transforms:
                aug = self.transforms['val'](image=image, mask=mask)
                image = aug['image']
                mask = aug['mask']
            
            # Ensure mask is long tensor
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            elif isinstance(mask, torch.Tensor):
                mask = mask.long()
            
            return image, mask
