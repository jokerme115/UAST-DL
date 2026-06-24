import os
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from .dataset import CassavaDataset, CassavaSemiDataset, get_transforms, get_semi_transforms

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from semi_supervised_segmentation.config import IMAGES_DIR, LABELS_PNG_DIR, UNLABELED_DIR
import semi_supervised_segmentation.config as config_module


def prepare_data(k_fold=False, fold_idx=0, num_folds=5):
    """
    Prepare training, validation, and unlabeled data
    
    Args:
        k_fold (bool): Whether to use K-fold cross-validation
        fold_idx (int): Current fold index (0 to num_folds-1)
        num_folds (int): Total number of folds
        
    Returns:
        dict: Dictionary containing training, validation, and unlabeled data loaders
    """
    # Initialize data lists
    train_image_paths = []
    train_label_paths = []
    val_image_paths = []
    val_label_paths = []
    unlabeled_image_paths = []
    
    # Get data transforms (using new semi-supervised transforms)
    transforms = get_semi_transforms()
    
    # Collect all labeled data (from train and val folders)
    all_labeled_images = []
    all_labeled_labels = []
    
    # 1. Scan train directory
    train_images_dir = os.path.join(IMAGES_DIR, 'train')
    train_labels_dir = os.path.join(LABELS_PNG_DIR, 'train')
    
    if os.path.exists(train_images_dir) and os.path.exists(train_labels_dir):
        for filename in os.listdir(train_images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(train_images_dir, filename)
                base_name = os.path.splitext(filename)[0]
                for ext in ['.png', '.txt']:
                    label_filename = f"{base_name}{ext}"
                    label_path = os.path.join(train_labels_dir, label_filename)
                    if os.path.exists(label_path):
                        all_labeled_images.append(image_path)
                        all_labeled_labels.append(label_path)
                        if not k_fold: # If not K-fold, keep original split
                            train_image_paths.append(image_path)
                            train_label_paths.append(label_path)
                        break
    
    # 2. Scan val directory
    val_images_dir = os.path.join(IMAGES_DIR, 'val')
    val_labels_dir = os.path.join(LABELS_PNG_DIR, 'val')
    
    if os.path.exists(val_images_dir) and os.path.exists(val_labels_dir):
        for filename in os.listdir(val_images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(val_images_dir, filename)
                base_name = os.path.splitext(filename)[0]
                for ext in ['.png', '.txt']:
                    label_filename = f"{base_name}{ext}"
                    label_path = os.path.join(val_labels_dir, label_filename)
                    if os.path.exists(label_path):
                        all_labeled_images.append(image_path)
                        all_labeled_labels.append(label_path)
                        if not k_fold: # If not K-fold, keep original split
                            val_image_paths.append(image_path)
                            val_label_paths.append(label_path)
                        break

    # K-fold cross-validation split
    if k_fold:
        print(f"Running {num_folds}-fold cross-validation (fold {fold_idx+1}/{num_folds})")
        
        # Sort data to ensure reproducibility
        combined = list(zip(all_labeled_images, all_labeled_labels))
        combined.sort(key=lambda x: x[0])
        all_labeled_images, all_labeled_labels = zip(*combined)
        all_labeled_images = np.array(all_labeled_images)
        all_labeled_labels = np.array(all_labeled_labels)
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        splits = list(kf.split(all_labeled_images))
        
        if fold_idx >= len(splits):
            raise ValueError(f"Fold index {fold_idx} out of range for {num_folds} folds")
            
        train_idx, val_idx = splits[fold_idx]
        
        train_image_paths = all_labeled_images[train_idx].tolist()
        train_label_paths = all_labeled_labels[train_idx].tolist()
        val_image_paths = all_labeled_images[val_idx].tolist()
        val_label_paths = all_labeled_labels[val_idx].tolist()

    # Label efficiency experiment: optionally reduce the number of labeled training samples (supports single split and K-fold)
    labeled_ratio = float(getattr(config_module, 'LABELED_RATIO', 1.0))
    labeled_max_count = getattr(config_module, 'LABELED_MAX_COUNT', None)
    if labeled_ratio < 1.0 or labeled_max_count is not None:
        total_train = len(train_image_paths)
        if total_train > 0:
            indices = np.arange(total_train)
            rng_seed = int(getattr(config_module, 'SEED', 42))
            rng = np.random.RandomState(rng_seed)
            rng.shuffle(indices)
            target_count = total_train
            if labeled_ratio < 1.0:
                target_count = max(1, int(round(total_train * labeled_ratio)))
            if labeled_max_count is not None:
                target_count = min(target_count, int(labeled_max_count))
            selected = indices[:target_count]
            train_image_paths = [train_image_paths[i] for i in selected]
            train_label_paths = [train_label_paths[i] for i in selected]

    
    # Prepare unlabeled data
    unlabeled_images_dir = UNLABELED_DIR
    
    if os.path.exists(unlabeled_images_dir):
        for filename in os.listdir(unlabeled_images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(unlabeled_images_dir, filename)
                unlabeled_image_paths.append(image_path)
    
    # Create datasets
    train_dataset = CassavaSemiDataset(
        train_image_paths,
        train_label_paths,
        transforms=transforms,
        mode='labeled'
    )
    
    val_dataset = CassavaSemiDataset(
        val_image_paths,
        val_label_paths,
        transforms=transforms,
        mode='val'
    )
    
    unlabeled_dataset = CassavaSemiDataset(
        unlabeled_image_paths,
        transforms=transforms,
        mode='unlabeled'
    )
    
    # Create data loaders (batch size read dynamically from current config module to support different experiment overrides)
    train_loader = DataLoader(
        train_dataset,
        batch_size=getattr(config_module, 'TRAIN_BATCH_SIZE', 2),
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(config_module, 'VAL_BATCH_SIZE', 1),
        shuffle=False,
        num_workers=0
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=getattr(config_module, 'TRAIN_BATCH_SIZE', 2),
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    print(f"\nData prepared:")
    print(f"- Train samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Unlabeled samples: {len(unlabeled_dataset)}")
    
    # Verify datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Check data path and file format.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty! Check data path and file format.")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'unlabeled_loader': unlabeled_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'unlabeled_dataset': unlabeled_dataset
    }
