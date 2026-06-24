import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                           accuracy_score, jaccard_score, matthews_corrcoef,
                           cohen_kappa_score, f1_score)
from scipy.ndimage import binary_erosion, binary_dilation, label, find_objects, distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from scipy import stats
import torch
import torch.nn.functional as F
import cv2
# Due to import issues, NUM_CLASSES is temporarily defined directly here; in actual use, it should be imported from config
NUM_CLASSES = 2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import os


def extract_boundary(mask, boundary_width=1):
    """
    Extract boundaries of a segmentation mask

    Args:
        mask (numpy.ndarray): Segmentation mask, shape (H, W) or (H, W, D)
        boundary_width (int, optional): Boundary width. Defaults to 1.

    Returns:
        numpy.ndarray: Boundary mask
    """
    # Create corresponding structuring element based on mask dimensionality
    if mask.ndim == 2:
        structure = np.ones((2 * boundary_width + 1, 2 * boundary_width + 1))
    elif mask.ndim == 3:
        structure = np.ones((2 * boundary_width + 1, 2 * boundary_width + 1, 2 * boundary_width + 1))
    else:
        # Default to 2D structure for most cases
        structure = np.ones((2 * boundary_width + 1, 2 * boundary_width + 1))

    # Extract interior via erosion, then subtract from original mask to get boundary
    if np.max(mask) > 0:  # Ensure mask is not all zeros
        eroded = binary_erosion(mask, structure=structure)
        boundary = mask - eroded
        return boundary > 0
    return np.zeros_like(mask, dtype=bool)


def compute_jaccard_score(preds, labels, num_classes=NUM_CLASSES, smooth=1e-5):
    """
    Compute Jaccard Index (IoU)

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.
        smooth (float, optional): Smoothing factor. Defaults to 1e-5.

    Returns:
        tuple: (Overall Jaccard Index, per-class Jaccard Index, FWIoU, class frequencies)
    """
    jaccard_scores = []
    class_frequencies = []

    # Ensure preds and labels are 1D
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    total_pixels = len(labels_flat)

    for cls in range(num_classes):
        # Compute TP, FP, FN for current class
        tp = np.sum((preds_flat == cls) & (labels_flat == cls))
        fp = np.sum((preds_flat == cls) & (labels_flat != cls))
        fn = np.sum((preds_flat != cls) & (labels_flat == cls))

        # Compute class frequency
        class_freq = np.sum(labels_flat == cls) / total_pixels
        class_frequencies.append(class_freq)

        # Compute Jaccard Index
        jaccard = (tp + smooth) / (tp + fp + fn + smooth)
        jaccard_scores.append(jaccard)

    # Compute Frequency Weighted Intersection over Union (FWIoU)
    fwiou = np.sum([freq * score for freq, score in zip(class_frequencies, jaccard_scores)])

    return np.mean(jaccard_scores), jaccard_scores, fwiou, class_frequencies


def compute_dice_score(preds, labels, num_classes=NUM_CLASSES, smooth=1e-5):
    """
    Compute Dice Coefficient

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.
        smooth (float, optional): Smoothing factor. Defaults to 1e-5.

    Returns:
        tuple: (Overall Dice Coefficient, per-class Dice Coefficient, mean VOE, per-class VOE, mean RVD, per-class RVD)
    """
    dice_scores = []
    voe_scores = []  # Volumetric Overlap Error
    rvd_scores = []  # Relative Volume Difference

    # Ensure preds and labels are 1D
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    for cls in range(num_classes):
        # Compute TP, FP, FN for current class
        tp = np.sum((preds_flat == cls) & (labels_flat == cls))
        fp = np.sum((preds_flat == cls) & (labels_flat != cls))
        fn = np.sum((preds_flat != cls) & (labels_flat == cls))

        # Compute Dice Coefficient
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        dice_scores.append(dice)

        # Compute Volumetric Overlap Error (VOE)
        voe = 1 - dice
        voe_scores.append(voe)

        # Compute Relative Volume Difference (RVD)
        pred_volume = np.sum(preds_flat == cls)
        gt_volume = np.sum(labels_flat == cls)
        if gt_volume > 0:
            rvd = (pred_volume - gt_volume) / gt_volume
        else:
            rvd = 0.0
        rvd_scores.append(rvd)

    return np.mean(dice_scores), dice_scores, np.mean(voe_scores), voe_scores, np.mean(rvd_scores), rvd_scores


def compute_hausdorff_distance(preds, labels, num_classes=NUM_CLASSES, boundary_width=1):
    """
    Compute Hausdorff Distance and HD95

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.
        boundary_width (int, optional): Boundary width. Defaults to 1.

    Returns:
        tuple: (Overall Hausdorff Distance, per-class Hausdorff Distance, Overall HD95, per-class HD95)
    """
    hausdorff_distances = []
    hd95_distances = []

    # Optimization: limit boundary point count to avoid performance issues
    MAX_BOUNDARY_POINTS = 1000  # Set maximum boundary point count

    for cls in range(num_classes):
        # Extract mask for current class
        pred_mask = (preds == cls).astype(np.int32)
        label_mask = (labels == cls).astype(np.int32)

        # If current class does not exist in prediction or label, distance is 0
        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            hausdorff_distances.append(0.0)
            hd95_distances.append(0.0)
            continue

        # Extract boundaries
        pred_boundary = extract_boundary(pred_mask, boundary_width)
        label_boundary = extract_boundary(label_mask, boundary_width)

        # Get boundary point coordinates
        pred_points = np.argwhere(pred_boundary)
        label_points = np.argwhere(label_boundary)

        # If no boundary points, distance is 0
        if len(pred_points) == 0 or len(label_points) == 0:
            hausdorff_distances.append(0.0)
            hd95_distances.append(0.0)
            continue

        # Optimization: if too many boundary points, randomly sample a subset to improve efficiency
        if len(pred_points) > MAX_BOUNDARY_POINTS:
            indices = np.random.choice(len(pred_points), MAX_BOUNDARY_POINTS, replace=False)
            pred_points = pred_points[indices]

        if len(label_points) > MAX_BOUNDARY_POINTS:
            indices = np.random.choice(len(label_points), MAX_BOUNDARY_POINTS, replace=False)
            label_points = label_points[indices]

        # Compute all pairwise distances for HD95
        all_distances = []
        # Min distance from predicted boundary points to label boundary points
        for p_point in pred_points:
            dists = np.sqrt(np.sum((p_point - label_points) ** 2, axis=1))
            min_dist = np.min(dists)
            all_distances.append(min_dist)
        # Min distance from label boundary points to predicted boundary points
        for l_point in label_points:
            dists = np.sqrt(np.sum((l_point - pred_points) ** 2, axis=1))
            min_dist = np.min(dists)
            all_distances.append(min_dist)

        # Compute HD95
        hd95 = np.percentile(all_distances, 95)
        hd95_distances.append(hd95)

        # Compute traditional Hausdorff distance
        d1 = directed_hausdorff(pred_points, label_points)[0]
        d2 = directed_hausdorff(label_points, pred_points)[0]
        hausdorff_distance = max(d1, d2)
        hausdorff_distances.append(hausdorff_distance)

    # Remove zeros before computing mean
    non_zero_hd = [d for d in hausdorff_distances if d > 0]
    non_zero_hd95 = [d for d in hd95_distances if d > 0]

    if non_zero_hd:
        mean_hausdorff = np.mean(non_zero_hd)
    else:
        mean_hausdorff = 0.0

    if non_zero_hd95:
        mean_hd95 = np.mean(non_zero_hd95)
    else:
        mean_hd95 = 0.0

    return mean_hausdorff, hausdorff_distances, mean_hd95, hd95_distances

def compute_boundary_iou(preds, labels, num_classes=NUM_CLASSES, boundary_width=1):
    """
    Compute Boundary IoU

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.
        boundary_width (int, optional): Boundary width. Defaults to 1.

    Returns:
        tuple: (Overall Boundary IoU, per-class Boundary IoU)
    """
    boundary_iou_scores = []

    for cls in range(num_classes):
        # Extract mask for current class
        pred_mask = (preds == cls).astype(np.int32)
        label_mask = (labels == cls).astype(np.int32)

        # If current class does not exist in prediction or label, IoU is 0
        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            boundary_iou_scores.append(0.0)
            continue

        # Extract boundaries
        pred_boundary = extract_boundary(pred_mask, boundary_width)
        label_boundary = extract_boundary(label_mask, boundary_width)

        # Compute boundary intersection and union
        intersection = np.sum(pred_boundary & label_boundary)
        union = np.sum(pred_boundary | label_boundary)

        # Compute Boundary IoU
        if union > 0:
            boundary_iou = intersection / union
        else:
            boundary_iou = 0.0

        boundary_iou_scores.append(boundary_iou)

    return np.mean(boundary_iou_scores), boundary_iou_scores

def compute_boundary_f1(preds, labels, num_classes=NUM_CLASSES, boundary_width=1):
    """
    Compute Boundary F1 Score

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.
        boundary_width (int, optional): Boundary width. Defaults to 1.

    Returns:
        tuple: (Overall Boundary F1 Score, per-class Boundary F1 Score)
    """
    boundary_f1_scores = []

    for cls in range(num_classes):
        # Extract mask for current class
        pred_mask = (preds == cls).astype(np.int32)
        label_mask = (labels == cls).astype(np.int32)

        # If current class does not exist in prediction or label, F1 is 0
        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            boundary_f1_scores.append(0.0)
            continue

        # Extract boundaries
        pred_boundary = extract_boundary(pred_mask, boundary_width)
        label_boundary = extract_boundary(label_mask, boundary_width)

        # Compute boundary TP, FP, FN
        tp = np.sum(pred_boundary & label_boundary)
        fp = np.sum(pred_boundary & ~label_boundary)
        fn = np.sum(~pred_boundary & label_boundary)

        # Compute Boundary F1 Score
        if tp + fp + fn > 0:
            boundary_f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            boundary_f1 = 0.0

        boundary_f1_scores.append(boundary_f1)

    return np.mean(boundary_f1_scores), boundary_f1_scores

def compute_connectivity_error(preds, labels, num_classes=NUM_CLASSES):
    """
    Compute Connectivity Error

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.

    Returns:
        tuple: (Overall Connectivity Error, per-class Connectivity Error)
    """
    connectivity_errors = []

    for cls in range(num_classes):
        # Extract mask for current class
        pred_mask = (preds == cls).astype(np.int32)
        label_mask = (labels == cls).astype(np.int32)

        # If current class does not exist in prediction or label, error is 0
        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            connectivity_errors.append(0.0)
            continue

        # Compute connected regions
        pred_labels, pred_num_features = label(pred_mask)
        true_labels, true_num_features = label(label_mask)

        # Compute connectivity error
        if true_num_features > 0:
            error = abs(pred_num_features - true_num_features) / true_num_features
        else:
            error = 0.0

        connectivity_errors.append(error)

    return np.mean(connectivity_errors), connectivity_errors

def compute_compactness(preds, labels, num_classes=NUM_CLASSES):
    """
    Compute Compactness (ratio of area to perimeter)

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.

    Returns:
        tuple: (Overall Compactness, per-class Compactness)
    """
    compactness_scores = []

    for cls in range(num_classes):
        # Extract mask for current class
        pred_mask = (preds == cls).astype(np.int32)

        # If current class does not exist in prediction, compactness is 0
        if np.sum(pred_mask) == 0:
            compactness_scores.append(0.0)
            continue

        # Extract boundaries
        pred_boundary = extract_boundary(pred_mask, boundary_width=1)

        # Compute area and perimeter
        area = np.sum(pred_mask)
        perimeter = np.sum(pred_boundary)

        # Compute compactness (smaller values indicate more compact shapes)
        if perimeter > 0:
            compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0
        else:
            compactness = 0.0

        compactness_scores.append(compactness)

    return np.mean(compactness_scores), compactness_scores

def compute_fragmentation_index(preds, labels, num_classes=NUM_CLASSES):
    """
    Compute Fragmentation Index

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.

    Returns:
        tuple: (Overall Fragmentation Index, per-class Fragmentation Index)
    """
    fragmentation_scores = []

    for cls in range(num_classes):
        # Extract mask for current class
        pred_mask = (preds == cls).astype(np.int32)

        # If current class does not exist in prediction, fragmentation index is 0
        if np.sum(pred_mask) == 0:
            fragmentation_scores.append(0.0)
            continue

        # Compute connected regions
        pred_labels, pred_num_features = label(pred_mask)

        # Compute average region size
        if pred_num_features > 0:
            avg_region_size = np.sum(pred_mask) / pred_num_features
            # Fragmentation Index = number of connected regions / average region size
            fragmentation = pred_num_features / (avg_region_size + 1e-5)
        else:
            fragmentation = 0.0

        fragmentation_scores.append(fragmentation)

    return np.mean(fragmentation_scores), fragmentation_scores

# Statistical significance metrics
def compute_stability_metrics(results_list, metric_name='mIoU'):
    """
    Compute stability evaluation metrics

    Args:
        results_list (list): List of results from multiple runs, each element is a dict containing evaluation metrics
        metric_name (str, optional): Name of the metric to analyze. Defaults to 'mIoU'.

    Returns:
        dict: Stability metrics
    """
    metrics = {}

    # Extract specified metric values
    values = []
    for result in results_list:
        if metric_name in result:
            values.append(result[metric_name])

    if not values:
        return metrics

    # Compute stability metrics
    values_array = np.array(values)
    metrics[f'{metric_name}_Std'] = np.std(values_array)
    metrics[f'{metric_name}_CV'] = np.std(values_array) / np.mean(values_array) if np.mean(values_array) > 0 else 0.0
    metrics[f'{metric_name}_Range'] = np.max(values_array) - np.min(values_array)
    metrics[f'{metric_name}_IQR'] = stats.iqr(values_array)

    return metrics

def compute_statistical_tests(group1_results, group2_results, metric_name='mIoU', test_type='t'):
    """
    Perform statistical tests

    Args:
        group1_results (list): List of results for group 1
        group2_results (list): List of results for group 2
        metric_name (str, optional): Name of the metric to compare. Defaults to 'mIoU'.
        test_type (str, optional): Test type ('t', 'anova', 'wilcoxon', 'friedman'). Defaults to 't'.

    Returns:
        dict: Statistical test results
    """
    metrics = {}

    # Extract specified metric values
    group1_values = [result[metric_name] for result in group1_results if metric_name in result]
    group2_values = [result[metric_name] for result in group2_results if metric_name in result]

    if not group1_values or not group2_values:
        return metrics

    try:
        if test_type.lower() == 't':
            # t-test
            t_stat, p_value = stats.ttest_ind(group1_values, group2_values)
            metrics[f'Test_{metric_name}_T_Stat'] = t_stat
            metrics[f'Test_{metric_name}_P_Value'] = p_value

        elif test_type.lower() == 'anova':
            # ANOVA test
            f_stat, p_value = stats.f_oneway(group1_values, group2_values)
            metrics[f'Test_{metric_name}_F_Stat'] = f_stat
            metrics[f'Test_{metric_name}_P_Value'] = p_value

        elif test_type.lower() == 'wilcoxon':
            # Wilcoxon signed-rank test (requires equal sample sizes)
            if len(group1_values) == len(group2_values):
                w_stat, p_value = stats.wilcoxon(group1_values, group2_values)
                metrics[f'Test_{metric_name}_W_Stat'] = w_stat
                metrics[f'Test_{metric_name}_P_Value'] = p_value

        elif test_type.lower() == 'friedman':
            # Friedman test (requires multiple groups)
            # Simplified here to compare only two groups
            # For true multi-group comparison, more groups of data are needed
            if len(group1_values) == len(group2_values):
                # Friedman test requires transposed data format
                data = np.array([group1_values, group2_values]).T
                f_stat, p_value = stats.friedmanchisquare(*data.T)
                metrics[f'Test_{metric_name}_F_Stat'] = f_stat
                metrics[f'Test_{metric_name}_P_Value'] = p_value
    except Exception as e:
        print(f"Statistical test error: {e}")

    return metrics

# Compute efficiency metrics
def compute_model_complexity(model, input_size=(1, 3, 256, 256)):
    """
    Compute model complexity metrics

    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple, optional): Input size. Defaults to (1, 3, 256, 256).

    Returns:
        dict: Complexity metrics
    """
    import torch
    metrics = {}

    # Compute parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics['Total_Parameters_M'] = total_params / 1e6  # Convert to millions
    metrics['Trainable_Parameters_M'] = trainable_params / 1e6

    # Compute computational cost (FLOPs)
    try:
        # Use hook method to count FLOPs
        class FlopCounter:
            def __init__(self):
                self.flops = 0

            def __call__(self, module, input, output):
                # Simple FLOPs calculation based on different layer types
                if isinstance(module, torch.nn.Conv2d):
                    batch_size = input[0].size(0)
                    output_channels = module.out_channels
                    output_size = output.size(2) * output.size(3)
                    kernel_size = module.kernel_size[0] * module.kernel_size[1]
                    in_channels = module.in_channels
                    groups = module.groups

                    flops = batch_size * output_channels * output_size * kernel_size * in_channels / groups
                    self.flops += flops

                elif isinstance(module, torch.nn.Linear):
                    batch_size = input[0].size(0)
                    in_features = module.in_features
                    out_features = module.out_features

                    flops = batch_size * in_features * out_features
                    self.flops += flops

        counter = FlopCounter()
        handles = []

        # Add hooks for all Conv2d and Linear layers
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(counter))

        # Forward pass once
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        with torch.no_grad():
            model(dummy_input)

        # Remove hooks
        for handle in handles:
            handle.remove()

        metrics['GFLOPs'] = counter.flops / 1e9  # Convert to GFLOPs
    except Exception as e:
        print(f"Error computing FLOPs: {e}")
        metrics['GFLOPs'] = None

    return metrics

def compute_efficiency_metrics(train_time=None, inference_time=None, memory_usage=None, convergence_epochs=None):
    """
    Compute efficiency metrics

    Args:
        train_time (float, optional): Training time (hours). Defaults to None.
        inference_time (float, optional): Inference time (ms/image). Defaults to None.
        memory_usage (float, optional): Memory usage (GB). Defaults to None.
        convergence_epochs (int, optional): Epochs to convergence. Defaults to None.

    Returns:
        dict: Efficiency metrics
    """
    metrics = {}

    if train_time is not None:
        metrics['Training_Time_Hours'] = train_time

    if inference_time is not None:
        metrics['Inference_Time_ms'] = inference_time

    if memory_usage is not None:
        metrics['Memory_Usage_GB'] = memory_usage

    if convergence_epochs is not None:
        metrics['Convergence_Epochs'] = convergence_epochs

    return metrics

# Semi-supervised specific metrics

def compute_pseudo_label_quality(pseudo_labels, teacher_preds, teacher_probs=None, conf_threshold=0.7):
    """
    Compute pseudo-label quality evaluation metrics

    Args:
        pseudo_labels (numpy.ndarray): Pseudo labels
        teacher_preds (numpy.ndarray): Teacher model predictions
        teacher_probs (numpy.ndarray, optional): Teacher model probabilities. Defaults to None.
        conf_threshold (float, optional): High confidence threshold. Defaults to 0.7.

    Returns:
        dict: Pseudo-label quality metrics
    """
    metrics = {}

    # Compute pseudo-label accuracy (consistency with teacher predictions)
    pseudo_acc = accuracy_score(teacher_preds.flatten(), pseudo_labels.flatten())
    metrics['Pseudo_Label_Accuracy'] = pseudo_acc

    # Compute pseudo-label confidence
    if teacher_probs is not None:
        # For each pixel, get the probability of the predicted class as confidence
        if teacher_probs.ndim == 4:  # (batch, classes, H, W)
            batch_size = teacher_probs.shape[0]
            height = teacher_probs.shape[2]
            width = teacher_probs.shape[3]

            # Flatten to (batch*H*W, classes)
            probs_flat = teacher_probs.transpose(0, 2, 3, 1).reshape(-1, teacher_probs.shape[1])
            labels_flat = teacher_preds.flatten()

            # Get predicted class probability for each pixel
            confidences = probs_flat[np.arange(len(labels_flat)), labels_flat]

            metrics['Pseudo_Label_Confidence'] = np.mean(confidences)

            # Compute ratio of high-confidence pixels
            high_conf_ratio = np.mean(confidences >= conf_threshold)
            metrics['High_Confidence_Pixel_Ratio'] = high_conf_ratio

            # Compute entropy reduction rate (relative to uniform distribution)
            uniform_entropy = -np.log(1 / teacher_probs.shape[1])
            pixel_entropy = -np.sum(probs_flat * np.log(probs_flat + 1e-10), axis=1)
            entropy_reduction = np.mean((uniform_entropy - pixel_entropy) / uniform_entropy)
            metrics['Entropy_Reduction_Rate'] = entropy_reduction

            # Compute calibration error
            calibration_error = np.mean(np.abs(confidences - (pseudo_labels.flatten() == teacher_preds.flatten()).astype(float)))
            metrics['Calibration_Error'] = calibration_error

    return metrics

def compute_consistency_metrics(preds_list, teacher_preds=None, student_preds=None, prev_epoch_preds=None):
    """
    Compute consistency evaluation metrics

    Args:
        preds_list (list): List of prediction results from multiple augmentations
        teacher_preds (numpy.ndarray, optional): Teacher model predictions. Defaults to None.
        student_preds (numpy.ndarray, optional): Student model predictions. Defaults to None.
        prev_epoch_preds (numpy.ndarray, optional): Predictions from previous epoch. Defaults to None.

    Returns:
        dict: Consistency metrics
    """
    metrics = {}

    # Compute prediction consistency (consistency across multiple augmentations)
    if len(preds_list) > 1:
        # Flatten prediction results from the list
        flattened_preds = [pred.flatten() for pred in preds_list]

        # Compute prediction variance (measure of inconsistency)
        pred_var = np.var(np.stack(flattened_preds), axis=0)
        mean_var = np.mean(pred_var)

        # Prediction Consistency = 1 - normalized prediction variance
        max_var = np.max(pred_var)
        if max_var > 0:
            metrics['Prediction_Consistency'] = 1.0 - (mean_var / max_var)
        else:
            metrics['Prediction_Consistency'] = 1.0

    # Compute teacher-student consistency
    if teacher_preds is not None and student_preds is not None:
        # Use Dice coefficient to measure consistency
        teacher_flat = teacher_preds.flatten()
        student_flat = student_preds.flatten()

        # Compute per-class Dice, then average
        dice_scores = []
        classes = np.unique(np.concatenate([teacher_flat, student_flat]))
        for cls in classes:
            if cls >= 0:  # Exclude background or invalid labels
                teacher_binary = (teacher_flat == cls).astype(int)
                student_binary = (student_flat == cls).astype(int)
                if np.sum(teacher_binary) > 0 or np.sum(student_binary) > 0:
                    dice = f1_score(teacher_binary, student_binary)
                    dice_scores.append(dice)

        if dice_scores:
            metrics['Teacher_Student_Consistency'] = np.mean(dice_scores)
        else:
            metrics['Teacher_Student_Consistency'] = 1.0  # If no predictions, set to 1

    # Compute temporal consistency
    if prev_epoch_preds is not None:
        current_flat = preds_list[0].flatten() if preds_list else None
        prev_flat = prev_epoch_preds.flatten()

        if current_flat is not None:
            # Compute IoU as temporal consistency
            iou_scores = []
            classes = np.unique(np.concatenate([current_flat, prev_flat]))
            for cls in classes:
                if cls >= 0:  # Exclude background or invalid labels
                    current_binary = (current_flat == cls).astype(int)
                    prev_binary = (prev_flat == cls).astype(int)
                    if np.sum(current_binary) > 0 or np.sum(prev_binary) > 0:
                        iou = jaccard_score(current_binary, prev_binary)
                        iou_scores.append(iou)

            if iou_scores:
                metrics['Temporal_Consistency'] = np.mean(iou_scores)
            else:
                metrics['Temporal_Consistency'] = 1.0  # If no predictions, set to 1

    return metrics

def compute_semi_supervised_gain(ssl_metrics, supervised_metrics, unlabeled_data_size=0, effective_pseudo_pixels=0, convergence_epochs=0, supervised_convergence_epochs=0):
    """
    Compute semi-supervised gain metrics

    Args:
        ssl_metrics (dict): Evaluation metrics from semi-supervised learning
        supervised_metrics (dict): Evaluation metrics from supervised learning
        unlabeled_data_size (int, optional): Amount of unlabeled data. Defaults to 0.
        effective_pseudo_pixels (int, optional): Number of effective pseudo-label pixels. Defaults to 0.
        convergence_epochs (int, optional): Epochs to convergence for semi-supervised. Defaults to 0.
        supervised_convergence_epochs (int, optional): Epochs to convergence for supervised. Defaults to 0.

    Returns:
        dict: Semi-supervised gain metrics
    """
    metrics = {}

    # Compute supervision efficiency ratio
    if 'mIoU' in ssl_metrics and 'mIoU' in supervised_metrics and supervised_metrics['mIoU'] > 0:
        metrics['Supervision_Efficiency_Ratio'] = ssl_metrics['mIoU'] / supervised_metrics['mIoU']

    # Compute unlabeled data utilization rate
    if unlabeled_data_size > 0:
        # Assume pixel count per unlabeled sample
        pixels_per_sample = 256 * 256  # Assume image size is 256x256
        total_unlabeled_pixels = unlabeled_data_size * pixels_per_sample
        if total_unlabeled_pixels > 0:
            metrics['Unlabeled_Utilization_Rate'] = effective_pseudo_pixels / total_unlabeled_pixels

    # Compute convergence speedup
    if supervised_convergence_epochs > 0 and convergence_epochs > 0:
        speedup = (supervised_convergence_epochs - convergence_epochs) / supervised_convergence_epochs
        metrics['Convergence_Speedup'] = speedup

    # Compute generalization gain (if target domain metrics exist)
    if 'Target_Domain_mIoU' in ssl_metrics and 'Target_Domain_mIoU' in supervised_metrics:
        generalization_gain = ssl_metrics['Target_Domain_mIoU'] - supervised_metrics['Target_Domain_mIoU']
        metrics['Generalization_Gain'] = generalization_gain

    return metrics

# Visualization analysis metrics
def create_error_heatmap(preds, labels, save_path=None, show=True):
    """
    Create error heatmap showing spatial distribution of false positives and false negatives

    Args:
        preds (np.ndarray): Predictions, shape (H, W) or (B, H, W)
        labels (np.ndarray): Ground truth labels, shape (H, W) or (B, H, W)
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        np.ndarray: Error heatmap matrix
    """
    # Ensure 2D image (take the first sample)
    if len(preds.shape) == 3:
        preds = preds[0]
    if len(labels.shape) == 3:
        labels = labels[0]

    # Compute error type
    # 0: correct, 1: false positive, 2: false negative
    error_map = np.zeros_like(preds, dtype=np.uint8)
    error_map[(preds == 1) & (labels == 0)] = 1  # False positive
    error_map[(preds == 0) & (labels == 1)] = 2  # False negative

    # Create custom colormap
    colors = ['gray', 'red', 'blue']  # Correct (gray), False Positive (red), False Negative (blue)
    cmap = ListedColormap(colors)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap=cmap)
    plt.colorbar(ticks=[0, 1, 2], label='Error Type')
    plt.title('Error Heatmap')
    plt.xlabel('Width')
    plt.ylabel('Height')

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return error_map

def plot_confidence_distribution(probs, save_path=None, show=True, num_bins=20):
    """
    Plot confidence distribution histogram

    Args:
        probs (np.ndarray): Prediction probabilities, shape (H, W) or (B, H, W) or (B, C, H, W)
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.
        num_bins (int, optional): Number of histogram bins. Defaults to 20.

    Returns:
        tuple: (bin_edges, bin_counts)
    """
    # Ensure probability values
    if len(probs.shape) == 4:  # (B, C, H, W)
        # Take max probability for each pixel
        probs = np.max(probs, axis=1)

    # Flatten to 1D array
    probs_flat = probs.flatten()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(probs_flat, bins=num_bins, range=(0, 1),
                               alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.grid(True, alpha=0.3)

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return bins, n

def create_boundary_error_map(preds, labels, boundary_width=1, save_path=None, show=True):
    """
    Create boundary error map

    Args:
        preds (np.ndarray): Predictions, shape (H, W) or (B, H, W)
        labels (np.ndarray): Ground truth labels, shape (H, W) or (B, H, W)
        boundary_width (int, optional): Boundary width. Defaults to 1.
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        np.ndarray: Boundary error map
    """
    # Ensure 2D image
    if len(preds.shape) == 3:
        preds = preds[0]
    if len(labels.shape) == 3:
        labels = labels[0]

    # Extract boundaries
    pred_boundary = extract_boundary(preds, boundary_width)
    gt_boundary = extract_boundary(labels, boundary_width)

    # Compute boundary error
    # 0: non-boundary, 1: correct boundary, 2: incorrect boundary
    boundary_map = np.zeros_like(preds, dtype=np.uint8)
    boundary_map[pred_boundary & gt_boundary] = 1  # Correct boundary
    boundary_map[(pred_boundary ^ gt_boundary) & (pred_boundary | gt_boundary)] = 2  # Incorrect boundary

    # Create custom colormap
    colors = ['white', 'green', 'red']  # Non-Boundary (white), Correct Boundary (green), Incorrect Boundary (red)
    cmap = ListedColormap(colors)

    # Plot boundary error map
    plt.figure(figsize=(10, 8))
    plt.imshow(boundary_map, cmap=cmap)
    plt.colorbar(ticks=[0, 1, 2], label='Boundary Type')
    plt.title('Boundary Error Map')
    plt.xlabel('Width')
    plt.ylabel('Height')

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return boundary_map

def plot_confusion_matrix_heatmap(preds, labels, class_names=None, save_path=None, show=True):
    """
    Plot class confusion matrix heatmap

    Args:
        preds (np.ndarray): Predictions
        labels (np.ndarray): Ground truth labels
        class_names (list, optional): List of class names. Defaults to None.
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        np.ndarray: Confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels.flatten(), preds.flatten())

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Class Confusion Matrix')

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return cm

def plot_performance_distribution(results_list, metric_name='mIoU', class_names=None, save_path=None, show=True):
    """
    Plot performance distribution (box plot or violin plot)

    Args:
        results_list (list): List of results from multiple runs or results for different classes
        metric_name (str, optional): Metric name. Defaults to 'mIoU'.
        class_names (list, optional): List of class names. Defaults to None.
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        dict: Distribution plot data
    """
    # Extract data
    data = []
    labels = []

    if isinstance(results_list, list) and all(isinstance(item, dict) for item in results_list):
        # Results from multiple runs
        data.append([result[metric_name] for result in results_list if metric_name in result])
        labels = [metric_name]
    elif isinstance(results_list, dict) and class_names:
        # Results for different classes
        for i, class_name in enumerate(class_names):
            if f'{metric_name}_{i}' in results_list:
                data.append(results_list[f'{metric_name}_{i}'])
                labels.append(class_name)

    if not data:
        print(f"Cannot extract data for {metric_name}")
        return {}

    # Plot box plot and violin plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Box plot
    axes[0].boxplot(data, labels=labels)
    axes[0].set_title(f'{metric_name} Box Plot')
    axes[0].set_ylabel(metric_name)
    axes[0].grid(True, alpha=0.3)

    # Violin plot
    axes[1].violinplot(data, showmeans=True, showmedians=True)
    axes[1].set_xticks(range(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    axes[1].set_title(f'{metric_name} Violin Plot')
    axes[1].set_ylabel(metric_name)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return {'data': data, 'labels': labels}

def plot_difficulty_curve(metrics_list, difficulty_scores, metric_name='mIoU', save_path=None, show=True):
    """
    Plot difficulty curve showing performance sorted by sample difficulty

    Args:
        metrics_list (list): List of metric values per sample
        difficulty_scores (list): List of difficulty scores per sample
        metric_name (str, optional): Metric name. Defaults to 'mIoU'.
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        tuple: (sorted_difficulty, sorted_metrics)
    """
    # Sort by difficulty scores
    sorted_indices = np.argsort(difficulty_scores)
    sorted_difficulty = np.array(difficulty_scores)[sorted_indices]
    sorted_metrics = np.array(metrics_list)[sorted_indices]

    # Plot difficulty curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_metrics)), sorted_metrics, 'b-', alpha=0.7)
    plt.fill_between(range(len(sorted_metrics)), sorted_metrics - np.std(sorted_metrics),
                    sorted_metrics + np.std(sorted_metrics), alpha=0.2, color='blue')
    plt.xlabel('Sample Index (sorted by difficulty)')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Difficulty Curve')
    plt.grid(True, alpha=0.3)

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return sorted_difficulty, sorted_metrics

def plot_threshold_sensitivity(prob_thresholds, metrics_dict, metric_names=None, save_path=None, show=True):
    """
    Plot threshold sensitivity curve showing metric changes under different thresholds

    Args:
        prob_thresholds (list): List of probability thresholds
        metrics_dict (dict): Metric dictionary under different thresholds, format {threshold: {metric_name: value}}
        metric_names (list, optional): List of metric names to display. Defaults to None.
        save_path (str, optional): Save path. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        dict: Threshold sensitivity data
    """
    # If no metrics specified, use all available metrics
    if metric_names is None:
        if prob_thresholds and prob_thresholds[0] in metrics_dict:
            metric_names = list(metrics_dict[prob_thresholds[0]].keys())

    # Plot sensitivity curve
    plt.figure(figsize=(12, 6))

    for metric_name in metric_names:
        metric_values = [metrics_dict[thresh].get(metric_name, 0) for thresh in prob_thresholds]
        plt.plot(prob_thresholds, metric_values, marker='o', label=metric_name)

    plt.xlabel('Probability Threshold')
    plt.ylabel('Metric Value')
    plt.title('Threshold Sensitivity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)

    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return {metric: [metrics_dict[thresh].get(metric, 0) for thresh in prob_thresholds]
            for metric in metric_names}

def generate_visualization_metrics(preds, labels, probs=None, save_dir=None, show=False):
    """
    Generate all visualization analysis metrics

    Args:
        preds (np.ndarray): Predictions
        labels (np.ndarray): Ground truth labels
        probs (np.ndarray, optional): Prediction probabilities. Defaults to None.
        save_dir (str, optional): Save directory. Defaults to None.
        show (bool, optional): Whether to display images. Defaults to False.

    Returns:
        dict: Visualization results dictionary
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Error analysis metrics
    print("\nGenerating error analysis visualization metrics...")

    # Error heatmap
    error_map_path = os.path.join(save_dir, 'error_heatmap.png') if save_dir else None
    error_map = create_error_heatmap(preds, labels, error_map_path, show)
    results['error_map'] = error_map

    # Confidence distribution
    if probs is not None:
        conf_dist_path = os.path.join(save_dir, 'confidence_distribution.png') if save_dir else None
        bins, counts = plot_confidence_distribution(probs, conf_dist_path, show)
        results['confidence_distribution'] = {'bins': bins, 'counts': counts}

    # Boundary error map
    boundary_error_path = os.path.join(save_dir, 'boundary_error_map.png') if save_dir else None
    boundary_map = create_boundary_error_map(preds, labels, 1, boundary_error_path, show)
    results['boundary_error_map'] = boundary_map

    # Class confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
    cm = plot_confusion_matrix_heatmap(preds, labels, ['Background', 'Target'], cm_path, show)
    results['confusion_matrix'] = cm

    return results

# Convenience function for computing statistical significance and efficiency metrics
def compute_statistical_and_efficiency_metrics(results_list=None, model=None,
                                              train_time=None, inference_time=None,
                                              memory_usage=None, convergence_epochs=None):
    """
    Compute statistical significance and efficiency metrics (convenience function)

    Args:
        results_list (list, optional): List of results from multiple runs. Defaults to None.
        model (torch.nn.Module, optional): PyTorch model. Defaults to None.
        train_time (float, optional): Training time (hours). Defaults to None.
        inference_time (float, optional): Inference time (ms/image). Defaults to None.
        memory_usage (float, optional): Memory usage (GB). Defaults to None.
        convergence_epochs (int, optional): Epochs to convergence. Defaults to None.

    Returns:
        dict: Combined statistical and efficiency metrics
    """
    metrics = {}

    # Compute stability metrics
    if results_list and len(results_list) > 1:
        # Compute stability for main metrics
        for metric_name in ['mIoU', 'Dice', 'Accuracy', 'F1_Score']:
            stability_metrics = compute_stability_metrics(results_list, metric_name)
            metrics.update(stability_metrics)

    # Compute model complexity
    if model is not None:
        complexity_metrics = compute_model_complexity(model)
        metrics.update(complexity_metrics)

    # Compute efficiency metrics
    efficiency_metrics = compute_efficiency_metrics(
        train_time, inference_time, memory_usage, convergence_epochs
    )
    metrics.update(efficiency_metrics)

    return metrics


def compute_average_surface_distance(preds, labels, num_classes=NUM_CLASSES, boundary_width=1):
    """
    Compute Average Surface Distance (ASD)

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.
        boundary_width (int, optional): Boundary width. Defaults to 1.

    Returns:
        tuple: (Overall Average Surface Distance, per-class Average Surface Distance)
    """
    asd_scores = []

    for cls in range(num_classes):
        pred_mask = (preds == cls).astype(np.int32)
        label_mask = (labels == cls).astype(np.int32)

        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            asd_scores.append(0.0)
            continue

        pred_boundary = extract_boundary(pred_mask, boundary_width)
        label_boundary = extract_boundary(label_mask, boundary_width)

        if not (np.any(pred_boundary) and np.any(label_boundary)):
            asd_scores.append(0.0)
            continue

        pred_dt = distance_transform_edt(~pred_boundary)
        label_dt = distance_transform_edt(~label_boundary)

        pred_to_label = label_dt[pred_boundary]
        label_to_pred = pred_dt[label_boundary]

        all_distances = np.concatenate([pred_to_label, label_to_pred])
        if all_distances.size == 0:
            asd_scores.append(0.0)
            continue

        asd_scores.append(float(all_distances.mean()))

    non_zero_distances = [d for d in asd_scores if d > 0]
    if non_zero_distances:
        mean_asd = float(np.mean(non_zero_distances))
    else:
        mean_asd = 0.0

    return mean_asd, asd_scores


def compute_fbeta_score(precision, recall, beta=2.0):
    """
    Compute F-beta Score

    Args:
        precision (float): Precision
        recall (float): Recall
        beta (float, optional): Beta value. Defaults to 2.0.

    Returns:
        float: F-beta Score
    """
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def compute_balanced_accuracy(recall, specificity):
    """
    Compute Balanced Accuracy

    Args:
        recall (float): Recall
        specificity (float): Specificity

    Returns:
        float: Balanced Accuracy
    """
    return (recall + specificity) / 2

def compute_g_mean(recall, specificity):
    """
    Compute Geometric Mean

    Args:
        recall (float): Recall
        specificity (float): Specificity

    Returns:
        float: Geometric Mean
    """
    return np.sqrt(recall * specificity)

def compute_metrics(preds, labels, num_classes=NUM_CLASSES,
                   pseudo_labels=None, teacher_preds=None, teacher_probs=None,
                   preds_list=None, student_preds=None, prev_epoch_preds=None,
                   supervised_metrics=None, unlabeled_data_size=0, effective_pseudo_pixels=0,
                   convergence_epochs=0, supervised_convergence_epochs=0,
                   results_list=None, model=None, train_time=None, inference_time=None,
                   memory_usage=None):
    """
    Compute evaluation metrics

    Args:
        preds (numpy.ndarray): Predictions
        labels (numpy.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    metrics = {}

    # Convert continuous prediction values to class labels
    # Check if predictions are continuous (float type)
    if preds.dtype in (np.float32, np.float64):
        if preds.ndim == 4 and preds.shape[1] > 1:  # Multi-channel output (num_samples, num_classes, H, W)
            preds = np.argmax(preds, axis=1)
        elif preds.ndim == 4 and preds.shape[1] == 1:  # Single-channel output (num_samples, 1, H, W)
            preds = (preds > 0.5).astype(np.int32).squeeze(1)
        elif preds.ndim == 3:  # Already processed as (num_samples, H, W) but values are continuous
            preds = (preds > 0.5).astype(np.int32)

    # Ensure labels are also integer type
    labels = labels.astype(np.int32)

    # Flatten arrays for computation
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    # Compute overall accuracy
    metrics['Accuracy'] = accuracy_score(labels_flat, preds_flat)

    # Compute precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average='weighted', labels=np.arange(num_classes), zero_division=1
    )
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1_Score'] = f1_score

    # Compute F-beta scores (beta=0.5, 1, 2)
    metrics['F0.5_Score'] = compute_fbeta_score(precision, recall, beta=0.5)
    metrics['F2_Score'] = compute_fbeta_score(precision, recall, beta=2.0)

    # Compute confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat, labels=np.arange(num_classes))
    metrics['Confusion_Matrix'] = cm.tolist()

    # Compute Jaccard Index (IoU) and Frequency Weighted IoU (FWIoU)
    mean_jaccard, jaccard_per_class, fwiou, class_frequencies = compute_jaccard_score(preds, labels, num_classes)
    metrics['Jaccard'] = mean_jaccard
    metrics['Jaccard_per_class'] = jaccard_per_class
    metrics['FWIoU'] = fwiou
    metrics['Class_Frequencies'] = class_frequencies

    # Keep backward compatibility, use Jaccard as IoU
    metrics['mIoU'] = mean_jaccard
    metrics['IoU_per_class'] = jaccard_per_class

    # Compute Dice coefficient, Volumetric Overlap Error (VOE), and Relative Volume Difference (RVD)
    mean_dice, dice_per_class, mean_voe, voe_per_class, mean_rvd, rvd_per_class = compute_dice_score(preds, labels, num_classes)
    metrics['Dice'] = mean_dice
    metrics['Dice_per_class'] = dice_per_class
    metrics['VOE'] = mean_voe
    metrics['VOE_per_class'] = voe_per_class
    metrics['RVD'] = mean_rvd
    metrics['RVD_per_class'] = rvd_per_class

    # Compute Hausdorff Distance and HD95
    mean_hausdorff, hausdorff_per_class, mean_hd95, hd95_per_class = compute_hausdorff_distance(preds, labels, num_classes)
    metrics['Hausdorff'] = mean_hausdorff
    metrics['Hausdorff_per_class'] = hausdorff_per_class
    metrics['HD95'] = mean_hd95
    metrics['HD95_per_class'] = hd95_per_class

    # Compute Average Surface Distance (ASD)
    mean_asd, asd_per_class = compute_average_surface_distance(preds, labels, num_classes)

    # Compute Boundary IoU
    mean_boundary_iou, boundary_iou_per_class = compute_boundary_iou(preds, labels, num_classes)
    metrics['Boundary_IoU'] = mean_boundary_iou
    metrics['Boundary_IoU_per_class'] = boundary_iou_per_class

    # Compute Boundary F1 Score
    mean_boundary_f1, boundary_f1_per_class = compute_boundary_f1(preds, labels, num_classes)
    metrics['Boundary_F1'] = mean_boundary_f1
    metrics['Boundary_F1_per_class'] = boundary_f1_per_class

    # Compute Connectivity Error
    mean_connectivity_error, connectivity_error_per_class = compute_connectivity_error(preds, labels, num_classes)
    metrics['Connectivity_Error'] = mean_connectivity_error
    metrics['Connectivity_Error_per_class'] = connectivity_error_per_class

    # Compute Compactness
    mean_compactness, compactness_per_class = compute_compactness(preds, labels, num_classes)
    metrics['Compactness'] = mean_compactness
    metrics['Compactness_per_class'] = compactness_per_class

    # Compute Fragmentation Index
    mean_fragmentation, fragmentation_per_class = compute_fragmentation_index(preds, labels, num_classes)
    metrics['Fragmentation_Index'] = mean_fragmentation
    metrics['Fragmentation_per_class'] = fragmentation_per_class
    metrics['ASD'] = mean_asd
    metrics['ASD_per_class'] = asd_per_class

    # Compute Specificity
    specificity_scores = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp + 1e-5)
        specificity_scores.append(specificity)
    metrics['Specificity'] = np.mean(specificity_scores)
    metrics['Specificity_per_class'] = specificity_scores

    # Compute Balanced Accuracy (BA)
    metrics['Balanced_Accuracy'] = compute_balanced_accuracy(metrics['Recall'], metrics['Specificity'])

    # Compute Geometric Mean (G-Mean)
    metrics['G_Mean'] = compute_g_mean(metrics['Recall'], metrics['Specificity'])

    # Compute Matthews Correlation Coefficient (MCC)
    # For multi-class case, use weighted MCC
    if num_classes == 2:
        # Binary classification: compute directly
        metrics['MCC'] = matthews_corrcoef(labels_flat, preds_flat)
    else:
        # Multi-class case: compute MCC for each pair of classes and average
        mcc_scores = []
        for i in range(num_classes):
            # Binarize each class
            binary_labels = (labels_flat == i).astype(int)
            binary_preds = (preds_flat == i).astype(int)
            if len(np.unique(binary_labels)) > 1 and len(np.unique(binary_preds)) > 1:
                mcc = matthews_corrcoef(binary_labels, binary_preds)
                mcc_scores.append(mcc)
        if mcc_scores:
            metrics['MCC'] = np.mean(mcc_scores)
        else:
            metrics['MCC'] = 0.0

    # Compute Cohen's Kappa coefficient
    try:
        metrics['Cohen_Kappa'] = cohen_kappa_score(labels_flat, preds_flat, labels=np.arange(num_classes), zero_division=1)
    except:
        metrics['Cohen_Kappa'] = 0.0

    # Compute semi-supervised specific metrics

    # 1. Pseudo-label quality evaluation
    if pseudo_labels is not None and teacher_preds is not None:
        pseudo_quality_metrics = compute_pseudo_label_quality(
            pseudo_labels, teacher_preds, teacher_probs
        )
        metrics.update(pseudo_quality_metrics)

    # 2. Consistency evaluation
    if preds_list is None:
        preds_list = [preds]  # If no augmented prediction list provided, use current predictions

    consistency_metrics = compute_consistency_metrics(
        preds_list, teacher_preds, student_preds, prev_epoch_preds
    )
    metrics.update(consistency_metrics)

    # 3. Semi-supervised gain metrics
    if supervised_metrics is not None:
        ssl_gain_metrics = compute_semi_supervised_gain(
            metrics, supervised_metrics,
            unlabeled_data_size, effective_pseudo_pixels,
            convergence_epochs, supervised_convergence_epochs
        )
        metrics.update(ssl_gain_metrics)

    # Compute statistical significance and efficiency metrics
    statistical_efficiency_metrics = compute_statistical_and_efficiency_metrics(
        results_list, model, train_time, inference_time, memory_usage, convergence_epochs
    )
    metrics.update(statistical_efficiency_metrics)

    return metrics


def evaluate_model_detailed(model, dataloader, criterion, device, num_classes=NUM_CLASSES):
    """
    Detailed model evaluation

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device
        num_classes (int, optional): Number of classes. Defaults to NUM_CLASSES.

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    import torch

    model.eval()
    total_loss = 0.0

    # Initialize lists for storing predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())

            # Accumulate loss
            total_loss += loss.item()

            # Save predictions and labels
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.squeeze(1).cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute average loss
    avg_loss = total_loss / len(dataloader)

    # Compute evaluation metrics
    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics['Loss'] = avg_loss
    return metrics


def sliding_window_inference(model, image, device, patch_size, stride, num_classes=NUM_CLASSES):
    model.eval()
    _, h, w = image.shape

    # Pad image to be divisible by 16 (requirement for DeepLabv3+)
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    padded_image = image
    if pad_h > 0 or pad_w > 0:
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        padded_image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode='constant', value=0).squeeze(0)

    _, new_h, new_w = padded_image.shape

    ph = min(patch_size[0], new_h)
    pw = min(patch_size[1], new_w)
    sh = stride[0]
    sw = stride[1]

    prob_map = torch.zeros(num_classes, new_h, new_w, device=device)
    count_map = torch.zeros(1, new_h, new_w, device=device)

    ys = list(range(0, max(new_h - ph + 1, 1), sh))
    xs = list(range(0, max(new_w - pw + 1, 1), sw))

    if ys[-1] != new_h - ph:
        ys.append(new_h - ph)
    if xs[-1] != new_w - pw:
        xs.append(new_w - pw)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                if y < 0 or x < 0:
                    continue
                patch = padded_image[:, y:y + ph, x:x + pw].unsqueeze(0).to(device)

                # Double check patch size (must be divisible by 16)
                if patch.shape[2] % 16 != 0 or patch.shape[3] % 16 != 0:
                     # This should theoretically not happen if image is padded to 16 and patch_size is divisible by 16
                     # But if patch_size is NOT divisible by 16, we might have issues.
                     # Assuming patch_size is valid (e.g. 512x512).
                     pass

                logits = model(patch)
                probs = torch.softmax(logits, dim=1)[0]
                prob_map[:, y:y + ph, x:x + pw] += probs
                count_map[:, y:y + ph, x:x + pw] += 1

    prob_map = prob_map / count_map.clamp_min(1.0)

    # Crop back to original size
    prob_map = prob_map[:, :h, :w]

    pred = torch.argmax(prob_map, dim=0)
    return pred


def evaluate_model_sliding_full(model, image_paths, label_paths, device, num_classes=NUM_CLASSES,
                                patch_size=(256, 256), stride=(128, 128)):
    import semi_supervised_segmentation.config as config_module
    mean = np.array(config_module.MEAN).reshape(1, 1, 3)
    std = np.array(config_module.STD).reshape(1, 1, 3)
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img_path, lbl_path in zip(image_paths, label_paths):
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_f = image.astype(np.float32) / 255.0
            image_n = (image_f - mean) / std
            image_t = torch.from_numpy(image_n.transpose(2, 0, 1)).float().to(device)
            mask = None
            _, ext = os.path.splitext(lbl_path)
            ext = ext.lower()
            if ext == '.png':
                m = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if m.shape[:2] != image.shape[:2]:
                    h, w = image.shape[:2]
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = m.astype(np.uint8)
                if m.max() > 1:
                    m = (m > 0).astype(np.uint8)
                mask = m
            elif ext == '.txt':
                m = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                try:
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            data = list(map(float, line.split()))
                            if len(data) < 3:
                                continue
                            class_id = int(data[0])
                            coords = np.array(data[1:]).reshape(-1, 2)
                            h, w = image.shape[:2]
                            pixel_coords = coords * np.array([w, h])
                            pixel_coords = pixel_coords.astype(np.int32)
                            cv2.fillPoly(m, [pixel_coords], color=class_id)
                except Exception:
                    m = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask = m
            else:
                continue
            pred = sliding_window_inference(model, image_t, device, patch_size, stride, num_classes)
            all_preds.append(pred.cpu().numpy().reshape(-1))
            all_labels.append(mask.astype(np.int32).reshape(-1))
    if not all_preds:
        return {}
    preds_flat = np.concatenate(all_preds)
    labels_flat = np.concatenate(all_labels)
    preds_arr = preds_flat.reshape(1, -1)
    labels_arr = labels_flat.reshape(1, -1)
    metrics = compute_metrics(preds_arr, labels_arr, num_classes)
    return metrics


def print_evaluation_metrics(metrics):
    """
    Print evaluation metrics

    Args:
        metrics (dict): Evaluation metrics dictionary
    """
    print("\nDetailed Evaluation Results:")
    print("=" * 50)

    # Print scalar metrics first, sorted by importance
    scalar_metrics = ['Loss', 'Accuracy', 'Balanced_Accuracy', 'G_Mean', 'MCC', 'Cohen_Kappa',
                      'mIoU', 'Jaccard', 'FWIoU', 'Dice', 'VOE', 'RVD',
                      'Precision', 'Recall', 'F1_Score', 'F0.5_Score', 'F2_Score', 'Specificity',
                      'Hausdorff', 'HD95', 'ASD', 'Boundary_IoU', 'Boundary_F1',
                      'Connectivity_Error', 'Compactness', 'Fragmentation_Index',
                      # Semi-supervised specific metrics
                      'Pseudo_Label_Accuracy', 'Pseudo_Label_Confidence',
                      'High_Confidence_Pixel_Ratio', 'Entropy_Reduction_Rate', 'Calibration_Error',
                      'Prediction_Consistency', 'Teacher_Student_Consistency', 'Temporal_Consistency',
                      'Supervision_Efficiency_Ratio', 'Unlabeled_Utilization_Rate',
                      'Convergence_Speedup', 'Generalization_Gain',
                      # Computational efficiency metrics
                      'Total_Parameters_M', 'Trainable_Parameters_M', 'GFLOPs',
                      'Training_Time_Hours', 'Inference_Time_ms', 'Memory_Usage_GB', 'Convergence_Epochs']

    # Print basic evaluation metrics
    print("\nBasic Evaluation Metrics:")
    print("-" * 30)
    for metric_name in scalar_metrics:
        if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
            print(f"{metric_name}: {metrics[metric_name]:.4f}")

    # Print per-class detailed metrics
    class_metrics = ['IoU_per_class', 'Jaccard_per_class', 'Dice_per_class', 'VOE_per_class', 'RVD_per_class',
                    'Specificity_per_class', 'Hausdorff_per_class', 'HD95_per_class', 'ASD_per_class',
                    'Boundary_IoU_per_class', 'Boundary_F1_per_class',
                    'Connectivity_Error_per_class', 'Compactness_per_class', 'Fragmentation_per_class']
    for metric_name in class_metrics:
        if metric_name in metrics and isinstance(metrics[metric_name], (list, np.ndarray)):
            # Format class metric names
            display_name = metric_name.replace('_per_class', ' (per class)')
            formatted_values = "[" + ", ".join([f"{val:.4f}" for val in metrics[metric_name]]) + "]"
            print(f"\n{display_name}: {formatted_values}")

    # Finally print confusion matrix
    if 'Confusion_Matrix' in metrics:
        print(f"\nConfusion Matrix:")
        for row in metrics['Confusion_Matrix']:
            print("  ".join([f"{int(val):4d}" for val in row]))

    print("=" * 50)
