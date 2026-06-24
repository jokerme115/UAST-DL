import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def create_dirs(dirs):
    """
    Create directory list
    
    Args:
        dirs (list): List of directory paths to create
    """
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def append_epoch_metrics(row, filename='training_metrics_live.csv'):
    import semi_supervised_segmentation.config as config_module
    metrics_dir = config_module.METRICS_DIR
    os.makedirs(metrics_dir, exist_ok=True)

    csv_file = os.path.join(metrics_dir, filename)
    file_exists = os.path.exists(csv_file)

    fieldnames = list(row.keys())
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def save_training_metrics(train_history, val_history):
    """
    Save training metrics to CSV file
    
    Args:
        train_history (dict): Training history records
        val_history (dict): Validation history records
    """
    # Dynamically import config module to get the latest directory settings
    import semi_supervised_segmentation.config as config_module
    metrics_dir = config_module.METRICS_DIR
    
    # Ensure directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, 'training_metrics.csv')
    
    # metrics_dir existence already ensured above, no additional operation needed
    
    # Write to CSV file
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Epoch', 'Train Loss', 'Train mIoU', 'Train Precision', 
                         'Val Loss', 'Val mIoU', 'Val Precision'])
        
        train_loss = train_history.get('loss', [])
        train_miou = train_history.get('miou', [])
        train_precision = train_history.get('precision', [])
        val_loss = val_history.get('loss', [])
        val_miou = val_history.get('miou', [])
        val_precision = val_history.get('precision', [])

        num_epochs = max(len(train_loss), len(val_loss), len(train_miou), len(val_miou))

        def safe_get(arr, idx):
            if idx < len(arr):
                return arr[idx]
            return float('nan')

        for epoch in range(num_epochs):
            writer.writerow([
                epoch + 1,
                safe_get(train_loss, epoch),
                safe_get(train_miou, epoch),
                safe_get(train_precision, epoch),
                safe_get(val_loss, epoch),
                safe_get(val_miou, epoch),
                safe_get(val_precision, epoch)
            ])
    
    print(f"Training metrics saved to: {metrics_file}")


def save_history(train_history, val_history):
    """
    Save history records to file
    
    Args:
        train_history (dict): Training history records
        val_history (dict): Validation history records
    """
    # Dynamically import config module to get the latest directory settings
    import semi_supervised_segmentation.config as config_module
    results_dir = config_module.RESULTS_DIR
    
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    metrics_dir = config_module.METRICS_DIR
    
    # Ensure directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"Saving experiment results to: {results_dir}")
    
    # Save as npz file
    npz_file = os.path.join(metrics_dir, 'training_history.npz')
    np.savez(
        npz_file,
        train_loss=np.array(train_history.get('loss', [])),
        train_miou=np.array(train_history.get('miou', [])),
        train_precision=np.array(train_history.get('precision', [])),
        val_loss=np.array(val_history.get('loss', [])),
        val_miou=np.array(val_history.get('miou', [])),
        val_precision=np.array(val_history.get('precision', [])),
    )
    
    print(f"History saved to: {npz_file}")
    
    # Save to CSV
    save_training_metrics(train_history, val_history)
    
    # Visualize training curves
    plot_training_curves(train_history, val_history)


def plot_training_curves(train_history, val_history):
    """
    Plot training curves
    
    Args:
        train_history (dict): Training history records
        val_history (dict): Validation history records
    """
    # Dynamically import config module to get the latest directory settings
    import semi_supervised_segmentation.config as config_module
    results_dir = config_module.RESULTS_DIR
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss curve
    plt.subplot(1, 3, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.plot(val_history['loss'], label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot mIoU curve
    plt.subplot(1, 3, 2)
    plt.plot(train_history['miou'], label='Train mIoU')
    plt.plot(val_history['miou'], label='Val mIoU')
    plt.title('mIoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    
    # Plot precision curve
    plt.subplot(1, 3, 3)
    plt.plot(train_history['precision'], label='Train Precision')
    plt.plot(val_history['precision'], label='Val Precision')
    plt.title('Precision vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    curves_file = os.path.join(results_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(curves_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {curves_file}")


def save_evaluation_metrics_csv(metrics, filename='evaluation_metrics.csv'):
    """
    Save evaluation metrics as CSV format
    
    Args:
        metrics (dict): Evaluation metrics dictionary
        filename (str, optional): CSV filename. Defaults to 'evaluation_metrics.csv'.
    """
    # Dynamically import config module to get the latest directory settings
    import semi_supervised_segmentation.config as config_module
    metrics_dir = config_module.METRICS_DIR
    
    # Ensure directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    csv_file = os.path.join(metrics_dir, filename)
    
    # Define main metrics list (in order of terminal output)
    main_metrics_order = [
        'Loss', 'Accuracy', 'Balanced_Accuracy', 'G_Mean', 'MCC', 'Cohen_Kappa',
        'mIoU', 'Jaccard', 'FWIoU', 'Dice', 'VOE', 'RVD', 'Precision', 'Recall',
        'F1_Score', 'F0.5_Score', 'F2_Score', 'Specificity', 'Hausdorff', 'HD95',
        'ASD', 'Boundary_IoU', 'Boundary_F1', 'Connectivity_Error', 'Compactness',
        'Fragmentation_Index', 'Convergence_Epochs'
    ]
    
    # Define per-class metrics list
    per_class_metrics = [
        'IoU_per_class', 'Jaccard_per_class', 'Dice_per_class', 'VOE_per_class',
        'RVD_per_class', 'Specificity_per_class', 'Hausdorff_per_class', 
        'HD95_per_class', 'ASD_per_class', 'Boundary_IoU_per_class', 
        'Boundary_F1_per_class', 'Connectivity_Error_per_class', 
        'Compactness_per_class', 'Fragmentation_per_class'
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Get experiment name
        import semi_supervised_segmentation.config as config_module
        experiment_name = config_module.EXPERIMENT_NAME
        
        # Write experiment information
        writer.writerow(['Experiment Name:', experiment_name])
        writer.writerow([''])
        
        # Write main metrics (flat format)
        writer.writerow(['Main Evaluation Metrics'])
        
        # Write header and data
        headers = []
        values = []
        
        for metric_name in main_metrics_order:
            if metric_name in metrics:
                headers.append(metric_name)
                values.append(f"{metrics[metric_name]:.6f}")
        
        # Add other metrics not in the predefined list
        for metric_name, metric_value in metrics.items():
            if metric_name not in main_metrics_order and metric_name not in per_class_metrics:
                if not isinstance(metric_value, (list, dict)) or len(metric_value) <= 10:  # Avoid overly long lists
                    headers.append(metric_name)
                    if isinstance(metric_value, (int, float)):
                        values.append(f"{metric_value:.6f}")
                    else:
                        values.append(str(metric_value))
        
        writer.writerow(headers)
        writer.writerow(values)
        writer.writerow([''])
        
        # Write per-class metrics (if present)
        for per_class_name, csv_col_name in zip(
            ['IoU_per_class', 'Jaccard_per_class', 'Dice_per_class', 'VOE_per_class', 
             'RVD_per_class', 'Specificity_per_class'],
            ['IoU (per class)', 'Jaccard (per class)', 'Dice (per class)', 'VOE (per class)', 
             'RVD (per class)', 'Specificity (per class)']
        ):
            if per_class_name in metrics:
                writer.writerow([csv_col_name])
                writer.writerow([f"{val:.6f}" for val in metrics[per_class_name]])
                writer.writerow([''])
    
    print(f"Evaluation metrics CSV saved to: {csv_file}")


def save_evaluation_results(metrics, filename='evaluation_metrics.txt'):
    """
    Save evaluation results to text file
    
    Args:
        metrics (dict): Evaluation metrics dictionary
        filename (str, optional): Save filename. Defaults to 'evaluation_metrics.txt'.
    """
    # Dynamically import config module to get the latest directory settings
    import semi_supervised_segmentation.config as config_module
    metrics_dir = config_module.METRICS_DIR
    
    # Ensure directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    
    eval_file = os.path.join(metrics_dir, filename)
    
    with open(eval_file, 'w') as f:
        f.write("Detailed Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: ")
            
            if metric_name == "Confusion_Matrix":
                # Handle confusion matrix (2D list)
                f.write("\n")
                for row in metric_value:
                    f.write("  ".join([f"{int(val):4d}" for val in row]) + "\n")
            elif isinstance(metric_value, list):
                # Handle 1D list
                formatted_values = [f"{val:.4f}" for val in metric_value]
                f.write("[" + ", ".join(formatted_values) + "]\n")
            elif isinstance(metric_value, (int, float)):
                # Handle scalar values
                f.write(f"{metric_value:.4f}\n")
            else:
                # Other types
                f.write(f"{metric_value}\n")
        
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"Evaluation results saved to: {eval_file}")
    
    # Also save as CSV format
    save_evaluation_metrics_csv(metrics)
