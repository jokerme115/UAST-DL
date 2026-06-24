import numpy as np
import torch
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Due to relative import issues, we need to import modules directly instead of using relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    # Main functions (use these functions to compute all metrics)
    compute_metrics, evaluate_model_detailed, print_evaluation_metrics,
    compute_statistical_and_efficiency_metrics,
    # Visualization metrics
    create_error_heatmap, plot_confidence_distribution, create_boundary_error_map,
    plot_confusion_matrix_heatmap, generate_visualization_metrics
)

# Create mock data
def create_mock_data(batch_size=2, height=64, width=64, num_classes=2):
    """Create mock prediction results and ground truth labels"""
    # Create ground truth labels (binary classification scenario, target class is 1)
    labels = np.zeros((batch_size, height, width), dtype=np.int32)
    # Add some target regions
    for i in range(batch_size):
        # Add rectangular regions of different sizes as targets in each sample
        h_start, w_start = 10 + i*5, 15 + i*3
        h_end, w_end = h_start + 20, w_start + 25
        labels[i, h_start:h_end, w_start:w_end] = 1
        
        # Add some noise points
        noise_points = np.random.randint(0, min(height, width), (20, 2))
        for h, w in noise_points:
            labels[i, h, w] = 1
    
    # Create predictions (add some errors based on labels)
    preds = labels.copy()
    for i in range(batch_size):
        # Add false positive errors
        fp_points = np.random.randint(0, min(height, width), (15, 2))
        for h, w in fp_points:
            if preds[i, h, w] == 0:
                preds[i, h, w] = 1
        
        # Add false negative errors
        fn_mask = (preds[i] == 1) & (np.random.random((height, width)) < 0.1)
        preds[i, fn_mask] = 0
    
    # Create prediction probabilities
    probs = np.random.random((batch_size, num_classes, height, width))
    # Ensure predictions are consistent with probabilities
    for i in range(batch_size):
        for h in range(height):
            for w in range(width):
                probs[i, preds[i, h, w], h, w] = 0.7 + np.random.random() * 0.3
                probs[i, 1-preds[i, h, w], h, w] = 1 - probs[i, preds[i, h, w], h, w]
    
    # Create teacher model predictions (for semi-supervised metrics)
    teacher_preds = labels.copy()
    for i in range(batch_size):
        # Teacher model has slightly higher accuracy than student model
        fn_mask = (teacher_preds[i] == 1) & (np.random.random((height, width)) < 0.05)
        teacher_preds[i, fn_mask] = 0
    
    # Multi-run results (for stability metrics)
    results_list = []
    for _ in range(5):
        jaccard = np.random.normal(0.85, 0.02)
        dice = np.random.normal(0.92, 0.015)
        accuracy = np.random.normal(0.94, 0.01)
        results_list.append({
            'Jaccard': min(1.0, max(0.0, jaccard)),
            'Dice': min(1.0, max(0.0, dice)),
            'Accuracy': min(1.0, max(0.0, accuracy))
        })
    
    return {
        'labels': labels,
        'preds': preds,
        'probs': probs,
        'teacher_preds': teacher_preds,
        'results_list': results_list
    }

# Simple mock model class
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 2, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

def test_all_metrics():
    """Test all evaluation metrics"""
    print("Starting evaluation metrics test...")
    
    # 1. Create test data
    mock_data = create_mock_data()
    labels = mock_data['labels']
    preds = mock_data['preds']
    probs = mock_data['probs']
    teacher_preds = mock_data['teacher_preds']
    results_list = mock_data['results_list']
    
    # Create mock model
    model = MockModel()
    
    print("\nTest data creation complete, shape info:")
    print(f"- Label shape: {labels.shape}")
    print(f"- Prediction shape: {preds.shape}")
    print(f"- Probability shape: {probs.shape}")
    print(f"- Teacher prediction shape: {teacher_preds.shape}")
    
    # 2. Test compute_metrics main function (this will test all basic, segmentation, and semi-supervised metrics)
    print("\n=== Testing compute_metrics main function ===")
    metrics = compute_metrics(
        preds=preds,
        labels=labels,
        teacher_preds=teacher_preds,
        teacher_probs=probs,
        results_list=results_list,
        model=model,
        train_time=10.5,
        inference_time=0.02,
        memory_usage=4.2
    )
    
    # Print some key metrics
    print("Key metric results:")
    key_metrics_groups = [
        # Basic classification metrics
        ['Accuracy', 'Balanced_Accuracy', 'G_Mean', 'MCC', 'Cohen_Kappa'],
        # Segmentation metrics
        ['mIoU', 'Jaccard', 'FWIoU', 'Dice', 'VOE', 'RVD'],
        # Boundary metrics
        ['Hausdorff', 'HD95', 'ASD', 'Boundary_IoU', 'Boundary_F1'],
        # Semi-supervised metrics
        ['Pseudo_Label_Accuracy', 'Pseudo_Label_Confidence', 
         'High_Confidence_Pixel_Ratio', 'Teacher_Student_Consistency']
    ]
    
    for group in key_metrics_groups:
        print("")
        for metric in group:
            if metric in metrics:
                print(f"  {metric}: {metrics[metric]:.4f}")
    
    # 3. Test statistical and efficiency metrics
    print("\n=== Testing statistical and efficiency metrics ===")
    statistical_metrics = compute_statistical_and_efficiency_metrics(
        results_list=results_list,
        model=model,
        train_time=10.5,
        inference_time=0.02,
        memory_usage=4.2,
        convergence_epochs=50
    )
    
    print("Statistical and efficiency metric results:")
    for key, value in statistical_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 4. Test print_evaluation_metrics function
    print("\n=== Testing print_evaluation_metrics function ===")
    print_evaluation_metrics(metrics)
    
    # 5. Test visualization metrics (only test functions, do not display images)
    print("\n=== Testing visualization metrics ===")
    # Save to temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Generating visualizations to temp directory: {temp_dir}")
        
        # Test error heatmap
        error_map = create_error_heatmap(preds, labels, show=False)
        print(f"  Error heatmap created successfully, shape: {error_map.shape}")
        
        # Test confidence distribution
        bins, counts = plot_confidence_distribution(probs, show=False)
        print(f"  Confidence distribution created successfully, bin count: {len(bins)-1}")
        
        # Test boundary error map
        boundary_map = create_boundary_error_map(preds, labels, show=False)
        print(f"  Boundary error map created successfully, shape: {boundary_map.shape}")
        
        # Test confusion matrix heatmap
        cm = plot_confusion_matrix_heatmap(preds, labels, ['Background', 'Target'], show=False)
        print(f"  Confusion matrix heatmap created successfully, shape: {cm.shape}")
        
        # Test full visualization generation
        viz_results = generate_visualization_metrics(
            preds=preds,
            labels=labels,
            probs=probs,
            show=False
        )
        print(f"  Full visualization generation successful, result count: {len(viz_results)}")
    
    print("\nAll metrics tests complete!")

if __name__ == "__main__":
    test_all_metrics()
