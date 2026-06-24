import torch
import argparse
import sys
import os
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semi_supervised_segmentation.data.dataset import CassavaSemiDataset, get_semi_transforms
from semi_supervised_segmentation.data.data_preparation import prepare_data
from semi_supervised_segmentation.models.model import DeepLabV3Plus
from semi_supervised_segmentation.train.trainer import train_model
from semi_supervised_segmentation.evaluate.metrics import evaluate_model_detailed, evaluate_model_sliding_full, print_evaluation_metrics
from semi_supervised_segmentation.utils.logger import create_dirs, save_evaluation_results, save_history


def load_config_from_file(config_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def apply_config_to_module(config_source, config_module):
    config_vars = [attr for attr in dir(config_source)
                   if attr.isupper() and not attr.startswith('_')]
    for var in config_vars:
        setattr(config_module, var, getattr(config_source, var))
    print(f"Loaded config: {len(config_vars)} parameters")


def run_training_cycle(config_module, fold_idx=None, num_folds=1):
    print(f"Starting UAST-DL training — experiment: {config_module.EXPERIMENT_NAME}")
    print(f"Results directory: {config_module.RESULTS_DIR}")
    print("=" * 70)

    if hasattr(config_module, 'ENABLE_WANDB') and config_module.ENABLE_WANDB:
        if wandb is None:
            print("Warning: W&B enabled but wandb not installed. Run `pip install wandb`.")
        else:
            config_dict = {}
            for key in dir(config_module):
                if key.isupper() and not key.startswith('_'):
                    config_dict[key] = getattr(config_module, key)
            if fold_idx is not None:
                config_dict['fold'] = fold_idx
                config_dict['num_folds'] = num_folds
            wandb.init(
                project=config_module.WANDB_PROJECT,
                entity=getattr(config_module, 'WANDB_ENTITY', None),
                name=config_module.EXPERIMENT_NAME,
                config=config_dict,
                reinit=True
            )

    print("\n1. Data preparation")
    k_fold = (fold_idx is not None)
    current_fold = fold_idx if k_fold else 0
    total_folds = num_folds if k_fold else 1
    data_dict = prepare_data(k_fold=k_fold, fold_idx=current_fold, num_folds=total_folds)
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    unlabeled_loader = data_dict['unlabeled_loader']
    val_dataset = data_dict.get('val_dataset')

    use_sliding = getattr(config_module, 'USE_SLIDING_WINDOW_EVAL', False)
    if use_sliding and val_dataset is not None:
        if hasattr(val_dataset, 'transforms') and 'val_sliding' in val_dataset.transforms:
            print("Sliding-window evaluation mode enabled")
            val_dataset.transforms['val'] = val_dataset.transforms['val_sliding']
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=0
            )

    print("\n2. Model initialization")
    model = DeepLabV3Plus(num_classes=config_module.NUM_CLASSES)
    print(f"Model: {model.__class__.__name__} (encoder: {config_module.ENCODER_NAME})")

    if hasattr(config_module, 'LOAD_CHECKPOINT') and config_module.LOAD_CHECKPOINT:
        checkpoint_path = getattr(config_module, 'CHECKPOINT_PATH', None)
        if checkpoint_path:
            full_checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                checkpoint_path
            )
            if os.path.exists(full_checkpoint_path):
                print(f"Loading checkpoint: {full_checkpoint_path}")
                map_location = None if torch.cuda.is_available() else torch.device('cpu')
                model.load_state_dict(torch.load(full_checkpoint_path, map_location=map_location))
                print("Checkpoint loaded successfully")

    print("\n3. Training")
    train_history, val_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        unlabeled_loader=unlabeled_loader,
        num_epochs=config_module.NUM_EPOCHS,
        enable_early_stopping=getattr(config_module, 'ENABLE_EARLY_STOPPING', False),
        early_stopping_patience=getattr(config_module, 'EARLY_STOPPING_PATIENCE', 10),
        early_stopping_monitor=getattr(config_module, 'EARLY_STOPPING_MONITOR', 'val_miou')
    )

    save_history(train_history, val_history)

    if getattr(config_module, 'DEBUG_SKIP_FINAL_EVAL', False):
        return {'mIoU': 0.0, 'Dice': 0.0, 'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1_Score': 0.0, 'ASD': 0.0}

    print("\n4. Evaluation")
    best_model_path = os.path.join(config_module.MODEL_DIR, 'best_model.pth')
    print(f"Loading best model: {best_model_path}")
    if os.path.exists(best_model_path):
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model.load_state_dict(torch.load(best_model_path, map_location=map_location))
    else:
        print(f"Warning: best model file not found at {best_model_path}")

    model = model.to(config_module.DEVICE)

    use_sliding = getattr(config_module, 'USE_SLIDING_WINDOW_EVAL', False)
    if use_sliding and val_dataset is not None:
        window_size = getattr(config_module, 'SLIDING_WINDOW_SIZE', config_module.IMAGE_SIZE)
        stride = getattr(config_module, 'SLIDING_WINDOW_STRIDE', (window_size[0] // 2, window_size[1] // 2))
        eval_metrics = evaluate_model_sliding_full(
            model=model,
            image_paths=val_dataset.image_paths,
            label_paths=val_dataset.label_paths,
            device=config_module.DEVICE,
            num_classes=config_module.NUM_CLASSES,
            patch_size=window_size,
            stride=stride
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()
        eval_metrics = evaluate_model_detailed(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=config_module.DEVICE,
            num_classes=config_module.NUM_CLASSES
        )

    save_evaluation_results(eval_metrics)
    print_evaluation_metrics(eval_metrics)

    if hasattr(config_module, 'ENABLE_WANDB') and config_module.ENABLE_WANDB and wandb and wandb.run:
        test_log_dict = {}
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                test_log_dict[f"test/{k}"] = v
        wandb.log(test_log_dict)
        wandb.finish()

    print(f"\nFinal results: mIoU={eval_metrics['mIoU']:.4f}, Dice={eval_metrics['Dice']:.4f}")
    print(f"Saved to: {config_module.METRICS_DIR}")
    print("\n" + "=" * 70)
    print("UAST-DL training and evaluation completed")

    return eval_metrics


def main():
    import semi_supervised_segmentation.config as config_module

    parser = argparse.ArgumentParser(description='UAST-DL: Uncertainty-Aware Student-Teacher Deep Learning for Cassava PPD Segmentation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to experiment config file (e.g., configs/release/USTMT_Best.py)')
    parser.add_argument('--name', type=str, default=None, help='Override experiment name')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--fold', type=int, default=None, help='Run only specified fold (1~NUM_FOLDS)')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
        if not os.path.exists(config_path):
            print(f"Error: config file not found: {config_path}")
            sys.exit(1)
        print(f"Loading config: {config_path}")
        experiment_config = load_config_from_file(config_path)
        apply_config_to_module(experiment_config, config_module)
    else:
        print("No config specified. Use --config configs/release/USTMT_Best.py")

    if args.name:
        config_module.EXPERIMENT_NAME = args.name
    if args.epochs:
        config_module.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config_module.TRAIN_BATCH_SIZE = args.batch_size
        config_module.VAL_BATCH_SIZE = args.batch_size

    if hasattr(config_module, 'initialize_config_with_experiment_name'):
        config_module.initialize_config_with_experiment_name(config_module.EXPERIMENT_NAME)

    enable_k_fold = getattr(config_module, 'ENABLE_K_FOLD', False)
    num_folds = getattr(config_module, 'NUM_FOLDS', 5)

    if enable_k_fold:
        print(f"\nStarting {num_folds}-fold cross-validation")
        print("=" * 70)

        original_exp_name = config_module.EXPERIMENT_NAME
        fold_metrics_list = []

        if args.fold is not None:
            if args.fold < 1 or args.fold > int(num_folds):
                raise ValueError(f"--fold must be 1~{num_folds}, got {args.fold}")
            folds_to_run = [int(args.fold) - 1]
            print(f"Running only fold {args.fold}/{num_folds}")
        else:
            folds_to_run = list(range(num_folds))

        for fold in folds_to_run:
            print(f"\n>>> Fold {fold+1}/{num_folds} <<<")
            fold_exp_name = f"{original_exp_name}_fold{fold+1}"
            config_module.EXPERIMENT_NAME = fold_exp_name
            config_module.RESULTS_DIR = os.path.join(config_module.BASE_RESULTS_DIR, fold_exp_name)
            config_module.MODEL_DIR = os.path.join(config_module.RESULTS_DIR, 'models')
            config_module.METRICS_DIR = os.path.join(config_module.RESULTS_DIR, 'metrics')
            create_dirs([config_module.RESULTS_DIR, config_module.MODEL_DIR, config_module.METRICS_DIR])
            metrics = run_training_cycle(config_module, fold_idx=fold, num_folds=num_folds)
            fold_metrics_list.append(metrics)

        print("\n" + "=" * 70)
        ran_folds = len(fold_metrics_list)
        print(f"{ran_folds}/{num_folds} folds completed. Summary:")

        avg_metrics = {}
        metric_keys = ['mIoU', 'Dice', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ASD']
        for key in metric_keys:
            values = [m.get(key, 0.0) for m in fold_metrics_list]
            avg_val = sum(values) / len(values)
            std_val = np.std(values)
            avg_metrics[key] = (avg_val, std_val)
            print(f"  {key}: {avg_val:.4f} +/- {std_val:.4f}")

        summary_path = os.path.join(config_module.BASE_RESULTS_DIR, f"{original_exp_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {original_exp_name}\n")
            f.write(f"{ran_folds}/{num_folds}-Fold Cross Validation Summary\n")
            f.write("=" * 50 + "\n")
            for key, (avg, std) in avg_metrics.items():
                f.write(f"{key}: {avg:.4f} +/- {std:.4f}\n")
        print(f"Summary saved to: {summary_path}")

    else:
        create_dirs([config_module.RESULTS_DIR, config_module.MODEL_DIR, config_module.METRICS_DIR])
        run_training_cycle(config_module)


if __name__ == "__main__":
    main()
