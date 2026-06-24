import os
import torch
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = PROJECT_ROOT
ORGANIZED_DATA_DIR = os.path.join(DATA_DIR, 'dataset')
IMAGES_DIR = os.path.join(ORGANIZED_DATA_DIR, 'images')
LABELS_DIR = os.path.join(ORGANIZED_DATA_DIR, 'labels')
LABELS_PNG_DIR = os.path.join(ORGANIZED_DATA_DIR, 'labels_png')
UNLABELED_DIR = os.path.join(ORGANIZED_DATA_DIR, 'unlabeled')

DEFAULT_EXPERIMENT_NAME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
EXPERIMENT_NAME = DEFAULT_EXPERIMENT_NAME

ENABLE_WANDB = False
WANDB_PROJECT = "Cassava-Segmentation-SSL"
WANDB_ENTITY = None
WANDB_RUN_NAME = None

BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, EXPERIMENT_NAME)
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')


def create_experiment_dirs():
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    print(f"Experiment directories created:")
    print(f"- Results: {RESULTS_DIR}")
    print(f"- Models: {MODEL_DIR}")
    print(f"- Metrics: {METRICS_DIR}")


def initialize_config_with_experiment_name(experiment_name):
    global EXPERIMENT_NAME, RESULTS_DIR, MODEL_DIR, METRICS_DIR
    EXPERIMENT_NAME = experiment_name
    RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, experiment_name)
    MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    create_experiment_dirs()
    print(f"Config initialized with experiment '{experiment_name}'")


os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
print(f"Base results directory created: {BASE_RESULTS_DIR}")

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 2
IMAGE_SIZE = (256, 256)
ENCODER_NAME = "resnet101"
ENCODER_WEIGHTS = "imagenet"

SUPERVISED_WEIGHT = 1.0
UNSUPERVISED_WEIGHT = 0.1

CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
CONSISTENCY_WEIGHT = 0.5

ENABLE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

DICE_LOSS_WEIGHT_C0 = 1.0
DICE_LOSS_WEIGHT_C1 = 3.0
ENABLE_BOUNDARY_LOSS = True
BOUNDARY_LOSS_WEIGHT = 0.1

THRESHOLD_C0 = 0.95
THRESHOLD_C1 = 0.80

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ENABLE_TPRAM = True
ENABLE_SYMGD = True
ENABLE_UCP = True
LB = 0.005
CUTMIX_PROB = 1.0
THRESHOLD = 0.95
CONSISTENCY_RAMPUP = 200.0
INCREASE = 1.0003
QUEUE_LEN = 200

EVALUATE_METRICS = ['accuracy', 'miou', 'dice', 'precision', 'recall', 'f1_score', 'specificity']
LOG_INTERVAL = 10
SAVE_INTERVAL = 1

OPTIMIZER = 'adam'
LR_SCHEDULER = 'cosine'
POWER = 0.9
STEP_SIZE = 50
GAMMA = 0.1

SEED = 42

ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MONITOR = 'val_iou_c1'
