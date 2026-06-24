<div align="center">

# UAST-DL

**Uncertainty-Aware Student–Teacher Deep Learning for Cassava PPD Segmentation**

[![Paper](https://img.shields.io/badge/Paper-Submitted-orange)](https://github.com/jokerme115/UAST-DL)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

Official code repository for the paper:

> **Detection and Quantification of Cassava PPD with Semi-supervised Student–Teacher Deep Learning and RGB Images**
>
> Tao He, Hao Zhang, Xingmingyue Chen, Haixia Long
>
> *Submitted to Food Control*

</div>

---

## Overview

Cassava roots are susceptible to postharvest physiological deterioration (PPD), which develops within 24–72 h after harvest and manifests as blue-black or brown vascular streaking. Manual inspection is subjective, inefficient, and cannot reliably quantify affected areas. **UAST-DL** addresses this through a semi-supervised segmentation framework that requires only **110 annotated images** while leveraging **1,494 unannotated images** for PPD detection and area quantification.

### Key Contributions

- **Uncertainty-Aware Screening Threshold (UST)**: Filters low-confidence pseudo-labels (τ = 0.95) with class-balanced weighting (w = 0.1 for background, 1.0 for PPD) to prevent high-confidence background domination
- **EMA Mean Teacher**: Provides temporally stable pseudo-label targets via exponential moving average (decay = 0.999)
- **Warm-up + Sigmoid Ramp-up**: First 5 epochs train only on annotated data; unsupervised weight then increases via sigmoid schedule (λ_max = 1.0)
- **False-Negative-Sensitive Loss**: 0.5·CE + 1.0·Tversky(α=0.3, β=0.7) + 1.0·Dice(ω₀=1, ω₁=10)

### Results

| Metric | UAST-DL | PLMT | Δ |
|:---|:---:|:---:|:---:|
| mIoU (5-fold CV) | **72.95%** | — | — |
| Dice (5-fold CV) | **81.97%** | — | — |
| mIoU (independent n=20) | **70.64%** | 60.53% | +10.11 pp |
| Dice (independent n=20) | **79.52%** | 67.98% | +11.54 pp |
| PPD IoU (foreground, n=20) | **48.36%** | 35.97% | +12.39 pp |
| PPD-area MAE (pp) | **2.11** | 2.97 | −0.86 |
| PPD-area R² | **0.885** | 0.744 | +0.141 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    UAST-DL Framework                     │
├────────────────────────┬────────────────────────────────┤
│    Annotated Path       │     Unannotated Path           │
│    (x_l, y_l)          │     x_u                        │
│         │               │         │                      │
│    T_geo(x_l, y_l)     │    T_geo(x_u) → x̄_u          │
│         │               │    ┌────┴────┐                 │
│         │               │  T_weak  T_strong              │
│         │               │    │       │                   │
│         │               │  x_u^w  x_u^s                 │
│         │               │    │       │                   │
│         ▼               │    ▼       ▼                   │
│   ┌─────────┐           │ ┌───────┐ ┌─────────┐         │
│   │ Student  │◄─────────│ │Teacher│ │ Student  │         │
│   │ (θ_S)    │  L_unsup │ │(θ_T)  │ │ (θ_S)   │         │
│   └────┬────┘           │ └───┬───┘ └────┬────┘         │
│        │                │     │          │               │
│   L_sup = 0.5·CE        │  Pseudo-label  │               │
│   + 1.0·Tversky         │  + UST(τ=0.95) │               │
│   + 1.0·Dice(w)         │  + Class-bal.  │               │
│        │                │     │          │               │
│        ▼                │     ▼          ▼               │
│   EMA update: θ_T ← α·θ_T + (1−α)·θ_S                 │
│   α = min(1−1/(k+1), 0.999)                            │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/jokerme115/UAST-DL.git
cd UAST-DL
pip install -r requirements.txt
```

---

## Dataset Preparation

Organize your data as follows:

```
dataset/
├── images/
│   ├── train/    # 88 annotated training images (per fold)
│   └── val/      # 22 annotated validation images (per fold)
├── labels_png/
│   ├── train/    # Binary masks (0 = healthy, 1 = PPD)
│   └── val/      # Binary masks for validation
└── unlabeled/    # 1,494 unannotated images
```

- **Image format**: JPG or PNG
- **Label format**: PNG with pixel values 0 (background) and 1 (PPD)
- **Input size**: Images are padded and cropped to 512×512 during training

The training code resolves dataset paths relative to the repository root by default. To customize, modify `DATA_DIR` in `semi_supervised_segmentation/config.py`.

---

## Training

### Primary Configuration (5-Fold Cross-Validation)

```bash
python train.py
```

This runs the primary UAST-DL configuration with 5-fold cross-validation using `configs/release/USTMT_Best.py`.

### Run a Single Fold

```bash
python train.py --fold 1
```

### Override Parameters

```bash
python train.py --epochs 100 --batch_size 4
```

### Configuration Summary (Paper Table 1)

| Item | Setting |
|:---|:---|
| Cross-validation & seed | 5 folds; seed 42 |
| Encoder & initialization | ResNet101; ImageNet; output stride 16 |
| Input & batch size | 512×512; train batch 2; val batch 1 |
| Training duration | Max 200 epochs; early-stop patience 30 |
| Optimizer | SGD; momentum 0.9; weight decay 1e-4 |
| LR schedule | Initial 1e-3; polynomial decay; power 0.9 |
| Teacher & screening | EMA decay 0.999; τ = 0.95 |
| Semi-supervised schedule | Warm-up 5; ramp-up 5; λ_max = 1.0 |
| Supervised loss | 0.5·CE + 1.0·Tversky(α=0.3,β=0.7) + 1.0·Dice(ω₀=1,ω₁=10) |
| Inference | Sliding window 512×512; stride 256×256 |

---

## Inference

### Single Image

```bash
python predict.py -i path/to/image.jpg -w models/best_model.pth
```

### Batch Directory

```bash
python predict.py -i path/to/image_folder/ -w models/best_model.pth -o predict_results/
```

### Disable Test-Time Augmentation

```bash
python predict.py -i test/ -w models/best_model.pth --no-tta
```

### Output

Each prediction generates an overlay image named `{basename}_PPD-{ppd_value}_overlay.png`, where `PPD` is the PPD-affected area ratio (percentage of cassava tissue classified as deteriorated).

---

## Repository Structure

```
UAST-DL/
├── configs/
│   └── release/
│       └── USTMT_Best.py         # Primary configuration
├── semi_supervised_segmentation/
│   ├── __init__.py
│   ├── config.py                  # Default configuration & paths
│   ├── main.py                    # Training & evaluation entry point
│   ├── losses.py                  # Focal, Dice, Boundary losses
│   ├── ust_utils.py               # Frequency-domain mixing utilities
│   ├── data/
│   │   ├── dataset.py             # CassavaDataset, CassavaSemiDataset, transforms
│   │   └── data_preparation.py    # Data loading, K-fold, label-ratio sampling
│   ├── evaluate/
│   │   ├── metrics.py             # Comprehensive evaluation metrics
│   │   └── test_metrics.py        # Metric unit tests
│   ├── models/
│   │   └── model.py               # DeepLabV3Plus (ResNet101)
│   ├── train/
│   │   ├── trainer.py             # UAST-DL training loop (EMA, UST, warm-up)
│   │   └── losses_semi.py         # Focal, Tversky, Dice loss implementations
│   └── utils/
│       ├── logger.py              # Metrics logging, CSV export, curve plots
│       ├── utils_semi.py          # EMA update, sigmoid ramp-up, CutMix, TP-RAM
│       └── visualization.py       # Training visualization utilities
├── model.py                        # Standalone DeepLabV3Plus + load_model()
├── network.py                      # Inference helpers, cassava ROI extraction, TTA
├── predict.py                      # Prediction & PPD quantification script
├── train.py                        # Top-level training launcher
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Method Details

### Supervised Loss

$$\mathcal{L}_{\text{sup}} = 0.5\,\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Tv}} + \mathcal{L}_{\text{Dice}}^{(w)}$$

where Tversky loss uses α = 0.3, β = 0.7 to penalize false negatives more heavily, and weighted Dice uses class weights (ω₀, ω₁) = (1, 10).

### Unsupervised Consistency Loss

$$\mathcal{L}_{\text{unsup}} = \frac{\sum_i m_i w_i \|\mathbf{p}_i^S - \text{sg}(\mathbf{p}_i^T)\|_2^2}{\sum_i m_i w_i + \varepsilon}$$

where:
- $\tilde{y}_i = \arg\max_c p_{i,c}^T$ (pseudo-label)
- $q_i = \max_c p_{i,c}^T$ (confidence)
- $m_i = \mathbb{I}(q_i \geq \tau)$ with τ = 0.95
- $w_i = 0.1$ if $\tilde{y}_i = 0$; $w_i = 1.0$ if $\tilde{y}_i = 1$

### Warm-up & Ramp-up Schedule

$$\lambda(e) = \begin{cases} 0, & e < E_w \\ \lambda_{\max}\exp[-5(1-r_e)^2], & e \geq E_w \end{cases}$$

with $E_w = 5$, $E_r = 5$, $\lambda_{\max} = 1.0$.

### EMA Teacher Update

$$\theta_T^{(k)} = \alpha_k\,\theta_T^{(k-1)} + (1-\alpha_k)\,\theta_S^{(k)}, \quad \alpha_k = \min\!\left(1 - \frac{1}{k+1},\, 0.999\right)$$

---

## Citation

```bibtex
@article{he2025uastdl,
  title   = {Detection and Quantification of Cassava PPD with Semi-supervised Student--Teacher Deep Learning and RGB Images},
  author  = {He, Tao and Zhang, Hao and Chen, Xingmingyue and Long, Haixia},
  journal = {Food Control},
  note    = {Submitted}
}
```

---

## Data Availability

Some datasets, model weights, and code are available at this repository. The full homemade datasets (36 total) are available by contacting the corresponding author (longhx@hainnu.edu.cn).

---

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

---

<div align="center">

**School of Artificial Intelligence, Hainan Normal University, Haikou, China**

**Key Laboratory of Data Science and Smart Education, Ministry of Education**

</div>
