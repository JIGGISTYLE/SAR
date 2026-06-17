# Multi-Modal Satellite Land Cover Classification

> Dual-branch CNN that fuses Sentinel-1 SAR and Sentinel-2 optical imagery for terrain classification — **98.38% SAR · 98.08% Optical · 99.75% Fusion (val)**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Overview

This project trains three CNN models on paired Sentinel-1/2 satellite images to classify terrain into four categories: **agricultural land**, **barrenland**, **grassland**, and **urban**. The highlight is a **late-fusion model** that combines learned feature vectors from both modalities, consistently outperforming either single-modal model alone.

| Model | Modality | Test Accuracy |
|---|---|---|
| SAR CNN | Sentinel-1 (grayscale radar) | **98.38%** |
| Optical CNN | Sentinel-2 (RGB) | **98.08%** |
| Fusion CNN | SAR + Optical (late fusion) | **≥ 99.75%** (val @ epoch 4) |

---

## Why This Problem

SAR radar images penetrate clouds and work at night but look noisy. Optical images are photorealistic but blocked by weather. Fusing both gives an all-weather, high-accuracy classifier — relevant for precision agriculture, urban planning, and disaster response.

---

## Architecture

```
Sentinel-1 SAR (1-ch, 256×256)         Sentinel-2 Optical (3-ch, 256×256)
         │                                          │
    ┌────▼────┐                              ┌──────▼──────┐
    │ SAR CNN │  5 conv blocks               │ Optical CNN │  5 conv blocks
    │         │  (1→32→64→64→128→128)        │             │  (3→32→64→64→128→128)
    └────┬────┘                              └──────┬──────┘
         │  flatten → 8192-dim                      │  flatten → 8192-dim
         └──────────────────┬───────────────────────┘
                            │  torch.cat → 16384-dim
                     ┌──────▼──────┐
                     │ Fusion Head │  Linear(16384→128) → ReLU → Dropout(0.5) → Linear(128→4)
                     └──────┬──────┘
                            │
                     4-class logits
              (agri · barrenland · grassland · urban)
```

Each conv block: `Conv2d(3×3, pad=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2×2)`

Five blocks halve the spatial resolution five times: **256 → 128 → 64 → 32 → 16 → 8**, yielding a **128 × 8 × 8 = 8,192**-dimensional feature vector per branch.

---

## Dataset

**[Sentinel-1/2 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)** (Kaggle)

```
v_2/
├── agri/
│   ├── s1/   ← SAR  (grayscale PNG, 256×256)
│   └── s2/   ← Optical (RGB PNG, 256×256)
├── barrenland/
├── grassland/
└── urban/
```

- **16,000 paired images** (4,000 per class)
- Split: **70% train / 15% val / 15% test** (fixed seed = 42, reproducible)
- SAR and Optical paired by filename: `ROIs1868_summer_s1_59_p100.png` ↔ `ROIs1868_summer_s2_59_p100.png`

---

## Project Structure

```
├── dataset.py     # PyTorch Dataset classes (SAR_Dataset, Optical_Dataset, Fusion_Dataset)
├── model.py       # CNN architectures (SAR_model, Optical_model, Fusion_model)
├── utils.py       # Transforms, dataloaders, train/val/test split logic
├── train.py       # train_one_epoch() and evaluate() — modality-agnostic
├── main.py        # Orchestrates training runs with checkpointing
└── inference.py   # load_model(), predict_fusion(), predict_sar(), predict_optical()
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision pillow tqdm
```

### 2. Download the dataset

Place the Kaggle dataset at the path expected by `utils.py`:

```
/kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2/
```

Or update `BASE_DIR` in `utils.py` to your local path.

### 3. Train

```python
# In main.py — uncomment the run you want:

sar_main()          # Train SAR-only model     → saves best_model.pth
optical_main()      # Train Optical-only model → saves best_model_optical.pth
fusion_main(pretrained=True)   # Train fusion model, warm-started from above checkpoints
```

### 4. Run inference

```python
model = load_model("fusion", "best_model_fusion.pth")

result = predict_fusion(
    model,
    sar_path="path/to/image_s1.png",
    opt_path="path/to/image_s2.png"
)

print_result(result)
# Predicted : agri  (conf=1.0000)
#   agri         1.0000  ████████████████████████████████████████
#   barrenland   0.0000
#   grassland    0.0000
#   urban        0.0000
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 20 |
| Loss | CrossEntropyLoss |
| Dropout | 0.5 |
| Augmentation | RandomHorizontalFlip |
| Input size | 256 × 256 |

**SAR normalisation:** `mean=0.5, std=0.5` (grayscale radar backscatter)

**Optical normalisation:** ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

---

## Fusion Strategy

This project uses **late feature-level fusion** (also called decision-level fusion at the feature stage):

1. Both branches train to convergence as single-modal models first.
2. The fusion model is **warm-started** by copying the trained branch weights — the fusion head trains from scratch.
3. During fusion training, gradients flow through all parameters jointly, allowing the branches to co-adapt.

This approach was chosen over early fusion (channel stacking) because SAR and optical images have fundamentally different statistical distributions — radar backscatter vs. natural light — making it difficult for a single early CNN to learn useful joint representations.

---

## Checkpointing

Each training run saves the **best validation accuracy** checkpoint (not the last epoch):

```python
{
    "epoch":           int,
    "model_state":     state_dict,
    "optimizer_state": state_dict,
    "best_val_acc":    float,
}
```

Training can be **resumed** from any checkpoint — the loop will pick up from the saved epoch with the optimizer's internal state (momentum, adaptive LR) fully restored.

---

## Results

```
============================================================
  RESULTS SUMMARY
============================================================
  SAR only     : 98.38%
  Optical only : 98.08%
  Fusion        : ≥ 99.75%  (val acc @ epoch 4, training interrupted)
============================================================
```

The fusion model reached 99.75% validation accuracy by epoch 4 — already surpassing both single-modal models — before training was manually interrupted.

---

## Possible Improvements

- **Attention-based fusion** — cross-attention between SAR and optical feature maps instead of simple concatenation
- **Learning rate scheduling** — cosine annealing or ReduceLROnPlateau
- **Stronger augmentation** — random rotations, colour jitter (optical only), Gaussian noise (SAR)
- **Pretrained encoders** — replace custom CNN backbone with ResNet-18 or EfficientNet-B0
- **Per-class metrics** — confusion matrix, per-class F1, to identify which terrain types are hardest

---

## References

- [Sentinel-1/2 Dataset on Kaggle](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Dimitrovski et al., *Deep Multimodal Fusion for Semantic Segmentation of Remote Sensing Earth Observation Data*, arXiv:2410.00469

---

## License

MIT License — see [LICENSE](LICENSE) for details.
