# Classification of Laguerre-Gauss Spatial Light Modes with CNNs (PyTorch)

This repository is a PyTorch reimplementation of the MATLAB neural networks originally developed for the undergraduate [thesis](https://ri-ng.uaq.mx/handle/123456789/10410) *"Reconstrucción de modos espaciales de luz a partir de redes neuronales profundas"* ([Bruno Salgado](https://brunosalgado.website/)). It reproduces, appendix by appendix, the multilayer perceptron, the custom 3-layer CNN, and the transfer-learning networks (EfficientNet-B0, ResNet18/50/101) used to classify intensity patterns of Laguerre-Gauss (LG) light modes after propagation through free space or multimode optical fiber.

---
*July 20th, 2026 (A PyTorch reinterpretation of a thesis project originally implemented in MATLAB)*

## Table of Contents

1. [Project Overview](#project-overview)
2. [Physical Background](#physical-background)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Installation & Usage](#installation--usage)
6. [Training](#training)
7. [Prediction & GUI](#prediction--gui)
8. [Differences from the Original MATLAB Implementation](#differences-from-the-original-matlab-implementation)
9. [Repository Structure](#repository-structure)
10. [Limitations and Technical Notes](#limitations-and-technical-notes)
11. [References](#references)

---

## Project Overview

Light beams carrying orbital angular momentum (OAM), such as Laguerre-Gauss modes, are of interest for optical communication protocols because their quantum numbers ($p$, $l$) can, in principle, be used to encode information. This project trains convolutional neural networks to recognize which LG mode produced a given intensity pattern after the beam has traveled through a dispersive medium (multimode optical fiber) or free space, using only the resulting 2D intensity image as input.

The original work (2024-2025) was implemented in MATLAB using the Deep Learning Toolbox and App Designer (see `matlab_reference/` for the appendices from the thesis). This repository reproduces the same architectures and training procedures using **PyTorch** and **torchvision**, so the networks can be retrained with the user's own dataset outside of MATLAB.

> **Note:** The experimental image dataset was collected independently by the Laboratorio de Micro- y Nanofotónica, ICN-UNAM, and is **not included** in this repository. See [Dataset](#dataset) for the expected folder layout so you can plug in your own copy of the data and retrain the networks.

---

## Physical Background

Laguerre-Gauss modes are solutions to the paraxial wave equation in cylindrical coordinates, characterized by a radial index $p$ and an azimuthal (topological) index $l$. The azimuthal phase term $e^{il\phi}$ carries $l\hbar$ of orbital angular momentum per photon. Because modes with different $(p, l)$ have visually distinct intensity distributions (and because propagation through a dispersive medium such as multimode fiber scrambles the transverse phase, changing the recorded pattern), a CNN trained on intensity images can, without any explicit phase information, learn to classify the underlying $(p, l)$ combination.

---

## Dataset

The thesis used **1,900 experimental images** (100 per class) distributed over **19 classes**:

- **13 fiber-propagated modes** — $LG_{0,l}$ with $l \in \{-6, ..., 6\}$, recorded after propagation through multimode optical fiber.
- **6 free-space modes** — $LG_{p,\pm1}$ with $p \in \{0, 3, 5\}$, recorded after propagation through free space.

Two resolutions of the same dataset were used, matching the input layer of each network family:

| Folder | Resolution | Channels | Used by |
|---|---|---|---|
| `Fotos 128x192/` | 128 x 192 | 1 (grayscale) | Simple 3-layer CNN (Appendix 3) |
| `Fotos 224x224x3/` | 224 x 224 | 3 (RGB) | EfficientNet-B0 / ResNet18 / ResNet50 / ResNet101 (Appendices 4-5) |

Each of these two folders must contain one subfolder per class, following the original MATLAB naming convention so the class label can be parsed automatically from the folder name:

```
Fotos 128x192/
├── m__-6/        # fiber mode, l = -6
├── m__-5/
│   ...
├── m__6/         # fiber mode, l = 6
├── LG_-10/       # free-space mode, l = -1, p = 0
├── LG_10/        # free-space mode, l = 1,  p = 0
├── LG_-13/       # free-space mode, l = -1, p = 3
├── LG_13/
├── LG_-15/       # free-space mode, l = -1, p = 5
└── LG_15/
```

`m__<l>` folders hold fiber-propagated modes labeled only by their azimuthal number $l$. `LG_<l><p>` folders hold free-space modes, where the sign of $l$ is written directly before $p$ (e.g. `LG_-15` is $l=-1, p=5$; `LG_13` is $l=1, p=3$).

Place your own copy of the dataset under `data/`, e.g.:

```
data/
├── Fotos 128x192/
└── Fotos 224x224x3/
```

The `--data-dir` flag of every training script points at one of these two folders.

### Preprocessing pipeline (Appendix 1 / Appendix 2)

The very first network reproduced in the thesis (Appendix 1, a reimplementation of Lollie et al., 2022) does **not** use the resized JPEGs above; instead it works from full-resolution raw photographs, which it crops and downsamples on the fly:

1. Convert to grayscale.
2. Crop a fixed region `[x=1900, y=1100, width=2899, height=2499]` that discards the black borders of the raw photograph.
3. Downsample by block-averaging (default block size 20x20 pixels).
4. Flatten into a single feature vector.

This is reproduced in [`src/datasets/downsampling.py`](src/datasets/downsampling.py) (`down_sampling`, equivalent to `DownSampling_v2.m`) and [`src/datasets/mlp_dataset.py`](src/datasets/mlp_dataset.py).

---

## Models

| Appendix | Original MATLAB | PyTorch equivalent | File |
|---|---|---|---|
| 1 | `patternnet(5)` (single hidden layer, 5 neurons) | `SimpleMLP` | `src/models/simple_mlp.py` |
| 2 | `DownSampling_v2.m` | `down_sampling()` | `src/datasets/downsampling.py` |
| 3 | Custom CNN, 3x(conv-batchnorm-relu), 2 max-pools, 8 filters/layer | `SimpleCNN` | `src/models/simple_cnn.py` |
| 4 | `efficientnetb0` (transfer learning) | `build_efficientnet_b0()` | `src/models/transfer_learning.py` |
| 5 | `resnet18` / `resnet50` / `resnet101` (transfer learning) | `build_resnet()` | `src/models/transfer_learning.py` |
| 6 | App Designer GUI (`GUILaguerreGaussBeams`) | Tkinter app | `gui/app.py` |

### Appendix 3 — Simple CNN

Three convolutional blocks with 8 filters of size 3x3 each (`padding='same'`), batch normalization and ReLU; the first two blocks are followed by 2x2 max pooling, the third is not; the result is flattened into a fully connected layer with one output per class.

### Appendices 4-5 — Transfer learning

The pretrained ImageNet backbones (EfficientNet-B0, ResNet18/50/101) have their classification head replaced with a fully connected layer of `num_classes` outputs, and their earliest layers frozen (kept at their pretrained weights) so only the last blocks and the new head are fine-tuned — mirroring `freezeWeights(layers(1:flayer))` in the MATLAB code.

---

## Installation & Usage

### Python Version
- Python 3.9 or higher recommended.
- Check your version: `python --version` or `python3 --version`

### Setup

```bash
git clone https://github.com/b-salgado13/lg-mode-cnn-pytorch.git
cd lg-mode-cnn-pytorch
python -m venv .venv
source .venv/bin/activate       # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place your dataset under `data/` as described in [Dataset](#dataset).

---

## Training

Each appendix has its own training entry point under `src/training/`. All scripts save a checkpoint (`.pt` file) plus a `class_to_idx.json` mapping so predictions can be mapped back to LG mode labels.

### Appendix 1 — Simple MLP

```bash
python -m src.training.train_mlp \
    --data-dir "data/Fotos" \
    --output checkpoints/mlp.pt \
    --epochs 400 \
    --hidden-units 5 \
    --down-square 20
```

### Appendix 3 — Simple 3-layer CNN

```bash
python -m src.training.train_cnn \
    --data-dir "data/Fotos 128x192" \
    --output checkpoints/cnn_8_8_8.pt \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-3
```

### Appendices 4-5 — Transfer learning (EfficientNet-B0 / ResNet)

```bash
python -m src.training.train_transfer \
    --data-dir "data/Fotos 224x224x3" \
    --backbone resnet18 \
    --freeze-fraction 0.7 \
    --output checkpoints/resnet18.pt \
    --epochs 10 \
    --batch-size 70 \
    --lr 1e-3
```

`--backbone` accepts `efficientnet_b0`, `resnet18`, `resnet50`, or `resnet101`.

All scripts print training/validation loss and accuracy per epoch and use a 70/15/15 (MLP) or 70/30 (CNN/transfer) train/validation(/test) split, mirroring `Net.divideParam` and `splitEachLabel` from the MATLAB code respectively.

---

## Prediction & GUI

### Command line

```bash
python -m src.predict \
    --checkpoint checkpoints/resnet18.pt \
    --backbone resnet18 \
    --image path/to/pattern.jpg
```

This prints the predicted mode (fiber or free-space, with its $l$ and $p$ values) and the confidence percentage, mirroring `PredictLGModeButtonPushed` in Appendix 6.

### GUI

```bash
python -m gui.app
```

The Tkinter app reproduces the core workflow of the original App Designer interface (Appendix 6):

1. Choose a network from the dropdown (Simple CNN or a transfer-learning backbone).
2. Load a pattern image through the file picker.
3. Run the prediction and display the reconstructed mode and confidence percentage.

The decorative fiber-propagation animation and the reference mode images from the original GUI were left out since they depend on institution-specific static assets not included in this repository; the classification logic itself is fully reproduced.

---

## Differences from the Original MATLAB Implementation

- **Optimizer:** the MATLAB scripts use `trainscg`/`sgdm`; this repository uses PyTorch's `SGD` with momentum by default (configurable via `--optimizer`) to stay close to the original `sgdm`, but the MLP (Appendix 1, originally trained with `trainscg`) uses `Adam` here since PyTorch does not ship a scaled-conjugate-gradient optimizer.
- **Layer freezing:** MATLAB freezes an explicit number of *layers* (`freezeWeights(layers(1:flayer))`); this implementation freezes a *fraction* of the backbone's parameters (`--freeze-fraction`), which is architecture-agnostic and easier to tune for different torchvision versions.
- **Preprocessing:** `padding='same'` in MATLAB's `convolution2dLayer` is reproduced with equivalent explicit padding in PyTorch's `nn.Conv2d`.
- **GUI:** reimplemented in Tkinter rather than MATLAB App Designer; the propagation animation is omitted (see [Prediction & GUI](#prediction--gui)).

---

## Repository Structure

```
lg-mode-cnn-pytorch/
├── data/                          # place your dataset here (not tracked by git)
├── gui/
│   └── app.py                     # Appendix 6 — Tkinter GUI
├── matlab_reference/               # original MATLAB code, transcribed from the thesis appendices
├── src/
│   ├── datasets/
│   │   ├── downsampling.py        # Appendix 2 — DownSampling_v2.m
│   │   ├── mlp_dataset.py         # Appendix 1 preprocessing pipeline
│   │   └── image_dataset.py       # ImageFolder-based dataset for CNN/transfer learning
│   ├── models/
│   │   ├── simple_mlp.py          # Appendix 1
│   │   ├── simple_cnn.py          # Appendix 3
│   │   └── transfer_learning.py   # Appendices 4-5
│   ├── training/
│   │   ├── train_mlp.py
│   │   ├── train_cnn.py
│   │   └── train_transfer.py
│   ├── mode_labels.py             # folder-name <-> (l, p, medium) parsing
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## Limitations and Technical Notes

- The dataset is **not included** in this repository for institutional/privacy reasons; you must supply your own copy following the folder layout above.
- Results (accuracy, training time) will depend on your hardware, hyperparameters, and dataset size, and are therefore not reported here — retrain and evaluate with your own data.
- The frozen-layer counts reported in the thesis (e.g. 50 layers for ResNet18, 286 for EfficientNet-B0) are specific to MATLAB's internal layer graph representation and do not map one-to-one onto PyTorch's module structure; `--freeze-fraction` is provided as an equivalent, adjustable control instead.

---

## References

1. Allen, L., Padgett, M. J., & Babiker, M. (1999). IV The Orbital Angular Momentum of Light. Progress in Optics, 39, 291-372.
2. Lollie, et al. (2022). Reconstruction of Laguerre-Gauss modes with a shallow neural network (as reproduced in Appendices 1-2 of the original thesis).
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv:1512.03385*.
5. Salgado Molina, B. I. (2024). *Reconstrucción de modos espaciales de luz a partir de redes neuronales profundas*(undergraduate thesis, original MATLAB implementation and dataset).
