# HiFi-ST

HiFi-ST is a deep learning framework for spatial gene expression prediction from histology image patches and spatial transcriptomics data. The model integrates multi-scale histology image encoding, spatial positional modeling, scale-aware feature modulation, neural-field-based gene expression reconstruction, and optional high-expression-gene prediction for spot-level gene expression estimation across tissue sections.

The current implementation supports three dataset settings:

- HER2ST breast cancer spatial transcriptomics data
- Human cutaneous squamous cell carcinoma spatial transcriptomics data from GSE144240
- Alex+10x spatial transcriptomics data




## Overview

HiFi-ST contains the following components:

- Multi-scale histology image encoder for 112, 224, and 448 pixel patches
- Fourier positional encoding for spatial coordinates
- Scale-aware FiLM modulation for adaptive multi-scale feature fusion
- Neural field decoder for continuous spatial gene expression reconstruction
- Monte Carlo spot-level aggregation for robust prediction
- Optional expression-PCA and spatial-PCA conditioning
- Optional high-expression-gene branch for cSCC experiments

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
python >= 3.9
torch
numpy
pandas
Pillow
scipy
scikit-learn
tqdm
```

A CUDA-enabled GPU is recommended for training.

## Repository Structure

```text
HiFi-ST/
├── dataset.py          # Dataset loading and dataloader construction
├── model.py            # HiFi-ST model architecture
├── train.py            # Training, loss, and metric utilities
├── eval.py             # Evaluation utilities for trained checkpoints
├── main.py             # Command-line entry point
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Training

Train one fold on cSCC:

```bash
python main.py \
  --dataset cSCC \
  --fold 0 \
  --epochs 100 \
  --batch_size 512 \
  --device cuda \
  --output_dir outputs
```

Train one fold on HER2ST:

```bash
python main.py \
  --dataset HER2+ \
  --fold 0 \
  --epochs 100 \
  --batch_size 512 \
  --device cuda \
  --output_dir outputs
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python main.py \
  --evaluate \
  --dataset cSCC \
  --fold 0 \
  --model_path outputs/cSCC_0/best_model.pth \
  --device cuda \
  --output_dir outputs
```
