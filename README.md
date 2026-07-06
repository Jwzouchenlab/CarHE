# CarHE: Cross-modal Alignment of Histology and Expression

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**CarHE** is a deep learning framework that aligns H&E-stained histology images with spatial transcriptomics gene expression via contrastive learning. It leverages a **Hierarchical Image Pyramid Transformer (HIPT)** as the image encoder and a **CLIP-style** contrastive objective to learn joint embeddings of tissue morphology and molecular profiles.

<div align="center">
  <img src="https://coresg-normal.trae.ai/api/ide/v1/text_to_image?prompt=Scientific+diagram+illustrating+a+deep+learning+framework+that+aligns+H%26E+histology+images+with+gene+expression+data+using+contrastive+learning.+Left+side+shows+a+tissue+slide+and+256x256+image+patches+entering+a+Vision+Transformer+encoder.+Right+side+shows+gene+expression+vectors+entering+a+projection+head.+Both+outputs+are+L2+normalized+and+fed+into+a+contrastive+loss+that+maximizes+cosine+similarity+for+matched+pairs.+Clean+minimal+vector+style+on+white+background.&image_size=landscape_16_9" width="100%" alt="CarHE Architecture">
</div>

---

## Features

- **HIPT ViT-256 encoder** — extracts hierarchical visual features from gigapixel H&E whole-slide images
- **CLIP contrastive learning** — aligns image patches with matched gene expression profiles in a shared embedding space
- **Projection head** — Linear → GELU → Linear → Dropout → Residual → LayerNorm
- **GradCAM visualization** — interpret which tissue regions drive gene expression predictions
- **Unified AnnData format** — all datasets use the standard `.h5ad` format; no special-case branches
- **Multi-platform support** — compatible with 10X Visium, Visium HD, Xenium, and custom ST platforms

---

## Project Structure

```
CarHE/
├── run_pipeline.py                # One-click pipeline script
├── TUTORIAL.md                    # Full tutorial (Chinese)
├── README.md                      # This file
│
├── img_encoder/                   # HIPT ViT encoder
│   └── HIPT_4K/
│       ├── Checkpoints/
│       │   └── vit256_small_dino.pth   # Pretrained ViT-256 weights
│       ├── hipt_model_utils.py
│       └── vision_transformer.py
│
├── model/                         # Core model code
│   ├── config.py                  # All configuration paths
│   ├── model.py                   # HIPT + ProjectionHead + CLIP loss
│   ├── train.py                   # Training entry point
│   ├── inference.py               # Inference (evaluation / prediction / GradCAM)
│   ├── evaluate.py                # Comprehensive evaluation metrics
│   ├── gradcam.py                 # GradCAM core implementation
│   ├── run_gradcam.py             # Standalone GradCAM runner
│   ├── get_HE_model.py            # HIPT encoder loader
│   ├── get_adata.py               # Unified AnnData data loader
│   ├── utils.py                   # Utility functions
│   └── checkpoint/                # Saved model checkpoints
│
└── data/                          # Data directory
    ├── convert_to_h5ad.py         # Universal format converter
    ├── quick_xenium_to_h5ad.py    # Fast Xenium → h5ad converter
    ├── run_xenium_pipeline.py     # Xenium preprocessing pipeline
    ├── *.py                       # Xenium preprocessing scripts (1-4)
    └── xenium_prostate.h5ad       # Example converted dataset
```

---

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with 16GB+ VRAM (recommended: RTX 3090, A100) — CPU training is possible but slow
- 32GB+ system RAM

### 1. Clone and setup

```bash
git clone https://github.com/Jwzouchenlab/CarHE.git
cd CarHE
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy pandas scanpy anndata opencv-python \
    Pillow tqdm tifffile matplotlib scikit-learn scipy seaborn \
    webdataset einops requests scikit-image h5py zarr
```

### 3. Download HIPT pretrained weights

The image encoder uses **HIPT** (Hierarchical Image Pyramid Transformer). Download the ViT-256 pretrained checkpoint:

| File | Size | Link |
|------|------|------|
| `vit256_small_dino.pth` | ~350 MB | [HIPT GitHub](https://github.com/mahmoodlab/HIPT) → `HIPT_4K/Checkpoints/` |

Place it at: `img_encoder/HIPT_4K/Checkpoints/vit256_small_dino.pth`

> **Reference:** Chen, R.J., et al. "Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning." CVPR 2022. [[Paper]](https://arxiv.org/abs/2206.02647) [[Code]](https://github.com/mahmoodlab/HIPT)

### 4. scGPT (Optional — for reference-based gene embedding)

If you plan to use **scGPT** for gene expression embedding (recommended for large gene panels):

```bash
pip install scgpt
```

> **Reference:** Cui, H., et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods 2024. [[Paper]](https://www.nature.com/articles/s41592-024-02201-0) [[Code]](https://github.com/bowang-lab/scGPT)

### 5. Edit configuration

Edit `model/config.py` to set your data paths. All paths are relative to the `model/` directory by default — no changes needed if you follow the recommended layout.

---

## Quick Start

### Step 1: Prepare your data

CarHE uses the **AnnData (`.h5ad`)** format exclusively. Convert your data first:

```bash
# For Xenium data
cd data
python quick_xenium_to_h5ad.py          # Generates xenium_prostate.h5ad

# For 10X Visium
python convert_to_h5ad.py visium --spaceranger_dir /path/to/spaceranger_output

# For CSV-format datasets (BRCA/DLPFC/CCRCC)
python convert_to_h5ad.py csv --image_dir /path/to/images --sample_ids H1,H2,H3

# Validate your h5ad
python convert_to_h5ad.py validate --adata your_data.h5ad
```

**Required AnnData structure:**

```
adata.X              : [N_cells, N_genes]  float32  gene expression matrix
adata.obs:
  sample_id           : str    sample identifier
  image_path          : str    path to the whole-slide H&E image
adata.obsm:
  spatial             : [N, 2] spatial coordinates in image pixels
  X_scGPT             : [N, D] (optional) precomputed scGPT embeddings
adata.var             : gene names
```

### Step 2: Train

```bash
cd model
python train.py \
    --adata ../data/xenium_prostate.h5ad \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --exp_name ./checkpoint/xenium_model \
    --device cuda
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--adata` | `../data/xenium_prostate.h5ad` | Path to h5ad file |
| `--batch_size` | `32` | Batch size (reduce if OOM) |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `0.002` | Weight decay |
| `--exp_name` | `./checkpoint/train` | Output directory for model checkpoints |
| `--device` | `cuda:0` | Device (`cuda:0` or `cpu`) |
| `--num_workers` | `4` | Data loading workers |

### Step 3: Evaluate

```bash
python evaluate.py \
    --adata ../data/xenium_prostate.h5ad \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --output ./evaluation_results
```

**Metrics reported:**
- Gene-level Pearson / Spearman correlation
- Spot-level PCC, MSE, MAE
- Cosine similarity (diagonal vs off-diagonal)
- Top-K retrieval accuracy (image ↔ expression)

### Step 4: GradCAM Visualization

```bash
# Single-cell GradCAM
python run_gradcam.py \
    --adata ../data/xenium_prostate.h5ad \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --spot_idx 0 \
    --output ./gradcam_output

# Multi-layer GradCAM
python run_gradcam.py \
    --adata ../data/xenium_prostate.h5ad \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --spot_idx 0 \
    --multi_layer \
    --output ./gradcam_output
```

**Output:** A three-panel figure (Original H&E | GradCAM heatmap | Overlay) for each cell.

---

## One-Click Pipeline

```bash
# Check environment
python run_pipeline.py --check

# Run all steps (preprocess → train → evaluate → gradcam)
python run_pipeline.py --all --epochs 50

# Run individual steps
python run_pipeline.py --step train
python run_pipeline.py --step evaluate
python run_pipeline.py --step gradcam
```

---

## Model Architecture

```
┌─────────────────────────────────────────────────────┐
│                  H&E Image (256×256)                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│          HIPT ViT-256 Encoder (frozen ~backbone)     │
│   Patch Embed → Positional Embed → 12 ViT Blocks     │
│   → LayerNorm → Linear Head → 384-dim embedding      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼  L2 Normalize
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   Image Embedding           Spot Embedding
   (384-dim)                 (384-dim)
         │                         ▲
         │                         │
         │                  ┌──────┴──────────┐
         │                  │ Projection Head  │
         │                  │   Linear(2000,   │
         │                  │     384)         │
         │                  │   GELU           │
         │                  │   Linear(384,    │
         │                  │     384)         │
         │                  │   Dropout        │
         │                  │   Residual       │
         │                  │   LayerNorm      │
         │                  └──────┬───────────┘
         │                         │
         │                  Gene Expression
         │                  (2000-dim)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│           Contrastive Loss (Symmetric CE)            │
│    Adjusted similarity matrix + softmax targets      │
│    Temperature-scaled cosine logits                  │
└─────────────────────────────────────────────────────┘
```

---

## Datasets

| Dataset | Platform | Format | Status |
|---------|----------|--------|--------|
| BRCA (HER2ST) | Visium | 36 slides | Via h5ad conversion |
| DLPFC | Visium | 12 slides | Via h5ad conversion |
| CCRCC | Visium (FFPE) | 23 slides | Via h5ad conversion |
| Xenium Prostate | Xenium Prime | 1 slide | Built-in quick converter |
| Custom | Any | `.h5ad` | Use `convert_to_h5ad.py` |

---

## Configuration

All paths are configured in [`model/config.py`](model/config.py):

```python
# HIPT encoder
hipt_module_path = "../img_encoder/HIPT_4K"
hipt_vit256_checkpoint = "../img_encoder/HIPT_4K/Checkpoints/vit256_small_dino.pth"

# Default h5ad path
default_adata_path = "../data/xenium_prostate.h5ad"

# Model parameters
image_embedding = 384
spot_embedding = 2000
projection_dim = 384
```

---

## Citation

If you use CarHE in your research, please cite:

```bibtex
@article{zou2024carhe,
  title   = {CarHE: Cross-modal Alignment of Histology and Expression},
  author  = {Jiawei Zou and Kai Xiao and Zhiyuan Yuan and Luonan Chen},
  journal = {bioRxiv},
  year    = {2024}
}
```

---

## License

This project is licensed under the MIT License.

## Acknowledgements

- [HIPT](https://github.com/mahmoodlab/HIPT) — Hierarchical Image Pyramid Transformer
- [CLIP](https://github.com/openai/CLIP) — Contrastive Language-Image Pre-training
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) — Gradient-weighted Class Activation Mapping
- [scGPT](https://github.com/bowang-lab/scGPT) — Single-cell foundation model
