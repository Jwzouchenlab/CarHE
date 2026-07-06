# CarHE Complete Tutorial

## Full Pipeline: From Xenium Data to Gene Expression Prediction

---

## Project Structure

```
CarHE/                              # Project root directory
├── run_pipeline.py                # ★ One-click run script
├── TUTORIAL.md                    # ★ This document
│
├── img_encoder/                   # HIPT image encoder
│   └── HIPT_4K/
│       ├── Checkpoints/
│       │   └── vit256_small_dino.pth   # ViT-256 pretrained weights
│       ├── hipt_model_utils.py
│       └── vision_transformer.py
│
├── model/                         # Core model code
│   ├── config.py                  # Configuration file (change all paths here)
│   ├── model.py                   # Model definition (HIPT + ProjectionHead + CLIP)
│   ├── train.py                   # Training entry point
│   ├── inference.py               # Inference entry point
│   ├── evaluate.py                # Evaluation entry point
│   ├── gradcam.py                # GradCAM visualization
│   ├── run_gradcam.py            # GradCAM runner script
│   ├── get_xenium.py             # Xenium data loader
│   ├── get_HE_model.py           # HIPT encoder loader
│   ├── Dataset.py                # Generic dataset class
│   └── checkpoint/               # Model save directory
│
└── data/                          # Data directory
    ├── run_xenium_pipeline.py    # Xenium preprocessing script
    ├── *.py                      # Preprocessing scripts (1-4)
    ├── Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif  # H&E image
    ├── Xenium_Prime_Human_Prostate_FFPE_he_imagealignment.csv
    └── Xenium_Prime_Human_Prostate_FFPE_outs/             # Xenium raw output
```

---

## Step 1: Environment Check

```bash
# Run from the CarHE directory
python run_pipeline.py --check
```

This checks:
- Python packages: PyTorch, numpy, pandas, scanpy, opencv, etc.
- HIPT ViT-256 pretrained weights
- Xenium H&E image
- Xenium raw data
- GPU availability

---

## Step 2: Xenium Data Preprocessing

```bash
# Option A: Use the one-click script
python run_pipeline.py --step preprocess

# Option B: Run the preprocessing script directly
cd data
python run_xenium_pipeline.py --all
```

Preprocessing steps:
1. **Coordinate alignment** — Xenium micron coordinates → H&E pixel coordinates
2. **Gene matrix extraction** — Extract and filter from cell_feature_matrix
3. **Nuclei segmentation mask** — Generate Xenium nuclei segmentation image
4. **Cell matching** — Xenium nuclei ↔ H&E nuclei one-to-one matching

Output files (under `data/data_processing/`):
- `matched_nuclei_filtered.csv` — Matched nuclei correspondence table
- `cell_gene_matrix_filtered.csv` — Filtered gene expression matrix
- `xenium_nuclei.tif` — Xenium nuclei segmentation mask
- `he_image_nuclei_seg_microns.tif` — H&E nuclei segmentation mask (requires HoverNet)

---

## Step 3: Model Training

```bash
# Option A: Use the one-click script (default 50 epochs)
python run_pipeline.py --step train

# Option B: Custom parameters
cd model
python train.py \
    --dataset xenium \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 50 \
    --exp_name ./checkpoint/xenium_model
```

Training parameter reference:
| Parameter | Default | Description |
|------|--------|------|
| `--dataset` | xenium | Dataset type |
| `--batch_size` | 32 | Batch size (reduce if GPU memory is insufficient) |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 50 | Number of training epochs |
| `--exp_name` | checkpoint/xenium_model | Model save directory |

Model save locations:
- `checkpoint/xenium_model/best_model.pt` — Best model on validation set
- `checkpoint/xenium_model/model.pt` — Final model

---

## Step 4: Model Evaluation

```bash
# Option A: Use the one-click script
python run_pipeline.py --step evaluate

# Option B: Specify checkpoint
cd model
python evaluate.py \
    --dataset xenium \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --output ./evaluation_results/xenium
```

Evaluation metrics (output in `evaluation_results/xenium/`):
- **Gene-level PCC** — Pearson correlation per gene
- **Gene-level Spearman** — Spearman correlation per gene
- **Spot-level PCC/MSE/MAE** — Prediction accuracy per cell
- **Cosine Similarity** — Diagonal/off-diagonal cosine similarity in embedding space
- **Top-K Retrieval** — Image→gene and gene→image retrieval accuracy

---

## Step 5: GradCAM Visualization

```bash
# Option A: Use the one-click script (default: cell 0)
python run_pipeline.py --step gradcam

# Option B: Specify a cell
cd model
python run_gradcam.py \
    --dataset xenium \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --spot_idx 0 \
    --output ./gradcam_output

# Multi-cell GradCAM
python run_gradcam.py \
    --dataset xenium \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --spot_idx 0 \
    --multi_layer \
    --output ./gradcam_output
```

Output files (under `gradcam_output/`):
- `gradcam_X.png` — GradCAM triptych for cell X (original + heatmap + overlay)

---

## Full Pipeline One-Click Run

```bash
# Environment check + preprocessing + training + evaluation + GradCAM
python run_pipeline.py --all

# Custom training parameters
python run_pipeline.py --all --epochs 100 --batch_size 64

# Stop on first error
python run_pipeline.py --all --stop_on_error
```

---

## Environment Dependencies

```bash
pip install torch torchvision numpy pandas scanpy anndata opencv-python \
    Pillow tqdm tifffile matplotlib scikit-learn scipy seaborn \
    python-docx zarr
```

---

## FAQ

### Q: "HIPT module path not found"
Make sure the `img_encoder/HIPT_4K/` directory exists and contains `hipt_model_utils.py`.

### Q: Insufficient GPU memory
- Reduce `--batch_size` (e.g., 16 or 8)
- Or use `--device cpu` (slower)

### Q: Xenium gene count mismatch with model
Xenium typically has around 400 genes; when fewer than 2000, zeros are automatically padded. To modify, edit `xenium_ngenes` in `config.py`.

### Q: How to use other checkpoints?
```bash
python run_pipeline.py --step evaluate --checkpoint ./checkpoint/Final_scgpt/model.pt
```

---

## All Available Path Configurations (config.py)

| Variable | Current Value | Description |
|------|--------|------|
| `hipt_module_path` | `../img_encoder/HIPT_4K` | HIPT module path |
| `hipt_vit256_checkpoint` | `../img_encoder/HIPT_4K/Checkpoints/vit256_small_dino.pth` | ViT-256 weights |
| `xenium_he_image` | `../data/Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif` | H&E image |
| `xenium_matched_nuclei` | `../data/data_processing/matched_nuclei_filtered.csv` | Matched cells |
| `xenium_cell_gene_matrix` | `../data/data_processing/cell_gene_matrix_filtered.csv` | Gene matrix |
| `xenium_seg_mask` | `../data/data_processing/he_image_nuclei_seg_microns.tif` | Nuclei segmentation |
| `checkpoint_dir` | `./checkpoint` | Model save directory |
| `log_dir` | `./logs` | Log directory |
