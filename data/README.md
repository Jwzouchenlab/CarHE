# Datasets & Download Links

This directory contains data conversion scripts. Raw data files are **not** included in the repository due to size constraints. Download them from the sources below.

---

## Downloadable Datasets

| Dataset | Platform | Samples | Download Link |
|---------|----------|---------|---------------|
| **BRCA (HER2ST)** | Visium | 36 slides | [GitHub](https://github.com/almaan/her2st) |
| **DLPFC** | Visium | 12 slides | [SpatialLIBD](http://research.libd.org/spatialLIBD/) |
| **CCRCC** | Visium (FFPE) | 23 slides | [GEO GSE240773](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240773) |
| **HEST-1k** | Visium/HD/Xenium | 1000+ slides | [HuggingFace](https://huggingface.co/datasets/MahmoodLab/hest) |
| **10X Xenium** | Xenium Prime | Various | [10X Genomics](https://www.10xgenomics.com/datasets) |
| **10X Visium HD** | Visium HD | Various | [10X Genomics](https://www.10xgenomics.com/datasets) |

---

## Third-Party Methods Referenced

### HIPT (Hierarchical Image Pyramid Transformer)

The image encoder backbone used by CarHE.

- **Paper:** Chen, R.J. et al. "Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning." CVPR 2022.
- **Code:** https://github.com/mahmoodlab/HIPT
- **Download:** `vit256_small_dino.pth` from `HIPT_4K/Checkpoints/`
- **Repo location:** `../img_encoder/HIPT_4K/`

### scGPT (Single-Cell Generative Pre-trained Transformer)

Pre-trained foundation model for gene expression embedding (optional but recommended).

- **Paper:** Cui, H. et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods 2024.
- **Code:** https://github.com/bowang-lab/scGPT
- **Usage:** `pip install scgpt`; produces `adata.obsm['X_scGPT']` embeddings

### Hover-Net (optional — for H&E nuclei segmentation in Xenium pipeline)

- **Paper:** Graham, S. et al. "Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical Image Analysis 2019.
- **Code:** https://github.com/vqdang/hover_net
- **Note:** Only needed if running the full Xenium preprocessing pipeline with H&E nuclei segmentation. The quick converter (`quick_xenium_to_h5ad.py`) does not require this.

### Grad-CAM

- **Paper:** Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
- **Integrated implementation:** `model/gradcam.py`

---

## Data Format After Conversion

All datasets are converted to the unified **AnnData (.h5ad)** format:

```python
adata.X              # [N, G] float32 — gene expression matrix
adata.obs["sample_id"]   # sample identifier
adata.obs["image_path"]  # absolute path to H&E whole-slide image
adata.obsm["spatial"]    # [N, 2] pixel coordinates (x, y)
adata.obsm["X_scGPT"]    # (optional) scGPT embeddings
adata.var                  # gene names
```

## Conversion Scripts

| Script | Input | Output |
|--------|-------|--------|
| `quick_xenium_to_h5ad.py` | Xenium `outs/` folder | `.h5ad` |
| `convert_to_h5ad.py xenium` | Preprocessed Xenium CSVs | `.h5ad` |
| `convert_to_h5ad.py csv` | CSV-format (BRCA/DLPFC) | `.h5ad` |
| `convert_to_h5ad.py visium` | 10X spaceranger output | `.h5ad` |
| `convert_to_h5ad.py validate` | Any `.h5ad` | Format validation report |
| `scgpt_finetune.py` | Reference scRNA-seq h5ad | Fine-tuned scGPT model |
| `scgpt_embed.py` | Spatial h5ad + scGPT model | `.h5ad` with `obsm['X_scGPT']` |

> **Note:** For full Xenium preprocessing with H&E nuclei segmentation (requires HoverNet), see the original scripts at the [Zenodo release](https://doi.org/10.5281/zenodo.15668517).

## scGPT Integration

CarHE can use **scGPT** embeddings for gene expression representation. Two-step workflow:

### Step 1: Fine-tune scGPT

```bash
python scgpt_finetune.py \
    --adata reference_scrnaseq.h5ad \
    --checkpoint ./scgpt_checkpoints/scgpt/pan_cancer \
    --epochs 10
```

### Step 2: Generate embeddings

```bash
python scgpt_embed.py \
    --adata ../data/xenium_prostate.h5ad \
    --checkpoint ./scgpt_checkpoints/scgpt/pan_cancer \
    --finetuned ./scgpt_finetuned/best_model.pt
```

### scGPT Checkpoint Downloads

| Checkpoint | Size | Link |
|------------|------|------|
| `pan_cancer` (pre-trained) | ~400 MB | [scGPT GitHub](https://github.com/bowang-lab/scGPT) |
| `whole_blood` (pre-trained) | ~400 MB | [scGPT GitHub](https://github.com/bowang-lab/scGPT) |

Download and extract to `scgpt_checkpoints/scgpt/`. Expected structure:

```
scgpt_checkpoints/scgpt/pan_cancer/
├── best_model.pt
├── args.json
└── vocab.json
```

> **Reference:** Cui, H., et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods 2024. [[Paper]](https://www.nature.com/articles/s41592-024-02201-0) [[Code]](https://github.com/bowang-lab/scGPT)

## Example: Full Xenium Pipeline

```bash
# 1. Download Xenium prostate dataset from 10X:
#    https://www.10xgenomics.com/datasets

# 2. Place files in data/
#    data/Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif
#    data/Xenium_Prime_Human_Prostate_FFPE_he_imagealignment.csv
#    data/Xenium_Prime_Human_Prostate_FFPE_outs/

# 3. Quick convert to h5ad
python quick_xenium_to_h5ad.py

# 4. Train
cd ../model
python train.py --adata ../data/xenium_prostate.h5ad --device cuda --epochs 50
```
