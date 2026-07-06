# -*- coding: utf-8 -*-
"""CarHE GradCAM Runner Script
Run GradCAM analysis on H&E images to visualize image regions the model attends to.

Features:
    1. GradCAM analysis for specified spot/gene
    2. Batch analysis for multiple marker genes
    3. Compare GradCAM heatmaps across different genes
    4. Multi-layer GradCAM analysis

Usage:
    # Single spot-level GradCAM
    python run_gradcam.py --checkpoint ./checkpoint/model.pt \
        --dataset BRCA --sample_id H1 --spot_idx 0
    
    # Gene-level GradCAM (multiple marker genes)
    python run_gradcam.py --checkpoint ./checkpoint/model.pt \
        --dataset adata --adata data.h5ad --marker_genes ERBB2,ESR1,PGR
    
    # GradCAM from full-image coordinates
    python run_gradcam.py --checkpoint ./checkpoint/model.pt \
        --image_path image.tif --centers_path centers.txt \
        --ref_adata ref.h5ad --spot_idx 0
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import scanpy as sc
import anndata as ad
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

from config import CFG
from model import HIPT_CLIP_Model
from gradcam import (
    GradCAM_CarHE, ViT_GradCAM, MultiLayerGradCAM,
    visualize_heatmap, visualize_multi_gene_heatmap,
    save_heatmap_as_image, overlay_heatmap,
)


# ==================== Logging ====================
def setup_logger(save_dir: str):
    logger = logging.getLogger("CarHE_GradCAM")
    logger.setLevel(logging.INFO)
    
    os.makedirs(save_dir, exist_ok=True)
    
    fh = logging.FileHandler(os.path.join(save_dir, "gradcam_log.txt"), mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ==================== Data Loading ====================
def load_data_from_dataset(args, logger):
    """Load image and expression data from adata"""
    from get_adata import build_loaders_adata
    import scanpy as sc
    
    logger.info(f"Loading data from {args.adata}...")
    adata = sc.read_h5ad(args.adata)
    _, test_loader = build_loaders_adata(adata=adata, batch_size=1)
    
    # Get sample
    data_iter = iter(test_loader)
    batch = next(data_iter)
    
    image_tensor = batch['image']  # [1, 3, 256, 256]
    spot_expression = batch['reduced_expression']  # [1, n_genes]
    barcode = batch.get('barcode', ['unknown'])[0] if 'barcode' in batch else 'unknown'
    
    logger.info(f"Loaded sample: barcode={barcode}, image shape={image_tensor.shape}")
    
    return image_tensor, spot_expression, barcode, None


def load_data_from_image(args, logger):
    """Load data from raw image file and coordinates"""
    import tifffile
    
    logger.info(f"Loading image: {args.image_path}")
    
    # Load image
    whole_image = cv2.imread(args.image_path)
    if whole_image is None:
        whole_image = tifffile.imread(args.image_path)
        if whole_image is None:
            raise ValueError(f"Cannot load image: {args.image_path}")
    
    if whole_image.ndim == 2:
        whole_image = cv2.cvtColor(whole_image, cv2.COLOR_GRAY2RGB)
    
    # Load coordinates
    if args.centers_path:
        centers = pd.read_csv(args.centers_path, sep=',', header=None)
        if args.spot_idx >= len(centers):
            raise ValueError(f"spot_idx {args.spot_idx} out of range (max {len(centers)-1})")
        center_x, center_y = centers.iloc[args.spot_idx, 0], centers.iloc[args.spot_idx, 1]
    elif args.spatial_csv:
        spatial = pd.read_csv(args.spatial_csv)
        if args.spot_idx >= len(spatial):
            raise ValueError(f"spot_idx {args.spot_idx} out of range")
        center_x = int(spatial.iloc[args.spot_idx]['pixel_x'])
        center_y = int(spatial.iloc[args.spot_idx]['pixel_y'])
    else:
        raise ValueError("Need --centers_path or --spatial_csv")
    
    # Load reference expression
    if args.ref_adata:
        adata_ref = sc.read_h5ad(args.ref_adata)
        if args.spot_idx >= adata_ref.shape[0]:
            raise ValueError(f"spot_idx {args.spot_idx} out of reference range")
        if args.spot_col and args.spot_col in adata_ref.obs.columns:
            # Filter by sample
            ref_subset = adata_ref[adata_ref.obs[args.spot_col] == args.spot_id]
            if len(ref_subset) > 0:
                spot_expr = ref_subset[args.spot_idx % len(ref_subset)]
                spot_expression = torch.tensor(
                    spot_expr.X.toarray().flatten()[:CFG.spot_embedding]
                    if hasattr(spot_expr.X, 'toarray')
                    else spot_expr.X.flatten()[:CFG.spot_embedding]
                ).float()
            else:
                spot_expression = torch.tensor(
                    adata_ref[args.spot_idx].X.toarray().flatten()[:CFG.spot_embedding]
                    if hasattr(adata_ref[args.spot_idx].X, 'toarray')
                    else adata_ref[args.spot_idx].X.flatten()[:CFG.spot_embedding]
                ).float()
        else:
            idx = args.spot_idx % adata_ref.shape[0]
            x = adata_ref[idx].X
            if hasattr(x, 'toarray'):
                x = x.toarray()
            spot_expression = torch.tensor(x.flatten()[:CFG.spot_embedding]).float()
    else:
        # Use zero vector (demo only)
        spot_expression = torch.zeros(CFG.spot_embedding)
    
    # Extract image patch
    half = 128
    patch = whole_image[
        max(0, center_y-half):min(whole_image.shape[0], center_y+half),
        max(0, center_x-half):min(whole_image.shape[1], center_x+half)
    ]
    
    # Padding
    pad_top = max(0, half - center_y)
    pad_bottom = max(0, center_y + half - whole_image.shape[0])
    pad_left = max(0, half - center_x)
    pad_right = max(0, center_x + half - whole_image.shape[1])
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    if patch.shape[0] != 256 or patch.shape[1] != 256:
        patch = cv2.resize(patch, (256, 256))
    
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if patch.ndim == 3 and patch.shape[2] == 3 else patch
    patch_pil = Image.fromarray(patch_rgb)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(patch_pil).unsqueeze(0)
    spot_expression = spot_expression.unsqueeze(0)
    
    logger.info(f"Extracted patch at ({center_x}, {center_y})")
    
    return image_tensor, spot_expression, f"spot_{args.spot_idx}", patch_rgb


# ==================== GradCAM Analysis ====================
def run_single_gradcam(args, logger):
    """Single-spot GradCAM analysis"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = HIPT_CLIP_Model().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded.")
    
    # Load data
    if args.image_path:
        image_tensor, spot_expression, barcode, original_image = load_data_from_image(args, logger)
    else:
        image_tensor, spot_expression, barcode, original_image = load_data_from_dataset(args, logger)
    
    # Create GradCAM
    gradcam = GradCAM_CarHE(
        model,
        target_block_index=args.target_block,
        multi_layer=args.multi_layer,
    )
    
    # Compute GradCAM
    logger.info("Computing GradCAM...")
    
    if args.marker_genes:
        # Gene-level GradCAM
        gene_list = [g.strip().upper() for g in args.marker_genes.split(',')]
        logger.info(f"Target genes: {gene_list}")
        
        # Load gene name list
        gene_names = None
        if args.gene_names_file:
            gene_names = pd.read_csv(args.gene_names_file).values.flatten().tolist()
        elif args.ref_adata:
            adata_ref = sc.read_h5ad(args.ref_adata)
            gene_names = adata_ref.var_names.tolist()
        
        if gene_names:
            gene_list_upper = [g.upper() for g in gene_names]
            gene_indices = [gene_list_upper.index(g) for g in gene_list if g in gene_list_upper]
            logger.info(f"Found {len(gene_indices)}/{len(gene_list)} genes")
        else:
            gene_indices = list(range(min(len(gene_list), CFG.spot_embedding)))
        
        # Compute for each gene
        gene_results = []
        for i, gidx in enumerate(gene_indices):
            gname = gene_list[i] if i < len(gene_list) else f"gene_{gidx}"
            logger.info(f"  Gene: {gname} (index {gidx})")
            heatmap, score = gradcam.compute(image_tensor, spot_expression, gene_indices=[gidx])
            gene_results.append((gname, heatmap, score))
            
            # Save single gene result
            save_path = os.path.join(args.output, f"gradcam_{barcode}_{gname}.png")
            if original_image is not None:
                visual_img = original_image
            else:
                visual_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                visual_img = ((visual_img * 0.5 + 0.5) * 255).astype(np.uint8)
            save_heatmap_as_image(heatmap, visual_img, save_path)
        
        # Multi-gene comparison figure
        if original_image is not None:
            vis_img = original_image
        else:
            vis_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            vis_img = ((vis_img * 0.5 + 0.5) * 255).astype(np.uint8)
        
        multi_path = os.path.join(args.output, f"gradcam_{barcode}_multi_gene.png")
        visualize_multi_gene_heatmap(vis_img, gene_results, save_path=multi_path)
        logger.info(f"Multi-gene comparison saved: {multi_path}")
    else:
        # Spot-level GradCAM
        heatmap, score = gradcam.compute(image_tensor, spot_expression)
        logger.info(f"GradCAM score: {score:.4f}")
        
        if original_image is not None:
            vis_img = original_image
        else:
            vis_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            vis_img = ((vis_img * 0.5 + 0.5) * 255).astype(np.uint8)
        
        save_path = os.path.join(args.output, f"gradcam_{barcode}.png")
        visualize_heatmap(
            heatmap, vis_img,
            save_path=save_path,
            title=f"GradCAM - {barcode}",
            score=score,
        )
        logger.info(f"Saved: {save_path}")
    
    logger.info("Done!")


# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarHE GradCAM Analysis")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    
    # Data source
    parser.add_argument("--adata", type=str, default=CFG.default_adata_path,
                        help="AnnData h5ad 文件路径")
    
    # Direct image mode
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to H&E image (direct image mode)")
    parser.add_argument("--centers_path", type=str, default=None,
                        help="Path to valid coordinate file (direct image mode)")
    parser.add_argument("--spatial_csv", type=str, default=None,
                        help="Spatial coordinate CSV (direct image mode, alternative)")
    parser.add_argument("--ref_adata", type=str, default=None,
                        help="Reference gene expression h5ad (direct image mode)")
    parser.add_argument("--spot_col", type=str, default=None,
                        help="Column name in reference adata identifying the sample")
    parser.add_argument("--spot_id", type=str, default=None,
                        help="Sample ID in reference adata")
    
    # GradCAM parameters
    parser.add_argument("--spot_idx", type=int, default=0,
                        help="Target spot index")
    parser.add_argument("--target_block", type=int, default=-1,
                        help="Target ViT block index (-1 = last)")
    parser.add_argument("--multi_layer", action="store_true",
                        help="Use multi-layer GradCAM")
    
    # Gene parameters
    parser.add_argument("--marker_genes", type=str, default=None,
                        help="Target gene list (comma-separated, e.g. ERBB2,ESR1,PGR)")
    parser.add_argument("--gene_names_file", type=str, default=None,
                        help="Path to gene name CSV file")
    
    # Output parameters
    parser.add_argument("--output", type=str, default="./gradcam_output",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.adata and not args.image_path:
        parser.error("Need --adata or --image_path")
    
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(args.output)
    logger.info(f"Arguments: {args}")
    
    run_single_gradcam(args, logger)
