# -*- coding: utf-8 -*-
"""CarHE Inference Script
统一使用 AnnData (h5ad) 格式。所有数据必须先转换为 h5ad。

Usage:
    # 评估模式
    python inference.py --mode eval --adata ../data/xenium_prostate.h5ad --checkpoint ./checkpoint/model.pt
    
    # 预测模式 (KNN)
    python inference.py --mode predict --image_path image.tif --centers_path centers.txt \
        --ref_adata ref.h5ad --checkpoint ./checkpoint/model.pt

    # GradCAM 模式
    python inference.py --mode gradcam --adata ../data/xenium_prostate.h5ad \
        --checkpoint ./checkpoint/model.pt --spot_idx 0
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import logging

from model import HIPT_CLIP_Model
from config import CFG
from utils import AvgMeter

# Import GradCAM modules
try:
    from gradcam import (
        GradCAM_CarHE,
        visualize_heatmap,
        visualize_multi_gene_heatmap,
        save_heatmap_as_image,
    )
    HAS_GRADCAM_AVAILABLE = True
except ImportError:
    HAS_GRADCAM_AVAILABLE = False
    print("Warning: GradCAM module not available")


# ==================== Logging ====================
def setup_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger("CarHE_Inference")
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# ==================== Dataset for Image-only Inference ====================
class ImageInferenceDataset(torch.utils.data.Dataset):
    """Create inference dataset from H&E image and coordinate files"""
    def __init__(self, image_path, centers_txt_path, patch_size=256):
        self.patch_size = patch_size
        # Loading image
        self.whole_image = cv2.imread(image_path)
        if self.whole_image is None:
            # Trying tifffile fallback
            import tifffile
            self.whole_image = tifffile.imread(image_path)
            if self.whole_image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            if self.whole_image.ndim == 2:
                self.whole_image = cv2.cvtColor(self.whole_image, cv2.COLOR_GRAY2RGB)
        
        # Loading valid center coordinates
        centers = pd.read_csv(centers_txt_path, sep=',', header=None)
        if centers.shape[1] >= 2:
            self.centers = centers.iloc[:, :2].values.astype(int)
        else:
            raise ValueError(f"Centers file needs at least 2 columns, got {centers.shape[1]}")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.centers)
    
    def __getitem__(self, idx):
        center_y, center_x = self.centers[idx]
        h, w = self.whole_image.shape[:2]
        half = self.patch_size // 2
        
        y_start = max(0, center_y - half)
        y_end = min(h, center_y + half)
        x_start = max(0, center_x - half)
        x_end = min(w, center_x + half)
        
        patch = self.whole_image[y_start:y_end, x_start:x_end]
        
        # Padding
        pad_top = max(0, half - center_y)
        pad_bottom = max(0, center_y + half - h)
        pad_left = max(0, half - center_x)
        pad_right = max(0, center_x + half - w)
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        
        # BGR -> RGB -> Tensor
        if patch.ndim == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_pil = Image.fromarray(patch)
        image_tensor = self.transform(patch_pil)
        
        return {
            'image': image_tensor,
            'idx': idx,
            'center': torch.tensor([center_x, center_y], dtype=torch.float32)
        }


# ==================== Encoding Functions ====================
def get_image_embeddings(model, dataloader, device):
    """Extract image embeddings"""
    model.eval()
    model.to(device)
    all_embeddings = []
    all_centers = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding images"):
            images = batch['image'].to(device)
            embeddings = model.encode_image(images)
            all_embeddings.append(embeddings.cpu())
            all_centers.append(batch['center'])
    if not all_embeddings:
        return torch.empty(0), torch.empty(0)
    return torch.cat(all_embeddings), torch.cat(all_centers)


def get_spot_embeddings(model, expression_data, device, batch_size=128):
    """Extract expression embeddings from reference expression matrix"""
    model.eval()
    model.to(device)
    all_embeddings = []
    
    # Processing expression data
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    expression_tensor = torch.tensor(expression_data, dtype=torch.float32)
    
    ds = torch.utils.data.TensorDataset(expression_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (batch,) in tqdm(dl, desc="Encoding reference spots"):
            batch = batch.to(device)
            embeddings = model.encode_spot(batch)
            all_embeddings.append(embeddings.cpu())
    if not all_embeddings:
        return torch.empty(0)
    return torch.cat(all_embeddings)


# ==================== KNN Prediction ====================
def knn_predict(query_embeddings, ref_embeddings, ref_expression, top_k=5):
    """Predict gene expression from reference using KNN"""
    n_query = query_embeddings.shape[0]
    n_ref = ref_embeddings.shape[0]
    
    if n_query == 0 or n_ref == 0:
        return np.empty((n_query, ref_expression.shape[1]))
    
    effective_k = min(top_k, n_ref)
    
    # Computing distances
    distances = torch.cdist(query_embeddings, ref_embeddings)
    _, indices = torch.topk(distances, k=effective_k, dim=1, largest=False)
    indices = indices.numpy()
    
    # Processing expression matrix
    if hasattr(ref_expression, 'toarray'):
        ref_expr = ref_expression.toarray()
    else:
        ref_expr = np.array(ref_expression)
    
    n_genes = ref_expr.shape[1]
    predicted = np.zeros((n_query, n_genes), dtype=np.float32)
    
    for i in range(n_query):
        neighbor_expr = ref_expr[indices[i]]
        predicted[i] = np.mean(neighbor_expr, axis=0)
    
    return predicted


# ==================== Pearson Correlation ====================
def compute_pearson_correlation(pred, true):
    """Compute Pearson correlation coefficient for each gene"""
    cors = []
    for i in range(pred.shape[1]):
        if np.std(pred[:, i]) < 1e-8 or np.std(true[:, i]) < 1e-8:
            cors.append(0)
        else:
            cor = np.corrcoef(pred[:, i], true[:, i])[0, 1]
            cors.append(cor if not np.isnan(cor) else 0)
    return np.array(cors)


# ==================== Evaluation Mode ====================
def run_evaluation(args, logger):
    """Evaluation mode: comprehensive evaluation on paired labeled data
    
    Uses run_full_evaluation from evaluate.py for:
    - Gene-level PCC / Spearman
    - Spot-level PCC / MSE / MAE
    - Cosine similarity (diagonal vs off-diagonal)
    - Top-K retrieval accuracy
    - SpaGCN spatial domain ARI (optional)
    """
    from get_adata import build_loaders_adata
    from evaluate import run_full_evaluation
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Loading model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = HIPT_CLIP_Model().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded.")
    
    # Loading data
    logger.info(f"Loading data from {args.adata}...")
    adata = sc.read_h5ad(args.adata)
    _, test_loader = build_loaders_adata(adata=adata, batch_size=args.batch_size)
    
    # Running comprehensive evaluation
    os.makedirs(args.output, exist_ok=True)
    gene_names = adata.var_names.tolist() if adata.var_names is not None else None
    
    metrics = run_full_evaluation(
        model, test_loader, device, args.output,
        gene_names=gene_names,
        run_spagcn=getattr(args, 'spagcn', False),
        logger=logger,
    )
    
    return metrics.get('mean_diag_cosine', 0.0)


# ==================== Prediction Mode ====================
def run_prediction(args, logger):
    """Prediction mode: predict gene expression for new images"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Loading model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = HIPT_CLIP_Model().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded.")
    
    # Loading reference data
    logger.info(f"Loading reference: {args.ref_adata}")
    adata_ref = sc.read_h5ad(args.ref_adata)
    ngenes = min(adata_ref.shape[1], CFG.spot_embedding)
    if adata_ref.shape[1] > ngenes:
        adata_ref = adata_ref[:, :ngenes].copy()
    gene_names = adata_ref.var_names.tolist()
    logger.info(f"Reference: {adata_ref.shape[0]} spots, {ngenes} genes")
    
    # Extracting reference embeddings
    ref_spot_embs = get_spot_embeddings(
        model, adata_ref.X, device, batch_size=args.batch_size
    )
    logger.info(f"Reference embeddings: {ref_spot_embs.shape}")
    
    # Loading target image
    logger.info(f"Loading target image: {args.image_path}")
    target_ds = ImageInferenceDataset(args.image_path, args.centers_path)
    target_dl = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logger.info(f"Target image: {len(target_ds)} patches")
    
    # Extracting target image embeddings
    target_img_embs, target_centers = get_image_embeddings(
        model, target_dl, device
    )
    logger.info(f"Target image embeddings: {target_img_embs.shape}")
    
    # KNN prediction
    logger.info(f"Running KNN prediction (k={args.top_k})...")
    predicted_expr = knn_predict(
        target_img_embs, ref_spot_embs, adata_ref.X, top_k=args.top_k
    )
    logger.info(f"Predicted expression: {predicted_expr.shape}")
    
    # Creating output AnnData
    barcodes = [f"spot_{i}" for i in range(predicted_expr.shape[0])]
    adata_pred = ad.AnnData(
        X=predicted_expr,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names)
    )
    adata_pred.obsm['spatial'] = target_centers.numpy()
    adata_pred.obs['x_coord'] = target_centers[:, 0].numpy()
    adata_pred.obs['y_coord'] = target_centers[:, 1].numpy()
    
    # Saving
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "predicted_expression.h5ad")
    adata_pred.write_h5ad(output_path)
    logger.info(f"Predicted expression saved to {output_path}")
    
    return adata_pred


# ==================== GradCAM Mode ====================
def run_gradcam_inference(args, logger):
    """Run GradCAM visualization
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    if not HAS_GRADCAM_AVAILABLE:
        logger.error("GradCAM module not available!")
        logger.error("Please ensure gradcam.py is in the same directory")
        sys.exit(1)
    
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
    if args.adata:
        from get_adata import build_loaders_adata
        logger.info(f"Loading data from {args.adata}...")
        adata = sc.read_h5ad(args.adata)
        _, test_loader = build_loaders_adata(adata=adata, batch_size=1)
        
        # Get sample
        data_iter = iter(test_loader)
        for i in range(args.spot_idx + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                logger.error(f"spot_idx {args.spot_idx} out of range")
                sys.exit(1)
        
        image_tensor = batch['image']
        spot_expression = batch['reduced_expression']
        barcode = batch.get('barcode', [f'spot_{args.spot_idx}'])[0] if 'barcode' in batch else f'spot_{args.spot_idx}'
        
        # Prepare visual image
        vis_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        vis_img = ((vis_img * 0.5 + 0.5) * 255).astype(np.uint8)
    else:
        # Load from image file
        logger.info(f"Loading image: {args.image_path}")
        import tifffile
        
        whole_image = cv2.imread(args.image_path)
        if whole_image is None:
            whole_image = tifffile.imread(args.image_path)
            if whole_image is None:
                raise ValueError(f"Cannot load image: {args.image_path}")
        
        if whole_image.ndim == 2:
            whole_image = cv2.cvtColor(whole_image, cv2.COLOR_GRAY2RGB)
        
        # Load coordinates
        centers = pd.read_csv(args.centers_path, sep=',', header=None)
        if args.spot_idx >= len(centers):
            raise ValueError(f"spot_idx {args.spot_idx} out of range")
        center_x, center_y = int(centers.iloc[args.spot_idx, 0]), int(centers.iloc[args.spot_idx, 1])
        
        # Load reference expression
        adata_ref = sc.read_h5ad(args.ref_adata)
        idx = args.spot_idx % adata_ref.shape[0]
        x = adata_ref[idx].X
        if hasattr(x, 'toarray'):
            x = x.toarray()
        spot_expression = torch.tensor(x.flatten()[:CFG.spot_embedding]).float().unsqueeze(0)
        
        # Extract image patch
        half = 128
        patch = whole_image[
            max(0, center_y-half):min(whole_image.shape[0], center_y+half),
            max(0, center_x-half):min(whole_image.shape[1], center_x+half)
        ]
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
        vis_img = patch_rgb
        barcode = f"spot_{args.spot_idx}"
    
    # Create GradCAM
    logger.info("Initializing GradCAM...")
    gradcam = GradCAM_CarHE(
        model,
        target_block_index=args.target_block,
        multi_layer=args.multi_layer,
    )
    
    # Compute and save results
    os.makedirs(args.output, exist_ok=True)
    
    if args.marker_genes:
        # Gene-level GradCAM
        gene_list = [g.strip() for g in args.marker_genes.split(',')]
        logger.info(f"Processing genes: {gene_list}")
        
        # Try to get gene names
        gene_indices = list(range(min(len(gene_list), CFG.spot_embedding)))
        gene_names = []
        if args.gene_names_file:
            try:
                gene_names = pd.read_csv(args.gene_names_file).values.flatten().tolist()
                gene_list_upper = [g.upper() for g in gene_names]
                found_indices = []
                for g in gene_list:
                    g_upper = g.upper()
                    if g_upper in gene_list_upper:
                        found_indices.append(gene_list_upper.index(g_upper))
                if found_indices:
                    gene_indices = found_indices
                    gene_list = [gene_names[i] for i in gene_indices]
            except:
                pass
        
        gene_results = []
        for i, gidx in enumerate(gene_indices):
            gname = gene_list[i] if i < len(gene_list) else f"gene_{gidx}"
            logger.info(f"Computing for gene: {gname}")
            heatmap, score = gradcam.compute(image_tensor, spot_expression, gene_indices=[gidx])
            gene_results.append((gname, heatmap, score))
            
            save_path = os.path.join(args.output, f"gradcam_{barcode}_{gname}.png")
            save_heatmap_as_image(heatmap, vis_img, save_path)
        
        multi_path = os.path.join(args.output, f"gradcam_{barcode}_multi_gene.png")
        visualize_multi_gene_heatmap(vis_img, gene_results, save_path=multi_path)
        logger.info(f"Multi-gene visualization saved: {multi_path}")
    else:
        # Spot-level GradCAM
        heatmap, score = gradcam.compute(image_tensor, spot_expression)
        logger.info(f"GradCAM score: {score:.4f}")
        
        save_path = os.path.join(args.output, f"gradcam_{barcode}.png")
        visualize_heatmap(
            heatmap, vis_img,
            save_path=save_path,
            title=f"GradCAM - {barcode}",
            score=score,
        )
        logger.info(f"Saved: {save_path}")
    
    logger.info("GradCAM completed!")


# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarHE: Inference, Evaluation & GradCAM")
    
    parser.add_argument("--mode", type=str, required=True, choices=['eval', 'predict', 'gradcam'],
                        help="Run mode: eval=evaluate, predict=predict, gradcam=visualization")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    
    # Evaluation mode arguments
    parser.add_argument("--adata", type=str, default=CFG.default_adata_path,
                        help="AnnData h5ad 文件路径 (eval/gradcam 模式)")
    
    # Prediction mode arguments
    parser.add_argument("--image_path", type=str, default=None,
                        help="Target H&E image path (predict/gradcam mode)")
    parser.add_argument("--centers_path", type=str, default=None,
                        help="Path to valid tile centers file (predict/gradcam mode)")
    parser.add_argument("--ref_adata", type=str, default=None,
                        help="Reference h5ad file path (predict/gradcam mode)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="K value for KNN prediction")
    
    # GradCAM arguments
    parser.add_argument("--spot_idx", type=int, default=0,
                        help="Target spot index for GradCAM")
    parser.add_argument("--target_block", type=int, default=-1,
                        help="Target ViT block for GradCAM (-1 = last)")
    parser.add_argument("--multi_layer", action="store_true",
                        help="Use multi-layer GradCAM")
    parser.add_argument("--marker_genes", type=str, default=None,
                        help="Comma-separated marker genes for gene-level GradCAM")
    parser.add_argument("--gene_names_file", type=str, default=None,
                        help="CSV file with gene names")
    
    # Common arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--spagcn", action="store_true",
                        help="Run SpaGCN evaluation (eval mode only)")
    parser.add_argument("--output", type=str, default="./inference_output",
                        help="Output directory")
    
    args = parser.parse_args()
    
    logger = setup_logger(os.path.join(args.output, "inference_log.txt"))
    logger.info(f"Arguments: {args}")
    
    if args.mode == 'eval':
        run_evaluation(args, logger)
    elif args.mode == 'predict':
        if args.image_path is None or args.centers_path is None or args.ref_adata is None:
            logger.error("Predict mode requires --image_path, --centers_path, and --ref_adata")
            sys.exit(1)
        run_prediction(args, logger)
    elif args.mode == 'gradcam':
        if not args.adata and not args.image_path:
            logger.error("GradCAM mode requires --adata or --image_path")
            sys.exit(1)
        run_gradcam_inference(args, logger)
