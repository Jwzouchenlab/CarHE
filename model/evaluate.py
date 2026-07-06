# -*- coding: utf-8 -*-
"""CarHE System Evaluation Script
Comprehensive quantitative evaluation of trained models, including:
  1. Gene-level Pearson / Spearman correlation
  2. Spot-level PCC, MSE, MAE
  3. Embedding space cosine similarity (diagonal vs off-diagonal)
  4. Top-K retrieval accuracy
  5. SpaGCN spatial domain ARI (requires ground truth)
  6. Multi-gene spatial visualization comparison
  7. Auto-generated results summary

Usage:
    # Basic evaluation
    python evaluate.py --checkpoint ./checkpoint/model.pt --dataset BRCA
    
    # With SpaGCN domain evaluation
    python evaluate.py --checkpoint ./checkpoint/model.pt --dataset BRCA --spagcn
    
    # Specify output directory and marker genes
    python evaluate.py --checkpoint ./checkpoint/model.pt --dataset adata \
        --adata data.h5ad --output ./eval_results --marker_genes ERBB2,ESR1,MKI67
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import cv2
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from config import CFG
from model import HIPT_CLIP_Model
from utils import AvgMeter


# ==================== Logger ====================
def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("CarHE_Eval")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(save_dir, "evaluation_log.txt"), mode='w')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ==================== Evaluation Metrics Computation ====================
def compute_gene_level_pcc(
    predicted: np.ndarray, ground_truth: np.ndarray, gene_names: Optional[List[str]] = None
) -> Dict:
    """Compute per-gene Pearson correlation coefficient
    
    Args:
        predicted: [N_spots, N_genes]
        ground_truth: [N_spots, N_genes]
        gene_names: Gene name list
    Returns:
        dict with per_gene_pcc, mean_pcc, median_pcc, std_pcc, top_genes, bottom_genes
    """
    n_genes = predicted.shape[1]
    gene_pccs = np.full(n_genes, np.nan)
    
    for g in range(n_genes):
        pred_g = predicted[:, g]
        true_g = ground_truth[:, g]
        if np.std(pred_g) < 1e-8 or np.std(true_g) < 1e-8:
            continue
        try:
            r, _ = pearsonr(pred_g, true_g)
            if not np.isnan(r) and not np.isinf(r):
                gene_pccs[g] = r
        except Exception:
            continue
    
    valid = gene_pccs[~np.isnan(gene_pccs)]
    
    result = {
        'per_gene_pcc': gene_pccs,
        'mean_pcc': np.mean(valid),
        'median_pcc': np.median(valid),
        'std_pcc': np.std(valid),
        'n_valid_genes': len(valid),
        'fraction_sig': np.mean(valid > 0.1),  # Proportion with PCC > 0.1
        'fraction_strong': np.mean(valid > 0.5),  # Proportion with PCC > 0.5
    }
    
    # Best/worst genes
    if gene_names is not None and len(valid) > 0:
        sorted_idx = np.argsort(valid)[::-1]
        top_n = min(10, len(valid))
        result['top_genes'] = [(gene_names[int(sorted_idx[i])], float(valid[int(sorted_idx[i])])) for i in range(top_n)]
        result['bottom_genes'] = [(gene_names[int(sorted_idx[-i-1])], float(valid[int(sorted_idx[-i-1])])) for i in range(top_n)]
    
    return result


def compute_gene_level_spearman(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> Dict:
    """Compute gene-level Spearman correlation"""
    n_genes = predicted.shape[1]
    gene_spr = np.full(n_genes, np.nan)
    
    for g in range(n_genes):
        pred_g = predicted[:, g]
        true_g = ground_truth[:, g]
        if np.std(pred_g) < 1e-8 or np.std(true_g) < 1e-8:
            continue
        try:
            r, _ = spearmanr(pred_g, true_g)
            if not np.isnan(r) and not np.isinf(r):
                gene_spr[g] = r
        except Exception:
            continue
    
    valid = gene_spr[~np.isnan(gene_spr)]
    return {
        'per_gene_spearman': gene_spr,
        'mean_spearman': np.mean(valid),
        'median_spearman': np.median(valid),
        'std_spearman': np.std(valid),
    }


def compute_spot_level_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> Dict:
    """Compute spot-level metrics: PCC, MSE, MAE per spot"""
    n_spots = predicted.shape[0]
    
    spot_pccs = np.full(n_spots, np.nan)
    spot_mse = np.full(n_spots, np.nan)
    spot_mae = np.full(n_spots, np.nan)
    
    for s in range(n_spots):
        pred_s = predicted[s, :]
        true_s = ground_truth[s, :]
        if np.std(pred_s) < 1e-8 or np.std(true_s) < 1e-8:
            continue
        try:
            r, _ = pearsonr(pred_s, true_s)
            if not np.isnan(r) and not np.isinf(r):
                spot_pccs[s] = r
        except Exception:
            continue
        spot_mse[s] = mean_squared_error(true_s, pred_s)
        spot_mae[s] = mean_absolute_error(true_s, pred_s)
    
    valid_pcc = spot_pccs[~np.isnan(spot_pccs)]
    valid_mse = spot_mse[~np.isnan(spot_mse)]
    valid_mae = spot_mae[~np.isnan(spot_mae)]
    
    return {
        'per_spot_pcc': spot_pccs,
        'mean_spot_pcc': np.mean(valid_pcc),
        'median_spot_pcc': np.median(valid_pcc),
        'mean_mse': np.mean(valid_mse),
        'median_mse': np.median(valid_mse),
        'mean_mae': np.mean(valid_mae),
        'median_mae': np.median(valid_mae),
    }


def compute_cosine_similarity_metrics(
    image_embeddings: np.ndarray, spot_embeddings: np.ndarray
) -> Dict:
    """Compute embedding space cosine similarity metrics
    
    Args:
        image_embeddings: [N, D]
        spot_embeddings: [N, D]
    """
    # L2 normalization
    img_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
    spot_norm = spot_embeddings / (np.linalg.norm(spot_embeddings, axis=1, keepdims=True) + 1e-8)
    
    cos_sim = spot_norm @ img_norm.T  # [N, N]
    diag = np.diag(cos_sim)
    
    # Off-diagonal (between different spots)
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    off_diag = cos_sim[mask]
    
    return {
        'mean_diag_cosine': np.mean(diag),
        'std_diag_cosine': np.std(diag),
        'median_diag_cosine': np.median(diag),
        'mean_offdiag_cosine': np.mean(off_diag),
        'std_offdiag_cosine': np.std(off_diag),
        'diag_offdiag_ratio': np.mean(diag) / (np.mean(off_diag) + 1e-8),
        'per_spot_cosine': diag,
    }


def compute_retrieval_accuracy(
    image_embeddings: np.ndarray, spot_embeddings: np.ndarray, top_k_list: List[int] = [1, 5, 10]
) -> Dict:
    """Compute Top-K retrieval accuracy (spot->image and image->spot)"""
    img_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
    spot_norm = spot_embeddings / (np.linalg.norm(spot_embeddings, axis=1, keepdims=True) + 1e-8)
    
    N = img_norm.shape[0]
    cos_sim = spot_norm @ img_norm.T  # [N, N]
    
    results = {}
    for k in top_k_list:
        # spot -> image: For each spot, whether its correct image is in top-k
        topk_indices = np.argsort(-cos_sim, axis=1)[:, :k]  # [N, k]
        s2i_match = np.array([i in topk_indices[i] for i in range(N)])
        results[f'top{k}_spot2image'] = np.mean(s2i_match)
        
        # image -> spot
        topk_indices_t = np.argsort(-cos_sim.T, axis=1)[:, :k]
        i2s_match = np.array([i in topk_indices_t[i] for i in range(N)])
        results[f'top{k}_image2spot'] = np.mean(i2s_match)
    
    return results


# ==================== SpaGCN Domain Evaluation ====================
def evaluate_spagcn_domains(
    adata_pred: ad.AnnData,
    adata_true: ad.AnnData,
    n_clusters: int = 7,
    save_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Evaluate predicted expression performance on SpaGCN spatial domains"""
    try:
        import SpaGCN as spg
    except ImportError:
        if logger:
            logger.warning("SpaGCN not installed, skipping domain evaluation")
        return {'spagcn_ari': None, 'note': 'SpaGCN not installed'}
    
    results = {}
    
    # Run SpaGCN on predicted and true expression separately
    for name, adata_obj in [('pred', adata_pred.copy()), ('true', adata_true.copy())]:
        try:
            if 'spatial' not in adata_obj.obsm:
                if logger:
                    logger.warning(f"No spatial coordinates for {name}, skipping")
                continue
            
            coords = adata_obj.obsm['spatial']
            x_pixel = coords[:, 0].astype(int)
            y_pixel = coords[:, 1].astype(int)
            
            # Compute adjacency matrix
            effective_n = min(n_clusters, adata_obj.shape[0])
            adj = spg.calculate_adj_matrix(
                x=x_pixel, y=y_pixel, histology=False
            )
            
            # Search hyperparameters
            l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
            
            if adata_obj.X.max() > 20:
                adata_temp = adata_obj.copy()
                sc.pp.log1p(adata_temp)
            else:
                adata_temp = adata_obj
            
            res = spg.search_res(
                adata_temp, adj, l, effective_n,
                start=0.4, step=0.1, tol=5e-3, lr=0.05, max_epochs=200,
                r_seed=100, t_seed=100, n_seed=100
            )
            
            # Train SpaGCN
            clf = spg.SpaGCN()
            clf.set_l(l)
            if adata_obj.X.dtype != np.float32:
                adata_obj.X = adata_obj.X.astype(np.float32)
            clf.train(adata_obj, adj, init_spa=True, init="louvain",
                     res=res, tol=5e-3, lr=0.05, max_epochs=200)
            y_pred, _ = clf.predict()
            
            results[f'{name}_domains'] = y_pred
            
            # If both complete, compute ARI
            if 'pred_domains' in results and 'true_domains' in results:
                ari = adjusted_rand_score(results['true_domains'], results['pred_domains'])
                results['spagcn_ari'] = ari
                if logger:
                    logger.info(f"SpaGCN ARI: {ari:.4f}")
        except Exception as e:
            if logger:
                logger.warning(f"SpaGCN failed for {name}: {e}")
            continue
    
    return results


# ==================== Main Evaluation Function ====================
def run_full_evaluation(
    model: HIPT_CLIP_Model,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    gene_names: Optional[List[str]] = None,
    run_spagcn: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Run full evaluation pipeline"""
    if logger is None:
        logger = logging.getLogger("eval")
    
    model.eval()
    
    # Collect all embeddings and expressions
    all_img_embs, all_spot_embs = [], []
    all_true_expr, all_pred_expr = [], []
    all_coords, all_barcodes = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            images = batch['image'].to(device)
            spots = batch['reduced_expression'].to(device)
            
            img_emb = model.encode_image(images)
            spot_emb = model.encode_spot(spots)
            
            all_img_embs.append(img_emb.cpu().numpy())
            all_spot_embs.append(spot_emb.cpu().numpy())
            all_true_expr.append(spots.cpu().numpy())
            
            if 'spatial_coords' in batch:
                all_coords.append(batch['spatial_coords'].cpu().numpy())
            if 'barcode' in batch:
                all_barcodes.extend(list(batch['barcode']) if isinstance(batch['barcode'], (list, tuple)) else [batch['barcode']])
    
    img_embs = np.concatenate(all_img_embs, axis=0)
    spot_embs = np.concatenate(all_spot_embs, axis=0)
    true_expr = np.concatenate(all_true_expr, axis=0)
    has_coords = len(all_coords) > 0
    if has_coords:
        all_coords = np.concatenate(all_coords, axis=0)
    
    N = img_embs.shape[0]
    logger.info(f"Evaluating on {N} spots")
    
    # ====== 1. KNN predicted expression ======
    from inference import knn_predict
    logger.info("Computing KNN predictions (k=5)...")
    spot_embs_tensor = torch.tensor(spot_embs)
    img_embs_tensor = torch.tensor(img_embs)
    pred_expr = knn_predict(img_embs_tensor, spot_embs_tensor, true_expr, top_k=5)
    
    # ====== 2. Cosine similarity ======
    logger.info("Computing cosine similarity metrics...")
    cosine_metrics = compute_cosine_similarity_metrics(img_embs, spot_embs)
    
    # ====== 3. Retrieval accuracy ======
    logger.info("Computing retrieval accuracy...")
    retrieval_metrics = compute_retrieval_accuracy(img_embs, spot_embs)
    
    # ====== 4. Gene-level PCC ======
    logger.info("Computing gene-level Pearson correlation...")
    gene_pcc = compute_gene_level_pcc(pred_expr, true_expr, gene_names)
    
    # ====== 5. Gene-level Spearman ======
    logger.info("Computing gene-level Spearman correlation...")
    gene_spearman = compute_gene_level_spearman(pred_expr, true_expr)
    
    # ====== 6. Spot-level metrics ======
    logger.info("Computing spot-level metrics...")
    spot_metrics = compute_spot_level_metrics(pred_expr, true_expr)
    
    # ====== 7. SpaGCN domain evaluation ======
    spagcn_results = {}
    if run_spagcn and has_coords:
        logger.info("Running SpaGCN domain evaluation...")
        if gene_names is not None:
            var_names = gene_names[:pred_expr.shape[1]]
        else:
            var_names = [f"gene_{i}" for i in range(pred_expr.shape[1])]
        
        adata_pred = ad.AnnData(X=pred_expr, var=pd.DataFrame(index=var_names))
        adata_pred.obsm['spatial'] = all_coords
        adata_true = ad.AnnData(X=true_expr, var=pd.DataFrame(index=var_names))
        adata_true.obsm['spatial'] = all_coords
        
        spagcn_results = evaluate_spagcn_domains(
            adata_pred, adata_true, n_clusters=7, save_dir=save_dir, logger=logger
        )
    
    # ====== Summary ======
    all_metrics = {
        'n_spots': N,
        'n_genes': true_expr.shape[1],
        **cosine_metrics,
        **retrieval_metrics,
        **{f'gene_{k}': v for k, v in gene_pcc.items() if k != 'per_gene_pcc'},
        **{f'gene_{k}': v for k, v in gene_spearman.items() if k != 'per_gene_spearman'},
        **spot_metrics,
        **spagcn_results,
    }
    
    # ====== Print results ======
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"  N spots: {N}, N genes: {true_expr.shape[1]}")
    logger.info(f"  Mean Diag Cosine:     {cosine_metrics['mean_diag_cosine']:.4f} \u00b1 {cosine_metrics['std_diag_cosine']:.4f}")
    logger.info(f"  Mean Off-Diag Cosine: {cosine_metrics['mean_offdiag_cosine']:.4f}")
    logger.info(f"  Diag/Off-Diag Ratio:  {cosine_metrics['diag_offdiag_ratio']:.2f}x")
    logger.info(f"  Top-1 Spot2Image:     {retrieval_metrics['top1_spot2image']:.4f}")
    logger.info(f"  Top-5 Spot2Image:     {retrieval_metrics['top5_spot2image']:.4f}")
    logger.info(f"  Mean Gene PCC:        {gene_pcc['mean_pcc']:.4f} \u00b1 {gene_pcc['std_pcc']:.4f}")
    logger.info(f"  Median Gene PCC:      {gene_pcc['median_pcc']:.4f}")
    logger.info(f"  Fraction PCC>0.1:     {gene_pcc['fraction_sig']:.4f}")
    logger.info(f"  Fraction PCC>0.5:     {gene_pcc['fraction_strong']:.4f}")
    logger.info(f"  Mean Gene Spearman:   {gene_spearman['mean_spearman']:.4f}")
    logger.info(f"  Mean Spot PCC:        {spot_metrics['mean_spot_pcc']:.4f}")
    logger.info(f"  Mean MSE:             {spot_metrics['mean_mse']:.4f}")
    logger.info(f"  Mean MAE:             {spot_metrics['mean_mae']:.4f}")
    if run_spagcn and 'spagcn_ari' in spagcn_results and spagcn_results['spagcn_ari'] is not None:
        logger.info(f"  SpaGCN ARI:           {spagcn_results['spagcn_ari']:.4f}")
    if 'top_genes' in gene_pcc:
        logger.info(f"\n  Top-5 genes: {gene_pcc['top_genes'][:5]}")
    logger.info("="*60)
    
    # ====== Save ======
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics CSV
    flat_metrics = {}
    for k, v in all_metrics.items():
        if isinstance(v, (int, float, str, bool)) and v is not None:
            flat_metrics[k] = v
    pd.DataFrame([flat_metrics]).to_csv(
        os.path.join(save_dir, "evaluation_metrics.csv"), index=False
    )
    
    # Save raw data
    np.save(os.path.join(save_dir, "image_embeddings.npy"), img_embs)
    np.save(os.path.join(save_dir, "spot_embeddings.npy"), spot_embs)
    np.save(os.path.join(save_dir, "predicted_expression.npy"), pred_expr)
    np.save(os.path.join(save_dir, "true_expression.npy"), true_expr)
    
    # Save gene-level PCC
    if hasattr(all_metrics.get('per_gene_pcc', None), '__len__'):
        pcc_df = pd.DataFrame({'gene_pcc': all_metrics['per_gene_pcc']})
        if gene_names is not None:
            pcc_df.index = gene_names[:len(pcc_df)]
        pcc_df.to_csv(os.path.join(save_dir, "gene_level_pcc.csv"))
    
    # Plot results
    plot_evaluation_results(all_metrics, save_dir, gene_names)
    
    logger.info(f"All results saved to {save_dir}")
    return all_metrics


# ==================== Visualization ====================
def plot_evaluation_results(
    metrics: Dict, save_dir: str, gene_names: Optional[List[str]] = None
):
    """Plot evaluation result charts"""
    
    # 1. Gene-level PCC histogram
    if 'per_gene_pcc' in metrics:
        pcc_vals = metrics['per_gene_pcc']
        valid_pcc = pcc_vals[~np.isnan(pcc_vals)]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(valid_pcc, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(np.mean(valid_pcc), color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {np.mean(valid_pcc):.4f}')
        axes[0].axvline(np.median(valid_pcc), color='orange', linestyle='--', linewidth=2,
                       label=f'Median = {np.median(valid_pcc):.4f}')
        axes[0].set_xlabel('Pearson Correlation')
        axes[0].set_ylabel('Number of Genes')
        axes[0].set_title('Gene-level PCC Distribution')
        axes[0].legend()
        
        # 2. Cosine similarity distribution
        if 'per_spot_cosine' in metrics:
            axes[1].hist(metrics['per_spot_cosine'], bins=50, color='coral', edgecolor='white', alpha=0.8)
            axes[1].axvline(metrics['mean_diag_cosine'], color='red', linestyle='--',
                           label=f'Mean = {metrics["mean_diag_cosine"]:.4f}')
            axes[1].set_xlabel('Cosine Similarity')
            axes[1].set_ylabel('Number of Spots')
            axes[1].set_title('Diagonal Cosine Similarity')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "evaluation_distributions.png"), dpi=150)
        plt.close()
    
    # 2. Top-20 gene barplot
    if 'per_gene_pcc' in metrics and gene_names is not None:
        pcc_vals = metrics['per_gene_pcc']
        valid_mask = ~np.isnan(pcc_vals)
        valid_pcc = pcc_vals[valid_mask]
        valid_names = np.array(gene_names[:len(pcc_vals)])[valid_mask]
        
        if len(valid_pcc) >= 20:
            sorted_idx = np.argsort(valid_pcc)[::-1]
            top_idx = sorted_idx[:20]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.RdYlGn(valid_pcc[top_idx])
            ax.barh(range(20), valid_pcc[top_idx][::-1], color=colors[::-1])
            ax.set_yticks(range(20))
            ax.set_yticklabels(valid_names[top_idx][::-1], fontsize=9)
            ax.set_xlabel('Pearson Correlation')
            ax.set_title('Top 20 Genes by PCC')
            ax.axvline(0, color='black', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "top20_genes_pcc.png"), dpi=150)
            plt.close()
    
    # 3. Metrics summary table
    summary_data = {
        'Metric': [
            'Mean Diag Cosine', 'Diag/Off-Diag Ratio',
            'Top-1 S2I', 'Top-5 S2I', 'Top-1 I2S', 'Top-5 I2S',
            'Mean Gene PCC', 'Median Gene PCC', 'Frac PCC>0.1',
            'Mean Gene Spearman', 'Mean Spot PCC', 'Mean MSE', 'Mean MAE'
        ],
        'Value': [
            metrics.get('mean_diag_cosine', 0),
            metrics.get('diag_offdiag_ratio', 0),
            metrics.get('top1_spot2image', 0),
            metrics.get('top5_spot2image', 0),
            metrics.get('top1_image2spot', 0),
            metrics.get('top5_image2spot', 0),
            metrics.get('gene_mean_pcc', metrics.get('mean_pcc', 0)),
            metrics.get('gene_median_pcc', metrics.get('median_pcc', 0)),
            metrics.get('gene_fraction_sig', metrics.get('fraction_sig', 0)),
            metrics.get('gene_mean_spearman', metrics.get('mean_spearman', 0)),
            metrics.get('mean_spot_pcc', 0),
            metrics.get('mean_mse', 0),
            metrics.get('mean_mae', 0),
        ]
    }
    df = pd.DataFrame(summary_data)
    df['Value'] = df['Value'].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Evaluation Summary', fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, "evaluation_summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarHE: Comprehensive Evaluation")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--adata", type=str, default=CFG.default_adata_path,
                        help="AnnData h5ad 文件路径")
    
    # --- 已废弃: --dataset 不再使用，统一用 --adata ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="./evaluation_results")
    parser.add_argument("--spagcn", action="store_true",
                        help="Run SpaGCN domain evaluation")
    parser.add_argument("--gene_names_file", type=str, default=None,
                        help="Gene names CSV file")
    
    args = parser.parse_args()
    
    logger = setup_logger(args.output)
    logger.info(f"Arguments: {args}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    logger.info(f"Loading model: {args.checkpoint}")
    model = HIPT_CLIP_Model().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded.")
    
    # Load data
    from get_adata import build_loaders_adata
    
    logger.info(f"Loading data from {args.adata}...")
    adata = sc.read_h5ad(args.adata)
    _, test_loader = build_loaders_adata(adata=adata, batch_size=args.batch_size)
    
    # Load gene names
    gene_names = adata.var_names.tolist() if adata.var_names is not None else None
    if args.gene_names_file:
        gene_names = pd.read_csv(args.gene_names_file).values.flatten().tolist()
    
    # Run evaluation
    run_full_evaluation(
        model, test_loader, device, args.output,
        gene_names=gene_names,
        run_spagcn=args.spagcn,
        logger=logger,
    )
