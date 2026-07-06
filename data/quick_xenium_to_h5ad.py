#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Xenium 原始数据 → h5ad 快速转换（跳过 HoverNet 等复杂步骤）

直接使用 Xenium 的 nucleus_boundaries 和 cell_feature_matrix 生成训练用的 h5ad。
"""

import os
import sys
import numpy as np
import pandas as pd
import tifffile
import anndata as ad
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "Xenium_Prime_Human_Prostate_FFPE_outs")
HE_IMAGE = os.path.join(os.path.dirname(__file__), "Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif")
ALIGNMENT = os.path.join(os.path.dirname(__file__), "Xenium_Prime_Human_Prostate_FFPE_he_imagealignment.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "xenium_prostate.h5ad")

print("=" * 60)
print("Xenium 原始数据 → h5ad 快速转换")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 1: 坐标对齐 (Xenium 微米 → H&E 像素)
# ---------------------------------------------------------------------------
print("\n[1/4] 坐标对齐...")
M = pd.read_csv(ALIGNMENT, header=None).values
M_inv = np.linalg.inv(M)

# 读取 nucleus_boundaries: cell_id, vertex_x(μm), vertex_y(μm)
fp_bounds = os.path.join(DATA_DIR, "nucleus_boundaries.csv.gz")
print(f"  读取: {fp_bounds}")
bounds = pd.read_csv(fp_bounds, compression='gzip')
print(f"  总顶点数: {len(bounds)}, 细胞数: {bounds['cell_id'].nunique()}")

# 为每个细胞计算质心（微米坐标）
print("  计算细胞质心...")
centroids_um = bounds.groupby("cell_id")[["vertex_x", "vertex_y"]].mean()
cell_ids = centroids_um.index.tolist()

# 微米 → Xenium 像素 → H&E 像素
x_um = centroids_um["vertex_x"].values
y_um = centroids_um["vertex_y"].values
x_xe = x_um / 0.2125
y_xe = y_um / 0.2125
ones = np.ones(len(x_um))
he_coords = (M_inv @ np.stack([x_xe, y_xe, ones], axis=0)).T
he_x = he_coords[:, 0].astype(np.float32)
he_y = he_coords[:, 1].astype(np.float32)
print(f"  坐标范围: x=[{he_x.min():.0f}, {he_x.max():.0f}], y=[{he_y.min():.0f}, {he_y.max():.0f}]")

# ---------------------------------------------------------------------------
# Step 2: 基因表达矩阵
# ---------------------------------------------------------------------------
print("\n[2/4] 提取基因表达矩阵...")

import h5py
fp_matrix = os.path.join(DATA_DIR, "cell_feature_matrix.h5")
print(f"  读取: {fp_matrix}")

# 读取 Xenium cell_feature_matrix.h5 (10X format)
with h5py.File(fp_matrix, "r") as f:
    # 基因名称
    features = f["matrix/features/name"][:]
    feature_types = f["matrix/features/feature_type"][:]
    
    # 筛选 Gene Expression
    gene_mask = np.array([ft.decode() == "Gene Expression" for ft in feature_types])
    gene_names = [gn.decode() for i, gn in enumerate(features) if gene_mask[i]]
    print(f"  总基因数: {len(gene_names)}")
    
    # 过滤负对照探针
    to_filter = ["NegControlProbe_", "antisense_", "NegControlCodeword_", "BLANK_", "Blank-", "NegPrb"]
    gene_names = [g for g in gene_names if not any(s in g for s in to_filter)]
    print(f"  过滤后基因数: {len(gene_names)}")
    
    # 细胞条码
    barcodes = [b.decode() for b in f["matrix/barcodes"][:]]
    print(f"  细胞数: {len(barcodes)}")
    
    # 稀疏矩阵 → 密集
    from scipy.sparse import csc_matrix
    data = f["matrix/data"][:]
    indices = f["matrix/indices"][:]
    indptr = f["matrix/indptr"][:]
    shape = f["matrix/shape"][:]
    
    mat = csc_matrix((data, indices, indptr), shape=shape)
    mat_dense = mat.toarray().T  # cells × genes
    print(f"  矩阵形状: {mat_dense.shape}")

# 只保留 Gene Expression 列
gene_cols = np.where(gene_mask)[0]
mat_dense = mat_dense[:, gene_cols]

# ---------------------------------------------------------------------------
# Step 2b: 将 Xenium cell_id 映射到 H&E 坐标
# ---------------------------------------------------------------------------
# nucleus_boundaries 中是 'aaaamcnn-1' 格式，cell_feature_matrix 中也是
# 取交集
boundary_cells = set(cell_ids)
matrix_cells = set(barcodes)
common_cells = sorted(boundary_cells & matrix_cells)
print(f"\n  坐标+表达都有的细胞: {len(common_cells)}")

# 创建映射
cell_to_he = dict(zip(cell_ids, zip(he_x, he_y)))
cell_to_expr = {b: mat_dense[i] for i, b in enumerate(barcodes)}

# 对齐
valid_cells = [c for c in common_cells if c in cell_to_he and c in cell_to_expr]
expr_list = [cell_to_expr[c] for c in valid_cells]
coords_list = [cell_to_he[c] for c in valid_cells]

X = np.array(expr_list, dtype=np.float32)
spatial = np.array(coords_list, dtype=np.float32)

# 过滤全零细胞
nonzero_mask = X.sum(axis=1) > 0
X = X[nonzero_mask]
spatial = spatial[nonzero_mask]
valid_cells = [c for c, m in zip(valid_cells, nonzero_mask) if m]
print(f"  有效细胞(表达>0): {len(valid_cells)}")

# ---------------------------------------------------------------------------
# Step 3: 创建 AnnData
# ---------------------------------------------------------------------------
print("\n[3/4] 创建 AnnData...")

adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame({
        "sample_id": ["xenium_prostate"] * len(valid_cells),
        "image_path": [os.path.abspath(HE_IMAGE)] * len(valid_cells),
    }, index=valid_cells),
    var=pd.DataFrame(index=gene_names),
)
adata.obsm["spatial"] = spatial
adata.uns["dataset_type"] = "xenium"
adata.uns["conversion_date"] = pd.Timestamp.now().isoformat()

# ---------------------------------------------------------------------------
# Step 4: 保存
# ---------------------------------------------------------------------------
print(f"\n[4/4] 保存 {OUTPUT}...")
adata.write_h5ad(OUTPUT)
print(f"  ✓ 完成!")
print(f"  细胞数: {adata.n_obs}")
print(f"  基因数: {adata.n_vars}")
print(f"  坐标范围: x=[{spatial[:,0].min():.0f}, {spatial[:,0].max():.0f}], y=[{spatial[:,1].min():.0f}, {spatial[:,1].max():.0f}]")
print("=" * 60)
