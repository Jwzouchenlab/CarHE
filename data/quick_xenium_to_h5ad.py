#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Xenium Raw Data → h5ad Quick Conversion (skips complex steps like HoverNet)

Directly uses Xenium nucleus_boundaries and cell_feature_matrix to generate 
training-ready h5ad.
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
print("Xenium Raw Data → h5ad Quick Conversion")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 1: Coordinate alignment (Xenium microns → H&E pixels)
# ---------------------------------------------------------------------------
print("\n[1/4] Coordinate alignment...")
M = pd.read_csv(ALIGNMENT, header=None).values
M_inv = np.linalg.inv(M)

# Read nucleus_boundaries: cell_id, vertex_x(μm), vertex_y(μm)
fp_bounds = os.path.join(DATA_DIR, "nucleus_boundaries.csv.gz")
print(f"  Reading: {fp_bounds}")
bounds = pd.read_csv(fp_bounds, compression='gzip')
print(f"  Total vertices: {len(bounds)}, Cells: {bounds['cell_id'].nunique()}")

# Compute centroids per cell (in micron coordinates)
print("  Computing cell centroids...")
centroids_um = bounds.groupby("cell_id")[["vertex_x", "vertex_y"]].mean()
cell_ids = centroids_um.index.tolist()

# Microns → Xenium pixels → H&E pixels
x_um = centroids_um["vertex_x"].values
y_um = centroids_um["vertex_y"].values
x_xe = x_um / 0.2125
y_xe = y_um / 0.2125
ones = np.ones(len(x_um))
he_coords = (M_inv @ np.stack([x_xe, y_xe, ones], axis=0)).T
he_x = he_coords[:, 0].astype(np.float32)
he_y = he_coords[:, 1].astype(np.float32)
print(f"  Coordinate range: x=[{he_x.min():.0f}, {he_x.max():.0f}], y=[{he_y.min():.0f}, {he_y.max():.0f}]")

# ---------------------------------------------------------------------------
# Step 2: Gene expression matrix
# ---------------------------------------------------------------------------
print("\n[2/4] Extracting gene expression matrix...")

import h5py
fp_matrix = os.path.join(DATA_DIR, "cell_feature_matrix.h5")
print(f"  Reading: {fp_matrix}")

# Read Xenium cell_feature_matrix.h5 (10X format)
with h5py.File(fp_matrix, "r") as f:
    # Gene names
    features = f["matrix/features/name"][:]
    feature_types = f["matrix/features/feature_type"][:]
    
    # Filter Gene Expression
    gene_mask = np.array([ft.decode() == "Gene Expression" for ft in feature_types])
    gene_names = [gn.decode() for i, gn in enumerate(features) if gene_mask[i]]
    print(f"  Total genes: {len(gene_names)}")
    
    # Filter negative control probes
    to_filter = ["NegControlProbe_", "antisense_", "NegControlCodeword_", "BLANK_", "Blank-", "NegPrb"]
    gene_names = [g for g in gene_names if not any(s in g for s in to_filter)]
    print(f"  Genes after filtering: {len(gene_names)}")
    
    # Cell barcodes
    barcodes = [b.decode() for b in f["matrix/barcodes"][:]]
    print(f"  Cells: {len(barcodes)}")
    
    # Sparse matrix → dense
    from scipy.sparse import csc_matrix
    data = f["matrix/data"][:]
    indices = f["matrix/indices"][:]
    indptr = f["matrix/indptr"][:]
    shape = f["matrix/shape"][:]
    
    mat = csc_matrix((data, indices, indptr), shape=shape)
    mat_dense = mat.toarray().T  # cells × genes
    print(f"  Matrix shape: {mat_dense.shape}")

# Keep only Gene Expression columns
gene_cols = np.where(gene_mask)[0]
mat_dense = mat_dense[:, gene_cols]

# ---------------------------------------------------------------------------
# Step 2b: Map Xenium cell_id to H&E coordinates
# ---------------------------------------------------------------------------
# nucleus_boundaries uses 'aaaamcnn-1' format, cell_feature_matrix uses the same
# Take the intersection
boundary_cells = set(cell_ids)
matrix_cells = set(barcodes)
common_cells = sorted(boundary_cells & matrix_cells)
print(f"\n  Cells with both coordinates and expression: {len(common_cells)}")

# Create mappings
cell_to_he = dict(zip(cell_ids, zip(he_x, he_y)))
cell_to_expr = {b: mat_dense[i] for i, b in enumerate(barcodes)}

# Align
valid_cells = [c for c in common_cells if c in cell_to_he and c in cell_to_expr]
expr_list = [cell_to_expr[c] for c in valid_cells]
coords_list = [cell_to_he[c] for c in valid_cells]

X = np.array(expr_list, dtype=np.float32)
spatial = np.array(coords_list, dtype=np.float32)

# Filter zero-expression cells
nonzero_mask = X.sum(axis=1) > 0
X = X[nonzero_mask]
spatial = spatial[nonzero_mask]
valid_cells = [c for c, m in zip(valid_cells, nonzero_mask) if m]
print(f"  Valid cells (expression > 0): {len(valid_cells)}")

# ---------------------------------------------------------------------------
# Step 3: Create AnnData
# ---------------------------------------------------------------------------
print("\n[3/4] Creating AnnData...")

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
# Step 4: Save
# ---------------------------------------------------------------------------
print(f"\n[4/4] Saving {OUTPUT}...")
adata.write_h5ad(OUTPUT)
print(f"  ✓ Done!")
print(f"  Cells: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")
print(f"  Coordinate range: x=[{spatial[:,0].min():.0f}, {spatial[:,0].max():.0f}], y=[{spatial[:,1].min():.0f}, {spatial[:,1].max():.0f}]")
print("=" * 60)
