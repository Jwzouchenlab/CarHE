#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Format Conversion Tool: Any Format → AnnData (h5ad)
========================================================

Convert all datasets to a unified AnnData format, eliminating special-case branches in CarHE.

Supported input formats:
    1. Xenium preprocessed data (matched_nuclei + cell_gene_matrix + HE image)
    2. CSV format (spot_*.csv + intdata/*.csv for BRCA/DLPFC/CCRCC)
    3. 10X Visium spaceranger output
    4. Existing h5ad files (format validation)

Output: Standard AnnData (h5ad) file

Usage:
    # Xenium → h5ad
    python convert_to_h5ad.py --input xenium

    # CSV dataset → h5ad
    python convert_to_h5ad.py --input csv --csv_dir ../data/BRCA

    # Validate existing h5ad
    python convert_to_h5ad.py --input validate --adata path.h5ad

    # Auto-detect
    python convert_to_h5ad.py --input auto --path ./some_data/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tifffile
import cv2
from tqdm import tqdm

# ==================== AnnData Format Specification ====================
"""
Standard AnnData structure:
    adata.X              : [N_cells, N_genes] float32 gene expression matrix
    adata.obs:
        sample_id         : str  sample identifier
        image_path        : str  corresponding H&E full image path
        (barcode)         : str  (index) cell/spot barcode
    adata.obsm:
        spatial           : [N, 2] float32 spatial coordinates (pixel_x, pixel_y)
        X_scGPT           : [N, D] (optional) scGPT dimensionality reduction embedding
    adata.var:
        (index)           : str  gene name
    adata.uns:
        dataset_type      : str  data source type
        conversion_date   : str  conversion date
"""


# ==================== Xenium → AnnData ====================
def xenium_to_h5ad(
    he_image_path: str,
    matched_nuclei_csv: str,
    cell_gene_matrix_csv: str,
    seg_mask_path: str = None,
    output_path: str = "xenium_data.h5ad",
    scale_factor: float = None,
):
    """Convert Xenium preprocessed data to AnnData"""
    import anndata as ad

    print("=" * 60)
    print("Xenium → AnnData Conversion")
    print("=" * 60)

    # 1. Load matched nuclei
    print(f"Loading: {matched_nuclei_csv}")
    matched = pd.read_csv(matched_nuclei_csv)
    matched = matched[matched["id_histology"] > 0].reset_index(drop=True)

    # 2. Load gene expression matrix
    print(f"Loading: {cell_gene_matrix_csv}")
    expr = pd.read_csv(cell_gene_matrix_csv, index_col=0)

    # 3. Align
    expr_ids = set(expr.index.astype(int))
    matched_ids = set(matched["id_histology"])
    valid_ids = sorted(expr_ids & matched_ids)
    matched = matched[matched["id_histology"].isin(valid_ids)].reset_index(drop=True)
    expr = expr.loc[valid_ids]

    # 4. Get spatial coordinates
    if seg_mask_path and os.path.exists(seg_mask_path):
        print(f"Computing centroids from segmentation mask: {seg_mask_path}")
        centroids = _compute_centroids(seg_mask_path, valid_ids)
        coords = np.array([centroids.get(nid, (0, 0)) for nid in valid_ids], dtype=np.float32)
    else:
        print("No segmentation mask provided, coordinates set to (0,0)")
        coords = np.zeros((len(valid_ids), 2), dtype=np.float32)

    # 5. Estimate scale factor
    if scale_factor is None and os.path.exists(he_image_path):
        scale_factor = _estimate_scale(he_image_path, seg_mask_path)
    coords_he = coords * (scale_factor or 1.0)

    # 6. Filter invalid coordinates
    valid_mask = np.all(coords_he > 0, axis=1)
    matched = matched[valid_mask].reset_index(drop=True)
    expr = expr.iloc[np.where(valid_mask)[0]]
    coords_he = coords_he[valid_mask]

    # 7. Create AnnData
    adata = ad.AnnData(
        X=expr.values.astype(np.float32),
        obs=pd.DataFrame({
            "sample_id": ["xenium_sample"] * len(expr),
            "image_path": [os.path.abspath(he_image_path)] * len(expr),
        }, index=[f"cell_{i}" for i in range(len(expr))]),
        var=pd.DataFrame(index=expr.columns),
    )
    adata.obsm["spatial"] = coords_he
    adata.uns["dataset_type"] = "xenium"
    adata.uns["conversion_date"] = pd.Timestamp.now().isoformat()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    return adata


# ==================== CSV (BRCA/DLPFC) → AnnData ====================
def csv_to_h5ad(
    image_dir: str,
    sample_ids: list,
    spot_prefix: str = "spot_",
    intdata_dir: str = None,
    barcode_prefix: str = "barcode_",
    image_ext: str = ".jpg",
    output_path: str = "csv_data.h5ad",
    ngenes: int = 2000,
):
    """Convert CSV-format datasets to AnnData (BRCA/DLPFC/CCRCC common)

    Expected directory structure:
        image_dir/        standardized_{id}.jpg
        st_dir/           spot_{id}.csv        (barcode, pixel_x, pixel_y)
        intdata_dir/      {id}.csv             (reduced expression, features x cells)
        st_dir/           barcode_{id}.csv     (barcode list)

    Args:
        image_dir:   image directory
        sample_ids:  list of sample IDs
        spot_prefix: spot CSV filename prefix
        intdata_dir: expression matrix directory (default: same as spot directory)
        barcode_prefix: barcode CSV prefix
        image_ext:   image extension
        output_path: output h5ad path
        ngenes:      number of genes
    """
    import anndata as ad

    print("=" * 60)
    print(f"CSV → AnnData Conversion ({len(sample_ids)} samples)")
    print("=" * 60)

    if intdata_dir is None:
        intdata_dir = image_dir

    all_expr = []
    all_coords = []
    all_barcodes = []
    all_sample_ids = []
    all_image_paths = []

    for sid in sample_ids:
        img_path = os.path.join(image_dir, f"standardized_{sid}{image_ext}")
        spot_path = os.path.join(image_dir, f"{spot_prefix}{sid}.csv")
        intdata_path = os.path.join(intdata_dir, f"{sid}.csv")
        barcode_path = os.path.join(image_dir, f"{barcode_prefix}{sid}.csv")

        # Check files
        missing = []
        for name, path in [("image", img_path), ("spot", spot_path), ("expression", intdata_path)]:
            if not os.path.exists(path):
                missing.append(name)
        if missing:
            print(f"  [{sid}] Skipped: missing {', '.join(missing)}")
            continue

        # Load spot coordinates
        spot_df = pd.read_csv(spot_path)
        if "pixel_x" in spot_df.columns and "pixel_y" in spot_df.columns:
            coords = spot_df[["pixel_x", "pixel_y"]].values.astype(np.float32)
            barcodes = spot_df["barcode"].tolist() if "barcode" in spot_df.columns else [f"{sid}_{i}" for i in range(len(coords))]
        else:
            coords = spot_df.iloc[:, :2].values.astype(np.float32)
            barcodes = [f"{sid}_{i}" for i in range(len(coords))]

        # Load expression matrix
        expr_df = pd.read_csv(intdata_path, index_col=0)
        if expr_df.shape[1] > expr_df.shape[0]:
            expr_df = expr_df.T
        expr = expr_df.values[:, :ngenes].astype(np.float32)

        n = min(len(coords), len(expr))
        all_coords.append(coords[:n])
        all_expr.append(expr[:n])
        all_barcodes.extend(barcodes[:n])
        all_sample_ids.extend([sid] * n)
        all_image_paths.extend([os.path.abspath(img_path)] * n)
        print(f"  [{sid}] {n} spots")

    if not all_expr:
        raise RuntimeError("No samples were loaded successfully")

    X = np.concatenate(all_expr, axis=0)
    spatial = np.concatenate(all_coords, axis=0)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({
            "sample_id": all_sample_ids,
            "image_path": all_image_paths,
        }, index=all_barcodes),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])]),
    )
    adata.obsm["spatial"] = spatial
    adata.uns["dataset_type"] = "csv"
    adata.uns["conversion_date"] = pd.Timestamp.now().isoformat()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"  Total spots: {adata.n_obs}, Genes: {adata.n_vars}")
    return adata


# ==================== Visium spaceranger → AnnData ====================
def visium_to_h5ad(
    spaceranger_dir: str,
    output_path: str = "visium_data.h5ad",
    ngenes: int = 2000,
):
    """Convert 10X Visium spaceranger output to AnnData"""
    import scanpy as sc
    import anndata as ad

    print("=" * 60)
    print("Visium spaceranger → AnnData Conversion")
    print("=" * 60)

    h5_path = os.path.join(spaceranger_dir, "filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # Load spatial coordinates
    tissue_pos = os.path.join(spaceranger_dir, "spatial", "tissue_positions_list.csv")
    if os.path.exists(tissue_pos):
        pos = pd.read_csv(tissue_pos, header=None, index_col=0)
        pos = pos.loc[adata.obs_names]
        adata.obsm["spatial"] = pos[[4, 5]].values.astype(np.float32)  # pixel coords

    # Load image path
    hires_img = os.path.join(spaceranger_dir, "spatial", "tissue_hires_image.png")
    if os.path.exists(hires_img):
        img_path = os.path.abspath(hires_img)
    else:
        img_path = ""

    adata.obs["sample_id"] = os.path.basename(spaceranger_dir)
    adata.obs["image_path"] = img_path
    adata.uns["dataset_type"] = "visium"

    # Truncate gene count
    if adata.n_vars > ngenes:
        adata = adata[:, :ngenes].copy()

    adata.write_h5ad(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"  spots: {adata.n_obs}, genes: {adata.n_vars}")
    return adata


# ==================== Validation Tool ====================
def validate_h5ad(adata_path: str) -> bool:
    """Validate whether an h5ad file meets CarHE format requirements"""
    import scanpy as sc

    print("=" * 60)
    print(f"Validating AnnData: {adata_path}")
    print("=" * 60)

    adata = sc.read_h5ad(adata_path)
    ok = True

    checks = [
        (".X", lambda a: a.X is not None),
        ("obs['sample_id']", lambda a: "sample_id" in a.obs),
        ("obs['image_path']", lambda a: "image_path" in a.obs),
        ("obsm['spatial']", lambda a: "spatial" in a.obsm),
    ]
    for name, check in checks:
        if check(adata):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} missing")
            ok = False

    print(f"\n  Shape: {adata.shape}")
    print(f"  Number of sample_ids: {adata.obs['sample_id'].nunique()}")
    if ok:
        print("  Validation passed ✓")
    return ok


# ==================== Helper Functions ====================
def _compute_centroids(seg_path: str, target_ids: list) -> dict:
    """Compute centroids from segmentation mask"""
    seg = tifffile.imread(seg_path)
    centroids = {}
    for nid in tqdm(target_ids, desc="Computing centroids"):
        ys, xs = np.where(seg == nid)
        if len(ys) == 0:
            continue
        centroids[nid] = (int(np.mean(xs)), int(np.mean(ys)))
    return centroids


def _estimate_scale(he_path: str, seg_path: str = None) -> float:
    """Estimate the scale ratio from H&E to segmentation mask"""
    if seg_path and os.path.exists(seg_path):
        seg = tifffile.imread(seg_path)
        he = tifffile.imread(he_path)
        he_h = he.shape[0] if he.shape[-1] in (3, 4) else he.shape[1]
        ratio = he_h / max(seg.shape[0], 1)
        print(f"  Estimated scale ratio: {ratio:.2f}")
        return ratio
    return 4.7  # Default Xenium: ~0.2125 μm/pixel → 1 μm/pixel


# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data format conversion: Any Format → AnnData (h5ad)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="Conversion commands")

    # Xenium
    p_xe = sub.add_parser("xenium", help="Xenium → h5ad")
    p_xe.add_argument("--he_image", default="../data/Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif")
    p_xe.add_argument("--matched_nuclei", default="../data/data_processing/matched_nuclei_filtered.csv")
    p_xe.add_argument("--cgm", default="../data/data_processing/cell_gene_matrix_filtered.csv")
    p_xe.add_argument("--seg_mask", default="../data/data_processing/he_image_nuclei_seg_microns.tif")
    p_xe.add_argument("--output", default="../data/xenium_prostate.h5ad")

    # CSV
    p_csv = sub.add_parser("csv", help="CSV (BRCA/DLPFC) → h5ad")
    p_csv.add_argument("--image_dir", required=True)
    p_csv.add_argument("--sample_ids", required=True, help="Comma-separated sample IDs")
    p_csv.add_argument("--intdata_dir", default=None)
    p_csv.add_argument("--spot_prefix", default="spot_")
    p_csv.add_argument("--image_ext", default=".jpg")
    p_csv.add_argument("--output", default="csv_data.h5ad")

    # Visium
    p_vis = sub.add_parser("visium", help="Visium spaceranger → h5ad")
    p_vis.add_argument("--spaceranger_dir", required=True)
    p_vis.add_argument("--output", default="visium_data.h5ad")

    # Validate
    p_val = sub.add_parser("validate", help="Validate h5ad format")
    p_val.add_argument("--adata", required=True)

    args = parser.parse_args()

    if args.command == "xenium":
        xenium_to_h5ad(
            he_image_path=args.he_image,
            matched_nuclei_csv=args.matched_nuclei,
            cell_gene_matrix_csv=args.cgm,
            seg_mask_path=args.seg_mask,
            output_path=args.output,
        )
    elif args.command == "csv":
        sample_ids = [s.strip() for s in args.sample_ids.split(",")]
        csv_to_h5ad(
            image_dir=args.image_dir,
            sample_ids=sample_ids,
            intdata_dir=args.intdata_dir,
            spot_prefix=args.spot_prefix,
            image_ext=args.image_ext,
            output_path=args.output,
        )
    elif args.command == "visium":
        visium_to_h5ad(args.spaceranger_dir, args.output)
    elif args.command == "validate":
        validate_h5ad(args.adata)
    else:
        parser.print_help()
