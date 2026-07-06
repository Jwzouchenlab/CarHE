# -*- coding: utf-8 -*-
"""
CarHE Unified Data Loader — AnnData Standard Format
====================================================

All datasets must be converted to AnnData (h5ad) format before use.
No more CSV / Xenium special-case branches are retained.

AnnData format requirements:
    adata.X              : [N_spots, N_genes] gene expression matrix (float32)
    adata.obs:
        - sample_id      : sample identifier (required for all datasets)
        - image_path     : corresponding full H&E image path (required for all datasets)
    adata.obsm:
        - spatial        : [N, 2] spatial coordinates (pixel_x, pixel_y)
    adata.obsm optional:
        - X_scGPT        : [N, D] scGPT pre-computed reduced embeddings (used preferentially if present)
    adata.var            : gene names

Conversion tool: data/convert_to_h5ad.py
"""

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import pandas as pd
import tifffile
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from typing import Tuple

from config import CFG


# ==================== Image Loading Helpers ====================
def _load_image_safe(image_path: str) -> np.ndarray:
    """Safely load H&E image (supports tif, jpg, png)"""
    if image_path.lower().endswith(('.btf', '.tif', '.tiff')):
        try:
            with tifffile.TiffFile(image_path) as tif:
                img = tif.pages[0].asarray()
        except Exception:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return img


# ==================== Unified Dataset Class ====================
class SpotDataset(Dataset):
    """General-purpose spatial transcriptomics Dataset — loads from AnnData

    Each item contains:
        - image:              [3, 256, 256] tensor
        - reduced_expression: [ngenes] tensor
        - barcode:            str
        - spatial_coords:     [2] tensor
    """

    def __init__(
        self,
        adata_subset: "anndata.AnnData",
        image_path: str,
        sample_id: str,
        ngenes: int = None,
        augment_factor: int = 2,
        patch_size: int = 256,
    ):
        self.sample_id = sample_id
        self.augment_factor = augment_factor
        self.patch_size = patch_size
        self.ngenes = ngenes or CFG.spot_embedding

        # Load H&E image
        self.whole_image = _load_image_safe(image_path)

        # Spatial coordinates
        if "spatial" not in adata_subset.obsm:
            raise ValueError(f"AnnData missing obsm['spatial']: {sample_id}")
        self.spatial_pos = adata_subset.obsm["spatial"]
        self.barcodes = adata_subset.obs_names.tolist()

        # Gene expression: prefer X_scGPT, fallback to .X
        if "X_scGPT" in adata_subset.obsm:
            X_data = adata_subset.obsm["X_scGPT"]
        else:
            X_data = adata_subset.X
        if hasattr(X_data, "toarray"):
            X_data = X_data.toarray()
        self.reduced_matrix = np.asarray(X_data, dtype=np.float32)

        if self.reduced_matrix.shape[0] != len(self.barcodes):
            raise ValueError(f"Expression matrix rows ({self.reduced_matrix.shape[0]}) mismatch with barcode count ({len(self.barcodes)})")

        # Data augmentation
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomResizedCrop(size=(patch_size, patch_size), scale=(0.8, 1.0)),
        ])
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _augment(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.augmentations(img)
        return np.asarray(img)

    def __len__(self):
        return int(len(self.barcodes) * self.augment_factor)

    def __getitem__(self, idx):
        while True:
            real_idx = idx % len(self.barcodes)
            aug_idx = idx // len(self.barcodes)

            barcode = self.barcodes[real_idx]
            cx, cy = self.spatial_pos[real_idx]
            cx, cy = int(cx), int(cy)

            # Extract 256x256 patch
            half = self.patch_size // 2
            h, w = self.whole_image.shape[:2]
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)

            patch = self.whole_image[y1:y2, x1:x2]

            # Padding
            pad_l = max(0, half - cx)
            pad_r = max(0, cx + half - w)
            pad_t = max(0, half - cy)
            pad_b = max(0, cy + half - h)
            if pad_t or pad_b or pad_l or pad_r:
                patch = cv2.copyMakeBorder(
                    patch, pad_t, pad_b, pad_l, pad_r,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0] if patch.ndim == 3 else 0
                )
            if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                patch = cv2.resize(patch, (self.patch_size, self.patch_size))

            if patch.ndim == 2:
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)

            if aug_idx > 0:
                patch = self._augment(patch)

            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_pil = Image.fromarray(patch_rgb)
            img_tensor = self.base_transform(patch_pil)

            # Gene expression
            expr = self.reduced_matrix[real_idx, :self.ngenes].astype(np.float32)
            if len(expr) < self.ngenes:
                expr = np.pad(expr, (0, self.ngenes - len(expr)), 'constant')

            return {
                "image": img_tensor,
                "reduced_expression": torch.tensor(expr),
                "barcode": str(barcode),
                "spatial_coords": torch.tensor([cx, cy], dtype=torch.float32),
            }


# ==================== Unified DataLoader Builder ====================
def build_loaders_adata(
    adata: "anndata.AnnData" = None,
    adata_path: str = None,
    batch_size: int = None,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 123,
    ngenes: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders from AnnData (unified entry point)

    Supports two modes:
        1. adata=        pass an already-loaded AnnData object
        2. adata_path=   pass a path to an h5ad file
    """
    import scanpy as sc

    if adata is None and adata_path is not None:
        adata = sc.read_h5ad(adata_path)
    elif adata is None:
        raise ValueError("Must provide either adata or adata_path")

    batch_size = batch_size or CFG.batch_size
    ngenes = ngenes or CFG.spot_embedding

    # Required columns
    required_cols = ["sample_id", "image_path"]
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"AnnData.obs missing required column: '{col}'. Please use convert_to_h5ad.py to convert the data.")

    # Group by sample_id, create one Dataset per sample
    sample_ids = adata.obs["sample_id"].unique().tolist()
    print(f"Detected {len(sample_ids)} samples: {sample_ids[:5]}{'...' if len(sample_ids) > 5 else ''}")

    datasets = []
    for sid in sample_ids:
        subset = adata[adata.obs["sample_id"] == sid].copy()
        img_path = subset.obs["image_path"].iloc[0]
        ds = SpotDataset(
            adata_subset=subset,
            image_path=img_path,
            sample_id=sid,
            ngenes=ngenes,
            augment_factor=CFG.augment_factor,
        )
        datasets.append(ds)

    # Merge and split
    full = ConcatDataset(datasets)
    n_train = int(train_ratio * len(full))
    n_test = len(full) - n_train
    train_ds, test_ds = random_split(
        full, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"Total spots/cells: {len(full)}, train: {n_train}, test: {n_test}")
    return train_loader, test_loader
