# -*- coding: utf-8 -*-
"""
CarHE 统一数据加载器 — AnnData 标准格式
==========================================

所有数据集必须转换为 AnnData (h5ad) 格式后使用。
不再保留 CSV / Xenium 等特例分支。

AnnData 格式要求:
    adata.X              : [N_spots, N_genes] 基因表达矩阵 (float32)
    adata.obs:
        - sample_id      : 样本标识 (所有数据集都需要)
        - image_path     : 对应的完整 H&E 图像路径 (所有数据集都需要)
    adata.obsm:
        - spatial        : [N, 2] 空间坐标 (pixel_x, pixel_y)
    adata.obsm 可选:
        - X_scGPT        : [N, D] scGPT 预计算的降维嵌入 (若存在则优先使用)
    adata.var            : 基因名称

转换工具: data/convert_to_h5ad.py
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


# ==================== 图像读取辅助 ====================
def _load_image_safe(image_path: str) -> np.ndarray:
    """安全加载 H&E 图像 (支持 tif, jpg, png)"""
    if image_path.lower().endswith(('.btf', '.tif', '.tiff')):
        try:
            with tifffile.TiffFile(image_path) as tif:
                img = tif.pages[0].asarray()
        except Exception:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return img


# ==================== 统一 Dataset 类 ====================
class SpotDataset(Dataset):
    """通用空间转录组 Dataset — 从 AnnData 加载

    每个 item 包含:
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

        # 加载 H&E 图像
        self.whole_image = _load_image_safe(image_path)

        # 空间坐标
        if "spatial" not in adata_subset.obsm:
            raise ValueError(f"AnnData 缺少 obsm['spatial']: {sample_id}")
        self.spatial_pos = adata_subset.obsm["spatial"]
        self.barcodes = adata_subset.obs_names.tolist()

        # 基因表达：优先使用 X_scGPT，否则使用 .X
        if "X_scGPT" in adata_subset.obsm:
            X_data = adata_subset.obsm["X_scGPT"]
        else:
            X_data = adata_subset.X
        if hasattr(X_data, "toarray"):
            X_data = X_data.toarray()
        self.reduced_matrix = np.asarray(X_data, dtype=np.float32)

        if self.reduced_matrix.shape[0] != len(self.barcodes):
            raise ValueError(f"表达矩阵行数 ({self.reduced_matrix.shape[0]}) 与 barcode 数 ({len(self.barcodes)}) 不一致")

        # 数据增强
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

            # 提取 256×256 patch
            half = self.patch_size // 2
            h, w = self.whole_image.shape[:2]
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)

            patch = self.whole_image[y1:y2, x1:x2]

            # padding
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

            # 基因表达
            expr = self.reduced_matrix[real_idx, :self.ngenes].astype(np.float32)
            if len(expr) < self.ngenes:
                expr = np.pad(expr, (0, self.ngenes - len(expr)), 'constant')

            return {
                "image": img_tensor,
                "reduced_expression": torch.tensor(expr),
                "barcode": str(barcode),
                "spatial_coords": torch.tensor([cx, cy], dtype=torch.float32),
            }


# ==================== 统一 DataLoader 构建 ====================
def build_loaders_adata(
    adata: "anndata.AnnData" = None,
    adata_path: str = None,
    batch_size: int = None,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 123,
    ngenes: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """从 AnnData 构建训练/验证 DataLoader（统一入口）

    支持两种方式：
        1. adata=        传入已加载的 AnnData 对象
        2. adata_path=   传入 h5ad 文件路径
    """
    import scanpy as sc

    if adata is None and adata_path is not None:
        adata = sc.read_h5ad(adata_path)
    elif adata is None:
        raise ValueError("需要提供 adata 或 adata_path")

    batch_size = batch_size or CFG.batch_size
    ngenes = ngenes or CFG.spot_embedding

    # 必须的列
    required_cols = ["sample_id", "image_path"]
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"AnnData.obs 缺少必要列: '{col}'。请用 convert_to_h5ad.py 转换数据。")

    # 按 sample_id 分组，每个样本创建一个 Dataset
    sample_ids = adata.obs["sample_id"].unique().tolist()
    print(f"检测到 {len(sample_ids)} 个样本: {sample_ids[:5]}{'...' if len(sample_ids) > 5 else ''}")

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

    # 合并、划分
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

    print(f"总 spots/cells: {len(full)}, train: {n_train}, test: {n_test}")
    return train_loader, test_loader


# ==================== 兼容旧接口（供 train/inference 调用） ====================
def build_loaders_csv(
    image_dir: str,
    csv_pattern: str,
    sample_ids: list,
    batch_size: int = None,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """从 CSV 数据集直接构建 DataLoader（向后兼容，内部先转 h5ad）
    
    这是 get_Data.py 的替代：接受 CSV 路径 → 自动转 h5ad → 调用 build_loaders_adata
    """
    import tempfile
    from convert_csv_to_h5ad import csv_folder_to_h5ad

    tmp_path = os.path.join(tempfile.gettempdir(), f"_carhe_auto_{hash(tuple(sample_ids))}.h5ad")
    if not os.path.exists(tmp_path):
        adata = csv_folder_to_h5ad(image_dir, csv_pattern, sample_ids, output_path=tmp_path)
    else:
        import scanpy as sc
        adata = sc.read_h5ad(tmp_path)

    return build_loaders_adata(adata=adata, batch_size=batch_size, num_workers=num_workers)
