import os
import cv2
import pandas as pd
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import anndata as ad
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp  # For sparse matrix handling
import config as CFG


def get_image_and_spatial_data(whole_image, spatial_pos_csv, barcode, augment_idx, augmentations):
    """Extracts image patch and spatial coords. Returns data if valid, else None."""
    spatial_info = spatial_pos_csv[spatial_pos_csv['barcode'] == barcode]
    if spatial_info.empty:
        return None
    try:
        v1 = int(spatial_info['pixel_x'].iloc[0])
        v2 = int(spatial_info['pixel_y'].iloc[0])
        image_patch = whole_image[(v1 - 128):(v1 + 128), (v2 - 128):(v2 + 128)]
        if image_patch.shape[0] != 256 or image_patch.shape[1] != 256:
            return None
    except (KeyError, IndexError, ValueError):
        return None

    if augment_idx > 0:
        image_patch = transform(image_patch, augmentations)

    image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
    image_patch_pil = Image.fromarray(image_patch_rgb)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = t(image_patch_pil)
    return {'image': image, 'spatial_coords': torch.tensor([v1, v2]).float(), 'barcode': barcode}

class ExpressionDataset(Dataset):
    """Dataset for loading reduced gene expression data."""
    def __init__(self, adata, sample_id, ngenes=512):
        self.adata = adata
        self.sample_id = sample_id
        self.ngenes = ngenes
        if 'sample' not in self.adata.obs.columns:
            raise ValueError("The 'sample' column is missing in adata.obs.")
        self.adata_subset = self.adata[self.adata.obs['sample'] == self.sample_id, :].copy()
        if self.adata_subset.n_obs == 0:
            all_samples = self.adata.obs['sample'].unique().tolist()
            raise ValueError(f"Sample ID '{sample_id}' not found. Available: {all_samples}")

        # Pre-calculate and store the reduced expressions.  This is the key optimization.
        self.reduced_expressions = {}
        for barcode in self.adata_subset.obs_names:
            expression_data = self._get_reduced_expression(barcode)  # Use internal method
            if expression_data is not None:
                self.reduced_expressions[barcode] = expression_data['reduced_expression'] #store the tensor


    def __len__(self):
        return len(self.reduced_expressions)

    def __getitem__(self, idx):
        barcode = list(self.reduced_expressions.keys())[idx]  # Get barcode by index
        expression_tensor = self.reduced_expressions[barcode]  # Retrieve pre-calculated tensor
        # Return a dictionary with the 'expression' key
        return {"barcode": barcode, "reduced_expression": expression_tensor}

    def _get_reduced_expression(self, barcode):
        """Gets reduced gene expression. Returns data if valid, else None."""
        if barcode not in self.adata_subset.obs_names:
            return None
        barcode_index = self.adata_subset.obs_names.get_loc(barcode)

        # Access 'X_scGPT' *once* and handle potential shapes
        expression_data = self.adata_subset.obsm['X_scGPT'][barcode_index, :self.ngenes]
        if expression_data.ndim == 2:
            expression_data = expression_data.squeeze(0)

        # Efficient type handling (avoid repeated isinstance checks)
        if isinstance(expression_data, np.ndarray):
            expression_data = torch.tensor(expression_data).float()
        elif isinstance(expression_data, (sp.csr_matrix, sp.csc_matrix)):
            expression_data = torch.from_numpy(expression_data.toarray()).float() # Use torch.from_numpy for efficiency
        elif not isinstance(expression_data, torch.Tensor): # if already tensor, do nothing
             raise TypeError(f"Unsupported data type: {type(expression_data)}")

        return {'reduced_expression': expression_data, 'barcode': barcode}

class ImageDataset(Dataset):
    """Dataset for loading image data and spatial coordinates."""
    def __init__(self, spatial_pos_csv_path, image_path, augment_idx=0):
        self.spatial_pos_csv = pd.read_csv(spatial_pos_csv_path)
        self.whole_image = cv2.imread(image_path)
        self.augment_idx = augment_idx
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        # Pre-filter valid barcodes based on image data availability
        self.valid_barcodes = []
        for barcode in self.spatial_pos_csv['barcode'].unique(): #iterate through unique barcodes
          data = get_image_and_spatial_data(self.whole_image, self.spatial_pos_csv, barcode, self.augment_idx, self.augmentations)
          if data is not None: #valid barcode
            self.valid_barcodes.append(barcode)
    def __len__(self):
        return len(self.valid_barcodes)

    def __getitem__(self, idx):
        barcode = self.valid_barcodes[idx]
        image_data = get_image_and_spatial_data(
            self.whole_image, self.spatial_pos_csv, barcode, self.augment_idx, self.augmentations
        )
        # Return a dictionary with 'barcode', 'image', and 'spatial' keys
        return {
            "barcode": barcode,
            "image": image_data['image'],  # Ensure correct shape (C, H, W) and type
            "spatial": image_data['spatial_coords'], # Ensure type

        }
def transform(image, augmentations):
    """Applies augmentations to an image."""
    image = Image.fromarray(image)
    image = augmentations(image)
    return np.asarray(image)