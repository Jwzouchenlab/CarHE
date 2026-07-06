# -*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
import torchvision.transforms as transforms
from config import CFG
import anndata as ad

# dataset for training with paired image and gene expression
class augmentDataset1(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path,ngenes=CFG.spot_embedding, augment_factor=2):
        self.whole_image = cv2.imread(image_path)
        if self.whole_image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=0)
        self.barcode_tsv = self.spatial_pos_csv['barcode']
        self.reduced_matrix = pd.read_csv(reduced_mtx_path, header=0)  # cell x features
        self.reduced_matrix = self.reduced_matrix.iloc[0:, 1:].values.T
        self.augment_factor = augment_factor
        self.ngenes = ngenes
        if len(self.spatial_pos_csv) == 0 or len(self.barcode_tsv) == 0:
            raise ValueError("CSV/TSV files are empty or improperly formatted.")
        
        if self.reduced_matrix.shape[0] != len(self.barcode_tsv):
            raise ValueError("Reduced matrix row count must match the number of barcodes.")
        
        #print("Finished loading all files")

        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        ])

    def transform(self, image):
        image = Image.fromarray(image)
        image = self.augmentations(image)
        return np.asarray(image)

    def __getitem__(self, idx):
        while True:
            actual_idx = idx % len(self.barcode_tsv)
            augment_idx = idx // len(self.barcode_tsv)

            item = {}
            barcode = self.barcode_tsv.iloc[actual_idx]
            spatial_info = self.spatial_pos_csv[self.spatial_pos_csv['barcode'] == barcode]

            if spatial_info.empty:
                print(f"Warning: No spatial information found for barcode {barcode}. Skipping.") #warn instead of raising error
                idx += 1  #skip this barcode
                continue


            try:
                v1 = int(spatial_info['pixel_x'].iloc[0])  # cell location x - uses column name
                v2 = int(spatial_info['pixel_y'].iloc[0])  # cell location y - uses column name
            except (KeyError, IndexError, ValueError) as e:
                print(f"Error processing barcode {barcode}: {e}. Skipping.")
                idx += 1
                continue


            # Extract image patch
            try:
                image_patch = self.whole_image[(v1 - 128):(v1 + 128), (v2 - 128):(v2 + 128)]
            except IndexError as e:
                print(f"Error extracting image patch for barcode {barcode}: {e}.  Likely coordinates are out of bounds.  Skipping.")
                idx += 1
                continue

            if image_patch.shape[0] != 256 or image_patch.shape[1] != 256:
                idx += 1
                continue

            if augment_idx > 0:
                image_patch = self.transform(image_patch)

            image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
            image_patch_pil = Image.fromarray(image_patch_rgb)
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            image = t(image_patch_pil)

            item['image'] = image  # color channel first, then XY
            item['reduced_expression'] = torch.tensor(self.reduced_matrix[actual_idx, :self.ngenes]).float()  # default with top 2000 hvg gene
            item['barcode'] = barcode
            item['spatial_coords'] = torch.tensor([v1, v2]).float()

            return item
    def __len__(self):  # This is the missing method
        return len(self.barcode_tsv) * self.augment_factor
    


# dataset for test dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path,ngenes=CFG.spot_embedding, augment_factor=1):
        self.whole_image = cv2.imread(image_path)
        if self.whole_image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=0)
        self.barcode_tsv = self.spatial_pos_csv['barcode']
        self.reduced_matrix = pd.read_csv(reduced_mtx_path, header=0)  # cell x features
        self.reduced_matrix = self.reduced_matrix.iloc[0:, 1:].values.T
        self.augment_factor = augment_factor
        self.ngenes = ngenes
        if len(self.spatial_pos_csv) == 0 or len(self.barcode_tsv) == 0:
            raise ValueError("CSV/TSV files are empty or improperly formatted.")
        
        if self.reduced_matrix.shape[0] != len(self.barcode_tsv):
            raise ValueError("Reduced matrix row count must match the number of barcodes.")
        
        #print("Finished loading all files")

        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        ])

    def transform(self, image):
        image = Image.fromarray(image)
        image = self.augmentations(image)
        return np.asarray(image)

    def __getitem__(self, idx):
        while True:
            actual_idx = idx % len(self.barcode_tsv)
            augment_idx = idx // len(self.barcode_tsv)

            item = {}
            barcode = self.barcode_tsv.iloc[actual_idx]
            spatial_info = self.spatial_pos_csv[self.spatial_pos_csv['barcode'] == barcode]

            if spatial_info.empty:
                print(f"Warning: No spatial information found for barcode {barcode}. Skipping.") #warn instead of raising error
                idx += 1  #skip this barcode
                continue


            try:
                v1 = int(spatial_info['pixel_x'].iloc[0])  # cell location x - uses column name
                v2 = int(spatial_info['pixel_y'].iloc[0])  # cell location y - uses column name
            except (KeyError, IndexError, ValueError) as e:
                print(f"Error processing barcode {barcode}: {e}. Skipping.")
                idx += 1
                continue


            # Extract image patch
            try:
                image_patch = self.whole_image[(v1 - 128):(v1 + 128), (v2 - 128):(v2 + 128)]
            except IndexError as e:
                print(f"Error extracting image patch for barcode {barcode}: {e}.  Likely coordinates are out of bounds.  Skipping.")
                idx += 1
                continue

            if image_patch.shape[0] != 256 or image_patch.shape[1] != 256:
                idx += 1
                continue

            if augment_idx > 0:
                image_patch = self.transform(image_patch)

            image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
            image_patch_pil = Image.fromarray(image_patch_rgb)
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            image = t(image_patch_pil)

            item['image'] = image  # color channel first, then XY
            item['reduced_expression'] = torch.tensor(self.reduced_matrix[actual_idx, :self.ngenes]).float()  # default with top 2000 hvg gene
            item['barcode'] = barcode
            item['spatial_coords'] = torch.tensor([v1, v2]).float()

            return item
    def __len__(self):  # This is the missing method
        return len(self.barcode_tsv) * self.augment_factor