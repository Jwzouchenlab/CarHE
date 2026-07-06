# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from PIL import Image
import tifffile
import cv2
import os
import math
import random
import warnings
from tqdm import tqdm
import re # Import regular expressions for filename parsing
import traceback # For printing full error tracebacks

# --- Set Matplotlib backend ---
# IMPORTANT: Do this BEFORE importing pyplot or calling any plotting functions
import matplotlib
matplotlib.use('Agg')
# -----------------------------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import torchvision.transforms as transforms
import SpaGCN as spg
from scipy.sparse import issparse, csr_matrix

# HIPT specific imports (ensure these paths are correct)
import sys
# Adjust the path to your HIPT installation if necessary
sys.path.append('/sibcb1/chenluonanlab8/zoujiawei/HIPT/HIPT_4K')
# Assuming HIPT_4K is not strictly needed based on model usage
# from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256

# --- Configuration ---
class CFG:
    # === Batch Processing Directories ===
    # Directory containing the input tif/txt pairs
    input_dir = '/sibcb1/chenluonanlab8/zoujiawei/data_lung/annotation/test/'
    # Base directory where results for all samples will be saved (each in its own subfolder)
    output_base_dir = '/sibcb1/chenluonanlab8/zoujiawei/data_lung/annotation/test/Results/'

    # === Fixed Reference and Model Paths ===
    ref_adata_path = "/sibcb1/chenluonanlab8/zoujiawei/data_lung/10x/10x/VHD/lung_ref_temp_300.h5ad"
    pretrained_weights_vit256 = '/sibcb1/chenluonanlab8/zoujiawei/HIPT/Checkpoints/vit256_small_dino.pth'
    pretrained_weights_clip = "/sibcb1/chenluonanlab8/zoujiawei/data_lung/10x/10x/VHD/model/HIPT_no_projection_200_gene_threshold.pt"

    # === Model & Embedding Config ===
    projection_dim = 256
    dropout = 0.1
    temperature = 0.07
    image_embedding_dim = 384
    ngenes = 2000 # Target number of genes from reference

    # === Training & Inference Config ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    num_workers = 4
    augment_factor = 1 # No augmentation needed for standard prediction run

    # === Matching & Prediction Config ===
    top_k_matches = 5
    matching_method = "average"

    # === SpaGCN Config ===
    spagcn_p = 0.5
    spagcn_n_clusters = 4 # Adjust if needed, maybe make dynamic later
    spagcn_refine_shape = "hexagon" # or "square"
    spagcn_beta_histology = 49
    spagcn_alpha_histology = 1
    spagcn_seed = 100
    spagcn_max_epochs = 200

    # === SVG Config ===
    svg_min_in_group_fraction = 0.8
    svg_min_in_out_group_ratio = 0.5
    svg_min_fold_change = 1.5
    svg_top_n_plot = 5
    svg_radius_neighbor = 512

# --- Model Definitions ---
class ProjectionHead(nn.Module):
    """Projection Head for embedding transformation."""
    # Corrected __init__ syntax
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout, eps=1e-06):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim, eps=eps)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class HIPT_CLIP_Model(nn.Module):
    """HIPT-based CLIP model."""
    # Corrected __init__ syntax
    def __init__(self, image_embedding_dim=CFG.image_embedding_dim, spot_embedding_dim=CFG.ngenes, projection_dim=CFG.image_embedding_dim, temperature=CFG.temperature):
        super().__init__()
        try:
            self.image_encoder = get_vit256(pretrained_weights=CFG.pretrained_weights_vit256, device=torch.device('cpu'))
        except FileNotFoundError: raise
        except Exception as e: raise

        for param in self.image_encoder.parameters(): param.requires_grad = False
        try:
            if hasattr(self.image_encoder, 'blocks') and len(self.image_encoder.blocks) > 0:
                for param in self.image_encoder.blocks[-1].parameters(): param.requires_grad = True
            if hasattr(self.image_encoder, 'norm'):
                for param in self.image_encoder.norm.parameters(): param.requires_grad = True
        except Exception as e: print(f"Warning: Could not unfreeze final layers of ViT: {e}")

        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding_dim, projection_dim=projection_dim)
        self.temperature = nn.Parameter(torch.tensor(np.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, batch):
        image = batch["image"].to(self.temperature.device)
        spot_expression = batch["reduced_expression"].to(self.temperature.device)
        image_features = self.encode_image(image)
        spot_embeddings = self.encode_spot(spot_expression)
        image_embeddings_norm = F.normalize(image_features, p=2, dim=1)
        spot_embeddings_norm = F.normalize(spot_embeddings, p=2, dim=1)
        logits = (spot_embeddings_norm @ image_embeddings_norm.T) * self.temperature.exp()
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def encode_image(self, image): return self.image_encoder(image)
    def encode_spot(self, spot_expression): return self.spot_projection(spot_expression)

# --- Dataset Definitions ---
class AugmentDatasetImage(torch.utils.data.Dataset):
    """Dataset for loading and augmenting image patches."""
    # Corrected __init__ syntax
    def __init__(self, image_path, valid_centers_path, augment_factor=1):
        self.image_path = image_path
        self.valid_centers_path = valid_centers_path
        self.augment_factor = max(1, int(augment_factor))
        self.patch_size = 256
        # --- Load Whole Slide Image ---
        try:
            self.whole_image = tifffile.imread(self.image_path)
            if self.whole_image is None: raise ValueError(f"Image None from {self.image_path}")
            if self.whole_image.ndim == 2: self.whole_image = cv2.cvtColor(self.whole_image, cv2.COLOR_GRAY2RGB)
            elif self.whole_image.shape[-1] == 4: self.whole_image = cv2.cvtColor(self.whole_image, cv2.COLOR_RGBA2RGB)
            elif self.whole_image.shape[-1] != 3: raise ValueError(f"Unsupported channels: {self.whole_image.shape[-1]}")
            if self.whole_image.dtype == np.uint16: self.whole_image = (self.whole_image / 256).astype(np.uint8)
            elif self.whole_image.dtype != np.uint8:
                 if np.issubdtype(self.whole_image.dtype, np.floating): self.whole_image = (self.whole_image * 255).clip(0, 255).astype(np.uint8)
                 else: self.whole_image = self.whole_image.astype(np.uint8)
        except FileNotFoundError: raise
        except Exception as e: raise ValueError(f"Error loading image {self.image_path}: {e}")
        # --- Load Valid Center Coordinates ---
        try:
            self.valid_centers = pd.read_csv(self.valid_centers_path, sep=',', header=None)
            if self.valid_centers.shape[1] < 2: raise ValueError("Centers file needs >= 2 columns")
            self.valid_centers = self.valid_centers.iloc[:, :2].values.astype(int)
        except FileNotFoundError: raise
        except Exception as e: raise ValueError(f"Error loading centers {self.valid_centers_path}: {e}")
        # --- Transforms ---
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 90)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomResizedCrop(size=(self.patch_size, self.patch_size), scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        ])
        self.standard_transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    # Corrected __len__ syntax
    def __len__(self): return len(self.valid_centers) * self.augment_factor
    # Corrected __getitem__ syntax
    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError("Index out of bounds")
        original_idx = idx % len(self.valid_centers)
        is_augmented = (idx // len(self.valid_centers)) > 0 and self.augment_factor > 1
        center_y, center_x = self.valid_centers[original_idx]
        h, w, _ = self.whole_image.shape
        half_patch = self.patch_size // 2
        y_start = max(0, center_y - half_patch)
        y_end = min(h, center_y + half_patch)
        x_start = max(0, center_x - half_patch)
        x_end = min(w, center_x + half_patch)
        image_patch_np = self.whole_image[y_start:y_end, x_start:x_end]
        pad_top = max(0, (center_y - half_patch) * -1)
        pad_bottom = max(0, (center_y + half_patch) - h)
        pad_left = max(0, (center_x - half_patch) * -1)
        pad_right = max(0, (center_x + half_patch) - w)
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
             image_patch_np = cv2.copyMakeBorder(image_patch_np, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if image_patch_np.shape[0] != self.patch_size or image_patch_np.shape[1] != self.patch_size:
             image_patch_np = cv2.resize(image_patch_np, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        image_patch_pil = Image.fromarray(image_patch_np)
        if is_augmented: image_patch_pil = self.augmentations(image_patch_pil)
        image_tensor = self.standard_transforms(image_patch_pil)
        item = {'image': image_tensor, 'spatial_coords': torch.tensor([center_y, center_x], dtype=torch.float32)}
        return item

class AugmentDatasetExpression(torch.utils.data.Dataset):
    """Dataset for loading spot expression data."""
    # Corrected __init__ syntax
    def __init__(self, expression_matrix):
        if isinstance(expression_matrix, csr_matrix): self.expression_matrix, self.is_sparse = expression_matrix, True
        elif isinstance(expression_matrix, np.ndarray): self.expression_matrix, self.is_sparse = expression_matrix, False
        else: raise TypeError("expression_matrix must be a NumPy array or csr_matrix.")
        if self.expression_matrix.ndim != 2: raise ValueError("expression_matrix must be 2D (spots x genes).")
    # Corrected __len__ syntax
    def __len__(self): return self.expression_matrix.shape[0]
    # Corrected __getitem__ syntax
    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError("Index out of bounds")
        if self.is_sparse: spot_expression = self.expression_matrix[idx].toarray().squeeze()
        else: spot_expression = self.expression_matrix[idx].squeeze()
        item = {'reduced_expression': torch.tensor(spot_expression, dtype=torch.float32)}
        return item

# --- Utility Functions ---
def get_embeddings(model, dataloader, encode_type="image", device="cuda"):
    model.eval()
    model.to(device)
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {encode_type}s", leave=False): # Set leave=False for nested loops
            if encode_type == "image": inputs = batch["image"].to(device); embeddings = model.encode_image(inputs)
            elif encode_type == "spot": inputs = batch["reduced_expression"].to(device); embeddings = model.encode_spot(inputs)
            else: raise ValueError("encode_type must be 'image' or 'spot'")
            all_embeddings.append(embeddings.cpu())
    if not all_embeddings: return torch.empty((0, CFG.image_embedding_dim))
    return torch.cat(all_embeddings)

def find_nearest_neighbors(query_embeddings, key_embeddings, top_k=5):
    n_query, n_key = query_embeddings.shape[0], key_embeddings.shape[0]
    # Corrected check for 0
    if n_query == 0 or n_key == 0: return np.empty((n_query, 0), dtype=int)
    effective_top_k = min(top_k, n_key)
    query_embeddings_cpu = query_embeddings.cpu(); key_embeddings_cpu = key_embeddings.cpu()
    try: distances = torch.cdist(query_embeddings_cpu, key_embeddings_cpu)
    except RuntimeError as e: print(f"ERROR: cdist failed: {e}"); raise
    _, indices = torch.topk(distances, k=effective_top_k, dim=1, largest=False)
    return indices.numpy()

def predict_expression_knn(neighbor_indices, reference_expression, method="average"):
    if neighbor_indices.size == 0: return np.empty((0, reference_expression.shape[1]))
    n_query, top_k = neighbor_indices.shape
    n_ref, n_genes = reference_expression.shape
    if isinstance(reference_expression, csr_matrix): ref_expr_dense = reference_expression.toarray()
    elif isinstance(reference_expression, np.ndarray): ref_expr_dense = reference_expression
    else: raise TypeError("reference_expression must be numpy array or csr_matrix")
    predicted_expression = np.zeros((n_query, n_genes), dtype=np.float32)
    if method == "average":
        for i in tqdm(range(n_query), desc="Averaging neighbor expression", leave=False): # Set leave=False for nested loops
            current_neighbor_indices = neighbor_indices[i, :]
            neighbor_expr = ref_expr_dense[current_neighbor_indices, :]
            predicted_expression[i, :] = np.mean(neighbor_expr, axis=0)
    else: raise ValueError(f"Unsupported prediction method: {method}")
    return predicted_expression

def plot_spatial_expression(coords, expression_data, gene_name, image_path, save_path, bg_scaling_factor=0.1, vmin_p=10, vmax_p=90, point_size=10, cmap='viridis'):
    if gene_name not in expression_data.columns: print(f"Warning: Gene '{gene_name}' not found. Skipping plot."); return
    if not coords.index.equals(expression_data.index):
         plot_data_merged = coords.join(expression_data[[gene_name]], how='inner')
         if plot_data_merged.empty: print(f"Warning: No matching spots for gene '{gene_name}'. Skipping plot."); return
         coords_filtered = plot_data_merged[['x', 'y']]; expression_filtered = plot_data_merged[[gene_name]]
    else: coords_filtered = coords; expression_filtered = expression_data[[gene_name]]
    plot_data = pd.concat([coords_filtered, expression_filtered], axis=1).dropna(subset=['x', 'y', gene_name])
    if plot_data.empty: print(f"Warning: No valid data points for gene '{gene_name}'. Skipping plot."); return

    # --- Add Data Checks ---
    # print(f"--- Checks for plot_spatial_expression: {gene_name} ---") # Reduce verbosity
    # print(f"Coordinates (head):\n{plot_data[['x', 'y']].head()}")
    # print(f"Coordinates describe:\n{plot_data[['x', 'y']].describe()}")
    # print(f"Expression data describe ('{gene_name}'):\n{plot_data[gene_name].describe()}")
    # print(f"NaN count in '{gene_name}': {plot_data[gene_name].isna().sum()}")
    # print(f"Plot data shape: {plot_data.shape}")
    # -----------------------

    thumbnail_rgb = None
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
             try: original_image = tifffile.imread(image_path)
             except Exception: pass # Ignore tifffile error if cv2 already failed
        if original_image is None: raise FileNotFoundError(f"Cannot load {image_path}")
        # Image processing
        if original_image.ndim == 3 and original_image.shape[0] in [3, 4]: original_image = np.moveaxis(original_image, 0, -1)
        if original_image.ndim == 2: original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[-1] == 4: original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        if original_image.dtype != np.uint8:
             if np.issubdtype(original_image.dtype, np.floating): original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
             elif np.issubdtype(original_image.dtype, np.integer): max_val = np.iinfo(original_image.dtype).max; original_image = (original_image / max_val * 255).astype(np.uint8)
        # Corrected multiplication
        thumbnail = cv2.resize(original_image, (int(original_image.shape[1] * bg_scaling_factor), int(original_image.shape[0] * bg_scaling_factor)), interpolation=cv2.INTER_AREA)
        # Corrected check
        thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB) if len(thumbnail.shape) == 3 and thumbnail.shape[2] == 3 else thumbnail # Basic BGR check
    except Exception as e: print(f"Warning: Could not load background '{image_path}'. Plotting without it. Error: {e}")

    # Corrected multiplication
    plot_data['x_scaled'] = plot_data['x'] * bg_scaling_factor; plot_data['y_scaled'] = plot_data['y'] * bg_scaling_factor
    expression_values = plot_data[gene_name].values
    try: vmin = np.percentile(expression_values[~np.isnan(expression_values)], vmin_p); vmax = np.percentile(expression_values[~np.isnan(expression_values)], vmax_p)
    except IndexError: vmin, vmax = 0, 1
    if vmin == vmax or np.isnan(vmin) or np.isnan(vmax):
        min_val, max_val = np.nanmin(expression_values), np.nanmax(expression_values)
        if min_val == max_val or np.isnan(min_val) or np.isnan(max_val): vmin, vmax = 0.0, 1.0
        else: vmin, vmax = min_val, max_val
    norm = Normalize(vmin=vmin, vmax=vmax)
    try: cmap_obj = cm.get_cmap(cmap)
    except ValueError: cmap_obj = cm.get_cmap('viridis')

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    if thumbnail_rgb is not None: ax.imshow(thumbnail_rgb, aspect='equal', extent=(0, thumbnail_rgb.shape[1], thumbnail_rgb.shape[0], 0))

    # Scatter plot
    try:
        # print(f"Plotting scatter for {gene_name}...") # Reduce verbosity
        sc_plot = ax.scatter(plot_data['x_scaled'], plot_data['y_scaled'], c=plot_data[gene_name], s=point_size, cmap=cmap_obj, norm=norm, edgecolor='none', alpha=0.85)
        # print(f"Scatter plot for {gene_name} created.")
    except Exception as scatter_e:
        print(f"ERROR during ax.scatter call for {gene_name}: {scatter_e}")
        plt.close(fig)
        return

    # Add colorbar
    try: cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04, shrink=0.7); cbar.set_label(f'{gene_name} Expression')
    except Exception as cbar_e: print(f"Warning: Could not create colorbar for {gene_name}. Error: {cbar_e}")

    # Titles, labels, limits
    ax.set_title(f'Predicted Spatial Expression: {gene_name}'); ax.set_xlabel('Spatial Coordinate X'); ax.set_ylabel('Spatial Coordinate Y')
    if thumbnail_rgb is not None: ax.set_xlim(0, thumbnail_rgb.shape[1]); ax.set_ylim(thumbnail_rgb.shape[0], 0)
    else: ax.autoscale_view(); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    # Save and close
    try:
        plt.tight_layout()
        # print(f"Attempting to save plot to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Saved spatial plot for {gene_name} to {save_path}")
    except Exception as save_e: print(f"ERROR saving plot {save_path}: {save_e}")
    finally:
        # print(f"Closing figure for {gene_name}.")
        plt.close(fig) # Ensure figure is closed


# === Helper Function to Find Sample Pairs ===
def find_sample_pairs(input_dir):
    """Scans input directory for matching tif and txt files."""
    samples = {}
    # Adjusted regex to be less strict about dashes/underscores within the ID part
    img_pattern = re.compile(r"standardized_(.+)\.(tif|tiff)$", re.IGNORECASE)
    txt_pattern = re.compile(r"valid_centers_standardized_(.+)\.txt$", re.IGNORECASE)

    print(f"Scanning directory: {input_dir}")
    try:
        files = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {input_dir}")
        return []
    except Exception as e:
        print(f"ERROR listing files in {input_dir}: {e}")
        return []
    print(f"Found {len(files)} files/dirs.")

    for f in files:
        # Ensure it's a file, not a directory (like 'Results')
        if not os.path.isfile(os.path.join(input_dir, f)):
            continue

        img_match = img_pattern.match(f)
        txt_match = txt_pattern.match(f)

        if img_match:
            sample_id = img_match.group(1)
            if sample_id not in samples: samples[sample_id] = {}
            samples[sample_id]['image'] = os.path.join(input_dir, f)
        elif txt_match:
            sample_id = txt_match.group(1)
            if sample_id not in samples: samples[sample_id] = {}
            samples[sample_id]['centers'] = os.path.join(input_dir, f)

    # Filter out incomplete pairs
    valid_samples = []
    for sample_id, paths in samples.items():
        if 'image' in paths and 'centers' in paths:
            valid_samples.append({
                'id': sample_id,
                'image_path': paths['image'],
                'centers_path': paths['centers']
            })
        else:
            print(f"Warning: Incomplete pair for sample ID '{sample_id}'. Found: {paths}. Skipping.")

    print(f"Found {len(valid_samples)} valid sample pairs in {input_dir}")
    if not valid_samples:
         print("Check file naming convention: expecting 'standardized_<ID>.tif' and 'valid_centers_standardized_<ID>.txt'")
    return valid_samples

# --- Main Execution ---
# Corrected __init__ and __name__ syntax
if __name__ == "__main__":
    print(f"Starting batch processing...") # Changed from "test run"
    print(f"Input directory: {CFG.input_dir}")
    print(f"Output base directory: {CFG.output_base_dir}") # Changed back to main output dir
    print(f"Using device: {CFG.device}")
    print(f"Scanpy version: {sc.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    # Set random seeds
    random.seed(CFG.spagcn_seed)
    np.random.seed(CFG.spagcn_seed)
    torch.manual_seed(CFG.spagcn_seed)
    if CFG.device == "cuda":
        torch.cuda.manual_seed_all(CFG.spagcn_seed)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # === Load Fixed Resources (Reference Data & Model) ONCE ===
    print("\n--- Loading Reference Data and Model (Once) ---")
    try:
        adata_ref = ad.read_h5ad(CFG.ref_adata_path)
        if adata_ref.shape[1] < CFG.ngenes: actual_ngenes = adata_ref.shape[1]
        elif adata_ref.shape[1] > CFG.ngenes: actual_ngenes = CFG.ngenes; adata_ref = adata_ref[:, :CFG.ngenes].copy()
        else: actual_ngenes = CFG.ngenes
        scaled_matrix_ref = adata_ref.X; gene_names_ref = adata_ref.var_names[:actual_ngenes].tolist()
        print(f"Reference expression matrix shape: {scaled_matrix_ref.shape}")
        ref_dataset_expr = AugmentDatasetExpression(scaled_matrix_ref)
        ref_dataloader_expr = DataLoader(ref_dataset_expr, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

        # Load Model
        hipt_model = HIPT_CLIP_Model(spot_embedding_dim=actual_ngenes).to(CFG.device)
        pretrained_state_dict = torch.load(CFG.pretrained_weights_clip, map_location=CFG.device)
        if all(k.startswith('module.') for k in pretrained_state_dict.keys()):
             pretrained_state_dict = {k[len('module.'):]: v for k, v in pretrained_state_dict.items()}
        missing_keys, unexpected_keys = hipt_model.load_state_dict(pretrained_state_dict, strict=False)
        if missing_keys: print("Warning: Missing keys:", missing_keys)
        if unexpected_keys: print("Warning: Unexpected keys:", unexpected_keys)
        hipt_model.eval()
        print("Reference data and HIPT model loaded successfully.")

        # Pre-generate reference spot embeddings
        print("Generating reference spot embeddings...")
        spot_embeddings_ref = get_embeddings(hipt_model, ref_dataloader_expr, encode_type="spot", device=CFG.device)
        print(f"Reference Spot Embeddings shape: {spot_embeddings_ref.shape}")
        if spot_embeddings_ref.shape[0] == 0:
             raise ValueError("Failed to generate reference spot embeddings.")

    except Exception as e:
        print(f"FATAL ERROR: Failed to load reference data or model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === Find Sample Pairs ===
    sample_pairs = find_sample_pairs(CFG.input_dir)
    if not sample_pairs:
        print("No valid sample pairs found. Exiting.")
        sys.exit(0)

    # === Loop Through Each Sample Pair ===
    print(f"\n--- Starting processing for {len(sample_pairs)} samples ---")
    for sample_info in sample_pairs: # Re-enabled the loop
        current_sample_id = sample_info['id']
        current_image_path = sample_info['image_path']
        current_centers_path = sample_info['centers_path']
        current_save_path = os.path.join(CFG.output_base_dir, current_sample_id)

        print(f"\n===== Processing Sample: {current_sample_id} =====")
        print(f"Image: {current_image_path}")
        print(f"Centers: {current_centers_path}")
        print(f"Output Dir: {current_save_path}")

        # Create output directory for the current sample
        os.makedirs(current_save_path, exist_ok=True)

        try: # Wrap processing for each sample in a try-except block
            # --- 2. Load Target Image Data (for current sample) ---
            print(f"\n--- 2. Loading Target Image Data for {current_sample_id} ---")
            target_dataset_img = AugmentDatasetImage(current_image_path, current_centers_path, CFG.augment_factor)
            target_dataloader_img = DataLoader(target_dataset_img, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=(CFG.device == "cuda"))
            print(f"Target image dataset size: {len(target_dataset_img)}")
            if len(target_dataset_img) == 0:
                 print(f"Warning: Image dataset for {current_sample_id} is empty. Skipping sample.")
                 continue # Skip to next sample

            # --- 4. Generate Embeddings (for current sample image) ---
            print(f"\n--- 4. Generating Image Embeddings for {current_sample_id} ---")
            img_embeddings_target = get_embeddings(hipt_model, target_dataloader_img, encode_type="image", device=CFG.device)
            print(f"Target Image Embeddings shape: {img_embeddings_target.shape}")
            if img_embeddings_target.shape[0] == 0:
                 print(f"Warning: Failed to generate image embeddings for {current_sample_id}. Skipping sample.")
                 continue # Skip to next sample
            # Save target embeddings
            img_embed_save_path = os.path.join(current_save_path, "target_img_embeddings.npy")
            np.save(img_embed_save_path, img_embeddings_target.numpy())

            # --- 5. Find Matches and Predict Expression ---
            print(f"\n--- 5. Finding Neighbors and Predicting Expression for {current_sample_id} ---")
            neighbor_indices = find_nearest_neighbors(img_embeddings_target, spot_embeddings_ref, top_k=CFG.top_k_matches)
            predicted_expression = predict_expression_knn(neighbor_indices, scaled_matrix_ref, method=CFG.matching_method)
            # Save predicted expression
            pred_expr_save_path = os.path.join(current_save_path, "predicted_expression.npy")
            np.save(pred_expr_save_path, predicted_expression)

            # --- 6. Prepare AnnData for Predicted Data ---
            print(f"\n--- 6. Creating AnnData for {current_sample_id} ---")
            # Assume CSV has x, y order based on previous working code? Double check this assumption.
            spatial_coords_df = pd.read_csv(current_centers_path, sep=',', header=None, names=['x_coord', 'y_coord'])
            if len(spatial_coords_df) != predicted_expression.shape[0]:
                raise ValueError(f"Coordinate count {len(spatial_coords_df)} from {current_centers_path} != Prediction count {predicted_expression.shape[0]}")
            spot_barcodes = [f"{current_sample_id}_spot_{i}" for i in range(predicted_expression.shape[0])] # Add sample ID prefix
            spatial_coords_df.index = spot_barcodes

            adata_pred = ad.AnnData(X=predicted_expression,
                                    obs=pd.DataFrame(index=spot_barcodes),
                                    var=pd.DataFrame(index=gene_names_ref))

            # --- ADD IMPUTATION HERE ---
            n_nans = np.isnan(adata_pred.X).sum()
            if n_nans > 0:
                print(f"WARNING: Predicted expression contains {n_nans} NaN values. Imputing NaNs with 0.")
                # Convert to dense if sparse, impute, then potentially back to sparse if memory is an issue
                if issparse(adata_pred.X):
                    print("Converting sparse matrix to dense for imputation...")
                    dense_x = adata_pred.X.toarray()
                    dense_x = np.nan_to_num(dense_x, nan=0.0) # Replace NaN with 0.0
                    # Optionally convert back to sparse if needed, e.g., csr_matrix(dense_x)
                    # For SpaGCN, keeping it dense is usually fine unless memory is very tight
                    adata_pred.X = dense_x
                    print("Imputation complete, matrix remains dense.")
                else:
                    adata_pred.X = np.nan_to_num(adata_pred.X, nan=0.0) # Replace NaN with 0.0 in dense array
                    print("Imputation complete.")
            else:
                print("Predicted expression contains no NaN values.")
            # --- END IMPUTATION ---


            # Assign coordinates based on CSV header assumption (x=col0, y=col1)
            adata_pred.obs['x_coord'] = spatial_coords_df['x_coord'].values
            adata_pred.obs['y_coord'] = spatial_coords_df['y_coord'].values
            # Standard spatial key uses [x, y] order
            adata_pred.obsm['spatial'] = spatial_coords_df[['x_coord', 'y_coord']].values
            adata_pred.var_names = [g.upper() for g in adata_pred.var_names]
            adata_pred.var_names_make_unique()
            adata_pred.var["genename"] = adata_pred.var.index.astype("str")

            adata_pred_save_path = os.path.join(current_save_path, "predicted_adata.h5ad")
            adata_pred.write_h5ad(adata_pred_save_path)
            print(f"Saved predicted AnnData object (with NaN imputation if needed) to {adata_pred_save_path}")


            # --- 7. Visualize Predicted Gene Expression ---
            print(f"\n--- 7. Visualizing Predicted Gene Expression for {current_sample_id} ---")
            genes_to_plot = []
            if gene_names_ref: genes_to_plot.append(gene_names_ref[0].upper())
            specific_genes_to_check = [
                'ANLN', 'ASPM', 'CDCA4', 'ERRFI1', 'FURIN', 'GOLGA8A', 'ITGA6',
                'JAG1', 'LRP12', 'MAFF', 'MRPS17', 'PLK1', 'PNP', 'PPP1R13L',
                'PRKCA', 'PTTG1', 'PYGB', 'RPP25', 'SCPEP1', 'SLC46A3', 'SNX7',
                'TPBG', 'XBP1'
            ]
            common_specific_genes = [g.upper() for g in specific_genes_to_check if g.upper() in adata_pred.var_names]
            print(f"Common specific genes found in prediction: {common_specific_genes}")
            genes_to_plot.extend(common_specific_genes[:3])
            genes_to_plot = sorted(list(set(genes_to_plot)))
            print(f"Plotting genes: {genes_to_plot}")

            expression_df_pred = pd.DataFrame(adata_pred.X, index=adata_pred.obs_names, columns=adata_pred.var_names)
            # Use x_coord and y_coord from .obs for consistency
            coords_df_pred = adata_pred.obs[['x_coord', 'y_coord']].rename(columns={'x_coord':'x', 'y_coord':'y'})

            for gene in genes_to_plot:
                if gene in expression_df_pred.columns:
                    plot_filename = os.path.join(current_save_path, f"predicted_spatial_{gene}.png")
                    plot_spatial_expression(coords=coords_df_pred,
                                            expression_data=expression_df_pred,
                                            gene_name=gene,
                                            image_path=current_image_path,
                                            save_path=plot_filename,
                                            point_size = max(1.0, 75000 / adata_pred.shape[0]))
                else: print(f"Skipping plot for {gene} as it's not in the predicted expression data columns.")


            # --- 8. SpaGCN Spatial Domain Identification ---
            print(f"\n--- 8. Running SpaGCN for {current_sample_id} ---")
            adata_spg = adata_pred.copy()
            # Assign pixel coordinates, assuming x_coord -> x_pixel, y_coord -> y_pixel
            adata_spg.obs["x_pixel"] = adata_spg.obs["x_coord"]
            adata_spg.obs["y_pixel"] = adata_spg.obs["y_coord"]

            # --- Load histology image ONCE for this sample ---
            histology_img = None
            try:
                histology_img = cv2.imread(current_image_path)
                if histology_img is None:
                     try: histology_img = tifffile.imread(current_image_path)
                     except Exception: pass
                if histology_img is None: raise FileNotFoundError(f"Cannot load histology {current_image_path}")
                # Processing...
                if histology_img.ndim == 2: histology_img = cv2.cvtColor(histology_img, cv2.COLOR_GRAY2BGR)
                elif histology_img.shape[-1]==4: histology_img = cv2.cvtColor(histology_img, cv2.COLOR_RGBA2BGR)
                if histology_img.dtype != np.uint8:
                     if np.issubdtype(histology_img.dtype, np.floating): histology_img = (histology_img*255).clip(0,255).astype(np.uint8)
                     else: histology_img=histology_img.astype(np.uint8)
            except Exception as img_load_e:
                 print(f"Warning: Failed to load histology image for SpaGCN/Marking: {img_load_e}")
                 histology_img = None # Ensure it's None if loading failed

            # --- Generate spatial_map.jpg ---
            try:
                print("Creating diagnostic image with spot centers marked (spatial_map)...")
                x_pixel_list = adata_spg.obs["x_pixel"].astype(int).values.tolist()
                y_pixel_list = adata_spg.obs["y_pixel"].astype(int).values.tolist()

                if histology_img is not None:
                    img_marked = histology_img.copy()
                    # Use black squares (BGR: 0,0,0) of size 20x20 pixels (radius=10?) - Adjust as needed
                    marker_half_size = 10 # Make the total size 2*marker_half_size = 20
                    marker_color = (0, 0, 0) # Black color in BGR for cv2

                    num_spots_to_mark = len(x_pixel_list)
                    h, w, _ = img_marked.shape

                    for i in range(num_spots_to_mark):
                        center_x = x_pixel_list[i]
                        center_y = y_pixel_list[i]

                        y_start = max(0, center_y - marker_half_size)
                        y_end = min(h, center_y + marker_half_size) # Non-inclusive end for slicing
                        x_start = max(0, center_x - marker_half_size)
                        x_end = min(w, center_x + marker_half_size) # Non-inclusive end

                        if y_start < y_end and x_start < x_end:
                             img_marked[y_start:y_end, x_start:x_end, :] = marker_color # Black out the square
                        # else: # Reduce verbosity
                             # print(f"Warning: Skipping marker for spot {i} at ({center_x}, {center_y}) - region invalid.")

                    marked_img_save_path = os.path.join(current_save_path, "spatial_map.jpg") # Save as spatial_map.jpg
                    success = cv2.imwrite(marked_img_save_path, img_marked)
                    if success: print(f"Saved diagnostic spatial_map.jpg to: {marked_img_save_path}")
                    else: print(f"Warning: Failed to save spatial_map.jpg.")
                else:
                    print("Warning: Histology image not available for creating spatial_map.jpg.")
            except Exception as diag_img_e:
                print(f"Warning: Error during spatial_map.jpg creation: {diag_img_e}")
            # --- End of spatial_map.jpg generation ---

            # Calculate Adjacency Matrix
            adj = None
            try:
                if histology_img is not None:
                    print("Calculating adjacency matrix using histology...")
                    adj = spg.calculate_adj_matrix(x=x_pixel_list, y=y_pixel_list,
                                                   x_pixel=x_pixel_list, y_pixel=y_pixel_list,
                                                   image=histology_img, beta=CFG.spagcn_beta_histology, alpha=CFG.spagcn_alpha_histology, histology=True)
                else:
                     print(f"Warning: Histology image not available for adj matrix. Calculating based on coordinates.")
                     adj = spg.calculate_adj_matrix(x=adata_spg.obs["x_coord"].values, y=adata_spg.obs["y_coord"].values, histology=False)
            except Exception as e:
                 print(f"ERROR during adjacency matrix calculation ({e}). Falling back to coordinate-based.")
                 adj = spg.calculate_adj_matrix(x=adata_spg.obs["x_coord"].values, y=adata_spg.obs["y_coord"].values, histology=False)

            adj_save_path = os.path.join(current_save_path, 'spagcn_adj.csv')
            np.savetxt(adj_save_path, adj, delimiter=',')

            # SpaGCN Training Parameters
            l = 0.5; res = 0.5 # Defaults
            try: # Search l
                 if np.all(np.sum(adj, axis=1) > 0): l = spg.search_l(CFG.spagcn_p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
                 else: print(f"Warning: Graph disconnected. Using default l={l}.")
                 print(f"Using SpaGCN l parameter: {l:.4f}")
            except Exception as e: print(f"Warning: search_l failed ({e}). Using default l={l}.")
            try: # Search res
                 if adata_spg.X.max() > 20: adata_temp = adata_spg.copy(); sc.pp.log1p(adata_temp)
                 else: adata_temp = adata_spg
                 effective_n_clusters = min(CFG.spagcn_n_clusters, adata_temp.shape[0])
                 print(f"Searching SpaGCN resolution for {effective_n_clusters} clusters...")
                 res = spg.search_res(adata_temp, adj, l, effective_n_clusters, start=0.4, step=0.1, tol=5e-3, lr=0.05, max_epochs=CFG.spagcn_max_epochs, r_seed=CFG.spagcn_seed, t_seed=CFG.spagcn_seed, n_seed=CFG.spagcn_seed)
                 print(f"Found SpaGCN resolution parameter: {res:.4f}")
                 del adata_temp
            except Exception as e: print(f"Warning: search_res failed ({e}). Using default res={res}.")

            # Train SpaGCN
            clf = spg.SpaGCN(); clf.set_l(l)
            if adata_spg.X.dtype != np.float32: adata_spg.X = adata_spg.X.astype(np.float32)
            initial_prediction_successful = False
            try:
                 clf.train(adata_spg, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=CFG.spagcn_max_epochs)
                 y_pred, prob = clf.predict()
                 print(f"SpaGCN y_pred (raw): {y_pred[:20]}...")
                 print(f"SpaGCN y_pred data type: {y_pred.dtype}")
                 adata_spg.obs["pred"] = pd.Categorical(y_pred.astype(str)) # Convert to string category
                 print(f"Assigned 'pred' column. Unique values: {adata_spg.obs['pred'].unique().tolist()}")
                 print("SpaGCN initial prediction done.")
                 initial_prediction_successful = True
            except Exception as train_e:
                print(f"ERROR: SpaGCN training failed: {train_e}");
                adata_spg.obs["pred"] = pd.Categorical(['0'] * adata_spg.shape[0]) # Assign dummy if fail
                initial_prediction_successful = False

            # Refine Clusters
            refinement_successful = False
            if initial_prediction_successful:
                try:
                    print("Refining SpaGCN clusters...")
                    adj_2d = spg.calculate_adj_matrix(x=adata_spg.obs["x_coord"].values, y=adata_spg.obs["y_coord"].values, histology=False)
                    pred_list_for_refine = adata_spg.obs["pred"].tolist()
                    refined_pred = spg.refine(sample_id=adata_spg.obs.index.tolist(), pred=pred_list_for_refine, dis=adj_2d, shape=CFG.spagcn_refine_shape)
                    print(f"SpaGCN refined_pred (raw list): {refined_pred[:20]}...")
                    adata_spg.obs["refined_pred"] = pd.Categorical(refined_pred) # Convert list directly
                    print(f"Assigned 'refined_pred' column. Unique values: {adata_spg.obs['refined_pred'].unique().tolist()}")
                    print("SpaGCN cluster refinement done.")
                    refinement_successful = True
                except Exception as e:
                    print(f"Warning: SpaGCN refinement failed ({e}). Skipping.")
                    # traceback.print_exc()
                    refinement_successful = False

            # Save final SpaGCN results AnnData
            spg_results_save_path = os.path.join(current_save_path, "spagcn_results.h5ad")
            adata_spg.write_h5ad(spg_results_save_path)
            print(f"Saved SpaGCN results AnnData to {spg_results_save_path}")

            # --- 9. Plot SpaGCN Domains ---
            print(f"\n--- 9. Plotting SpaGCN Domains for {current_sample_id} ---")
            # Add Matplotlib Test Plot
            test_plot_path = os.path.join(current_save_path, "matplotlib_test_plot.png")
            try:
                print(f"Attempting to save simple Matplotlib test plot to: {test_plot_path}")
                fig_test, ax_test = plt.subplots()
                ax_test.plot([0, 1], [0, 1], label='Test Line')
                ax_test.set_title("Matplotlib Backend Test")
                ax_test.legend()
                fig_test.savefig(test_plot_path, dpi=72)
                plt.close(fig_test)
                print(f"Simple test plot saved (check if file exists and is not empty): {test_plot_path}")
            except Exception as test_e:
                print(f"ERROR saving simple Matplotlib test plot: {test_e}")

            # Get Colormap
            try:
                num_pred_cats = len(adata_spg.obs['pred'].astype(str).unique())
                if num_pred_cats <= 10: cmap_qual = 'tab10'
                elif num_pred_cats <= 20: cmap_qual = 'tab20'
                else: cmap_qual = 'viridis'
                plot_colors_cat = plt.get_cmap(cmap_qual).colors
            except Exception: plot_colors_cat = plt.cm.get_cmap('tab10').colors

            # --- Add Data Checks ---
            print(f"--- Checks before domain plotting ---")
            print(f"adata_spg object:\n{adata_spg}")
            if 'spatial' in adata_spg.obsm:
                print(f"adata_spg.obsm['spatial'] (first 5):\n{adata_spg.obsm['spatial'][:5,:]}")
                print(f"adata_spg.obsm['spatial'] shape: {adata_spg.obsm['spatial'].shape}")
                print(f"NaN check in spatial coords: any NaN? {np.isnan(adata_spg.obsm['spatial']).any()}")
            else: print("ERROR: adata_spg.obsm['spatial'] not found!")
            if 'pred' in adata_spg.obs: print(f"adata_spg.obs['pred'] value counts:\n{adata_spg.obs['pred'].value_counts()}")
            if 'refined_pred' in adata_spg.obs: print(f"adata_spg.obs['refined_pred'] value counts:\n{adata_spg.obs['refined_pred'].value_counts()}")
            # -----------------------

            # Plot Initial Prediction
            if initial_prediction_successful:
                domain_col = "pred"
                if domain_col in adata_spg.obs and pd.api.types.is_categorical_dtype(adata_spg.obs[domain_col]):
                    n_doms = len(adata_spg.obs[domain_col].cat.categories)
                    print(f"Plotting '{domain_col}' with {n_doms} categories: {adata_spg.obs[domain_col].cat.categories.tolist()}")
                    adata_spg.uns[domain_col + "_colors"] = [plot_colors_cat[i % len(plot_colors_cat)] for i in range(n_doms)]
                    fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
                    save_path_pred = os.path.join(current_save_path, "spagcn_pred_spatial.png")
                    try:
                        print(f"Attempting SIMPLE sc.pl.spatial for '{domain_col}'...")
                        sc.pl.spatial(adata_spg, color=domain_col, ax=ax_pred, show=False) # Simplest call
                        print(f"SIMPLE sc.pl.spatial call for '{domain_col}' completed. Attempting save...")
                        fig_pred.savefig(save_path_pred, dpi=300, bbox_inches='tight')
                        print(f"Saved initial SpaGCN prediction plot to {save_path_pred}")
                    except Exception as plot_e: print(f"ERROR plotting/saving initial preds: {plot_e}")
                    finally: print(f"Closing figure for '{domain_col}'."); plt.close(fig_pred)
                else: print(f"Skipping plotting '{domain_col}' - column missing or not categorical.")

            # Plot Refined Prediction
            if refinement_successful:
                domain_col = "refined_pred"
                if domain_col in adata_spg.obs and pd.api.types.is_categorical_dtype(adata_spg.obs[domain_col]):
                    n_doms = len(adata_spg.obs[domain_col].cat.categories)
                    print(f"Plotting '{domain_col}' with {n_doms} categories: {adata_spg.obs[domain_col].cat.categories.tolist()}")
                    adata_spg.uns[domain_col + "_colors"] = [plot_colors_cat[i % len(plot_colors_cat)] for i in range(n_doms)]
                    fig_ref, ax_ref = plt.subplots(figsize=(8, 8))
                    save_path_ref = os.path.join(current_save_path, "spagcn_refined_pred_spatial.png")
                    try:
                        print(f"Attempting SIMPLE sc.pl.spatial for '{domain_col}'...")
                        sc.pl.spatial(adata_spg, color=domain_col, ax=ax_ref, show=False) # Simplest call
                        print(f"SIMPLE sc.pl.spatial call for '{domain_col}' completed. Attempting save...")
                        fig_ref.savefig(save_path_ref, dpi=300, bbox_inches='tight')
                        print(f"Saved refined SpaGCN prediction plot to {save_path_ref}")
                    except Exception as plot_e: print(f"ERROR plotting/saving refined preds: {plot_e}")
                    finally: print(f"Closing figure for '{domain_col}'."); plt.close(fig_ref)
                else: print(f"Skipping plotting '{domain_col}' - column missing or not categorical.")

            # --- 10. Spatially Variable Gene (SVG) Analysis ---
            print(f"\n--- 10. Running SVG Analysis for {current_sample_id} ---")
            if refinement_successful and "refined_pred" in adata_spg.obs: domain_col = "refined_pred"; print("Using 'refined_pred' column for SVG analysis.")
            elif initial_prediction_successful and "pred" in adata_spg.obs: domain_col = "pred"; print("Using 'pred' column for SVG analysis.")
            else: print("ERROR: No valid domain column for SVG. Skipping."); continue

            svg_adata = adata_spg.copy()
            if issparse(svg_adata.X): svg_adata.X = svg_adata.X.toarray()
            svg_adata.raw = svg_adata
            target_domains = sorted(svg_adata.obs[domain_col].cat.categories)
            print(f"Target domains for SVG analysis: {target_domains}")
            svg_cmap = LinearSegmentedColormap.from_list('pink_green', ['#008000', "#FFFFE0", "#FF00FF"], N=256) # Green-Yellow-Magenta

            # --- Add Checks before loop ---
            print(f"--- Checks before SVG plotting loop ---")
            print(f"svg_adata object:\n{svg_adata}")
            if 'spatial' in svg_adata.obsm:
                 print(f"svg_adata.obsm['spatial'] shape: {svg_adata.obsm['spatial'].shape}")
                 print(f"NaN check in spatial coords: any NaN? {np.isnan(svg_adata.obsm['spatial']).any()}")
            else: print("ERROR: svg_adata.obsm['spatial'] not found!")
            # ----------------------------

            try: adj_2d_svg = spg.calculate_adj_matrix(x=svg_adata.obs["x_coord"].values, y=svg_adata.obs["y_coord"].values, histology=False)
            except Exception as e: print(f"ERROR calculating SVG adj matrix: {e}"); continue

            all_svg_info_sample = []
            # Loop through each domain to find SVGs
            for target in target_domains:
                 print(f"\n--- Finding SVGs for Domain {target} ---")
                 target_spots = svg_adata.obs_names[svg_adata.obs[domain_col] == target]
                 if len(target_spots) == 0: print(f"Warning: Domain {target} has no spots. Skipping."); continue

                 # Find neighbors
                 nbr_domains = []
                 try:
                     nbr_indices = np.where(np.sum(adj_2d_svg[svg_adata.obs[domain_col] == target, :][:, svg_adata.obs[domain_col] != target], axis=0) > 0)[0]
                     if len(nbr_indices) > 0:
                         neighbor_spot_indices = np.where(svg_adata.obs[domain_col] != target)[0][nbr_indices]
                         nbr_domains = list(np.unique(svg_adata.obs[domain_col].iloc[neighbor_spot_indices]))
                     else: # Fallback radius search
                         nbr_domains = spg.find_neighbor_clusters(target_cluster=target, cell_id=svg_adata.obs.index.tolist(), x=svg_adata.obs["x_coord"].tolist(), y=svg_adata.obs["y_coord"].tolist(), pred=svg_adata.obs[domain_col].tolist(), radius=CFG.svg_radius_neighbor, ratio=0.5)
                     print(f"Neighbors for domain {target}: {nbr_domains}")
                     if not nbr_domains: print(f"Warning: No neighbors for domain {target}. Skipping SVG."); continue
                 except Exception as e: print(f"Error finding neighbors for domain {target}: {e}. Skipping."); continue

                 # Rank genes
                 de_genes_info = pd.DataFrame()
                 try:
                     if svg_adata.X.min() < 0 or svg_adata.X.max() > 25: sc.pp.log1p(svg_adata); use_log_in_rank = False
                     else: use_log_in_rank = True
                     print(f"Running rank_genes_groups for domain {target} vs neighbors {nbr_domains}...")
                     de_genes_info = spg.rank_genes_groups(input_adata=svg_adata, target_cluster=target, nbr_list=nbr_domains, label_col=domain_col, adj_nbr=True, log=use_log_in_rank)
                     if de_genes_info is None or de_genes_info.empty: print(f"Warning: No DE genes for domain {target}."); continue
                 except IndexError as ie: print(f"Warning: IndexError during rank_genes for domain {target}: {ie}. Skipping."); continue
                 except Exception as e: print(f"Error ranking genes for domain {target}: {e}. Skipping."); continue

                 # Filter SVGs
                 required_cols = ["pvals_adj", "in_out_group_ratio", "in_group_fraction", "fold_change", "genes"]
                 if not all(col in de_genes_info.columns for col in required_cols): print(f"Warning: Missing columns from rank_genes for {target}. Skipping."); continue
                 filtered_info = de_genes_info[(de_genes_info["pvals_adj"] < 0.05) & (de_genes_info["in_out_group_ratio"] > CFG.svg_min_in_out_group_ratio) & (de_genes_info["in_group_fraction"] > CFG.svg_min_in_group_fraction) & (de_genes_info["fold_change"] > CFG.svg_min_fold_change)].copy()
                 if filtered_info.empty: print(f"No SVGs passed filters for domain {target}."); continue

                 # Store and Save SVG list
                 filtered_info = filtered_info.sort_values(by="in_group_fraction", ascending=False)
                 filtered_info["target_domain"] = target; filtered_info["neighbors"] = str(nbr_domains)
                 all_svg_info_sample.append(filtered_info)
                 svg_csv_path = os.path.join(current_save_path, f'svgs_domain_{target}.csv')
                 filtered_info.to_csv(svg_csv_path, index=False)
                 print(f"Saved {len(filtered_info)} SVGs for domain {target} to {svg_csv_path}")

                 # Plot top N SVGs
                 top_n_genes = filtered_info["genes"].tolist()[:CFG.svg_top_n_plot]
                 for g in top_n_genes:
                     if g in svg_adata.var_names:
                         if svg_adata.raw is not None:
                             plot_adata = svg_adata.raw.to_adata()
                             plot_adata.obsm['spatial'] = svg_adata.obsm['spatial']
                             if domain_col in svg_adata.obs: plot_adata.obs[domain_col] = svg_adata.obs[domain_col]
                         else: plot_adata = svg_adata
                         if g not in plot_adata.var_names: print(f"Warning: Gene {g} missing from plot_adata. Skipping plot."); continue

                         plot_obs_col = f"{g}_exp"
                         try:
                            expression_values_g = plot_adata[:, g].X.toarray().flatten() if issparse(plot_adata[:, g].X) else plot_adata[:, g].X.flatten()
                            plot_adata.obs[plot_obs_col] = expression_values_g

                            # --- Add Detailed Check for SVG Data ---
                            print(f"\n--- Checks for SVG plot: Gene '{g}', Domain '{target}' ---")
                            # print(f"plot_adata object:\n{plot_adata}") # Reduce verbosity
                            print(f"plot_adata.obs['{plot_obs_col}'] (first 5): {expression_values_g[:5]}")
                            print(f"plot_adata.obs['{plot_obs_col}'] describe:\n{pd.Series(expression_values_g).describe()}")
                            print(f"NaN count in '{plot_obs_col}': {np.isnan(expression_values_g).sum()}")
                            print(f"Inf count in '{plot_obs_col}': {np.isinf(expression_values_g).sum()}")
                            if 'spatial' in plot_adata.obsm:
                                print(f"plot_adata.obsm['spatial'] shape: {plot_adata.obsm['spatial'].shape}")
                            else: print("ERROR: plot_adata.obsm['spatial'] missing!")
                            # ---------------------------------------

                            if np.any(np.isnan(expression_values_g)) or np.any(np.isinf(expression_values_g)): print(f"Warning: NaN/Inf values found for gene '{g}'.")

                         except Exception as data_prep_e: print(f"ERROR preparing data for SVG plot '{g}': {data_prep_e}"); continue

                         fig_svg, ax_svg = plt.subplots(figsize=(8, 8))
                         svg_plot_path = os.path.join(current_save_path, f"svg_domain_{target}_{g}.png")
                         try:
                             print(f"Attempting SIMPLE sc.pl.spatial for SVG '{g}'...")
                             sc.pl.spatial(plot_adata, color=plot_obs_col, cmap=svg_cmap, ax=ax_svg, show=False) # Simplest call
                             print(f"SIMPLE sc.pl.spatial call for SVG '{g}' completed. Attempting save...")
                             fig_svg.savefig(svg_plot_path, dpi=300, bbox_inches='tight')
                             print(f"Saved SVG plot for {g} to {svg_plot_path}")
                         except Exception as plot_e: print(f"ERROR plotting/saving SVG {g} for {target}: {plot_e}")
                         finally: print(f"Closing figure for SVG '{g}'."); plt.close(fig_svg) # Ensure closure
                     else: print(f"Warning: SVG gene {g} requested not found.")

            # Save combined SVG info for the current sample
            if all_svg_info_sample:
                 all_svg_df_sample = pd.concat(all_svg_info_sample, ignore_index=True)
                 all_svg_save_path = os.path.join(current_save_path, 'svgs_all_domains.csv')
                 all_svg_df_sample.to_csv(all_svg_save_path, index=False)
                 print(f"Saved combined SVG list for {current_sample_id} to {all_svg_save_path}")

            print(f"----- Finished processing sample {current_sample_id} -----")

        # Catch errors specific to the processing of one sample
        except Exception as sample_error:
             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print(f"ERROR processing sample {current_sample_id}. Skipping to next.")
             print(f"Error details: {sample_error}")
             traceback.print_exc() # Print full traceback for debugging
             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             # Optionally write error to a log file
             # error_log_path = os.path.join(CFG.output_base_dir, "error_log.txt")
             # with open(error_log_path, "a") as f:
             #     f.write(f"Error processing {current_sample_id}:\n{sample_error}\n{traceback.format_exc()}\n---\n")
             continue # Move to the next sample

    print("\n--- Batch Analysis Complete ---")