# -*- coding: utf-8 -*-
"""
CarHE GradCAM Module
Gradient-weighted Class Activation Mapping (GradCAM) for H&E images. Visualizes which tissue regions
drive predictions of specific genes or gene expression profiles.

ViT-based GradCAM:
    1. Register hooks to capture activations from the last transformer block
    2. Forward pass to obtain image embeddings and gene expression embeddings
    3. Compute similarity score and backprop to obtain gradients
    4. Convert gradient-weighted activations to a heatmap
    5. Overlay onto the original H&E stained image

Usage:
    from gradcam import GradCAM_CarHE, visualize_heatmap
    gradcam = GradCAM_CarHE(model, target_layer_name='blocks.-1')
    heatmap = gradcam(image_tensor, spot_expression_tensor)
    visualize_heatmap(heatmap, original_image, save_path='gradcam.png')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Optional, Tuple, List, Union
import warnings

from config import CFG


# ==================== ViT GradCAM Core Class ====================
class ViT_GradCAM:
    """GradCAM implementation for Vision Transformer
    
    For ViT architecture, we extract activations of all patch tokens (excluding CLS token)
    from the last transformer block output, compute gradient-weighted sum,
    and reshape to a 2D spatial heatmap.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_block_index: int = -1,  # Default: use the last block
        patch_size: int = 16,          # ViT patch size
        image_size: int = 256,         # Input image size
        use_cls_token: bool = False,   # Whether to include CLS token
    ):
        self.model = model
        self.target_block_index = target_block_index
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.use_cls_token = use_cls_token
        
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._target_layer = None
    
    def _find_target_layer(self) -> nn.Module:
        """Find the target ViT block layer"""
        if hasattr(self.model.image_encoder, 'blocks'):
            blocks = self.model.image_encoder.blocks
            self._target_layer = blocks[self.target_block_index]
        else:
            # Try to find all possible ViT block structures
            raise AttributeError(
                "Cannot find ViT blocks in image_encoder. "
                "Make sure the model has self.image_encoder.blocks"
            )
        return self._target_layer
    
    def _forward_hook(self, module, input, output):
        """Forward hook: capture activations"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook: capture gradients"""
        self.gradients = grad_output[0].detach()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        target_layer = self._find_target_layer()
        # Forward hook
        fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        # Backward hook
        bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)
        self.hooks = [fwd_hook, bwd_hook]
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __call__(
        self,
        image_tensor: torch.Tensor,
        spot_expression: torch.Tensor,
        target_gene_indices: Optional[List[int]] = None,
        return_heatmap_only: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Compute GradCAM heatmap
        
        Args:
            image_tensor: H&E image tensor [1, 3, 256, 256] or [3, 256, 256]
            spot_expression: Gene expression tensor [1, n_genes] or [n_genes]
            target_gene_indices: Optional, list of target gene indices (for gene-level GradCAM)
            return_heatmap_only: Whether to only return the heatmap
        
        Returns:
            heatmap: 256x256 numpy array
            If return_heatmap_only=False, also returns (heatmap, score)
        """
        # Ensure correct dimensions
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if spot_expression.dim() == 1:
            spot_expression = spot_expression.unsqueeze(0)
        
        image_tensor = image_tensor.detach().requires_grad_(True)
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass
            self.model.zero_grad()
            
            # Compute image embeddings
            image_embeddings = self.model.encode_image(image_tensor)  # [1, 384]
            
            # Compute expression embeddings
            spot_exp = spot_expression.to(image_embeddings.device)
            spot_embeddings = self.model.encode_spot(spot_exp)  # [1, 384]
            
            # L2 normalization
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            spot_embeddings = F.normalize(spot_embeddings, p=2, dim=1)
            
            # Compute similarity score
            if target_gene_indices is not None:
                # Gene-level GradCAM: focus on specific gene contributions
                score = self._gene_specific_score(
                    image_embeddings, spot_embeddings, spot_exp, target_gene_indices
                )
            else:
                # Spot-level GradCAM: overall image-expression matching
                score = (image_embeddings * spot_embeddings).sum(dim=1)
            
            # Backward pass
            self.model.zero_grad()
            score.backward(retain_graph=False)
            
            # Generate heatmap
            heatmap = self._compute_heatmap()
            
            score_val = score.item()
            
        finally:
            self._remove_hooks()
        
        if return_heatmap_only:
            return heatmap
        return heatmap, score_val
    
    def _gene_specific_score(
        self,
        image_embeddings: torch.Tensor,
        spot_embeddings: torch.Tensor,
        spot_expression: torch.Tensor,
        gene_indices: List[int],
    ) -> torch.Tensor:
        """Compute GradCAM score for specific genes
        
        Compare similarity changes by modifying target gene values.
        Implementation: zero out the target gene expression, compute the similarity difference,
        which reflects the marginal contribution of the target gene to the similarity.
        """
        # Original similarity
        original_score = (image_embeddings * spot_embeddings).sum(dim=1, keepdim=True)
        
        # Modify expression: set target genes to zero
        modified_expression = spot_expression.clone()
        modified_expression[:, gene_indices] = 0.0
        
        # Re-encode
        with torch.no_grad():
            modified_spot_embeddings = self.model.encode_spot(modified_expression)
            modified_spot_embeddings = F.normalize(modified_spot_embeddings, p=2, dim=1)
        
        # Difference score
        modified_score = (image_embeddings * modified_spot_embeddings).sum(dim=1, keepdim=True)
        gene_score = original_score - modified_score
        
        return gene_score
    
    def _compute_heatmap(self) -> np.ndarray:
        """Compute GradCAM heatmap from captured activations and gradients"""
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "No activations or gradients captured. "
                "Make sure to call __call__() first."
            )
        
        # activations: [1, num_tokens, embed_dim] (257, 384)
        # gradients:   [1, num_tokens, embed_dim]
        activations = self.activations
        gradients = self.gradients
        
        batch_size = activations.shape[0]
        
        # Remove CLS token (if needed)
        if not self.use_cls_token:
            activations = activations[:, 1:, :]  # [1, 256, 384]
            gradients = gradients[:, 1:, :]       # [1, 256, 384]
        else:
            # Keep all tokens (including CLS)
            pass
        
        # Compute gradient-weighted activations per token,
        # sum along embed_dim as per-token weight
        weights = gradients.mean(dim=-1, keepdim=True)  # [1, num_tokens, 1]
        cam = (weights * activations).sum(dim=-1)  # [1, num_tokens]
        
        # ReLU activation (keep only positive contributions)
        cam = F.relu(cam)
        
        # Reshape to 2D spatial map
        spatial_size = self.num_patches_per_side
        cam = cam.view(batch_size, spatial_size, spatial_size)  # [1, 16, 16]
        
        # Upsample to original image size
        cam = F.interpolate(
            cam.unsqueeze(1),                              # [1, 1, 16, 16]
            size=(self.image_size, self.image_size),       # 256x256
            mode='bilinear',
            align_corners=False
        ).squeeze()  # [256, 256]
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam


# ==================== Multi-layer GradCAM ====================
class MultiLayerGradCAM:
    """Multi-layer GradCAM: aggregate GradCAM results from multiple ViT blocks
    Lower layers capture fine-grained texture features, higher layers capture semantic features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        block_indices: List[int] = [-4, -3, -2, -1],
        patch_size: int = 16,
        image_size: int = 256,
    ):
        self.model = model
        self.block_indices = block_indices
        self.patch_size = patch_size
        self.image_size = image_size
        
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self._target_layers = []
    
    def _find_target_layers(self):
        """Find multiple target ViT blocks"""
        blocks = self.model.image_encoder.blocks
        self._target_layers = [blocks[i] for i in self.block_indices]
    
    def _make_forward_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook
    
    def _register_hooks(self):
        self._find_target_layers()
        for i, layer in enumerate(self._target_layers):
            name = f"block_{self.block_indices[i]}"
            fwd = layer.register_forward_hook(self._make_forward_hook(name))
            bwd = layer.register_full_backward_hook(self._make_backward_hook(name))
            self.hooks.extend([fwd, bwd])
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def __call__(
        self,
        image_tensor: torch.Tensor,
        spot_expression: torch.Tensor,
        return_per_layer: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """Compute multi-layer GradCAM heatmap
        
        Returns the average heatmap across all layers, and optionally per-layer heatmaps.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if spot_expression.dim() == 1:
            spot_expression = spot_expression.unsqueeze(0)
        
        image_tensor = image_tensor.detach().requires_grad_(True)
        
        self._register_hooks()
        
        try:
            self.model.zero_grad()
            
            image_embeddings = self.model.encode_image(image_tensor)
            spot_exp = spot_expression.to(image_embeddings.device)
            spot_embeddings = self.model.encode_spot(spot_exp)
            
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            spot_embeddings = F.normalize(spot_embeddings, p=2, dim=1)
            
            score = (image_embeddings * spot_embeddings).sum()
            
            self.model.zero_grad()
            score.backward(retain_graph=False)
            
            # Compute heatmap for each layer
            layer_heatmaps = []
            for name in sorted(self.activations.keys()):
                act = self.activations[name]
                grad = self.gradients[name]
                # Remove CLS token
                act = act[:, 1:, :]
                grad = grad[:, 1:, :]
                
                weights = grad.mean(dim=-1, keepdim=True)
                cam = (weights * act).sum(dim=-1)
                cam = F.relu(cam)
                
                spatial = self.image_size // self.patch_size
                cam = cam.view(1, spatial, spatial)
                cam = F.interpolate(
                    cam.unsqueeze(1), size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()
                
                # Normalize
                cmin, cmax = cam.min(), cam.max()
                if cmax - cmin > 1e-8:
                    cam = (cam - cmin) / (cmax - cmin)
                
                layer_heatmaps.append(cam)
            
            # Average fusion
            avg_heatmap = np.mean(layer_heatmaps, axis=0)
            
        finally:
            self._remove_hooks()
        
        if return_per_layer:
            return avg_heatmap, layer_heatmaps
        return avg_heatmap


# ==================== Convenience Wrapper Class ====================
class GradCAM_CarHE:
    """Convenience GradCAM wrapper for CarHE model
    
    Automatically handles ViT-256 parameter configuration, provides a clean API.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_block_index: int = -1,
        multi_layer: bool = False,
        multi_layer_indices: List[int] = [-4, -3, -2, -1],
    ):
        self.model = model
        self.image_size = 256
        self.patch_size = 16
        self.device = next(model.parameters()).device
        
        if multi_layer:
            self.gradcam = MultiLayerGradCAM(
                model,
                block_indices=multi_layer_indices,
                patch_size=self.patch_size,
                image_size=self.image_size,
            )
        else:
            self.gradcam = ViT_GradCAM(
                model,
                target_block_index=target_block_index,
                patch_size=self.patch_size,
                image_size=self.image_size,
            )
    
    def compute(
        self,
        image_tensor: torch.Tensor,
        spot_expression: torch.Tensor,
        gene_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Compute GradCAM heatmap
        
        Args:
            image_tensor: H&E image tensor [1,3,256,256] or [3,256,256]
            spot_expression: Gene expression tensor [1,n_genes] or [n_genes]
            gene_indices: Specific gene indices (optional, for gene-level GradCAM)
        
        Returns:
            (heatmap_256x256, similarity_score)
        """
        self.model.eval()
        self.model.zero_grad()
        
        image_tensor = image_tensor.to(self.device)
        spot_expression = spot_expression.to(self.device)
        
        result = self.gradcam(image_tensor, spot_expression,
                             target_gene_indices=gene_indices)
        
        if isinstance(result, tuple):
            return result
        return result, 0.0
    
    def compute_for_image_patch(
        self,
        full_image: np.ndarray,
        center_x: int,
        center_y: int,
        spot_expression: torch.Tensor,
        gene_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute GradCAM directly from full-image coordinates
        
        Args:
            full_image: Full H&E image (H, W, 3)
            center_x: Patch center x coordinate
            center_y: Patch center y coordinate
            spot_expression: Gene expression
            gene_indices: Specific gene indices
        
        Returns:
            (heatmap_256x256, image_patch_256x256, score)
        """
        half = 128
        patch = full_image[
            max(0, center_y-half):min(full_image.shape[0], center_y+half),
            max(0, center_x-half):min(full_image.shape[1], center_x+half)
        ]
        
        # Padding
        pad_top = max(0, half - center_y)
        pad_bottom = max(0, center_y + half - full_image.shape[0])
        pad_left = max(0, half - center_x)
        pad_right = max(0, center_x + half - full_image.shape[1])
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        if patch.shape[0] != 256 or patch.shape[1] != 256:
            patch = cv2.resize(patch, (256, 256))
        
        # BGR -> RGB -> tensor
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if patch.ndim == 3 else patch
        patch_pil = Image.fromarray(patch_rgb)
        
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(patch_pil)
        
        heatmap, score = self.compute(image_tensor, spot_expression, gene_indices)
        return heatmap, patch_rgb, score


# ==================== Visualization Functions ====================
def apply_colormap(heatmap: np.ndarray, colormap_name: str = 'jet') -> np.ndarray:
    """Map heatmap [0,1] to colored colormap
    
    Args:
        heatmap: [H, W] range [0, 1]
        colormap_name: matplotlib colormap name
    
    Returns:
        colored_heatmap: [H, W, 4] RGBA
    """
    cmap = cm.get_cmap(colormap_name)
    colored = cmap(heatmap)  # [H, W, 4] RGBA
    return (colored[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = 'jet',
) -> np.ndarray:
    """Overlay heatmap on original H&E image
    
    Args:
        original_image: Original image [H, W, 3] RGB
        heatmap: Heatmap [H, W] range [0, 1]
        alpha: Heatmap transparency
        colormap_name: Colormap name
    
    Returns:
        overlay: [H, W, 3] RGB overlay image
    """
    # Resize
    if original_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Generate colored heatmap
    colored_heatmap = apply_colormap(heatmap, colormap_name)
    
    # Ensure original image is RGB
    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Alpha blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay


def visualize_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    save_path: Optional[str] = None,
    alpha: float = 0.4,
    colormap_name: str = 'jet',
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None,
    score: Optional[float] = None,
) -> plt.Figure:
    """Visualize heatmap and overlay results
    
    Generate three-column figure: Original | Heatmap | Overlay
    
    Args:
        heatmap: Heatmap [H, W]
        original_image: Original image [H, W, 3] RGB
        save_path: Save path (None to skip saving)
        alpha: Overlay transparency
        colormap_name: Colormap
        figsize: Figure size
        title: Title
        score: GradCAM score
    
    Returns:
        matplotlib Figure
    """
    overlay = overlay_heatmap(original_image, heatmap, alpha, colormap_name)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original H&E')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap=colormap_name)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    if title:
        suptitle = title
        if score is not None:
            suptitle += f' (Score: {score:.4f})'
        fig.suptitle(suptitle, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_multi_gene_heatmap(
    original_image: np.ndarray,
    gene_heatmaps: List[Tuple[str, np.ndarray, float]],
    save_path: Optional[str] = None,
    alpha: float = 0.4,
    colormap_name: str = 'jet',
    n_cols: int = 4,
) -> plt.Figure:
    """Visualize GradCAM heatmaps for multiple genes
    
    Args:
        original_image: Original H&E image
        gene_heatmaps: List of [(gene_name, heatmap, score), ...]
        save_path: Save path
        alpha: Overlay transparency
        colormap_name: Colormap
        n_cols: Number of columns per row
    
    Returns:
        matplotlib Figure
    """
    n_genes = len(gene_heatmaps)
    n_rows = (n_genes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (gene_name, heatmap, score) in enumerate(gene_heatmaps):
        overlay = overlay_heatmap(original_image, heatmap, alpha, colormap_name)
        axes[i].imshow(overlay)
        axes[i].set_title(f'{gene_name}\nScore: {score:.4f}', fontsize=9)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(n_genes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_heatmap_as_image(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    save_path: str,
    alpha: float = 0.4,
    add_colorbar: bool = True,
):
    """Save high-quality GradCAM overlay image
    
    Args:
        heatmap: Heatmap
        original_image: Original image
        save_path: Save path
        alpha: Overlay transparency
        add_colorbar: Whether to add colorbar
    """
    overlay = overlay_heatmap(original_image, heatmap, alpha)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(overlay)
    
    if add_colorbar:
        # Add a dummy colorbar to represent heatmap range
        sm = plt.cm.ScalarMappable(
            cmap=cm.get_cmap('jet'),
            norm=Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.set_label('Importance', fontsize=10)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ==================== Batch Processing Utilities ====================
def compute_batch_gradcam(
    gradcam: GradCAM_CarHE,
    image_tensors: torch.Tensor,
    spot_expressions: torch.Tensor,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch compute GradCAM heatmaps
    
    Args:
        gradcam: GradCAM_CarHE instance
        image_tensors: [N, 3, 256, 256] image batch
        spot_expressions: [N, n_genes] expression batch
        batch_size: Process batch size
    
    Returns:
        (heatmaps [N, 256, 256], scores [N])
    """
    n_samples = image_tensors.shape[0]
    heatmaps = []
    scores = []
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        for j in range(i, end):
            hm, sc = gradcam.compute(
                image_tensors[j:j+1],
                spot_expressions[j:j+1]
            )
            heatmaps.append(hm)
            scores.append(sc)
    
    return np.stack(heatmaps), np.array(scores)
