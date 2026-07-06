# -*- coding: utf-8 -*-
"""
CarHE: Cross-modal Alignment of Histology and Expression

A CLIP-style contrastive learning framework that maps H&E-stained tissue images
to spatial transcriptomic gene expression using a HIPT ViT encoder.

Main Modules:
    - config: Global configuration
    - model: HIPT_CLIP_Model definition
    - get_HE_model: HIPT ViT encoder loader
    - get_adata: AnnData data loader (unified entry point)
    - train: Training script
    - inference: Inference and evaluation script
    - gradcam: GradCAM interpretability
    - evaluate: Comprehensive evaluation metrics
    - utils: Utility functions
"""

__version__ = "1.0.0"
