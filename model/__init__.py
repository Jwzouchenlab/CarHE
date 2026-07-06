# -*- coding: utf-8 -*-
"""
CarHE: Deep Learning Model for Predicting Spatial Transcriptomics from H&E Images

A CLIP-style contrastive learning framework that uses a HIPT ViT encoder
to predict spatial transcriptomic gene expression from H&E-stained tissue images.

Main Modules:
    - config: Global configuration
    - model: HIPT_CLIP_Model definition
    - get_HE_model: HIPT ViT encoder loader
    - Dataset: Training datasets (augmentDataset1)
    - get_Data: Data loaders (BRCA, DLPFC, CCRCC)
    - get_adata: h5ad data loader
    - train: Training script
    - inference: Inference & evaluation script
    - gradcam: GradCAM interpretability
    - evaluate: Comprehensive evaluation
    - utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "CarHE Team"
