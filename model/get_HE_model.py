# -*- coding: utf-8 -*-
import sys
import torch
import os

from config import CFG

def get_HE_encoder(model_256_path=None, device=None, freeze_backbone=True):
    """
    Load HIPT ViT-256 image encoder
    
    Parameters:
        model_256_path (str): ViT-256 pretrained weights path, defaults to CFG.hipt_vit256_checkpoint
        device (str): Device, defaults to 'cuda'
        freeze_backbone (bool): Whether to freeze backbone (only fine-tune last layers)
        
    Returns:
        model: HIPT ViT-256 model
    """
    if model_256_path is None:
        model_256_path = CFG.hipt_vit256_checkpoint
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Add HIPT module path
    if not os.path.exists(CFG.hipt_module_path):
        raise FileNotFoundError(
            f"HIPT module path not found: {CFG.hipt_module_path}\n"
            f"Please set CFG.hipt_module_path in config.py to the correct HIPT_4K directory"
        )
    sys.path.insert(0, CFG.hipt_module_path)
    
    from hipt_model_utils import get_vit256
    
    if not os.path.exists(model_256_path):
        raise FileNotFoundError(
            f"ViT-256 pretrained weights not found: {model_256_path}\n"
            f"Please set CFG.hipt_vit256_checkpoint in config.py to the correct weights file"
        )
    
    model = get_vit256(pretrained_weights=model_256_path, device=device)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze last two layers for fine-tuning
        for param in model.blocks[-2].parameters():
            param.requires_grad = True
        
        for param in model.norm.parameters():
            param.requires_grad = True
    
    print('model_HIPT_load_success!')
    return model
