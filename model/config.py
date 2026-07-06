import os

class CFG:
    # ==================== Model Parameters ====================
    image_embedding = 384       # HIPT ViT-256 output dimension
    spot_embedding = 2000       # Gene expression embedding dimension
    projection_dim = 384        # Projection head output dimension
    dropout = 0.1
    temperature = 0.07          # CLIP temperature parameter

    # ==================== Training Parameters ====================
    batch_size = 32
    augment_factor = 2
    learning_rate = 1e-4
    weight_decay = 0.002
    epochs = 100
    num_workers = 4

    # ==================== Device ====================
    device = "cuda"

    # ==================== HIPT Encoder ====================
    hipt_module_path = "../img_encoder/HIPT_4K"
    hipt_vit256_checkpoint = "../img_encoder/HIPT_4K/Checkpoints/vit256_small_dino.pth"

    # ==================== Default Data Path ====================
    default_adata_path = "../data/xenium_prostate.h5ad"

    # ==================== Output Paths ====================
    checkpoint_dir = "./checkpoint"
    log_dir = "./logs"
