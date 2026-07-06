import os

class CFG:
    # ==================== Model Parameters ====================
    image_embedding = 384       # HIPT ViT-256 output dimension
    spot_embedding = 2000       # Gene expression embedding dimension (compatible with checkpoints)
    projection_dim = 384        # Projection head output dimension
    dropout = 0.1               # Dropout rate
    temperature = 0.07          # CLIP temperature parameter

    # ==================== Training Parameters ====================
    batch_size = 32
    augment_factor = 2
    learning_rate = 1e-4
    weight_decay = 0.002
    epochs = 100
    num_workers = 4

    # ==================== Device ====================
    device = "cuda"  # "cuda" or "cpu"

    # ==================== Path Configuration ====================
    # --- HIPT encoder paths (relative to model/ dir) ---
    hipt_module_path = "../img_encoder/HIPT_4K"
    hipt_vit256_checkpoint = "../img_encoder/HIPT_4K/Checkpoints/vit256_small_dino.pth"

    # --- Legacy dataset base path (not used for Xenium) ---
    data_base_dir = "../data"

    # BRCA/DLPFC/CCRCC dataset paths (set to None if not available)
    brca_image_dir = None
    brca_st_dir = None
    brca_intdata_dir = None

    dlpfc_image_dir = None
    dlpfc_st_dir = None
    dlpfc_intdata_dir = None

    ccrcc_base_dir = None
    ccrcc_image_dir = None
    ccrcc_st_dir = None
    ccrcc_sp_dir = None

    # Adata mode default h5ad path
    default_adata_path = "../data/xenium_prostate.h5ad"

    # ==================== Xenium Dataset Paths ====================
    xenium_he_image = "../data/Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif"
    xenium_matched_nuclei = "../data/data_processing/matched_nuclei_filtered.csv"
    xenium_cell_gene_matrix = "../data/data_processing/cell_gene_matrix_filtered.csv"
    xenium_seg_mask = "../data/data_processing/he_image_nuclei_seg_microns.tif"
    xenium_adata_path = "../data/xenium_prostate.h5ad"
    xenium_ngenes = None  # auto-detect

    # ==================== Output Paths ====================
    checkpoint_dir = "./checkpoint"
    log_dir = "./logs"

    # ==================== Initialize derived paths ====================
    @classmethod
    def init_paths(cls):
        """Initialize derived paths (used by BRCA/DLPFC/CCRCC datasets only)"""
        if cls.data_base_dir:
            if cls.brca_image_dir is None:
                cls.brca_image_dir = os.path.join(cls.data_base_dir, "BRCA", "image_rename")
            if cls.brca_st_dir is None:
                cls.brca_st_dir = os.path.join(cls.data_base_dir, "BRCA", "st")
            if cls.brca_intdata_dir is None:
                cls.brca_intdata_dir = os.path.join(cls.data_base_dir, "BRCA", "st", "intdata")

            if cls.dlpfc_image_dir is None:
                cls.dlpfc_image_dir = os.path.join(cls.data_base_dir, "DLPFC", "image")
            if cls.dlpfc_st_dir is None:
                cls.dlpfc_st_dir = os.path.join(cls.data_base_dir, "DLPFC")
            if cls.dlpfc_intdata_dir is None:
                cls.dlpfc_intdata_dir = os.path.join(cls.data_base_dir, "DLPFC", "intdata")

            if cls.ccrcc_image_dir is None:
                cls.ccrcc_image_dir = os.path.join(cls.data_base_dir, "CCRCC", "images")
            if cls.ccrcc_st_dir is None:
                cls.ccrcc_st_dir = os.path.join(cls.data_base_dir, "CCRCC", "st")
            if cls.ccrcc_sp_dir is None:
                cls.ccrcc_sp_dir = os.path.join(cls.data_base_dir, "CCRCC", "sp")
