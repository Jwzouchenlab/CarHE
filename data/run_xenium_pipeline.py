# -*- coding: utf-8 -*-
"""
Xenium Data Preprocessing Unified Pipeline
==========================================

One-click execution of all preprocessing steps, converting Xenium raw data 
into CarHE-compatible format.

Pipeline steps:
    Step 1: Coordinate alignment (Xenium microns → H&E pixels)
    Step 2: Extract gene expression matrix
    Step 3: Generate Xenium nuclei segmentation mask
    Step 4: H&E nuclei segmentation (requires Hover-Net, optional)
    Step 5: Cell matching (Xenium ↔ H&E)

Usage:
    # Run all steps
    python run_xenium_pipeline.py --all

    # Run specific steps only
    python run_xenium_pipeline.py --step 1,2

    # Skip HoverNet (use pre-generated segmentation)
    python run_xenium_pipeline.py --all --skip_hovernet
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

# ==================== Configuration ====================
class PipelineConfig:
    """Pipeline path configuration"""
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.outs_dir = os.path.join(data_dir, "Xenium_Prime_Human_Prostate_FFPE_outs")
        self.output_dir = os.path.join(data_dir, "data_processing")
        self.scripts_dir = data_dir  # directory containing preprocessing scripts
        
        # Input files
        self.he_image = os.path.join(data_dir, "Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif")
        self.alignment_csv = os.path.join(data_dir, "Xenium_Prime_Human_Prostate_FFPE_he_imagealignment.csv")
        self.nucleus_boundaries = os.path.join(self.outs_dir, "nucleus_boundaries.csv.gz")
        self.cell_feature_matrix = os.path.join(self.outs_dir, "cell_feature_matrix")
        
        # Output files
        os.makedirs(self.output_dir, exist_ok=True)
        self.nucleus_boundaries_HE = os.path.join(self.output_dir, "nucleus_boundaries_HE_coords.csv.gz")
        self.cell_gene_matrix = os.path.join(self.output_dir, "cell_gene_matrix.csv")
        self.xenium_nuclei_seg = os.path.join(self.output_dir, "xenium_nuclei.tif")
        self.cell_ids_dict = os.path.join(self.output_dir, "xenium_cell_ids_dict.csv")
        self.he_nuclei_seg = os.path.join(self.output_dir, "he_image_nuclei_seg_microns.tif")
        self.matched_nuclei = os.path.join(self.output_dir, "matched_nuclei_filtered.csv")
        self.cell_gene_matrix_filtered = os.path.join(self.output_dir, "cell_gene_matrix_filtered.csv")


def run_cmd(cmd, cwd=None):
    """Run a command and check return code"""
    print(f"\n{'='*60}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Warning: command returned {result.returncode}")
    return result.returncode


# ==================== Step 1: Coordinate Alignment ====================
def step1_align_coordinates(cfg: PipelineConfig):
    """Convert Xenium nucleus boundary coordinates from microns to H&E pixel coordinates
    
    Uses alignment matrix: M_inv @ [x_micron/0.2125, y_micron/0.2125, 1]^T
    """
    print("\n" + "="*60)
    print("Step 1: Coordinate Alignment (Xenium microns → H&E pixels)")
    print("="*60)
    
    # Read alignment matrix
    M = pd.read_csv(cfg.alignment_csv, header=None).values
    M_inv = np.linalg.inv(M)
    
    # Read nucleus boundaries
    print(f"Reading nucleus boundaries: {cfg.nucleus_boundaries}")
    boundaries = pd.read_csv(cfg.nucleus_boundaries, compression='gzip')
    print(f"  Original coordinate range: x=[{boundaries['vertex_x'].min():.1f}, {boundaries['vertex_x'].max():.1f}] μm")
    
    # Convert: microns → Xenium pixels → H&E pixels
    x_xe_px = boundaries['vertex_x'] / 0.2125
    y_xe_px = boundaries['vertex_y'] / 0.2125
    
    xe_homogeneous = np.column_stack([x_xe_px, y_xe_px, np.ones(len(boundaries))])
    he_pixels = (M_inv @ xe_homogeneous.T).T
    
    # Update DataFrame
    output_df = boundaries.copy()
    output_df['vertex_x'] = he_pixels[:, 0].round(0).astype(int)
    output_df['vertex_y'] = he_pixels[:, 1].round(0).astype(int)
    
    # Remove unnecessary columns
    for col in ['label_id']:
        if col in output_df.columns:
            output_df = output_df.drop(columns=[col])
    
    # Save
    print(f"Saving: {cfg.nucleus_boundaries_HE}")
    output_df.to_csv(cfg.nucleus_boundaries_HE, index=False, compression='gzip')
    print(f"  H&E pixel coordinate range: x=[{output_df['vertex_x'].min()}, {output_df['vertex_x'].max()}]")
    print("Step 1 complete ✓")


# ==================== Step 2: Extract Gene Expression Matrix ====================
def step2_gene_matrix(cfg: PipelineConfig):
    """Extract and filter gene expression matrix from cell_feature_matrix"""
    print("\n" + "="*60)
    print("Step 2: Extract Gene Expression Matrix")
    print("="*60)
    
    script = os.path.join(cfg.scripts_dir, "2_get_xenium_cell_gene_matrix.py")
    cmd = [
        "python", script,
        f"--dir_feature_matrix={cfg.cell_feature_matrix}",
        f"--dir_output={cfg.output_dir}",
        f"--fp_out_matrix={os.path.basename(cfg.cell_gene_matrix)}",
        "--del_intm_files=True",
    ]
    run_cmd(cmd, cwd=cfg.data_dir)
    print("Step 2 complete ✓")


# ==================== Step 3: Xenium Nuclei Segmentation Mask ====================
def step3_xenium_nuclei_seg(cfg: PipelineConfig):
    """Generate nuclei segmentation mask from Xenium nucleus boundaries"""
    print("\n" + "="*60)
    print("Step 3: Generate Xenium Nuclei Segmentation Mask")
    print("="*60)
    
    script = os.path.join(cfg.scripts_dir, "1_get_xenium_nuclei_seg_image.py")
    cmd = [
        "python", script,
        f"--fp_boundaries={cfg.nucleus_boundaries_HE}",
        f"--fp_he_img={cfg.he_image}",
        f"--fp_out_nuclei_seg={os.path.basename(cfg.xenium_nuclei_seg)}",
        f"--dir_output={cfg.output_dir}",
        "--del_intm_files=True",
    ]
    run_cmd(cmd, cwd=cfg.data_dir)
    print("Step 3 complete ✓")


# ==================== Step 5: Cell Matching ====================
def step5_cell_matching(cfg: PipelineConfig):
    """Match Xenium cells with H&E segmented nuclei"""
    print("\n" + "="*60)
    print("Step 5: Cell Matching (Xenium ↔ H&E)")
    print("="*60)
    
    script = os.path.join(cfg.scripts_dir, "4_get_corresponding_cells.py")
    cmd = [
        "python", script,
        f"--fp_seg_hist={os.path.basename(cfg.he_nuclei_seg)}",
        f"--fp_seg_xenium={os.path.basename(cfg.xenium_nuclei_seg)}",
        f"--fp_cgm={os.path.basename(cfg.cell_gene_matrix)}",
        f"--dir_output={cfg.output_dir}",
    ]
    run_cmd(cmd, cwd=cfg.data_dir)
    print("Step 5 complete ✓")


# ==================== Main Entry Point ====================
def main():
    parser = argparse.ArgumentParser(
        description="Xenium Data Preprocessing Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_xenium_pipeline.py --all          # Run all preprocessing
  python run_xenium_pipeline.py --step 1,2     # Run coordinate alignment and gene matrix only
  python run_xenium_pipeline.py --all --skip_hovernet  # Skip HoverNet
        """
    )
    parser.add_argument("--data_dir", default=".",
                        help="Data directory path (contains Xenium output and scripts)")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps")
    parser.add_argument("--step", type=str, default=None,
                        help="Specify steps, comma-separated (e.g., '1,2,5')")
    parser.add_argument("--skip_hovernet", action="store_true",
                        help="Skip HoverNet step (Step 4)")
    
    args = parser.parse_args()
    cfg = PipelineConfig(args.data_dir)
    
    # Determine which steps to run
    if args.all:
        steps = [1, 2, 3, 5]  # HoverNet (Step 4) not run by default
        if not args.skip_hovernet:
            print("Note: Step 4 (HoverNet) requires a configured HoverNet environment, skipped by default.")
            print("To run, please manually execute: python 3.1_segment_nuclei_he_image.py --step 1")
    elif args.step:
        steps = [int(s.strip()) for s in args.step.split(",")]
    else:
        parser.print_help()
        return
    
    # Validate prerequisites
    for s in steps:
        if s == 1:
            assert os.path.exists(cfg.nucleus_boundaries), f"Nucleus boundary file not found: {cfg.nucleus_boundaries}"
            assert os.path.exists(cfg.alignment_csv), f"Alignment matrix not found: {cfg.alignment_csv}"
        elif s == 2:
            assert os.path.exists(cfg.cell_feature_matrix), f"Cell feature matrix directory not found: {cfg.cell_feature_matrix}"
        elif s == 3:
            assert os.path.exists(cfg.he_image), f"H&E image not found: {cfg.he_image}"
            assert os.path.exists(cfg.nucleus_boundaries_HE), f"Transformed nucleus boundaries not found: {cfg.nucleus_boundaries_HE}. Please run Step 1 first"
        elif s == 5:
            assert os.path.exists(cfg.xenium_nuclei_seg), f"Xenium nuclei segmentation not found: {cfg.xenium_nuclei_seg}. Please run Step 3 first"
            assert os.path.exists(cfg.cell_gene_matrix), f"Gene expression matrix not found: {cfg.cell_gene_matrix}. Please run Step 2 first"
    
    # Execute steps
    step_funcs = {
        1: step1_align_coordinates,
        2: step2_gene_matrix,
        3: step3_xenium_nuclei_seg,
        5: step5_cell_matching,
    }
    
    for s in steps:
        if s in step_funcs:
            step_funcs[s](cfg)
        else:
            print(f"Unknown step: {s} (available steps: 1, 2, 3, 5)")
    
    print("\n" + "="*60)
    print("Pipeline execution complete!")
    print(f"Output directory: {cfg.output_dir}")
    print("="*60)
    
    # Check final outputs
    final_files = [
        ("Gene expression matrix", cfg.cell_gene_matrix_filtered),
        ("Matched nuclei", cfg.matched_nuclei),
    ]
    for name, path in final_files:
        status = "✓" if os.path.exists(path) else "✗ (not yet generated)"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
