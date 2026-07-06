#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CarHE Full Pipeline Script — Data Preprocessing → Training → Evaluation → GradCAM
=================================================================================

Usage (run from the CarHE/ root directory):

    # Check environment
    python run_pipeline.py --check

    # Step 1: Xenium data preprocessing
    python run_pipeline.py --step preprocess

    # Step 2: Train the model
    python run_pipeline.py --step train

    # Step 3: Evaluate the model
    python run_pipeline.py --step evaluate

    # Step 4: GradCAM visualization
    python run_pipeline.py --step gradcam

    # Run the entire pipeline in one go
    python run_pipeline.py --all

All paths are relative. Run from the CarHE/ root directory.
"""

import os
import sys
import subprocess
import argparse

# ==================== Path Configuration ====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR = os.path.join(ROOT_DIR, "data")


def run_cmd(cmd, cwd=MODEL_DIR, description=""):
    """Run a command and print its status"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  !! Step failed (return code {result.returncode})")
        return False
    print(f"  ✓ Step completed")
    return True


# ==================== Environment Check ====================
def step_check_environment():
    """Check that all dependencies and data are ready"""
    print("\n" + "="*60)
    print("  Environment Check")
    print("="*60)

    checks = []

    # Python packages
    packages = ["torch", "numpy", "pandas", "scanpy", "anndata", "cv2", "PIL",
                "tqdm", "tifffile", "matplotlib"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
            checks.append(True)
        except ImportError:
            print(f"  ✗ {pkg} (needs installation: pip install {pkg})")
            checks.append(False)

    # HIPT encoder
    hipt_path = os.path.join(ROOT_DIR, "img_encoder", "HIPT_4K")
    vit_path = os.path.join(hipt_path, "Checkpoints", "vit256_small_dino.pth")
    if os.path.exists(vit_path):
        print(f"  ✓ HIPT ViT-256 weights: {vit_path}")
        checks.append(True)
    else:
        print(f"  ✗ HIPT ViT-256 weights not found: {vit_path}")
        checks.append(False)

    # Xenium data
    he_image = os.path.join(DATA_DIR, "Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif")
    if os.path.exists(he_image):
        print(f"  ✓ H&E image: {he_image}")
        checks.append(True)
    else:
        print(f"  ✗ H&E image not found: {he_image}")
        checks.append(False)

    # Xenium raw data
    outs_dir = os.path.join(DATA_DIR, "Xenium_Prime_Human_Prostate_FFPE_outs")
    nucleus_bounds = os.path.join(outs_dir, "nucleus_boundaries.csv.gz")
    if os.path.exists(nucleus_bounds):
        print(f"  ✓ Xenium nucleus boundaries: {nucleus_bounds}")
        checks.append(True)
    else:
        print(f"  ✗ Xenium nucleus boundaries not found: {nucleus_bounds}")
        checks.append(False)

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ! CPU mode (training will be slower)")
        checks.append(True)
    except:
        checks.append(False)

    all_ok = all(checks)
    print(f"\n  Environment check result: {'All passed ✓' if all_ok else 'Issues found ✗'}")
    return all_ok


# ==================== Step 1: Preprocess ====================
def step_preprocess():
    """Xenium data preprocessing: run quick_xenium_to_h5ad.py"""
    script = os.path.join(DATA_DIR, "quick_xenium_to_h5ad.py")
    if not os.path.exists(script):
        print(f"Preprocessing script not found: {script}")
        return False
    return run_cmd(
        ["python", script],
        cwd=DATA_DIR,
        description="Step 1/4: Xenium data preprocessing (raw -> h5ad)"
    )


# ==================== Step 2: Train ====================
def step_train(epochs=50, batch_size=32, lr=1e-4):
    """Train the CarHE model"""
    exp_name = os.path.join(MODEL_DIR, "checkpoint", "xenium_model")
    # Using Xenium data; gene count is small (Xenium panel ≈ 400 genes)
    return run_cmd(
        ["python", "train.py",
         "--adata", os.path.join("..", "data", "xenium_prostate.h5ad"),
         "--batch_size", str(batch_size),
         "--lr", str(lr),
         "--epochs", str(epochs),
         "--exp_name", exp_name,
        ],
        cwd=MODEL_DIR,
        description="Step 2/4: Train CarHE model"
    )


# ==================== Step 3: Evaluate ====================
def step_evaluate(checkpoint=None):
    """Evaluate model performance"""
    if checkpoint is None:
        checkpoint = os.path.join(MODEL_DIR, "checkpoint", "xenium_model", "best_model.pt")
    output = os.path.join(MODEL_DIR, "evaluation_results", "xenium")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        print("Please run the training step first, or specify the --checkpoint argument")
        return False
    return run_cmd(
        ["python", "evaluate.py",
         "--adata", os.path.join("..", "data", "xenium_prostate.h5ad"),
         "--checkpoint", checkpoint,
         "--output", output,
         "--batch_size", "16",
        ],
        cwd=MODEL_DIR,
        description="Step 3/4: Evaluate model performance"
    )


# ==================== Step 4: GradCAM ====================
def step_gradcam(checkpoint=None, spot_idx=0):
    """GradCAM visualization"""
    if checkpoint is None:
        checkpoint = os.path.join(MODEL_DIR, "checkpoint", "xenium_model", "best_model.pt")
    output = os.path.join(MODEL_DIR, "gradcam_output")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        print("Please run the training step first, or specify the --checkpoint argument")
        return False
    return run_cmd(
        ["python", "run_gradcam.py",
         "--adata", os.path.join("..", "data", "xenium_prostate.h5ad"),
         "--checkpoint", checkpoint,
         "--spot_idx", str(spot_idx),
         "--output", output,
        ],
        cwd=MODEL_DIR,
        description="Step 4/4: GradCAM visualization"
    )


# ==================== Full Pipeline ====================
def run_all(args):
    """Run the full pipeline"""
    print("="*60)
    print("  CarHE Full Pipeline: Xenium Data → Train → Evaluate → GradCAM")
    print("="*60)

    steps = [
        ("Environment Check", lambda: step_check_environment()),
        ("Preprocessing", lambda: step_preprocess()),
        ("Training", lambda: step_train(args.epochs, args.batch_size, args.lr)),
        ("Evaluation", lambda: step_evaluate(args.checkpoint)),
        ("GradCAM", lambda: step_gradcam(args.checkpoint, args.spot_idx)),
    ]

    for name, func in steps:
        ok = func()
        if not ok and args.stop_on_error:
            print(f"\n  Failed at '{name}' step, stopping execution")
            break

    print("\n" + "="*60)
    print("  Full pipeline execution complete!")
    print("="*60)


# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CarHE full pipeline script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --check              # Check environment only
  python run_pipeline.py --step preprocess    # Run preprocessing only
  python run_pipeline.py --step train         # Train only
  python run_pipeline.py --step evaluate      # Evaluate only
  python run_pipeline.py --step gradcam       # GradCAM only
  python run_pipeline.py --all                # Run full pipeline
  python run_pipeline.py --all --epochs 100   # Full pipeline, 100 epochs
        """
    )

    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--step", type=str, default=None,
                        choices=["preprocess", "train", "evaluate", "gradcam"],
                        help="Run a single step")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="Stop on error")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Inference parameters
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: checkpoint/xenium_model/best_model.pt)")
    parser.add_argument("--spot_idx", type=int, default=0, help="Target cell index for GradCAM")

    args = parser.parse_args()

    if args.check:
        step_check_environment()
    elif args.step == "preprocess":
        step_preprocess()
    elif args.step == "train":
        step_train(args.epochs, args.batch_size, args.lr)
    elif args.step == "evaluate":
        step_evaluate(args.checkpoint)
    elif args.step == "gradcam":
        step_gradcam(args.checkpoint, args.spot_idx)
    elif args.all:
        run_all(args)
    else:
        parser.print_help()
