#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CarHE 全流程运行脚本 — 从数据预处理到训练到推理
====================================================

使用方式 (在 CarHE/ 根目录下运行):

    # 检查环境
    python run_pipeline.py --check

    # 步骤 1: Xenium 数据预处理
    python run_pipeline.py --step preprocess

    # 步骤 2: 训练模型
    python run_pipeline.py --step train

    # 步骤 3: 评估模型
    python run_pipeline.py --step evaluate

    # 步骤 4: GradCAM 可视化
    python run_pipeline.py --step gradcam

    # 一键运行全流程
    python run_pipeline.py --all

所有路径均为相对路径，在 CarHE/ 根目录下运行即可。
"""

import os
import sys
import subprocess
import argparse

# ==================== 路径配置 ====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR = os.path.join(ROOT_DIR, "data")


def run_cmd(cmd, cwd=MODEL_DIR, description=""):
    """运行命令并打印状态"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  !! 步骤失败 (返回码 {result.returncode})")
        return False
    print(f"  ✓ 步骤完成")
    return True


# ==================== 环境检查 ====================
def step_check_environment():
    """检查所有依赖和数据是否就绪"""
    print("\n" + "="*60)
    print("  环境检查")
    print("="*60)

    checks = []

    # Python 包
    packages = ["torch", "numpy", "pandas", "scanpy", "anndata", "cv2", "PIL",
                "tqdm", "tifffile", "matplotlib"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
            checks.append(True)
        except ImportError:
            print(f"  ✗ {pkg} (需要安装: pip install {pkg})")
            checks.append(False)

    # HIPT encoder
    hipt_path = os.path.join(ROOT_DIR, "img_encoder", "HIPT_4K")
    vit_path = os.path.join(hipt_path, "Checkpoints", "vit256_small_dino.pth")
    if os.path.exists(vit_path):
        print(f"  ✓ HIPT ViT-256 权重: {vit_path}")
        checks.append(True)
    else:
        print(f"  ✗ HIPT ViT-256 权重不存在: {vit_path}")
        checks.append(False)

    # Xenium 数据
    he_image = os.path.join(DATA_DIR, "Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif")
    if os.path.exists(he_image):
        print(f"  ✓ H&E 图像: {he_image}")
        checks.append(True)
    else:
        print(f"  ✗ H&E 图像不存在: {he_image}")
        checks.append(False)

    # Xenium 原始数据
    outs_dir = os.path.join(DATA_DIR, "Xenium_Prime_Human_Prostate_FFPE_outs")
    nucleus_bounds = os.path.join(outs_dir, "nucleus_boundaries.csv.gz")
    if os.path.exists(nucleus_bounds):
        print(f"  ✓ Xenium 核边界: {nucleus_bounds}")
        checks.append(True)
    else:
        print(f"  ✗ Xenium 核边界不存在: {nucleus_bounds}")
        checks.append(False)

    # GPU 检查
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA GPU 可用: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ! CPU 模式 (训练会较慢)")
        checks.append(True)
    except:
        checks.append(False)

    all_ok = all(checks)
    print(f"\n  环境检查结果: {'全部通过 ✓' if all_ok else '有问题需要解决 ✗'}")
    return all_ok


# ==================== 步骤 1: 预处理 ====================
def step_preprocess():
    """Xenium 数据预处理"""
    script = os.path.join(DATA_DIR, "run_xenium_pipeline.py")
    if not os.path.exists(script):
        print(f"预处理脚本不存在: {script}")
        return False
    return run_cmd(
        ["python", script, "--all"],
        cwd=DATA_DIR,
        description="Step 1/4: Xenium 数据预处理"
    )


# ==================== 步骤 2: 训练 ====================
def step_train(epochs=50, batch_size=32, lr=1e-4):
    """训练 CarHE 模型"""
    exp_name = os.path.join(MODEL_DIR, "checkpoint", "xenium_model")
    # 使用 Xenium 数据，基因数设小（Xenium 面板约 400 基因）
    return run_cmd(
        ["python", "train.py",
         "--adata", os.path.join("..", "data", "xenium_prostate.h5ad"),
         "--batch_size", str(batch_size),
         "--lr", str(lr),
         "--epochs", str(epochs),
         "--exp_name", exp_name,
        ],
        cwd=MODEL_DIR,
        description="Step 2/4: 训练 CarHE 模型"
    )


# ==================== 步骤 3: 评估 ====================
def step_evaluate(checkpoint=None):
    """评估模型性能"""
    if checkpoint is None:
        checkpoint = os.path.join(MODEL_DIR, "checkpoint", "xenium_model", "best_model.pt")
    output = os.path.join(MODEL_DIR, "evaluation_results", "xenium")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint 不存在: {checkpoint}")
        print("请先运行训练步骤，或指定 --checkpoint 参数")
        return False
    return run_cmd(
        ["python", "evaluate.py",
         "--adata", os.path.join("..", "data", "xenium_prostate.h5ad"),
         "--checkpoint", checkpoint,
         "--output", output,
         "--batch_size", "16",
        ],
        cwd=MODEL_DIR,
        description="Step 3/4: 评估模型性能"
    )


# ==================== 步骤 4: GradCAM ====================
def step_gradcam(checkpoint=None, spot_idx=0):
    """GradCAM 可视化"""
    if checkpoint is None:
        checkpoint = os.path.join(MODEL_DIR, "checkpoint", "xenium_model", "best_model.pt")
    output = os.path.join(MODEL_DIR, "gradcam_output")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint 不存在: {checkpoint}")
        print("请先运行训练步骤，或指定 --checkpoint 参数")
        return False
    return run_cmd(
        ["python", "run_gradcam.py",
         "--adata", os.path.join("..", "data", "xenium_prostate.h5ad"),
         "--checkpoint", checkpoint,
         "--spot_idx", str(spot_idx),
         "--output", output,
        ],
        cwd=MODEL_DIR,
        description="Step 4/4: GradCAM 可视化"
    )


# ==================== 全流程 ====================
def run_all(args):
    """运行全流程"""
    print("="*60)
    print("  CarHE 全流程: Xenium 数据 → 训练 → 评估 → GradCAM")
    print("="*60)

    steps = [
        ("环境检查", lambda: step_check_environment()),
        ("预处理", lambda: step_preprocess()),
        ("训练", lambda: step_train(args.epochs, args.batch_size, args.lr)),
        ("评估", lambda: step_evaluate(args.checkpoint)),
        ("GradCAM", lambda: step_gradcam(args.checkpoint, args.spot_idx)),
    ]

    for name, func in steps:
        ok = func()
        if not ok and args.stop_on_error:
            print(f"\n  在「{name}」步骤失败，停止执行")
            break

    print("\n" + "="*60)
    print("  全流程执行完毕!")
    print("="*60)


# ==================== 入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CarHE 全流程运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py --check              # 只检查环境
  python run_pipeline.py --step preprocess    # 只运行预处理
  python run_pipeline.py --step train         # 只训练
  python run_pipeline.py --step evaluate      # 只评估
  python run_pipeline.py --step gradcam       # 只 GradCAM
  python run_pipeline.py --all                # 运行全流程
  python run_pipeline.py --all --epochs 100   # 全流程，100 epoch
        """
    )

    parser.add_argument("--check", action="store_true", help="环境检查")
    parser.add_argument("--step", type=str, default=None,
                        choices=["preprocess", "train", "evaluate", "gradcam"],
                        help="运行单个步骤")
    parser.add_argument("--all", action="store_true", help="运行全流程")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="遇到错误时停止")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")

    # 推理参数
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint 路径 (默认: checkpoint/xenium_model/best_model.pt)")
    parser.add_argument("--spot_idx", type=int, default=0, help="GradCAM 目标细胞索引")

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
