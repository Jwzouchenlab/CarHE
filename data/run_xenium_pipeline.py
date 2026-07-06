# -*- coding: utf-8 -*-
"""
Xenium 数据预处理统一管线
=========================

一键运行所有预处理步骤，将 Xenium 原始数据转换为 CarHE 可用的格式。

管线步骤：
    Step 1: 坐标对齐（Xenium 微米 → H&E 像素）
    Step 2: 提取基因表达矩阵
    Step 3: Xenium 核分割 mask 生成
    Step 4: H&E 核分割（需要 Hover-Net，可选）
    Step 5: 细胞匹配（Xenium ↔ H&E）

Usage:
    # 运行所有步骤
    python run_xenium_pipeline.py --all

    # 只运行特定步骤
    python run_xenium_pipeline.py --step 1,2

    # 跳过 HoverNet（使用预先生成的分割）
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

# ==================== 配置 ====================
class PipelineConfig:
    """管线路径配置"""
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.outs_dir = os.path.join(data_dir, "Xenium_Prime_Human_Prostate_FFPE_outs")
        self.output_dir = os.path.join(data_dir, "data_processing")
        self.scripts_dir = data_dir  # 预处理脚本所在目录
        
        # 输入文件
        self.he_image = os.path.join(data_dir, "Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif")
        self.alignment_csv = os.path.join(data_dir, "Xenium_Prime_Human_Prostate_FFPE_he_imagealignment.csv")
        self.nucleus_boundaries = os.path.join(self.outs_dir, "nucleus_boundaries.csv.gz")
        self.cell_feature_matrix = os.path.join(self.outs_dir, "cell_feature_matrix")
        
        # 输出文件
        os.makedirs(self.output_dir, exist_ok=True)
        self.nucleus_boundaries_HE = os.path.join(self.output_dir, "nucleus_boundaries_HE_coords.csv.gz")
        self.cell_gene_matrix = os.path.join(self.output_dir, "cell_gene_matrix.csv")
        self.xenium_nuclei_seg = os.path.join(self.output_dir, "xenium_nuclei.tif")
        self.cell_ids_dict = os.path.join(self.output_dir, "xenium_cell_ids_dict.csv")
        self.he_nuclei_seg = os.path.join(self.output_dir, "he_image_nuclei_seg_microns.tif")
        self.matched_nuclei = os.path.join(self.output_dir, "matched_nuclei_filtered.csv")
        self.cell_gene_matrix_filtered = os.path.join(self.output_dir, "cell_gene_matrix_filtered.csv")


def run_cmd(cmd, cwd=None):
    """运行命令并检查返回值"""
    print(f"\n{'='*60}")
    print(f"执行: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"警告: 命令返回 {result.returncode}")
    return result.returncode


# ==================== Step 1: 坐标对齐 ====================
def step1_align_coordinates(cfg: PipelineConfig):
    """将 Xenium 核边界坐标从微米转换为 H&E 像素坐标
    
    使用 alignment matrix: M_inv @ [x_micron/0.2125, y_micron/0.2125, 1]^T
    """
    print("\n" + "="*60)
    print("Step 1: 坐标对齐 (Xenium 微米 → H&E 像素)")
    print("="*60)
    
    # 读取对齐矩阵
    M = pd.read_csv(cfg.alignment_csv, header=None).values
    M_inv = np.linalg.inv(M)
    
    # 读取核边界
    print(f"读取核边界: {cfg.nucleus_boundaries}")
    boundaries = pd.read_csv(cfg.nucleus_boundaries, compression='gzip')
    print(f"  原始坐标范围: x=[{boundaries['vertex_x'].min():.1f}, {boundaries['vertex_x'].max():.1f}] μm")
    
    # 转换：微米 → Xenium 像素 → H&E 像素
    x_xe_px = boundaries['vertex_x'] / 0.2125
    y_xe_px = boundaries['vertex_y'] / 0.2125
    
    xe_homogeneous = np.column_stack([x_xe_px, y_xe_px, np.ones(len(boundaries))])
    he_pixels = (M_inv @ xe_homogeneous.T).T
    
    # 更新 DataFrame
    output_df = boundaries.copy()
    output_df['vertex_x'] = he_pixels[:, 0].round(0).astype(int)
    output_df['vertex_y'] = he_pixels[:, 1].round(0).astype(int)
    
    # 删除不需要的列
    for col in ['label_id']:
        if col in output_df.columns:
            output_df = output_df.drop(columns=[col])
    
    # 保存
    print(f"保存: {cfg.nucleus_boundaries_HE}")
    output_df.to_csv(cfg.nucleus_boundaries_HE, index=False, compression='gzip')
    print(f"  H&E 像素坐标范围: x=[{output_df['vertex_x'].min()}, {output_df['vertex_x'].max()}]")
    print("Step 1 完成 ✓")


# ==================== Step 2: 提取基因表达矩阵 ====================
def step2_gene_matrix(cfg: PipelineConfig):
    """从 cell_feature_matrix 提取并过滤基因表达矩阵"""
    print("\n" + "="*60)
    print("Step 2: 提取基因表达矩阵")
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
    print("Step 2 完成 ✓")


# ==================== Step 3: Xenium 核分割 mask ====================
def step3_xenium_nuclei_seg(cfg: PipelineConfig):
    """从 Xenium 核边界生成核分割 mask"""
    print("\n" + "="*60)
    print("Step 3: 生成 Xenium 核分割 mask")
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
    print("Step 3 完成 ✓")


# ==================== Step 5: 细胞匹配 ====================
def step5_cell_matching(cfg: PipelineConfig):
    """将 Xenium 细胞与 H&E 分割核匹配"""
    print("\n" + "="*60)
    print("Step 5: 细胞匹配 (Xenium ↔ H&E)")
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
    print("Step 5 完成 ✓")


# ==================== 主入口 ====================
def main():
    parser = argparse.ArgumentParser(
        description="Xenium 数据预处理统一管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_xenium_pipeline.py --all          # 运行全部预处理
  python run_xenium_pipeline.py --step 1,2     # 只运行坐标对齐和基因矩阵
  python run_xenium_pipeline.py --all --skip_hovernet  # 跳过 HoverNet
        """
    )
    parser.add_argument("--data_dir", default=".",
                        help="数据目录路径（包含 Xenium 输出和脚本）")
    parser.add_argument("--all", action="store_true",
                        help="运行所有步骤")
    parser.add_argument("--step", type=str, default=None,
                        help="指定步骤，逗号分隔 (如 '1,2,5')")
    parser.add_argument("--skip_hovernet", action="store_true",
                        help="跳过 HoverNet 步骤（Step 4）")
    
    args = parser.parse_args()
    cfg = PipelineConfig(args.data_dir)
    
    # 确定要运行的步骤
    if args.all:
        steps = [1, 2, 3, 5]  # 默认不运行 HoverNet (Step 4)
        if not args.skip_hovernet:
            print("注意: Step 4 (HoverNet) 需要配置 HoverNet 环境，默认跳过。")
            print("如需运行，请手动执行：python 3.1_segment_nuclei_he_image.py --step 1")
    elif args.step:
        steps = [int(s.strip()) for s in args.step.split(",")]
    else:
        parser.print_help()
        return
    
    # 验证前置条件
    for s in steps:
        if s == 1:
            assert os.path.exists(cfg.nucleus_boundaries), f"核边界文件不存在: {cfg.nucleus_boundaries}"
            assert os.path.exists(cfg.alignment_csv), f"对齐矩阵不存在: {cfg.alignment_csv}"
        elif s == 2:
            assert os.path.exists(cfg.cell_feature_matrix), f"细胞特征矩阵目录不存在: {cfg.cell_feature_matrix}"
        elif s == 3:
            assert os.path.exists(cfg.he_image), f"H&E 图像不存在: {cfg.he_image}"
            assert os.path.exists(cfg.nucleus_boundaries_HE), f"转换后的核边界不存在: {cfg.nucleus_boundaries_HE}。请先运行 Step 1"
        elif s == 5:
            assert os.path.exists(cfg.xenium_nuclei_seg), f"Xenium 核分割不存在: {cfg.xenium_nuclei_seg}。请先运行 Step 3"
            assert os.path.exists(cfg.cell_gene_matrix), f"基因表达矩阵不存在: {cfg.cell_gene_matrix}。请先运行 Step 2"
    
    # 执行步骤
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
            print(f"未知步骤: {s}（可用步骤: 1,2,3,5）")
    
    print("\n" + "="*60)
    print("管线执行完毕!")
    print(f"输出目录: {cfg.output_dir}")
    print("="*60)
    
    # 检查最终输出
    final_files = [
        ("基因表达矩阵", cfg.cell_gene_matrix_filtered),
        ("匹配细胞核", cfg.matched_nuclei),
    ]
    for name, path in final_files:
        status = "✓" if os.path.exists(path) else "✗ (尚未生成)"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
