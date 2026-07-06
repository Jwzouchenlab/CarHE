# CarHE 完整使用教程

## 从 Xenium 数据到基因表达预测的全流程

---

## 项目结构

```
CarHE/                              # 项目根目录
├── run_pipeline.py                # ★ 一键运行脚本
├── TUTORIAL.md                    # ★ 本文档
│
├── img_encoder/                   # HIPT 图像编码器
│   └── HIPT_4K/
│       ├── Checkpoints/
│       │   └── vit256_small_dino.pth   # ViT-256 预训练权重
│       ├── hipt_model_utils.py
│       └── vision_transformer.py
│
├── model/                         # 核心模型代码
│   ├── config.py                  # 配置文件（所有路径在这里改）
│   ├── model.py                   # 模型定义（HIPT + ProjectionHead + CLIP）
│   ├── train.py                   # 训练入口
│   ├── inference.py               # 推理入口
│   ├── evaluate.py                # 评估入口
│   ├── gradcam.py                # GradCAM 可视化
│   ├── run_gradcam.py            # GradCAM 运行脚本
│   ├── get_xenium.py             # Xenium 数据加载器
│   ├── get_HE_model.py           # HIPT 编码器加载
│   ├── Dataset.py                # 通用数据集类
│   └── checkpoint/               # 模型保存目录
│
└── data/                          # 数据目录
    ├── run_xenium_pipeline.py    # Xenium 预处理脚本
    ├── *.py                      # 预处理脚本 (1-4)
    ├── Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif  # H&E 图像
    ├── Xenium_Prime_Human_Prostate_FFPE_he_imagealignment.csv
    └── Xenium_Prime_Human_Prostate_FFPE_outs/             # Xenium 原始输出
```

---

## 第一步：环境检查

```bash
# 在 CarHE 目录下运行
python run_pipeline.py --check
```

这会检查：
- PyTorch, numpy, pandas, scanpy, opencv 等 Python 包
- HIPT ViT-256 预训练权重
- Xenium H&E 图像
- Xenium 原始数据
- GPU 是否可用

---

## 第二步：Xenium 数据预处理

```bash
# 方式 A: 使用一键脚本
python run_pipeline.py --step preprocess

# 方式 B: 直接运行预处理脚本
cd data
python run_xenium_pipeline.py --all
```

预处理步骤：
1. **坐标对齐** — Xenium 微米坐标 → H&E 像素坐标
2. **基因矩阵提取** — 从 cell_feature_matrix 提取并过滤
3. **核分割 mask** — 生成 Xenium 核分割图像
4. **细胞匹配** — Xenium 核 ↔ H&E 核一一对应

输出文件（在 `data/data_processing/` 下）：
- `matched_nuclei_filtered.csv` — 匹配后的细胞核对应表
- `cell_gene_matrix_filtered.csv` — 过滤后的基因表达矩阵
- `xenium_nuclei.tif` — Xenium 核分割 mask
- `he_image_nuclei_seg_microns.tif` — H&E 核分割 mask（需要 HoverNet）

---

## 第三步：训练模型

```bash
# 方式 A: 使用一键脚本（默认 50 epoch）
python run_pipeline.py --step train

# 方式 B: 自定义参数
cd model
python train.py \
    --dataset xenium \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 50 \
    --exp_name ./checkpoint/xenium_model
```

训练参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | xenium | 数据集类型 |
| `--batch_size` | 32 | 批大小（GPU 显存不足可改小） |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 50 | 训练轮数 |
| `--exp_name` | checkpoint/xenium_model | 模型保存目录 |

模型保存位置：
- `checkpoint/xenium_model/best_model.pt` — 验证集最优模型
- `checkpoint/xenium_model/model.pt` — 最终模型

---

## 第四步：评估模型

```bash
# 方式 A: 使用一键脚本
python run_pipeline.py --step evaluate

# 方式 B: 指定 checkpoint
cd model
python evaluate.py \
    --dataset xenium \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --output ./evaluation_results/xenium
```

评估指标（输出在 `evaluation_results/xenium/`）：
- **Gene-level PCC** — 每个基因的 Pearson 相关性
- **Gene-level Spearman** — 每个基因的 Spearman 相关性
- **Spot-level PCC/MSE/MAE** — 每个细胞的预测精度
- **Cosine Similarity** — 嵌入空间的对角线/非对角线余弦相似度
- **Top-K Retrieval** — 图像→基因、基因→图像的检索准确率

---

## 第五步：GradCAM 可视化

```bash
# 方式 A: 使用一键脚本（默认第 0 个细胞）
python run_pipeline.py --step gradcam

# 方式 B: 指定细胞
cd model
python run_gradcam.py \
    --dataset xenium \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --spot_idx 0 \
    --output ./gradcam_output

# 多细胞的 GradCAM
python run_gradcam.py \
    --dataset xenium \
    --checkpoint ./checkpoint/xenium_model/best_model.pt \
    --spot_idx 0 \
    --multi_layer \
    --output ./gradcam_output
```

输出文件（在 `gradcam_output/` 下）：
- `gradcam_X.png` — 细胞 X 的 GradCAM 三连图（原图 + 热图 + 叠加）

---

## 全流程一键运行

```bash
# 环境检查 + 预处理 + 训练 + 评估 + GradCAM
python run_pipeline.py --all

# 自定义训练参数
python run_pipeline.py --all --epochs 100 --batch_size 64

# 停止在第一个错误
python run_pipeline.py --all --stop_on_error
```

---

## 环境依赖

```bash
pip install torch torchvision numpy pandas scanpy anndata opencv-python \
    Pillow tqdm tifffile matplotlib scikit-learn scipy seaborn \
    python-docx zarr
```

---

## 常见问题

### Q: 提示 "HIPT module path not found"
确保 `img_encoder/HIPT_4K/` 目录存在且包含 `hipt_model_utils.py`。

### Q: GPU 显存不足
- 减小 `--batch_size`（如 16 或 8）
- 或使用 `--device cpu`（较慢）

### Q: Xenium 基因数与模型不匹配
Xenium 通常有约 400 个基因，不足 2000 时会自动补零。如需修改，编辑 `config.py` 中的 `xenium_ngenes`。

### Q: 如何用其他 checkpoints？
```bash
python run_pipeline.py --step evaluate --checkpoint ./checkpoint/Final_scgpt/model.pt
```

---

## 所有可用的路径配置 (config.py)

| 变量 | 当前值 | 说明 |
|------|--------|------|
| `hipt_module_path` | `../img_encoder/HIPT_4K` | HIPT 模块路径 |
| `hipt_vit256_checkpoint` | `../img_encoder/HIPT_4K/Checkpoints/vit256_small_dino.pth` | ViT-256 权重 |
| `xenium_he_image` | `../data/Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif` | H&E 图像 |
| `xenium_matched_nuclei` | `../data/data_processing/matched_nuclei_filtered.csv` | 匹配细胞 |
| `xenium_cell_gene_matrix` | `../data/data_processing/cell_gene_matrix_filtered.csv` | 基因矩阵 |
| `xenium_seg_mask` | `../data/data_processing/he_image_nuclei_seg_microns.tif` | 核分割 |
| `checkpoint_dir` | `./checkpoint` | 模型保存目录 |
| `log_dir` | `./logs` | 日志目录 |
