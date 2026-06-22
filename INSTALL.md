# MAISI (NV-Generate-CTMR) 安装与使用指南

## 系统要求

- Python 3.10+
- CUDA 11.8+ (GPU 显存 ≥ 16GB 推荐)
- 系统内存 ≥ 32GB
- 磁盘空间 ≥ 20GB

## 安装步骤

### 1. Clone 代码

```bash
git clone https://github.com/<your-username>/NV-Generate-CTMR.git
cd NV-Generate-CTMR
```

### 2. 下载大文件（从 GitHub Release）

前往 GitHub Release 页面，下载以下附件并解压到项目根目录：

| 文件 | 解压目标位置 | 说明 |
|------|-------------|------|
| `maisi_weights.zip` | `models/` | 5 个模型权重文件 (~3.2GB) |
| `maisi_masks_part1.zip` | `datasets/` | 候选 mask 数据库 第1部分 |
| `maisi_masks_part2.zip` | `datasets/` | 候选 mask 数据库 第2部分 |
| `maisi_masks_part3.zip` | `datasets/` | 候选 mask 数据库 第3部分 |
| `maisi_masks_part4.zip` | `datasets/` | 候选 mask 数据库 第4部分 |

解压后目录结构应为：
```
NV-Generate-CTMR/
├── models/
│   ├── autoencoder_v1.pt            (80MB)
│   ├── diff_unet_3d_rflow-ct.pt     (2.1GB)
│   ├── controlnet_3d_rflow-ct.pt    (275MB)
│   ├── mask_generation_autoencoder.pt (20MB)
│   └── mask_generation_diffusion_unet.pt (752MB)
├── datasets/
│   ├── all_anatomy_size_conditions.json
│   ├── candidate_masks_flexible_size_and_spacing_4000.json
│   └── all_masks_flexible_size_and_spacing_4000/
│       ├── AbdomenCT-1K/
│       ├── AMOS22/
│       ├── ... (21 个子数据集)
```

### 3. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate       # Windows

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 额外依赖（用于肿瘤生成管线，可选）
pip install SimpleITK python-docx
```

### 4. 设置环境变量

```bash
# Linux/Mac
export MONAI_DATA_DIRECTORY=<项目绝对路径>

# Windows
set MONAI_DATA_DIRECTORY=<项目绝对路径>
```

### 5. 验证安装

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI version: {monai.__version__}')"
python -m scripts.infer_image_from_mask --help
```

## 快速开始

### 生成单张合成 CT

```bash
python -m scripts.infer_image_from_mask \
    --output_size 256 256 128 \
    --spacing 1.7 1.7 2.0 \
    --body_region chest abdomen \
    --anatomy_list liver hepatic tumor \
    --num_output_samples 1 \
    --device cuda
```

### 使用肿瘤生成管线（需额外下载 DiffTumor）

如果要使用 MAISI + DiffTumor 肿瘤生成管线，需额外：

1. 下载 DiffTumor 仓库：
```bash
git clone https://github.com/MrGiovanni/DiffTumor.git
```

2. 修改 `configs/tumor_paths.json`，将路径指向你本地的 DiffTumor 目录

3. 运行肿瘤生成：
```bash
python -m scripts.tumor_prompt_runner --quick --organ liver --size medium
```

## 模型权重文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `autoencoder_v1.pt` | 80MB | VQGAN 3D 自编码器 |
| `diff_unet_3d_rflow-ct.pt` | 2.1GB | Rectified Flow 扩散 UNet (主模型) |
| `controlnet_3d_rflow-ct.pt` | 275MB | ControlNet 条件控制网络 |
| `mask_generation_autoencoder.pt` | 20MB | Mask 生成自编码器 |
| `mask_generation_diffusion_unet.pt` | 752MB | Mask 生成扩散 UNet |

## 常见问题

### Q: 模型权重从哪里下载？
A: 从本仓库的 GitHub Release 页面下载 `maisi_weights.zip`，解压到 `models/` 目录。
   也可以让 `download_model_data.py` 从 HuggingFace 自动下载（需网络连接）。

### Q: 显存不够怎么办？
A: 选择对应的 `config_infer_16g_*.json` 配置文件（16GB 显存版本），
   或减小 `output_size` 到 `[128, 128, 64]`。

### Q: 候选 mask 数据库是什么？
A: MAISI 使用真实医学影像的分割 mask 作为候选，通过 ControlNet 引导 CT 生成。
   这些 mask 来自 21 个公开数据集（BTCV, AMOS22, TotalSegmentator 等）。

### Q: 可以不用 DiffTumor 单独使用 MAISI 吗？
A: 可以。MAISI 本身就能生成含器官+粗略肿瘤的 CT。
   DiffTumor 只是额外注入真实肿瘤纹理，是可选增强。

## 参考文档

- 项目 README: `README.md`
- 分割验证实验报告: `docs/experiment_report.md`
- MAISI+DiffTumor 对接技术文档: `docs/model_integration_details.md`
- MAISI 官方论文: [NV-Generate-CTMR (arXiv)](https://arxiv.org/abs/2503.22622)
- DiffTumor 论文: [DiffTumor (CVPR 2024)](https://arxiv.org/abs/2403.06527)
