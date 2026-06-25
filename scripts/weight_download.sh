#!/bin/bash
# MAISI 权重下载脚本
# 用法: bash scripts/weight_download.sh
#
# 此脚本下载所有 MAISI 推理所需的权重文件到 models/ 目录。
# - 17个文件 (≤2GB) 已通过 Git LFS 在仓库中，git clone 时自动下载
# - 3个文件 (>2GB) 需从 GitHub Release 手动下载

set -e

MODELS_DIR="models"
RELEASE_TAG="weights-v1"
REPO="komeji1/tumor-generator"

mkdir -p "$MODELS_DIR"

echo "=== MAISI 权重下载 ==="
echo ""

# 检查 LFS 权重是否已存在（git clone --lfs 后自动下载）
echo "[Step 1] 检查 Git LFS 权重 (17个文件, ≤2GB)..."
LFS_FILES=(
    autoencoder_v1.pt
    controlnet_3d_rflow-ct.pt
    mask_generation_autoencoder.pt
    mask_generation_diffusion_unet.pt
    diff_unet_3d_ddpm-ct.pt
    autoencoder.pt
    autoencoder_epoch0.pt
    autoencoder_v2.pt
    discriminator.pt
    diff_unet_3d_rflow-ct_finetuned.pt
    lidc_controlnet_training_best.pt
    lidc_controlnet_training_current.pt
    tutorial_training_example_best.pt
    tutorial_training_example_current.pt
    controlnet_3d_ddpm-ct.pt
    seg_model_v2.pth
    seg_model_v3.pth
)

MISSING=0
for f in "${LFS_FILES[@]}"; do
    if [ ! -f "$MODELS_DIR/$f" ]; then
        echo "  ❌ $f — 缺失"
        MISSING=$((MISSING+1))
    else
        echo "  ✅ $f — 已存在"
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  有 $MISSING 个 LFS 文件缺失。请确保 clone 时使用 --lfs 选项:"
    echo "    git lfs install"
    echo "    git clone git@github.com:$REPO.git"
    echo ""
fi

# 下载 >2GB 的 Release 权重
echo ""
echo "[Step 2] 下载 GitHub Release 权重 (3个文件, >2GB)..."
echo "  这些文件超出 GitHub LFS 2GB 限制，存储在 Release 页面。"
echo ""

BIG_FILES=(
    "diff_unet_3d_rflow-ct.pt|核心权重|CT Diffusion U-Net (rectified-flow)"
    "diff_unet_3d_rflow-mr.pt|实验权重|MR Diffusion U-Net (rectified-flow)"
    "diff_unet_3d_rflow-mr-brain_v0.pt|实验权重|MR Brain Diffusion U-Net"
)

MISSING_BIG=0
for entry in "${BIG_FILES[@]}"; do
    IFS='|' read -r FILE TYPE DESC <<< "$entry"
    if [ ! -f "$MODELS_DIR/$FILE" ]; then
        echo "  ❌ $FILE — $TYPE: $DESC"
        MISSING_BIG=$((MISSING_BIG+1))
    else
        echo "  ✅ $FILE — 已存在 ($TYPE: $DESC)"
    fi
done

if [ $MISSING_BIG -gt 0 ]; then
    echo ""
    echo "  有 $MISSING_BIG 个大文件缺失。请手动从 GitHub Release 下载:"
    echo ""
    echo "    https://github.com/$REPO/releases/tag/$RELEASE_TAG"
    echo ""
    echo "  下载后将文件放到 models/ 目录下。"
    echo ""
    echo "  或使用 curl 下载 (需要 GitHub token):"
    for entry in "${BIG_FILES[@]}"; do
        IFS='|' read -r FILE TYPE DESC <<< "$entry"
        if [ ! -f "$MODELS_DIR/$FILE" ]; then
            echo "    curl -L -o $MODELS_DIR/$FILE https://github.com/$REPO/releases/download/$RELEASE_TAG/$FILE"
        fi
    done
fi

echo ""
echo "=== 权重总览 ==="
echo ""
echo "核心权重 (5, MAISI 推理必需):"
echo "  1. autoencoder_v1.pt               (80MB)   — CT图像 VAE 编解码器"
echo "  2. diff_unet_3d_rflow-ct.pt        (2.1GB)  — CT Diffusion U-Net  [Release]"
echo "  3. controlnet_3d_rflow-ct.pt        (275MB)  — ControlNet 条件控制"
echo "  4. mask_generation_autoencoder.pt   (21MB)   — 132类 mask VAE"
echo "  5. mask_generation_diffusion_unet.pt (753MB) — mask Diffusion U-Net"
echo ""
echo "实验权重 (15, 验证/微调/探索):"
echo "  6.  autoencoder.pt                  (80MB)   — 我们训练的 VAE"
echo "  7.  autoencoder_epoch0.pt           (80MB)   — VAE epoch0 快照"
echo "  8.  autoencoder_v2.pt               (80MB)   — VAE v2 (MR支持)"
echo "  9.  discriminator.pt                (11MB)   — GAN 判别器"
echo "  10. diff_unet_3d_ddpm-ct.pt         (654MB)  — DDPM版 Diffusion U-Net"
echo "  11. controlnet_3d_ddpm-ct.pt         (266MB)  — DDPM版 ControlNet"
echo "  12. diff_unet_3d_rflow-mr.pt        (2.1GB)  — MR版 Diffusion U-Net  [Release]"
echo "  13. diff_unet_3d_rflow-mr-brain_v0.pt (2.1GB)— MR脑版 Diffusion U-Net [Release]"
echo "  14. diff_unet_3d_rflow-ct_finetuned.pt (689MB)— 我们微调的 Diffusion U-Net"
echo "  15. lidc_controlnet_training_best.pt (275MB) — LIDC ControlNet 微调"
echo "  16. lidc_controlnet_training_current.pt (275MB) — LIDC ControlNet 中间快照"
echo "  17. tutorial_training_example_best.pt (275MB) — tutorial 训练最佳"
echo "  18. tutorial_training_example_current.pt (275MB) — tutorial 训练当前"
echo "  19. seg_model_v2.pth                (19MB)   — 分割验证模型 v2"
echo "  20. seg_model_v3.pth                (19MB)   — 分割验证模型 v3"
echo ""
echo "总计: 20个文件, ~10.1GB"
echo "  LFS: 17个文件, ~4.8GB (git clone 自动下载)"
echo "  Release: 3个文件, ~6.3GB (需手动下载)"
