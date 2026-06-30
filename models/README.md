# MAISI 模型权重说明

本项目的 `models/` 目录包含 20 个权重文件，分为 **NVIDIA 官方预训练权重** 和 **我们训练/微调的权重** 两类。

> 17 个 ≤2GB 的文件已通过 Git LFS 入库，`git lfs pull` 后即可获取。
> 3 个 >2GB 的文件超出 LFS 限制，需从 NVIDIA HuggingFace 手动下载（见文末）。

---

## 一、NVIDIA MAISI 官方预训练权重（14 个）

来源：NVIDIA HuggingFace，由 MAISI 团队在数千个医学影像体量上训练，覆盖 24+ 个公开数据集。

### 核心权重（5 个 — MAISI CT 推理必需）

| # | 文件名 | 大小 | 用途 | HuggingFace 来源 |
|---|--------|------|------|------------------|
| 1 | `autoencoder_v1.pt` | 80MB | **CT 图像 VAE**：将 CT 图像编码到 4 通道 latent 空间，再解码回 CT 图像。37,243 CT + 17,887 MRI 训练。 | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |
| 2 | `diff_unet_3d_rflow-ct.pt` | 2.1GB | **CT Diffusion U-Net (rflow)**：在 latent 空间生成 CT 图像的噪声预测网络，rectified-flow 调度器。10,277 CT + 1,225 HNSCC 训练。**核心推理权重，缺此无法生成 CT。** | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) ⚠️ >2GB，见文末 |
| 3 | `controlnet_3d_rflow-ct.pt` | 275MB | **ControlNet (rflow)**：以 132 类器官 mask 为条件，引导 Diffusion U-Net 生成与 mask 匹配的 CT 图像。 | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |
| 4 | `mask_generation_autoencoder.pt` | 21MB | **Mask VAE**：将 132 类器官 mask（8 通道输入）编码到 latent 空间，解码回 125 类 mask。仅用于 mask 生成子管线。 | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |
| 5 | `mask_generation_diffusion_unet.pt` | 753MB | **Mask Diffusion U-Net**：在 latent 空间生成 132 类器官 mask 的噪声预测网络，DDPM 调度器。用于步骤1 mask 生成。 | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |

### 备选/其他模态权重（9 个 — 非核心推理必需）

| # | 文件名 | 大小 | 用途 | HuggingFace 来源 |
|---|--------|------|------|------------------|
| 6 | `autoencoder_v2.pt` | 80MB | **VAE v2**：在 v1 基础上增加 8 个额外数据源（CT+MR），39,831 CT / 20,024 MRI。研究用途（非商业授权）。用于 `rflow-mr` 版本。 | [nvidia/NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) |
| 7 | `diff_unet_3d_ddpm-ct.pt` | 654MB | **CT Diffusion U-Net (DDPM)**：DDPM 调度器版本的 CT 生成网络，10,277 CT 训练。与 rflow 版功能相同，调度器不同。 | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |
| 8 | `controlnet_3d_ddpm-ct.pt` | 266MB | **ControlNet (DDPM)**：DDPM 版 ControlNet，6,330 CT 训练（20 数据集）。与 rflow 版功能相同。 | [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |
| 9 | `diff_unet_3d_rflow-mr.pt` | 2.1GB | **MR Diffusion U-Net**：rectified-flow 版 MRI 生成网络，16,291 MR 图像训练。用于 MRI 生成。 | [nvidia/NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) ⚠️ >2GB，见文末 |
| 10 | `diff_unet_3d_rflow-mr-brain_v0.pt` | 2.1GB | **MR Brain Diffusion U-Net**：MR 脑部专用生成网络。v0 版本。 | [nvidia/NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) ⚠️ >2GB，见文末 |
| 11 | `mask_generation_autoencoder.pt` | 21MB | 已列入上方核心权重（与 #4 同文件） | — |
| 12 | `mask_generation_diffusion_unet.pt` | 753MB | 已列入上方核心权重（与 #5 同文件） | — |

> 注：#6-#10 是其他模态/调度器版本的权重，我们项目使用 `rflow-ct` 版本，DDPM/MR 版仅作参考或对比实验用。

---

## 二、我们训练/微调的权重（6 个）

| # | 文件名 | 大小 | 用途 | 说明 |
|---|--------|------|------|------|
| 13 | `autoencoder.pt` | 80MB | 我们训练的 VAE | 在本地数据上训练的图像 VAE，结构与 v1 相同。主管线使用 v1，此为备选。 |
| 14 | `autoencoder_epoch0.pt` | 80MB | VAE 训练中间快照 | epoch 0 的 checkpoint，仅用于分析训练过程，不用于推理。 |
| 15 | `discriminator.pt` | 11MB | GAN 判别器 | VAE 训练时配套的对抗网络判别器，仅训练阶段使用，推理不需要。 |
| 16 | `diff_unet_3d_rflow-ct_finetuned.pt` | 689MB | 微调的 Diffusion U-Net | 在官方 `diff_unet_3d_rflow-ct.pt` 基础上微调的版本。文件较小可能是不同配置或部分训练。 |
| 17 | `seg_model_v2.pth` | 19MB | 分割验证模型 v2 | 3-class UNet（背景+肝脏+肿瘤），用合成 CT 训练，验证合成数据质量。DiceCELoss 加权。 |
| 18 | `seg_model_v3.pth` | 19MB | 分割验证模型 v3 | v2 的改进版，更强化肿瘤采样策略。 |

> 注：#13-#16 是训练中间产物或备选权重，主管线不依赖它们。#17-#18 是辅助验证工具。

---

## 三、其他训练快照（2 个）

| # | 文件名 | 大小 | 用途 | 说明 |
|---|--------|------|------|------|
| 19 | `lidc_controlnet_training_best.pt` | 275MB | LIDC ControlNet 微调最佳 | 在 LIDC 肺部数据集上微调 ControlNet 的最佳 checkpoint。 |
| 20 | `lidc_controlnet_training_current.pt` | 275MB | LIDC ControlNet 微调当前 | 同上，训练过程中的当前快照。 |

另有 2 个 tutorial 训练快照（`tutorial_training_example_best.pt` / `current.pt`，各 275MB），是 MAISI 官方 tutorial 脚本运行产生的示例 checkpoint，非自定义训练。

---

## 四、权重在 MAISI 管线中的角色

```
MAISI 步骤1: 生成 CT + 132类 mask
│
│  ┌─ mask 生成子管线 ──────────────────────────────┐
│  │  mask_generation_autoencoder.pt      (21MB)  #4 │ ← mask VAE
│  │  mask_generation_diffusion_unet.pt   (753MB) #5 │ ← mask Diffusion U-Net
│  │  → 输出: 132类器官 mask                         │
│  └──────────────────────────────────────────────────┘
│
│  ┌─ CT 生成子管线 ────────────────────────────────┐
│  │  autoencoder_v1.pt                  (80MB)  #1  │ ← 图像 VAE
│  │  diff_unet_3d_rflow-ct.pt          (2.1GB) #2  │ ← 图像 Diffusion U-Net
│  │  controlnet_3d_rflow-ct.pt          (275MB) #3  │ ← ControlNet (条件=mask)
│  │  → 输出: 合成 CT 图像                           │
│  └──────────────────────────────────────────────────┘
│
├─ 桥接步骤: bridge_maisi_mask.py (纯几何运算，不需要权重)
│  → 输出: 含新肿瘤的 mask
│
└─ 分割验证 (可选)
   seg_model_v2/v3.pth            (19MB) #17/#18
```

**运行 MAISI 推理只需 5 个核心权重（#1-#5），总计约 3.2GB。**

---

## 五、>2GB 文件下载地址

以下 3 个文件超出 GitHub LFS 2GB 限制，需从 NVIDIA HuggingFace 手动下载：

| 文件名 | 大小 | 下载地址 |
|--------|------|----------|
| `diff_unet_3d_rflow-ct.pt` | 2.1GB | https://huggingface.co/nvidia/NV-Generate-CT/tree/main/models |
| `diff_unet_3d_rflow-mr.pt` | 2.1GB | https://huggingface.co/nvidia/NV-Generate-MR/tree/main/models |
| `diff_unet_3d_rflow-mr-brain_v0.pt` | 2.1GB | https://huggingface.co/nvidia/NV-Generate-MR-Brain/tree/main/models |

下载方法：
1. 访问上方链接，找到对应文件
2. 点击文件名进入下载页面
3. 下载后放到项目的 `models/` 目录下

或使用 Python 自动下载（需安装 `huggingface_hub`）：
```python
from huggingface_hub import hf_hub_download

# 核心权重 (必须下载)
hf_hub_download("nvidia/NV-Generate-CT", "models/diff_unet_3d_rflow-ct.pt", local_dir="models/")

# 实验权重 (可选)
hf_hub_download("nvidia/NV-Generate-MR", "models/diff_unet_3d_rflow-mr.pt", local_dir="models/")
hf_hub_download("nvidia/NV-Generate-MR-Brain", "models/diff_unet_3d_rflow-mr-brain_v0.pt", local_dir="models/")
```

或使用项目自带脚本：
```bash
python scripts/download_model_data.py --version rflow-ct --root_dir ./ --model_only
```
