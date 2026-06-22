# NV-Generate-CTMR（肿瘤生成器）

[![许可证](https://img.shields.io/badge/许可证-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace CT](https://img.shields.io/badge/HuggingFace-NV--Generate--CT-yellow.svg)](https://huggingface.co/nvidia/NV-Generate-CT)
[![arXiv MAISI-v1](https://img.shields.io/badge/arXiv-2409.11169-red.svg)](https://arxiv.org/abs/2409.11169)
[![arXiv MAISI-v2](https://img.shields.io/badge/arXiv-2508.05772-red.svg)](https://arxiv.org/abs/2508.05772)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

基于 **MAISI（Medical AI for Synthetic Imaging）** 框架的三维潜扩散模型（LDM），用于生成高分辨率合成 CT 图像及配对的 132 类分割掩码。支持可变体积尺寸与体素间距，并可精确控制器官/肿瘤的大小。

| | | |
|:---:|:---:|:---:|
|![生成的CT与分割](assets/typical-generated-ct-image-corresponding-segmentation-condition.gif)| ![MR示例](assets/MR_example.png) | ![MR脑部示例](assets/combined_grid.gif) |
|*使用 `rflow-ct` 生成的 CT 图像/掩码配对*| *使用 `rflow-mr` 生成的 MR 图像* | *使用 `rflow-mr-brain` 生成的 MR 脑部图像* |

---

## 项目简介

NV-Generate-CTMR 利用基于 MAISI 框架的潜扩散模型生成高分辨率合成三维医学影像。它能产出 CT 图像及配对的 132 类分割掩码，也支持多种对比度的 MRI 体积生成——可用于合成训练数据、稀有病理的数据增强以及隐私保护的数据共享。

核心能力：

- **CT 生成**：配对 132 类分割掩码，支持最大 512×512×768 体素的体积，可控制器官和肿瘤尺寸
- **MRI 生成**：覆盖 T1、T2、FLAIR 等多种对比度，支持脑部、腹部、乳腺、前列腺等解剖区域
- **脑部 MRI 合成**：跨序列 ControlNet，生成配对的多对比度脑部体积（T1w、T2w、FLAIR、SWI）
- **可变分辨率**：每次生成均可配置体积尺寸和体素间距

---

## 生成原理详解

MAISI 采用**两阶段级联扩散模型管线**：先生成解剖结构掩码，再由掩码条件生成 CT 图像。核心思想是 **"结构先行，纹理后填"**。

### 第一阶段：掩码生成（结构蓝图）

这一阶段决定每个体素属于什么解剖结构——即生成一个 132 类的三维分割掩码。

**获取掩码有两条路径：**

#### 路径 A：从数据库查找现成掩码

当 `controllable_anatomy_size` 为空时，从预存的约 4000 个真实标注掩码数据库中查找最匹配的掩码，并进行弹性变形增强。

#### 路径 B：用 Mask DDPM 重新生成合成掩码

当指定了器官/肿瘤尺寸（如 `controllable_anatomy_size=[("liver", 0.5)]`）时，使用 Mask Diffusion Model 从纯随机噪声生成合成掩码：

```
anatomy_size（10维向量）→ Mask DDPM 去噪 → mask latent → Mask AE 解码 → 132 类掩码
```

- **`anatomy_size`** 是一个 10 维向量，控制 5 个器官 + 5 种肿瘤的相对尺寸
  - 器官：胆囊、肝脏、胃、胰腺、结肠
  - 肿瘤：肺肿瘤、胰腺肿瘤、肝肿瘤、结肠癌、骨病变
  - 值范围 `[0, 1.0]`，`-1` 表示不控制

### 第二阶段：由掩码生成 CT 图像（纹理填充）

根据掩码的解剖结构，填充真实的 CT 纹理（HU 值）：

```
132类掩码 → binarize_labels（转为8通道二进制）→ ControlNet 条件信号
                                                        │
纯噪声 latent → Image Diffusion UNet ←── ControlNet 注入 ←── spacing/modality/region
                      │
               去噪循环（30步 RFlow / 1000步 DDPM）
                      │
               干净 latent → Image AE 解码 → CT 图像（HU值）
                      │
               crop_img_body_mask → 背景设为 -1000 HU
```

关键步骤：

1. **`binarize_labels`**：将 1 通道的 132 类整数掩码转为 8 通道二进制 tensor（每个 voxel 的 label 值做 8-bit 二进制展开），作为 ControlNet 的条件输入
2. **ControlNet 注入**：每个去噪时间步，ControlNet 处理掩码条件，输出残差信号注入 Diffusion UNet 各层
3. **Diffusion UNet 去噪**：RFlow 版 30 步，DDPM 版 1000 步
4. **Image AE 解码**：潜变量 → CT 图像，HU 范围映射为 [-1000, 1000]
5. **背景清理**：掩码外的 voxel 强制设为 -1000 HU（空气的 CT 值）

### 掩码的作用总结

| 作用 | 说明 |
|------|------|
| 空间结构定义 | 告诉模型器官的位置、大小、形状 |
| ControlNet 条件信号 | 通过二进制化转为 8 通道，引导对应区域生成正确纹理 |
| 器官尺寸控制 | `anatomy_size` 向量控制各器官/肿瘤的相对大小 |
| 身体区域约束 | 从掩码推导 `top/bottom_region_index`，约束解剖范围 |
| 背景清理 | 将掩码外的 voxel 设为 -1000 HU |
| 质量检查 | 各器官区域的 CT median HU 值与真实统计对比 |

### 数据流全景

```
用户输入：
  body_region=["chest","abdomen"]  anatomy_list=["liver","spleen"]
  controllable_anatomy_size=[("liver", 0.5)]
  spacing=[1.5, 1.5, 2.0]  output_size=[256,256,512]  modality=1 (CT)
         │
         ▼
┌─────────────────────────────────┐
│  第一阶段：掩码生成              │
│  anatomy_size → Mask DDPM →    │
│  Mask AE → 132 类三维掩码       │
│  （256³, 1.5mm 间距）           │
└─────────────────────────────────┘
         │
         │  重采样至目标尺寸/间距
         │  binarize_labels → 8通道条件
         │  推导 top/bottom_region_index
         ▼
┌─────────────────────────────────┐
│  第二阶段：图像生成              │
│  噪声 → ControlNet+DM 30步    │
│  去噪 → 干净 latent → Image   │
│  AE 解码 → CT 体积（HU）       │
│  背景 → -1000 HU               │
└─────────────────────────────────┘
         │
         ▼
  输出：sample_xxx_image.nii.gz + sample_xxx_label.nii.gz
```

---

## 模型变体

本仓库提供四个模型变体：

| | `rflow-mr-brain` | `rflow-mr` | `rflow-ct` | `ddpm-ct` |
|---|---|---|---|---|
| **模态** | MRI（脑部） | MRI | CT | CT |
| **模型权重** | [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) | [NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) | [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) | [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) |
| **架构** | MAISI-v2（Rectified Flow） | MAISI-v2（Rectified Flow） | MAISI-v2（Rectified Flow） | MAISI-v1（DDPM） |
| **推理步数** | 30 | 30 | 30 | 1000 |
| **最大体积** | 512×512×256 | 512×512×128 | 512×512×768 | 512×512×768 |
| **用途** | MR 脑部多对比度合成 | MR 图像生成 | CT 图像/掩码配对生成 | CT 图像/掩码配对生成 |
| **许可证** | [NVIDIA 开源模型](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | [NVIDIA 非商用](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) | [NVIDIA 开源模型](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | [NVIDIA 开源模型](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

**简要推荐**：CT 用 `rflow-ct`，脑部 MRI 用 `rflow-mr-brain`，其他 MRI 用 `rflow-mr`（需在自己的数据上微调）。

---

## 快速开始（需要至少 16G GPU）

> ⚠️ **选择合适的 `dim` 和 `spacing` 是影响输出质量的最重要因素。** `dim × spacing` 定义了视野（FOV）。每个模型变体只见过其目标解剖的训练数据分布范围内的 FOV——超出分布的 FOV（例如 128mm 立方的全身 CT）会产生不可用的输出。请参考：
> - **CT**：[docs/inference.md](docs/inference.md)
> - **MR**：[docs/inference.md](docs/inference.md)

### 安装

```bash
pip install -r requirements.txt
```

### CT 图像/掩码配对生成

```bash
export MONAI_DATA_DIRECTORY="./temp_work_dir"
network="rflow"
generate_version="rflow-ct"  # 可改为 "ddpm-ct"
python -m scripts.inference \
  -t ./configs/config_network_${network}.json \
  -i ./configs/config_infer.json \
  -e ./configs/environment_${generate_version}.json \
  --random-seed 0 --version ${generate_version}
```

推理配置文件 [`configs/config_infer.json`](configs/config_infer.json) 中的关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `body_region` | 身体区域列表 | `["chest", "abdomen", "pelvis"]` |
| `anatomy_list` | 要生成的器官列表 | `["liver", "spleen", "heart"]` |
| `controllable_anatomy_size` | 器官/肿瘤尺寸控制（空=用现成掩码） | `[]` |
| `output_size` | 输出体积尺寸 | `[256, 256, 512]` |
| `spacing` | 体素间距（mm） | `[1.5, 1.5, 2.0]` |
| `modality` | 模态（CT=1） | `1` |
| `num_inference_steps` | 推理步数 | `30`（RFlow） |
| `cfg_guidance_scale` | CFG 引导强度（0=关闭） | `0.0` |

### CT 图像生成（无掩码）

```bash
network="rflow"
generate_version="rflow-ct"
python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer \
  -t ./configs/config_network_${network}.json \
  -e ./configs/environment_maisi_diff_model_${generate_version}.json \
  -c ./configs/config_maisi_diff_model_${generate_version}.json
```

### 使用自定义掩码生成 CT 图像

如果你已有 MAISI 132 类词汇表的三维分割掩码（含体包络 label `200`），可以直接用它生成配对 CT：

```bash
network="rflow"
generate_version="rflow-ct"
python -m scripts.download_model_data --version ${generate_version} --root_dir "./"
python -m scripts.infer_image_from_mask \
  -t ./configs/config_network_${network}.json \
  -i ./configs/config_infer.json \
  -e ./configs/environment_${generate_version}.json \
  --mask /你的掩码路径.nii.gz
```

> ⚠️ **掩码必须使用 MAISI 132 类标签词汇表，并包含体包络（label 200）。** 参见 [`configs/label_dict.json`](configs/label_dict.json)。

### MR 脑部图像生成

```bash
network="rflow"
generate_version="rflow-mr-brain"
python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer \
  -t ./configs/config_network_${network}.json \
  -e ./configs/environment_maisi_diff_model_${generate_version}.json \
  -c ./configs/config_maisi_diff_model_${generate_version}.json
```

在 [`configs/config_maisi_diff_model_rflow-mr-brain.json`](configs/config_maisi_diff_model_rflow-mr-brain.json) 中修改 `"modality"` 来控制 MR 对比度：

| 模态编号 | 说明 |
|---------|------|
| 8 | MRI（不指定对比度） |
| 9 | T1w 全脑 |
| 10 | T2w 全脑 |
| 11 | FLAIR 全脑 |
| 20 | SWI 全脑 |
| 29 | T1w 剥颅脑 |
| 30 | T2w 剥颅脑 |
| 31 | FLAIR 剥颅脑 |
| 32 | SWI 剥颅脑 |

---

## MAISI 132 类标签词汇表

掩码使用 132 类标签体系（参见 [`configs/label_dict.json`](configs/label_dict.json)），常用标签：

| 标签值 | 器官/结构 |
|-------|----------|
| 0 | 背景 |
| 1 | 肝脏 |
| 3 | 脾脏 |
| 4 | 胰腺 |
| 5 | 右肾 |
| 10 | 胆囊 |
| 12 | 胃 |
| 14 | 左肾 |
| 22 | 脑 |
| 23 | 肺肿瘤 |
| 24 | 胰腺肿瘤 |
| 26 | 肝肿瘤 |
| 27 | 结肠癌原发灶 |
| 28-32 | 肺叶 |
| 33-57 | 脊椎 |
| 128 | 骨病变 |
| 132 | 气管 |
| 200 | 体包络（特殊值，必须包含） |

---

## 项目结构

```
NV-Generate-CTMR/
├── configs/                    # 配置文件
│   ├── config_infer.json       # 推理参数
│   ├── config_network_rflow.json  # RFlow 网络定义
│   ├── config_network_ddpm.json   # DDPM 网络定义
│   ├── environment_rflow-ct.json  # RFlow-CT 环境路径
│   ├── environment_ddpm-ct.json   # DDPM-CT 环境路径
│   ├── label_dict.json         # 132 类标签词汇表
│   └── modality_mapping.json   # 模态编号映射
├── scripts/                    # 核心代码
│   ├── inference.py            # 主推理入口
│   ├── sample.py               # LDMSampler 编排器
│   ├── sample_mask.py          # 掩码生成管线
│   ├── infer_image_from_mask.py # 从掩码生成图像管线
│   ├── utils_infer.py          # 推理核心（去噪循环 + AE 解码）
│   ├── utils.py                # 工具函数（标签映射、二进制化等）
│   ├── augmentation.py         # 掩码增强
│   ├── quality_check.py        # 生成质量检查
│   ├── find_masks.py           # 掩码数据库查找
│   └── download_model_data.py  # 模型权重下载
├── inference_tutorial.py       # 推理教程脚本
├── inference_diff_unet_tutorial.py  # 图像生成教程脚本
├── docs/                       # 文档
├── skills/                     # 技能指南
├── figures/                    # 示例图片
└── requirements.txt            # 依赖列表
```

---

## 性能

在未见过的 [autoPET 2023](https://www.nature.com/articles/s41597-022-01718-3) 基准上：

| 模型 | FID 分数 | 推理步数 | 相对速度 |
|------|---------|---------|---------|
| `rflow-ct` | **5.124** | 30 | **比 ddpm-ct 快 33 倍** |
| `ddpm-ct` | 6.083 | 1000 | 基线 |

---

## 文档

| 指南 | 说明 |
|------|------|
| [安装](docs/setup.md) | 完整安装指南、依赖、模型权重下载 |
| [推理](docs/inference.md) | 推理参数详解、间距表 |
| [训练](docs/training.md) | VAE、Diffusion Model、ControlNet 训练指南 |
| [数据准备](docs/data.md) | 数据集格式与准备步骤 |
| [评估](docs/evaluation.md) | FID 评估工具与基准结果 |
| [故障排除](docs/troubleshooting.md) | 常见问题与解决方案 |

---

## 许可证

| 组成 | 许可证 |
|------|--------|
| 源代码 | [Apache 2.0](LICENSE) |
| NV-Generate-CT 权重 | [NVIDIA 开源模型](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| NV-Generate-MR 权重 | [NVIDIA 非商用](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |
| NV-Generate-MR-Brain 权重 | [NVIDIA 开源模型](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

---

## 引用

如果你使用了本仓库的代码或模型，请引用以下论文：

```bibtex
@article{zhao2026maisi,
  title={MAISI-v2: Accelerated 3D high-resolution medical image synthesis with rectified flow and region-specific contrastive loss},
  author={Zhao, Can and Guo, Pengfei and Yang, Dong and Tang, Yucheng and He, Yufan and Simon, Benjamin and Belue, Mason and Harmon, Stephanie and Turkbey, Baris and Xu, Daguang},
  journal={Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI 2026)},
  year={2026}
}
```

```bibtex
@inproceedings{guo2025maisi,
  title={MAISI: Medical AI for synthetic imaging},
  author={Guo, Pengfei and Zhao, Can and Yang, Dong and Xu, Ziyue and Nath, Vishwesh and Tang, Yucheng and Simon, Benjamin and Belue, Mason and Harmon, Stephanie and Turkbey, Baris and others},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={4430--4441},
  year={2025},
  organization={IEEE}
}
```

---

## 资源

- [NV-Generate-CT（HuggingFace）](https://huggingface.co/nvidia/NV-Generate-CT) — CT 模型权重
- [NV-Generate-MR（HuggingFace）](https://huggingface.co/nvidia/NV-Generate-MR) — MR 模型权重
- [NV-Generate-MR-Brain（HuggingFace）](https://huggingface.co/nvidia/NV-Generate-MR-Brain) — 脑部 MRI 模型权重
- [MAISI 在线演示](https://build.nvidia.com/nvidia/maisi) — 无需 GPU 即可试用
- [MAISI-v1 论文（WACV 2025）](https://arxiv.org/pdf/2409.11169)
- [MAISI-v2 论文（AAAI 2026）](https://arxiv.org/pdf/2508.05772)
- 基于 [MONAI](https://monai.io/) 构建

---

## 致谢

本项目由 NVIDIA 与苏黎世大学、伊斯坦布尔梅迪波尔大学、Forithmus 合作完成。

![合作机构](assets/github_logos.png)
