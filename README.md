# NV-Generate-CTMR（肿瘤生成器）— 核心代码分支

[![许可证](https://img.shields.io/badge/许可证-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace CT](https://img.shields.io/badge/HuggingFace-NV--Generate--CT-yellow.svg)](https://huggingface.co/nvidia/NV-Generate-CT)
[![arXiv MAISI-v1](https://img.shields.io/badge/arXiv-2409.11169-red.svg)](https://arxiv.org/abs/2409.11169)
[![arXiv MAISI-v2](https://img.shields.io/badge/arXiv-2508.05772-red.svg)](https://arxiv.org/abs/2508.05772)

> 📌 **本分支（`core-code`）仅包含推理核心代码，不含训练脚本、训练权重文件、数据集、图片资源等。**
> 模型权重请从 [HuggingFace](https://huggingface.co/nvidia/NV-Generate-CT) 下载，数据集掩码通过 `scripts/download_model_data.py` 自动获取。

基于 **MAISI（Medical AI for Synthetic Imaging）** 框架的三维潜扩散模型，用于生成高分辨率合成 CT 图像及配对的 132 类分割掩码。支持可变体积尺寸与体素间距，并可精确控制器官/肿瘤的大小。

---

## 生成原理详解

MAISI 采用**两阶段级联扩散模型管线**：先生成解剖结构掩码，再由掩码条件生成 CT 图像。核心思想是 **"结构先行，纹理后填"**。

### 第一阶段：掩码生成（结构蓝图）

这一阶段决定每个体素属于什么解剖结构——即生成一个 132 类的三维分割掩码。

**获取掩码有两条路径：**

#### 路径 A：从数据库查找现成掩码

当 `controllable_anatomy_size` 为空时，从预存的约 4000 个真实标注掩码数据库中查找最匹配的掩码，并进行弹性变形增强。

#### 路径 B：用 Mask DDPM 重新生成合成掩码

当指定了器官/肿瘤尺寸时，使用 Mask Diffusion Model 从纯随机噪声生成合成掩码：

```
anatomy_size（10维向量）→ Mask DDPM 去噪 → mask latent → Mask AE 解码 → 132 类掩码
```

- **`anatomy_size`** 是 10 维向量，控制 5 个器官 + 5 种肿瘤的相对尺寸
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

1. **`binarize_labels`**：将 1 通道的 132 类整数掩码转为 8 通道二进制 tensor（8-bit 二进制展开），作为 ControlNet 条件输入
2. **ControlNet 注入**：每个去噪时间步，ControlNet 处理掩码条件，输出残差信号注入 Diffusion UNet 各层
3. **Diffusion UNet 去噪**：RFlow 版 30 步，DDPM 版 1000 步
4. **Image AE 解码**：潜变量 → CT 图像，HU 范围 [-1000, 1000]
5. **背景清理**：掩码外的 voxel 强制设为 -1000 HU

### 数据流全景

```
用户输入：
  body_region / anatomy_list / controllable_anatomy_size
  spacing / output_size / modality
         │
         ▼
┌─────────────────────────────────┐
│  第一阶段：掩码生成              │
│  anatomy_size → Mask DDPM →    │
│  Mask AE → 132 类三维掩码       │
└─────────────────────────────────┘
         │
         │  binarize_labels → 8通道条件
         │  推导 top/bottom_region_index
         ▼
┌─────────────────────────────────┐
│  第二阶段：图像生成              │
│  噪声 → ControlNet+DM 去噪     │
│  → 干净 latent → AE 解码       │
│  → CT 体积（HU）               │
│  背景 → -1000 HU               │
└─────────────────────────────────┘
         │
         ▼
  输出：image.nii.gz + label.nii.gz
```

---

## 本分支包含的文件

```
NV-Generate-CTMR/  （core-code 分支）
├── scripts/                        # 核心推理代码
│   ├── inference.py                # 主推理入口
│   ├── sample.py                   # LDMSampler 编排器（两阶段管线串联）
│   ├── sample_mask.py              # 掩码生成管线（Mask DDPM + AE 解码）
│   ├── infer_image_from_mask.py    # 从掩码生成图像（ControlNet + DM 去噪）
│   ├── infer_image_from_mask_batch.py  # 批量掩码推理
│   ├── utils_infer.py              # 推理核心（去噪循环 + AE 解码 + ReconModel）
│   ├── utils.py                    # 工具函数（标签映射、二进制化、尺寸检查等）
│   ├── augmentation.py             # 掩码增强（弹性变形等）
│   ├── quality_check.py            # 生成质量检查（器官 HU median 统计）
│   ├── find_masks.py               # 掩码数据库查找
│   ├── diff_model_setting.py       # 模型配置加载
│   ├── diff_model_infer.py         # Diffusion Model 推理（无掩码模式）
│   ├── download_model_data.py      # 模型权重自动下载
│   ├── transforms.py               # 数据变换
│   ├── tumor_adapter.py            # 肿瘤适配器
│   ├── tumor_prompt_runner.py      # 肿瘤提示运行器
│   ├── utils_plot.py               # 可视化工具
│   ├── visualize_tumor.py          # 肿瘤可视化
│   └── __init__.py                 # 包初始化
├── configs/                        # 推理配置文件
│   ├── config_infer.json           # 推理参数（body_region, spacing, output_size 等）
│   ├── config_infer_16g_*.json     # 不同 GPU 显存的推理配置
│   ├── config_network_rflow.json   # RFlow 网络结构定义
│   ├── config_network_ddpm.json    # DDPM 网络结构定义
│   ├── config_maisi_diff_model_rflow-ct.json      # RFlow-CT Diffusion Model 配置
│   ├── config_maisi_diff_model_rflow-mr.json      # RFlow-MR Diffusion Model 配置
│   ├── config_maisi_diff_model_rflow-mr-brain.json # RFlow-MR-Brain 配置
│   ├── environment_rflow-ct.json   # RFlow-CT 环境（模型路径、数据路径）
│   ├── environment_ddpm-ct.json    # DDPM-CT 环境
│   ├── environment_rflow-mr.json   # RFlow-MR 环境
│   ├── environment_rflow-mr-brain.json # RFlow-MR-Brain 环境
│   ├── label_dict.json             # MAISI 132 类标签词汇表
│   ├── label_dict_124_to_132.json  # 124→132 标签映射
│   ├── label_dict_124_to_132_ctmr.json # CTMR 标签映射
│   ├── label_dict_ctmr.json        # CTMR 标签词汇表
│   ├── modality_mapping.json       # 模态编号映射（CT=1, MR variants 8-32）
│   ├── image_median_statistics_ct.json # CT 各器官 HU 统计（质量检查用）
│   └── config_tumor_pipeline.json  # 肿瘤管线配置
├── docs/                           # 核心文档（仅推理相关）
│   ├── inference.md                # 推理参数详解
│   ├── performance.md              # 性能指标
│   ├── setup.md                    # 安装指南
│   └── tumor_pipeline_guide.md     # 肿瘤管线指南
├── skills/                         # 技能指南
│   ├── download-models.md
│   ├── infer_image-from-mask.md
│   ├── infer_image-only.md
│   ├── infer_mask-generation.md
│   └── infer_mask-image-paired.md
├── inference_tutorial.py           # 推理教程脚本（CT 配对生成）
├── inference_diff_unet_tutorial.py # Diffusion 推理教程（无掩码生成）
├── requirements.txt                # 依赖列表
├── pyproject.toml                  # 项目元数据
├── .gitignore                      # Git 忽略规则（含训练文件排除）
├── LICENSE                         # Apache 2.0 许可证
├── LICENSE.weights                 # 模型权重许可证
├── INSTALL.md                      # 安装说明
└── README.md                       # 本文件
```

---

## 本分支不包含的文件

| 类别 | 排除的文件 | 原因 |
|------|-----------|------|
| 训练脚本 | `train_*_tutorial.py`, `scripts/diff_model_train.py` 等 | 非推理核心 |
| 训练配置 | `configs/config_maisi_*_train*.json` 等 | 训练专用，推理不需要 |
| 模型权重 | `models/*.pt`（需从 HuggingFace 下载） | 大文件，不适合 Git |
| 数据集 | `datasets/`（需通过 `download_model_data.py` 获取） | 大文件 |
| 图片资源 | `figures/`, `assets/*.png/*.gif` | 非代码，影响仓库大小 |
| 测试脚本 | `test_*.py`, `download_lidc*.py` 等 | 开发调试用 |
| CI/lint | `.github/`, `.markdownlint.yaml` 等 | 开发流程配置 |
| 训练文档 | `docs/training.md`, `docs/evaluation.md` 等 | 训练相关 |

---

## 快速开始（需要至少 16G GPU）

> ⚠️ **选择合适的 `dim` 和 `spacing` 是影响输出质量的最重要因素。** `dim × spacing` 定义了视野（FOV），请参考 [docs/inference.md](docs/inference.md)。

### 安装

```bash
pip install -r requirements.txt
```

### 下载模型权重

```bash
# CT 模型
python -m scripts.download_model_data --version rflow-ct --root_dir "./"

# DDPM-CT 模型
python -m scripts.download_model_data --version ddpm-ct --root_dir "./"

# MR 脑部模型
python -m scripts.download_model_data --version rflow-mr-brain --root_dir "./"

# MR 模型
python -m scripts.download_model_data --version rflow-mr --root_dir "./"
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

推理配置 [`configs/config_infer.json`](configs/config_infer.json) 中的关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `body_region` | 身体区域列表 | `["chest", "abdomen", "pelvis"]` |
| `anatomy_list` | 要生成的器官列表 | `["liver", "spleen", "heart"]` |
| `controllable_anatomy_size` | 器官/肿瘤尺寸控制 | `[]`（空=用现成掩码） |
| `output_size` | 输出体积尺寸 | `[256, 256, 512]` |
| `spacing` | 体素间距（mm） | `[1.5, 1.5, 2.0]` |
| `modality` | 模态（CT=1） | `1` |
| `num_inference_steps` | 推理步数 | `30`（RFlow） |
| `cfg_guidance_scale` | CFG 引导强度 | `0.0` |

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

```bash
python -m scripts.infer_image_from_mask \
  -t ./configs/config_network_rflow.json \
  -i ./configs/config_infer.json \
  -e ./configs/environment_rflow-ct.json \
  --mask /你的掩码路径.nii.gz
```

> ⚠️ 掩码必须使用 MAISI 132 类标签词汇表并包含体包络（label 200），参见 [`configs/label_dict.json`](configs/label_dict.json)。

### MR 脑部图像生成

修改 [`configs/config_maisi_diff_model_rflow-mr-brain.json`](configs/config_maisi_diff_model_rflow-mr-brain.json) 中的 `"modality"` 控制对比度：

| 编号 | 对比度 |
|------|--------|
| 8 | MRI（不指定对比度） |
| 9 | T1w 全脑 |
| 10 | T2w 全脑 |
| 11 | FLAIR 全脑 |
| 29 | T1w 剥颅脑 |
| 30 | T2w 剥颅脑 |

```bash
network="rflow"
generate_version="rflow-mr-brain"
python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer \
  -t ./configs/config_network_${network}.json \
  -e ./configs/environment_maisi_diff_model_${generate_version}.json \
  -c ./configs/config_maisi_diff_model_${generate_version}.json
```

---

## MAISI 132 类标签词汇表

掩码使用 132 类标签体系，参见 [`configs/label_dict.json`](configs/label_dict.json)，常用标签：

| 标签值 | 器官/结构 | 标签值 | 器官/结构 |
|-------|----------|-------|----------|
| 0 | 背景 | 22 | 脑 |
| 1 | 肝脏 | 23 | 肺肿瘤 |
| 3 | 脾脏 | 24 | 胰腺肿瘤 |
| 4 | 胰腺 | 26 | 肝肿瘤 |
| 5 | 右肾 | 27 | 结肠癌原发灶 |
| 10 | 胆囊 | 28-32 | 肺叶 |
| 12 | 胃 | 33-57 | 脊椎 |
| 14 | 左肾 | 128 | 骨病变 |
| 200 | **体包络（必须包含）** | 132 | 气管 |

---

## 性能

在未见过的 [autoPET 2023](https://www.nature.com/articles/s41597-022-01718-3) 基准上：

| 模型 | FID 分数 | 推理步数 | 相对速度 |
|------|---------|---------|---------|
| `rflow-ct` | **5.124** | 30 | **比 ddpm-ct 快 33 倍** |
| `ddpm-ct` | 6.083 | 1000 | 基线 |

---

## 许可证

| 组成 | 许可证 |
|------|--------|
| 源代码 | [Apache 2.0](LICENSE) |
| NV-Generate-CT 权重 | [NVIDIA 开源模型](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| NV-Generate-MR 权重 | [NVIDIA 非商用](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |

---

## 引用

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
