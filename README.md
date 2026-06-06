# Tumor Mask Generator — 项目文档

## 概述

本工程基于 DiffTumor (CVPR 2024) 和 Scaling Tumor (ICCV 2025) 论文，在真实腹部CT数据（AbdomenAtlas2.0）上自动生成肿瘤位置二值Mask (.nii.gz, 值域 {0,1})。

生成的肿瘤Mask可作为下游 DiffTumor 生成模型的位置约束条件。

## 参考文献

| 论文 | 会议 | 链接 |
|------|------|------|
| **DiffTumor** — Generating 3D Tumor-like Masks for CT Synthesis | CVPR 2024 | [arxiv.org/abs/2404.XXXXX](https://arxiv.org/abs/2404.XXXXX) |
| **Scaling Tumor** — Towards a Large-Scale Benchmark for Tumor Segmentation | ICCV 2025 | — |
| **AbdomenAtlas2.0** — A Large-Scale CT Dataset for Abdominal Anatomy | *HuggingFace* | [huggingface.co/datasets/MrGiovanni/AbdomenAtlas2.0Mini](https://huggingface.co/datasets/MrGiovanni/AbdomenAtlas2.0Mini) |
| **TotalSegmentator** — Robust Segmentation of 104 Anatomical Structures in CT | Radiology AI 2023 | [pubs.rsna.org/doi/10.1148/ryai.230024](https://pubs.rsna.org/doi/10.1148/ryai.230024) |

> 注：完整论文引用信息见 `PROJECT_OVERVIEW.md`

## 目录结构

```
Mask/
├── README.md                          # 本文档
├── PROJECT_OVERVIEW.md                # 工程概览（论文依据）
├── IMPLEMENTATION_PLAN.md             # 实现计划
│
├── data/                              # 原始数据
│   ├── ct/                            # ★ CT扫描数据
│   │   └── BDMAP_000000XX/            #   每个CT一个文件夹
│   │       └── ct.nii.gz              #   CT影像 (.nii.gz)
│   ├── organ_labels/                  # ★ 器官分割Mask
│   │   └── BDMAP_000000XX/            #   每个CT一个文件夹
│   │       └── segmentations/         #
│   │           ├── liver.nii.gz       #   肝脏
│   │           ├── pancreas.nii.gz    #   胰腺
│   │           ├── kidney_left.nii.gz #   左肾
│   │           ├── colon.nii.gz       #   结肠
│   │           ├── esophagus.nii.gz   #   食管
│   │           └── uterus.nii.gz      #   子宫（合成）
│   ├── tmp/                           #   下载缓存
│   └── manifest.csv                   #   样本索引表
│
├── output/real_ct/                    # ★ 生成的肿瘤Mask输出
│   ├── liver_lesion/                  #   肝脏肿瘤 (50个)
│   │   ├── liver_lesion_t00__BDMAP_00000012.nii.gz
│   │   └── ...
│   ├── pancreatic_lesion/             #   胰腺肿瘤 (50个)
│   ├── kidney_lesion/                 #   肾脏肿瘤 (50个)
│   ├── colon_lesion/                  #   结肠肿瘤 (50个)
│   ├── esophagus_tumor/               #   食管肿瘤 (50个)
│   ├── endometrioma_tumor/            #   子宫内膜瘤 (51个)
│   ├── generation_log.json            #   生成日志
│   └── statistics.json               #   统计数据
│
├── Step0/config/                      # 配置
│   └── generation_config.json         #   主要生成参数
├── Step1~7/                           # 实现模块
├── supplement_masks.py                # 补充生成脚本
├── validate_and_renumber.py           # 验证和重编号脚本
├── generate_videos.py                 # 视频生成脚本
└── generate_uterus_masks.py           # 子宫Mask生成脚本
```

## 文件命名规则

### 肿瘤Mask命名: `{器官类型}_t{序号}__{CT编号}.nii.gz`

| 部分 | 说明 | 示例 |
|------|------|------|
| `{器官类型}` | 6种器官之一 | `liver_lesion` |
| `t{序号}` | 该器官的第几个肿瘤(00-49)，按体积降序 | `t00` (最大) ~ `t49` (最小) |
| `__{CT编号}` | 肿瘤所在的源CT | `BDMAP_00000012` |

**完整示例:** `liver_lesion_t00__BDMAP_00000012.nii.gz`
- 器官: 肝脏肿瘤
- 序号: 该器官第0个(体积最大的)
- 源CT: BDMAP_00000012

### 6种器官类型

| 文件夹名 | 器官 | 器官Mask | 数量 |
|----------|------|----------|------|
| `liver_lesion` | 肝脏肿瘤 | `liver` (TotalSegmentator) | 50 |
| `pancreatic_lesion` | 胰腺肿瘤 | `pancreas` (TotalSegmentator) | 50 |
| `kidney_lesion` | 肾脏肿瘤 | `kidney_left` (TotalSegmentator) | 50 |
| `colon_lesion` | 结肠肿瘤 | `colon` (TotalSegmentator) | 50 |
| `esophagus_tumor` | 食管肿瘤 | `esophagus` (TotalSegmentator) | 50 |
| `endometrioma_tumor` | 子宫肿瘤 | `uterus` (合成) | 51 |

## 数据来源

### CT扫描
- **来源:** AbdomenAtlas2.0 (JHU/MrGiovanni)
- **数量:** 30个 (BDMAP_00000001 ~ 00000030)
- **格式:** .nii.gz, 各向异性间距
- **位置:** `data/ct/BDMAP_000000XX/`

### 器官分割
- **肝脏、胰腺、左肾、结肠、食管:** TotalSegmentator (fast模式) 自动分割
- **子宫:** 基于人体包围框(body-relative positioning)的合成椭球体Mask
- **位置:** `data/organ_labels/BDMAP_000000XX/segmentations/`

## 肿瘤生成方法

### 论文依据
1. **椭球体形状** — DiffTumor §3.3: "using ellipsoids"
2. **弹性形变** — DiffTumor §F.1: "elastic deformation"
3. **尺寸分布** — Scaling Tumor §4.1: small:medium:large = 4:2:1
4. **尺寸定义** — Scaling Tumor §3.2: tiny(r≤5), small(5<r≤10), medium(10<r≤20), large(r>20) mm

### 生成管线
1. 从30个CT中随机选取，加载器官Mask
2. 按4:2:1比例采样肿瘤尺寸（小器官回退到tiny，r=1-5mm）
3. 在器官有效区域内随机选择肿瘤中心
4. 生成椭球体Mask → 弹性变形 → 高斯平滑 → 裁剪到器官边界
5. 验证：肿瘤非空、完全在器官内、裁剪损失<20%

### 关于食管肿瘤尺寸
食管肿瘤几乎全部为"tiny"级别(r=1-5mm)。原因是TotalSegmentator对食管的自动分割质量参差不齐：
- 9/30个CT中食管体积<5,000体素（极薄管状结构）
- 最薄仅27个Z切片×5mm间距
- 即使最小肿瘤(r=2mm)加上margin后也能完全腐蚀掉器官
- 算法通过预过滤"可行CT"(29/30)，仅对有足够厚度的食管生成肿瘤

## 视频

视频文件在 `output/real_ct/{器官}/video/` 下，每种器官3个：
- `*_mask.mp4` — 纯肿瘤Mask扫描
- `*_ct.mp4` — CT背景扫描
- `*_overlay.mp4` — CT+肿瘤叠加显示
