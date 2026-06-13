# Tumor — DiffTumor 肿瘤合成工程

基于 [DiffTumor (CVPR 2024)](https://github.com/MrGiovanni/DiffTumor) 的腹部肿瘤 CT 合成管线，输入肿瘤 mask → 输出完整 CT（嵌入合成肿瘤）+ 96³ 各向同性 patch。

## 快速开始

```bash
# 1. 编辑 prompts.json
# 2. 运行
.\run prompts.json
```

输出：
```
output/
├── full_ct/{organ}/        ← 完整 CT，仅肿瘤区被合成纹理替换
└── synthetic_ct/{organ}/   ← 96³ patch (1mm³ 各向同性)
```

## 项目结构

```
Tumor/
├── config.py                  ← 路径解析器（自动检测项目根目录）
├── paths.json                 ← 外部路径配置（换机器只改这个）
├── prompts.json               ← ★ 你的工作配置文件
├── example_prompts.json       ← 参考示例（不要改）
├── requirements.txt           ← Python 依赖
├── run.bat                    ← Windows 启动脚本（自动找 Python）
├── PROMPT_RUNNER_使用指南.md   ← 完整使用文档
├── README.md                  ← 本文件
│
├── src/                       ← 源代码
│   ├── prompt_runner.py       ← 主入口：JSON/CLI 双接口
│   ├── embed_to_full_ct.py    ← 全 CT 肿瘤嵌入（核心管线）
│   ├── main.py                ← 批量 96³ patch 生成
│   ├── batch_full_ct.py       ← 批量全 CT 生成
│   ├── ct_preprocessor.py     ← CT 预处理（SimpleITK）
│   ├── condition_builder.py   ← 条件向量构造（VQGAN 编码）
│   ├── diffusion_engine.py    ← 扩散推理引擎（UNet+DDPM/DDIM）
│   ├── texture_blender.py     ← 纹理融合（论文公式）
│   ├── train_colon.py         ← 结肠模型训练脚本
│   ├── diagnose.py            ← 诊断工具
│   └── vqgan/                 ← 3D VQGAN 模型
│
├── checkpoints/               ← 预训练权重
│   ├── AutoencoderModel/      ← VQGAN 自编码器（232 MB，所有器官共用）
│   └── DiffusionModel/        ← 扩散模型权重（每个器官 277 MB）
│       ├── liver_early.pt / liver_noearly.pt
│       ├── pancreas_early.pt / pancreas_noearly.pt
│       ├── kidney_early.pt / kidney_noearly.pt
│       └── colon_early.pt     ← 本地训练的结肠权重
│
├── trained_weights/           ← 训练产物
│   └── colon_early.pt
│
└── output/                    ← 生成输出
    ├── full_ct/{organ}/       ← 全 CT 嵌入
    │   └── _batch_20260614/   ← 批量生成归档
    ├── synthetic_ct/{organ}/  ← 96³ patch
    └── tumor_labels/{organ}/  ← 肿瘤 mask
```

## 支持的器官与权重

| 器官 | 早期权重 | 中晚期权重 | 来源 |
|------|------|------|------|
| liver | liver_early.pt | liver_noearly.pt | DiffTumor 预训练 |
| pancreas | pancreas_early.pt | pancreas_noearly.pt | DiffTumor 预训练 |
| kidney | kidney_early.pt | kidney_noearly.pt | DiffTumor 预训练 |
| colon | colon_early.pt | colon_early.pt | 本地训练（MSD-Colon） |
| esophagus | liver_early.pt | — | zero-shot 跨器官 |
| uterus | liver_early.pt | — | zero-shot 跨器官 |

## JSON 配置字段

| 字段 | 说明 | 可选值 | 默认 |
|------|------|------|:--:|
| `organ` | 器官 | liver/pancreas/kidney/colon/esophagus/uterus | — |
| `size_category` | 肿瘤尺寸 | tiny/small/medium/large | small |
| `phase` | 采样策略 | early/noearly/null(自动) | null |
| `host_ct` | 宿主 CT | BDMAP_XXXXXXXX/null(随机) | null |
| `mask_index` | 第 N 个 mask | 0,1,2… | 0 |
| `output` | 输出格式 | both/full_ct/patch_96 | both |
| `repeat` | 重复次数 | 1,2,3… | 1 |

## 依赖

```
torch>=2.0
nibabel
SimpleITK
numpy
scipy
```

安装：`pip install -r requirements.txt`

## 换机器

只改 `paths.json` 的 4 行外部路径，内部路径自动相对项目根目录解析。

## 引用

- **DiffTumor**: Chen et al., "Towards Generalizable Tumor Synthesis," CVPR 2024
- **AbdomenAtlas 2.0**: Chen et al., "Scaling Tumor Segmentation," ICCV 2025
- **TotalSegmentator**: Wasserthal et al., Radiology: Artificial Intelligence, 2023
