# 肿瘤生成管线 — MAISI + DiffTumor

## 概述

三阶段管线，生成含真实肿瘤纹理的合成 CT：

1. **MAISI** 生成基础 CT + 132-class 分割 mask
2. **DiffTumor** 在肿瘤区域生成真实纹理（VQGAN 编码 → 扩散采样 → 解码）
3. **融合嵌入** 将合成纹理嵌入回完整 CT

## 环境要求

- Python 3.10+, PyTorch 2.x, CUDA GPU (≥16GB VRAM)
- 依赖：`monai`, `nibabel`, `SimpleITK`, `scipy`, `huggingface_hub`
- 磁盘：模型 + 数据集约 8.5 GB

## 快速开始

```bash
# 设置数据目录
set MONAI_DATA_DIRECTORY=D:\agent progress\CTMR\NV-Generate-CTMR-main

# 检查资源状态
python -m scripts.tumor_prompt_runner --list

# 单次快速生成
python -m scripts.tumor_prompt_runner --quick --organ liver --size medium --phase noearly --device cuda

# 查看示例配置
python -m scripts.tumor_prompt_runner --example
```

## 支持的器官

| 器官 | 肿瘤类型 | 标签 | 权重 |
|------|---------|------|------|
| liver | liver_lesion | 26 | liver_early.pt, liver_noearly.pt |
| pancreas | pancreatic_lesion | 24 | pancreas_early.pt, pancreas_noearly.pt |
| kidney | kidney_lesion | 116 | kidney_early.pt, kidney_noearly.pt |
| colon | colon_lesion | 27 | colon_early.pt |
| lung | lung_lesion | 23 | liver_early.pt (zero-shot) |
| bone | bone_lesion | 128 | liver_early.pt (zero-shot) |
| esophagus | esophagus_tumor | — | liver_early.pt (zero-shot) |
| uterus | endometrioma_tumor | — | liver_early.pt (zero-shot) |

## 肿瘤尺寸与阶段

| 尺寸 | 物理半径 | 默认阶段 | 扩散步数 |
|------|---------|---------|---------|
| tiny | 1-5 mm | early | DDPM 4 步 |
| small | 5-10 mm | early | DDPM 4 步 |
| medium | 10-20 mm | noearly | DDIM 50 步 |
| large | 20-50 mm | noearly | DDIM 50 步 |

## CLI 参数

```
python -m scripts.tumor_prompt_runner [config.json] [选项]

位置参数:
  config.json              JSON 配置文件路径

快速模式:
  --quick                  启用 CLI 快速模式
  --organ ORGAN            器官 (liver/pancreas/kidney/colon/lung/bone/esophagus/uterus)
  --size SIZE              尺寸 (tiny/small/medium/large, 默认: small)
  --phase PHASE            阶段 (early/noearly, 默认: 按尺寸自动选择)
  --output OUTPUT          输出 (full_ct/patch_96/both, 默认: both)
  --device DEVICE          设备 (默认: cuda)
  --eta ETA                DDIM 随机性 (0=确定性, 1=最大随机, 默认: 0)

其他:
  --list                   列出可用资源
  --example                输出示例 JSON 配置
```

## JSON 批量配置

```json
{
  "tasks": [
    {"organ": "liver",    "size_category": "medium", "phase": "noearly", "output": "both"},
    {"organ": "pancreas", "size_category": "small",  "phase": "early",   "output": "both"},
    {"organ": "kidney",   "size_category": "large",  "eta": 1.0,         "output": "both"}
  ],
  "maisi": {
    "generate_version": "rflow-ct",
    "output_size": [256, 256, 128],
    "spacing": [1.7, 1.7, 2.0],
    "num_output_samples": 1,
    "random_seed": null
  },
  "global": {
    "device": "cuda"
  }
}
```

运行：
```bash
python -m scripts.tumor_prompt_runner configs/batch_tumor_config.json
```

### 配置字段说明

**tasks（任务列表）：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| organ | string | 必填 | 器官名 |
| size_category | string | "small" | tiny/small/medium/large |
| phase | string/null | null | early/noearly，null=按尺寸自动 |
| output | string | "both" | full_ct/patch_96/both |
| eta | float | 0.0 | DDIM 随机性 (仅 noearly 有效) |
| repeat | int | 1 | 重复次数 |

**maisi（MAISI 参数）：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| generate_version | "rflow-ct" | 生成版本 |
| output_size | [256,256,128] | 输出体积大小 |
| spacing | [1.7,1.7,2.0] | 体素间距 (mm) |
| num_output_samples | 1 | 每任务生成数 |
| random_seed | null | 随机种子，null=每次不同 |

## 输出文件

输出目录结构：
```
output/
├── tumor_ct/
│   ├── liver_lesion/
│   │   ├── liver_small_early_20260619_HHMMSS.nii.gz      # 含肿瘤纹理的完整 CT
│   │   └── liver_small_early_20260619_HHMMSS_tumor_mask.nii.gz  # 肿瘤 mask
│   ├── pancreatic_lesion/
│   └── ...
├── tumor_patch/
│   ├── liver_lesion/
│   │   ├── liver_small_early_20260619_HHMMSS.nii.gz      # 96³ 肿瘤 patch (HU 值)
│   │   └── liver_small_early_20260619_HHMMSS_mask.nii.gz  # 96³ 肿瘤 mask
│   └── ...
└── sample_YYYYMMDD_HHMMSS_*.nii.gz                       # MAISI 中间输出
```

## 技术细节

### 融合逻辑（DiffTumor 原版）

- **liver/kidney**: Gaussian alpha 融合
  ```
  final = (1 - mask_blur) * orig_CT + mask_blur * synthetic
  sigma ~ Uniform(0, 4)  # 随机，每次边界锐利度不同
  ```
- **pancreas/esophagus**: 直接替换
  ```
  final = synthetic
  ```
- **HU 裁剪范围**: [-175, 250] → 归一化到 [0, 1]

### GPU 显存管理

批量生成时，每个任务结束后自动释放模型权重和 GPU 缓存，防止 OOM。

## 查看结果

推荐使用 ITK-SNAP（已安装在 `C:\Program Files\ITK-SNAP 4.0\bin\ITK-SNAP.exe`）：

```bash
# 打开含肿瘤的 CT + tumor mask
ITK-SNAP.exe -g tumor_ct.nii.gz -s tumor_mask.nii.gz
```

ITK-SNAP 操作：
- 按 `S` 键切换 mask overlay 显示/隐藏
- 右键拖拽调整窗宽窗位
- `Ctrl+L` 打开 Image Contrast 对话框（推荐 Window=400, Level=40）
