# 第二步实现计划：中间输出 + 分目录存放

## 目标

修改 `bridge_single()` 的保存逻辑，每次运行输出 4 组文件，每组独立存放在自己的子目录中。

## 输出结构

```
output/
└── {task_name}/                      ← 任务根目录 (如 sample_xxx_liver_large_tumor)
    ├── 01_original_ct/               ← 中间输出①: 清除自带肿瘤后的原始 CT + label
    │   ├── ct.nii.gz                 ← 原始 CT (内容不变)
    │   ├── label_no_tumor.nii.gz     ← 清除后的 label (肿瘤还原为器官)
    │   └── removal_info.json         ← 清除信息记录
    │
    ├── 02_tumor_mask/                ← 中间输出②: 单独的肿瘤二值 mask
    │   ├── tumor_mask.nii.gz         ← 二值 mask (0=无肿瘤, 1=肿瘤区域)
    │
    ├── 03_tumor_region_ct/           ← 中间输出③: 肿瘤区域的 CT 子体
    │   ├── tumor_region_ct.nii.gz    ← bbox 裁切后的 CT 子体 (仅肿瘤及其邻域)
    │   ├── tumor_region_label.nii.gz ← bbox 裁切后的 label 子体 (含肿瘤标签)
    │
    └── 04_final_merged/              ← 最终输出④: 合并后的完整 label
    │   ├── merged_label.nii.gz       ← 132类 label (含新肿瘤标签)
```

## 四组输出的数据来源和时机

| # | 子目录 | 数据来源 | 输出时机 | 内容 |
|---|--------|---------|---------|------|
| ① | `01_original_ct/` | Step 1 + Step 1.5 | Step 1.5 之后 | 清除自带肿瘤后的 CT + label，是桥接前的"干净基线" |
| ② | `02_tumor_mask/` | Step 4 的 `tumor_mask` | Step 4 之后 | 纯二值肿瘤 mask，不含器官/body标签，方便查看肿瘤形状位置 |
| ③ | `03_tumor_region_ct/` | `ct_data` + `tumor_mask` bbox 裁切 | Step 4 之后 | 肿瘤所在区域的小块 CT 子体，方便查看肿瘤区域的原始纹理 |
| ④ | `04_final_merged/` | Step 6 的 `merged` | Step 9 | 最终完整 132类 label，可直接用于 ControlNet |

## 改动内容

### 1. 修改 `bridge_single()` 的 Step 9 保存逻辑

当前 Step 9 只保存一个 `merged_label.nii.gz` 文件到一个平面目录。
改为：创建任务根目录 + 4 个子目录，每组各保存自己的文件。

### 2. 新增 bbox 裁切逻辑 (中间输出③)

对 `tumor_mask` 计算 bounding box，从 `ct_data` 和 `label_data` 中裁切该区域，
保存为 NIfTI 时调整 affine 的 origin（使坐标正确）。

### 3. 不改动的部分

- Step 1-8 的核心逻辑不变（清除、提取、采样、生成、合并等）
- `dry_run=True` 时跳过所有文件保存
- 返回字典新增 `output_dir` 字段（任务根目录路径），其他字段不变

## 任务根目录命名

沿用现有自动命名逻辑：`{label_basename}_{organ}_{size}_tumor`
加上版本化（_v2, _v3...）避免覆盖。
