# data_loader.py — 详细解读

> **所属步骤**: Step 2 — 数据加载层  
> **文件路径**: `Step2/src/data_loader.py`  
> **运行时路径**: `src/data_loader.py`  
> **依赖**: `utils.py` (Step 1), `numpy`, `nibabel`  
> **被依赖**: `position_selector.py` (Step 4), `mask_generator.py` (Step 5), `main.py` (Step 6)

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [数据结构](#二数据结构)
3. [函数详解](#三函数详解)
   - [3.1 文件加载 (3个函数)](#31-文件加载)
   - [3.2 校验 (1个函数)](#32-校验)
   - [3.3 辅助工具 (4个函数)](#33-辅助工具)
4. [数据流](#四数据流)
5. [异常处理策略](#五异常处理策略)
6. [与 AbdomenAtlas2.0 的目录适配](#六与-abdomenatlas20-的目录适配)
7. [自检说明](#七自检说明)

---

## 一、文件功能概述

`data_loader.py` 是项目中唯一涉及文件 I/O 的基础模块，负责：

1. **加载**: 读取 `.nii.gz` 格式的 CT 扫描和器官分割 mask
2. **标准化**: 统一封装为 `CTVolume` / `OrganMask` / `Sample` 数据结构
3. **校验**: 自动检查 CT 与 mask 的兼容性（shape / spacing / affine）
4. **索引**: 扫描数据目录构建样本清单 (manifest)

### 设计原则

| 原则 | 说明 |
|------|------|
| **统一接口** | 所有加载函数返回标准化 dataclass，下游不直接接触 nibabel |
| **防御性加载** | 每个加载步骤都有完整性检查，问题及早暴露 |
| **低侵入** | 不修改原始 .nii.gz 文件，所有处理在内存中进行 |
| **批量友好** | `build_manifest` 先构建索引，`load_sample` 批量调用时按需加载 |

---

## 二、数据结构

### 2.1 CTVolume

```python
@dataclass
class CTVolume:
    array: np.ndarray                          # (D, H, W) HU值数组, dtype=float64
    affine: np.ndarray                         # (4, 4) 仿射矩阵
    spacing: Tuple[float, float, float]        # (dz, dy, dx) mm/voxel
    shape: Tuple[int, int, int]                # (D, H, W)
    path: str                                  # 原始文件路径
```

**数据流向**:
```
nibabel.load(.nii.gz)
  → img.get_fdata() → array (HU值)
  → img.affine      → affine (4×4矩阵)
  → get_spacing(affine) → spacing
  → clip_hu(array)  → 裁剪后array
  → CTVolume 封装
```

**注意**: nibabel 的 `get_fdata()` 默认返回 `float64`。虽然 CT 原始数据为 `int16`，但 float64 对于后续计算（弹性形变、高斯滤波）更方便，不做额外 cast。

---

### 2.2 OrganMask

```python
@dataclass
class OrganMask:
    array: np.ndarray                      # (D, H, W) uint8, 值域 {0, 1}
    affine: np.ndarray                     # (4, 4) 仿射矩阵
    organ_type: str                        # 肿瘤类型名, 如 "liver_lesion"
    organ_label: str                       # 标签文件名, 如 "liver.nii.gz"
    path: str                              # 原始文件路径
```

**二值化处理**:
```
原始 label (可能含多类器官, 值 0~N)
  → ensure_uint8(array) 
  → 所有非零值 → 1, 零值 → 0
  → 仅保留靶器官的二值区域
```

---

### 2.3 Sample

```python
@dataclass
class Sample:
    ct: CTVolume                           # 正常CT扫描
    organ_mask: OrganMask                  # 靶器官二值mask
    sample_id: str                         # 如 "BDMAP_00000001"
```

`Sample` 是一对 `CT + 器官mask` 的打包，供 `main.py` 批量循环使用。

---

## 三、函数详解

### 3.1 文件加载

#### `load_ct(ct_path, hu_min=-175, hu_max=250) → CTVolume`

**流程**:
```
① os.path.exists(ct_path) → 不存在则 FileNotFoundError
② nibabel.load(ct_path)   → 读取 NIfTI
③ img.get_fdata()         → numpy 数组 (自动处理4D/5D)
④ get_spacing(affine)     → (dz, dy, dx)
⑤ clip_hu(array)          → [-175, 250] 裁剪
⑥ CTVolume(...)           → 封装返回
```

**4D/5D 处理**: nibabel 有时读取为 4D（如 `(D,H,W,1)`）或 5D。函数自动 squeeze 最后两个维度。

---

#### `load_organ_mask(mask_path, organ_type, organ_label="") → OrganMask`

**流程**:
```
① nibabel.load(mask_path)
② ensure_uint8(array > 0)  → 二值化
③ 非空检查 (sum > 0)       → 空则 ValueError
④ OrganMask(...)           → 封装返回
```

**二值化策略**: AbdomenAtlas2.0 的器官分割 mask 可能包含多个标签值（如 liver=5, tumor=27）。函数将所有非零标签统一转为 1，因为我们只需要器官的位置/范围，不区分子结构。

---

#### `load_sample(ct_path, organ_mask_path, organ_type, ...) → Sample`

**等价于**: `load_ct() + load_organ_mask() + validate_compatibility()`

一站式加载并校验，是 `main.py` 批量循环中主要的调用入口。

---

### 3.2 校验

#### `validate_compatibility(ct, organ_mask, strict_spacing=False) → None`

**检查项**:

| 检查 | 条件 | 失败处理 |
|------|------|----------|
| shape 一致性 | `ct.shape == mask.shape` | `ValueError` — 立即终止 |
| mask 非空 | `mask.sum() > 0` | `ValueError` — 立即终止 |
| spacing 差异 | `max(|ct_spacing - mask_spacing|) > 0.5mm` | `strict_spacing=True` 时抛 `ValueError`，否则 `warn` |
| 方向一致性 | `sign(det(ct_affine)) == sign(det(mask_affine))` | 不一致则 `warn` |

**为什么 spacing 默认不严格检查？** 不同版本的 NIfTI 在存储 affine 时可能有微小舍入误差（~0.01mm），严格检查会导致误报。

---

### 3.3 辅助工具

#### `get_organ_bbox(organ_mask) → (z_slice, y_slice, x_slice)`

获取器官在 3D 体积中的包围盒。用于裁剪加速——后续计算只需处理 bbox 范围，而非整个 CT。

**调用链**: `data_loader.get_organ_bbox() → utils.get_bbox()`

---

#### `build_manifest(ct_dir, organ_label_dir, organ_config) → List[Dict]`

扫描目录构建样本索引。**不加载任何 .nii.gz 文件**，只检查文件是否存在。

**输出格式**:
```python
{
    'sample_id':       'BDMAP_00000001',
    'ct_path':         'data/ct/BDMAP_00000001/ct.nii.gz',
    'organ_type':      'liver_lesion',
    'organ_label':     'liver.nii.gz',
    'organ_mask_path': 'data/organ_labels/BDMAP_00000001/segmentations/liver.nii.gz',
    'exists':          True
}
```

---

#### `save_manifest_csv(manifest, csv_path) / load_manifest_csv(csv_path)`

manifest 的 CSV 持久化。方便在数据预处理后保存索引，批量生成时直接读取。

---

## 四、数据流

```
main.py (Step 6)
  │
  ├── build_manifest() ──→ manifest.csv     [预处理阶段: 只扫描一次]
  │
  └── for each sample in manifest:          [批量生成阶段]
        │
        ├── load_sample(ct_path, mask_path, organ_type)
        │     │
        │     ├── load_ct()         → CTVolume
        │     │     ├── nibabel.load(.nii.gz)
        │     │     ├── get_spacing(affine)
        │     │     └── clip_hu(array, -175, 250)
        │     │
        │     ├── load_organ_mask() → OrganMask
        │     │     ├── nibabel.load(.nii.gz)
        │     │     └── ensure_uint8()
        │     │
        │     └── validate_compatibility()
        │           ├── shape check
        │           ├── spacing check (warn)
        │           └── orientation check (warn)
        │
        └── → Sample(ct, organ_mask) → 传递给 position_selector + mask_generator
```

---

## 五、异常处理策略

| 异常类型 | 触发条件 | 处理方式 |
|----------|---------|----------|
| `FileNotFoundError` | CT 或 mask 文件不存在 | 立即抛出，不重试。应在预处理阶段修复数据路径 |
| `IOError` | nibabel 无法解析 .nii.gz | 立即抛出，可能文件损坏 |
| `ValueError: ndim != 3` | CT/mask 不是3D体积 | 立即抛出 |
| `ValueError: empty mask` | 器官mask全为零 | 立即抛出，该样本不可用 |
| `ValueError: shape mismatch` | CT与mask尺寸不一致 | 立即抛出，可能来自不同扫描 |
| `Warning: spacing diff` | 间距差 > 0.5mm | 警告但继续，需人工确认 |
| `Warning: orientation` | 行列式符号不一致 | 警告但继续，可能左右翻转 |

### 为什么不在 data_loader 层重试？

文件 I/O 错误通常是**系统性**的（路径配置错误、数据损坏），重试不会改变结果。位置选择（Step 4）中的重试才是必要的——因为每次随机采样可能落在无效位置。

---

## 六、与 AbdomenAtlas2.0 的目录适配

### 预期目录结构

```
data/
├── ct/
│   └── BDMAP_XXXXXXXX/
│       └── ct.nii.gz              ← 正常CT扫描
│
├── organ_labels/
│   └── BDMAP_XXXXXXXX/
│       └── segmentations/
│           ├── liver.nii.gz       ← class 5
│           ├── pancreas.nii.gz    ← class 6
│           ├── kidney_left.nii.gz ← class 3
│           ├── colon.nii.gz       ← class 14
│           ├── esophagus.nii.gz   ← class 16
│           └── uterus.nii.gz      ← class 26
│
└── manifest.csv
```

### 路径构建逻辑

```python
# build_manifest() 中的路径拼接:
ct_path = f"{ct_dir}/{sample_id}/ct.nii.gz"
organ_mask_path = f"{organ_label_dir}/{sample_id}/segmentations/{organ_label}"

# 备选路径（无 segmentations 子目录）:
organ_mask_path = f"{organ_label_dir}/{sample_id}/{organ_label}"
```

---

## 七、自检说明

使用临时合成 `.nii.gz` 文件测试，不依赖真实数据：

| 测试 | 内容 | 验证方式 |
|------|------|----------|
| [1] | 创建合成CT (30,64,64), int16, spacing=(2,1,1) | nibabel 写入 |
| [2] | 创建合成器官mask，体积 38,720 voxels | nibabel 写入 |
| [3] | `load_ct()` — shape/spacing/HU范围 | assert |
| [4] | `load_organ_mask()` — dtype/值域/体积 | assert |
| [5] | `load_sample()` — 打包完整性 | assert |
| [6] | `validate_compatibility()` — 正常+shape不匹配 | assert ValueError |
| [7] | 文件不存在 → FileNotFoundError | assert |
| [8] | 空mask → ValueError | assert |
| [9] | `build_manifest()` + CSV 往返 | assert |

**运行方式**: `python Step2/src/data_loader.py`
