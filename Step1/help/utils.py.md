# utils.py — 详细解读

> **所属步骤**: Step 1 — 工具函数层  
> **文件路径**: `Step1/src/utils.py`  
> **运行时路径**: 项目运行时会复制/链接到 `src/utils.py`  
> **文件类型**: Python 源码模块  
> **依赖**: 仅 `numpy`, `scipy.ndimage`（零项目内依赖）

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [函数分类与索引](#二函数分类与索引)
3. [逐函数详解](#三逐函数详解)
   - [3.1 坐标变换 (3个函数)](#31-坐标变换)
   - [3.2 HU值处理 (2个函数)](#32-hu值处理)
   - [3.3 形态学操作 (3个函数)](#33-形态学操作)
   - [3.4 几何计算 (3个函数)](#34-几何计算)
   - [3.5 随机采样 (2个函数)](#35-随机采样)
   - [3.6 弹性形变 (3个函数)](#36-弹性形变)
   - [3.7 杂项工具 (2个函数)](#37-杂项工具)
4. [数据流与坐标约定](#四数据流与坐标约定)
5. [算法分析](#五算法分析)
6. [与其他模块的接口关系](#六与其他模块的接口关系)
7. [自检说明](#七自检说明)

---

## 一、文件功能概述

`utils.py` 是 Tumor Mask Generator 项目的基础设施层，提供所有其他模块共用的纯函数工具。

### 设计原则

| 原则 | 说明 |
|------|------|
| **零项目依赖** | 不依赖项目中任何其他模块，只依赖 numpy 和 scipy |
| **纯函数** | 无文件I/O，无全局状态，相同输入保证相同输出 |
| **类型安全** | 所有函数有完整 type hints |
| **自检完备** | `python utils.py` 直接运行可验证所有函数正确性 |

### 被依赖关系

```
utils.py  ← 本模块
  ↑
  ├── data_loader.py       (Step 2: 使用 clip_hu, get_spacing)
  ├── validator.py         (Step 3: 使用 erode_mask)
  ├── position_selector.py (Step 4: 使用 erode_mask, random_sample_valid)
  ├── mask_generator.py    (Step 5: 使用 compute_ellipsoid_dist, generate_elastic_deformation_field, apply_deformation)
  └── main.py              (Step 6: 使用所有函数)
```

---

## 二、函数分类与索引

```
utils.py
│
├── 坐标变换 ──────────────────────────────────────
│   ├── get_spacing(affine) → (dz, dy, dx)
│   ├── voxel_to_mm(voxel_coord, affine) → mm_coord
│   └── mm_to_voxel(mm_coord, affine) → voxel_coord
│
├── HU值处理 ─────────────────────────────────────
│   ├── clip_hu(ct_array, hu_min, hu_max) → clipped
│   └── normalize_hu(ct_array, hu_min, hu_max) → [0,1]
│
├── 形态学操作 ───────────────────────────────────
│   ├── _get_spherical_structure(radius_voxel) → structure  [内部]
│   ├── erode_mask(mask_3d, radius_voxel) → eroded
│   └── dilate_mask(mask_3d, radius_voxel) → dilated
│
├── 几何计算 ─────────────────────────────────────
│   ├── compute_ellipsoid_dist(shape, center, radii) → dist_field
│   ├── volume_from_radius(radius_mm, spacing) → voxel_count
│   └── compute_valid_region(organ_mask, margin) → valid_mask
│
├── 随机采样 ─────────────────────────────────────
│   ├── random_sample_valid(valid_mask, n, rng) → coords
│   └── random_axis_ratios(ratio_range, rng) → (rz, ry, rx)
│
├── 弹性形变 ─────────────────────────────────────
│   ├── generate_elastic_deformation_field(shape, α, σ, rng) → field
│   ├── apply_deformation(mask_3d, field) → deformed_mask
│   └── _zoom_3d(array, zoom_factors) → upsampled  [内部]
│
└── 杂项工具 ─────────────────────────────────────
    ├── ensure_uint8(array) → uint8_array
    └── get_bbox(mask_3d) → (z_slice, y_slice, x_slice)
```

---

## 三、逐函数详解

### 3.1 坐标变换

#### `get_spacing(affine) → (dz, dy, dx)`

从 4×4 nibabel affine 矩阵提取体素间距。

**算法**: 各轴方向向量（affine[:3, i]）的 L2 范数即为该轴的体素间距。

```
affine = | sx·rx   sy·ry   sz·rz   tx |
         | sx·rx   sy·ry   sz·rz   ty |
         | sx·rx   sy·ry   sz·rz   tz |
         | 0       0       0       1  |

spacing[2] = ‖affine[:3, 2]‖ = ‖(sz·rz, sz·rz, sz·rz)‖
spacing[1] = ‖affine[:3, 1]‖
spacing[0] = ‖affine[:3, 0]‖
```

**返回值顺序**: `(dz, dy, dx)` —— 对应 (z, y, x) 轴的物理间距。

---

#### `voxel_to_mm(voxel_coord, affine) → mm_coord`

体素坐标 → 物理坐标 (mm)。

**计算**: `[x, y, z, 1] @ affine.T → [mm_x, mm_y, mm_z, 1]`

**输入坐标约定**: `(z, y, x)` 顺序。函数内部转换为 `(x, y, z)` 与 affine 矩阵乘法，输出再转回 `(z, y, x)`。

**支持批量**: 输入 `(N, 3)` 或 `(3,)`，输出形状与输入一致。

---

#### `mm_to_voxel(mm_coord, affine) → voxel_coord`

物理坐标 (mm) → 体素坐标。

**计算**: `[x, y, z, 1] @ inv(affine).T → [vox_x, vox_y, vox_z, 1]`

**往返一致性**: `mm_to_voxel(voxel_to_mm(v, A), A) ≈ v`（浮点误差 < 1e-6）

---

### 3.2 HU值处理

#### `clip_hu(ct_array, hu_min=-175, hu_max=250) → clipped`

HU值裁剪到指定范围。

**论文依据**: DiffTumor §E.2 (P21): `"intensity in each scan is truncated to the range [−175, 250]"`

**实现**: `np.clip(ct_array, hu_min, hu_max)`，保持原 dtype。

**处理逻辑**:
```
HU < -175  →  -175    (空气、极低密度组织 → 截断)
-175 ≤ HU ≤ 250  →  不变  (软组织有效范围 → 保留)
HU > 250   →  250     (骨骼、金属 → 截断)
```

---

#### `normalize_hu(ct_array, hu_min=-175, hu_max=250) → [0, 1]`

线性归一化: `(clipped - hu_min) / (hu_max - hu_min)` → `[0, 1]`

输出 dtype 为 `float32`。

---

### 3.3 形态学操作

#### `_get_spherical_structure(radius_voxel) → structure` [内部函数]

生成近似球形结构元素。对非整数半径使用 `dist = sqrt(z²+y²+x²) ≤ radius` 的距离阈值近似。

---

#### `erode_mask(mask_3d, radius_voxel) → eroded`

3D二值mask腐蚀。核心作用：从器官mask边界向内收缩，得到安全的肿瘤放置区域。

**用途**: `position_selector.compute_valid_region()` 调用此函数计算有效采样区域。

**实现**: `scipy.ndimage.distance_transform_edt` → `dist > radius_voxel`。距离变换替代球形腐蚀，O(N)恒定时间，不受margin大小影响。

**腐蚀效果**:
```
半径 r 的侵蚀后:
  - 器官表面向内收缩 r 个体素
  - 厚度 < 2r 的细长结构会消失
  - 体积减小 ~ O(r × 表面积)
  - 距离变换对任意margin均为O(N)，远快于binary_erosion对大体素
```

---

#### `dilate_mask(mask_3d, radius_voxel) → dilated`

3D二值mask膨胀。逆操作，用于恢复或扩张mask。

---

### 3.4 几何计算

#### `compute_ellipsoid_dist(shape, center, radii) → dist_field`

计算椭球距离场。**这是 mask 形状生成的核心数学函数。**

**公式**:
```
dist(z, y, x) = sqrt( ((z - cz)/rz)² + ((y - cy)/ry)² + ((x - cx)/rx)² )

椭球内部:  dist ≤ 1
椭球表面:  dist = 1
椭球外部:  dist > 1
```

**论文依据**: 派生自 DiffTumor §3.3 (P5): `"using ellipsoids"`

**实现优化**: 使用 `np.ogrid` 避免完整 meshgrid 的内存开销。

---

#### `volume_from_radius(radius_mm, spacing) → voxel_count`

从等效半径估算椭球体积（体素数）。

**公式**: `V = (4/3)·π·r³ / (spacing_z · spacing_y · spacing_x)`

**用途**: 用于日志记录和质量统计，辅助判断肿瘤体积是否合理。

---

#### `compute_valid_region(organ_mask, margin_voxel) → valid_mask`

腐蚀器官mask得到可采样有效区域。

```
valid_region = erode(organ_mask, margin_voxel)
margin = max_tumor_radius + feather + safety
```

**优化**: 先裁剪到器官bounding box，再在裁剪区域上计算距离变换，然后放回原体积。对大体积CT显著加速。

**异常处理**: 如果腐蚀后有效区域为空（器官太小），抛出 `ValueError` 并给出调整建议。

---

### 3.5 随机采样

#### `random_sample_valid(valid_mask, n=1, rng=None) → coords`

在有效区域中等概率随机采样体素坐标。**这是位置选择策略的基础函数。**

**算法**:
```
① indices = argwhere(valid_mask > 0)  →  (N, 3) 有效体素坐标列表
② 如果 n=1: 随机选一个索引
   如果 n>1: 无放回随机选 n 个索引
③ 返回对应的 (z, y, x) 坐标
```

**随机数生成器**: 显式接受 `rng` 参数（`np.random.Generator`），确保可复现性。

---

#### `random_axis_ratios(ratio_range=(0.8, 1.2), rng=None) → (rz, ry, rx)`

生成随机三轴比例，保持体积守恒。

**算法**:
```
① rz = uniform(lo, hi)
   ry = uniform(lo, hi)
   rx = uniform(lo, hi)

② geo_mean = (rz * ry * rx)^(1/3)

③ rz /= geo_mean
   ry /= geo_mean
   rx /= geo_mean

④ 保证: rz * ry * rx = 1.0  (浮点误差 < 1e-6)
```

**体积守恒证明**:
```
V_ellipsoid = (4/3)·π·(r·rz)·(r·ry)·(r·rx)
            = (4/3)·π·r³ · (rz·ry·rx)
            = (4/3)·π·r³           [因为 rz·ry·rx = 1]
            = V_sphere
```

---

### 3.6 弹性形变

#### `generate_elastic_deformation_field(shape, alpha=15, sigma=3, rng=None) → field`

生成低频弹性形变位移场。**这是 mask 形状不规则化的核心函数。**

**论文依据**: DiffTumor §F.1 (P22): `"elastic deformation"`

**算法流程**:
```
① 在粗网格上生成标准正态随机位移
   grid_shape = max(3, shape / sigma)    ← sigma越大，粗网格越稀疏
   
② 上采样到原始尺寸 (scipy.ndimage.zoom)

③ 高斯滤波平滑 (sigma)
   → 确保位移场是"低频"的，变形看起来自然

④ 标准化 (零均值，单位方差)
   → 然后乘以 alpha 控制幅度

⑤ 输出: (3, D, H, W) 位移场
   field[0] = z方向位移
   field[1] = y方向位移
   field[2] = x方向位移
```

**参数含义**:

| 参数 | 默认值 | 含义 | 调大效果 |
|------|--------|------|----------|
| `alpha` | `15` | 变形幅度 | 肿瘤边界更不规则、扭曲更大 |
| `sigma` | `3` | 平滑度（体素） | 变形更宏观、更平滑 |

---

#### `apply_deformation(mask_3d, displacement_field, order=1, mode='nearest') → deformed_mask`

对mask施加弹性形变位移场。

**实现**: `scipy.ndimage.map_coordinates(mask, coords + field)`

**后处理**: 对二值输入，用阈值 0.5 恢复为 {0, 1}。

**体积保持**: 弹性形变不应大幅改变mask体积。测试中体积变化 < 15%。

---

### 3.7 杂项工具

#### `ensure_uint8(array) → uint8_array`

确保数组为 uint8 类型，值域 {0, 1}。实现: `(array > 0).astype(np.uint8)`

**用途**: NIfTI 保存前确保 dtype 正确。

---

#### `get_bbox(mask_3d) → (z_slice, y_slice, x_slice)`

获取二值mask在各维度的最小/最大索引范围。

**用途**: 裁剪加速计算（只处理bbox范围内的区域）。

---

## 四、数据流与坐标约定

### 坐标系统

```
本项目统一使用 (z, y, x) 顺序:
  z: 轴向 (superior → inferior)，对应 CT 扫描方向
  y: 冠状 (anterior → posterior)
  x: 矢状 (left → right)

shape: (D, H, W) = (n_slices, n_rows, n_cols)
affine: 4×4 矩阵，定义体素→物理坐标的映射
spacing: (dz, dy, dx) 各轴物理间距，单位 mm/voxel
```

### 数组 dtype 约定

| 数据类型 | 推荐 dtype | 值域 |
|----------|-----------|------|
| CT原始数据 | `int16` | HU值（通常 -1000~3000） |
| 裁剪后CT | `int16` | [-175, 250] |
| 归一化CT | `float32` | [0, 1] |
| 二值mask | `uint8` | {0, 1} |
| 距离场 | `float32` | [0, ∞) |
| 位移场 | `float32` | [-alpha, +alpha] |

---

## 五、算法分析

### 5.1 椭球距离场的数值稳定性

`compute_ellipsoid_dist` 使用 `np.ogrid` 而非 `np.meshgrid`，内存占用为 O(D+H+W) 而非 O(D×H×W)。对于典型 CT 体积 (512, 512, 400)，ogrid 节省 ~100MB 内存。

### 5.2 弹性形变的计算效率

- 在粗网格上生成随机位移 → 降低 O(Π(shape/sigma)) 的随机数生成量
- 上采样 + 高斯平滑 → O(n·log n) 的 FFT 实现
- 最终 map_coordinates → O(Π(shape)) 的重采样，不可进一步优化

### 5.3 形态学操作的边界处理

腐蚀操作使用 `binary_erosion` 的默认 `border_value=True`，意味着mask边界外的体素被视为 1（物体）。这确保腐蚀只在mask内部进行，不会从图像边界"腐蚀进来"。

---

## 六、与其他模块的接口关系

```
模块                  调用的 utils 函数
────────────────────────────────────────────────────────────
data_loader.py        get_spacing()        提取CT间距
(Step 2)              clip_hu()            裁剪HU值
                      ensure_uint8()       确保mask格式

validator.py          erode_mask()         计算器官边缘收缩参考
(Step 3)              compute_ellipsoid_dist()  验证肿瘤在器官内

position_selector.py  compute_valid_region()  计算有效采样区域
(Step 4)              random_sample_valid()   随机采样位置
                      erode_mask()            腐蚀器官mask

mask_generator.py     compute_ellipsoid_dist()  基础椭球体生成
(Step 5)              random_axis_ratios()    随机各轴比例
                      generate_elastic_deformation_field()  弹性形变位移场
                      apply_deformation()     施加形变
                      ensure_uint8()         输出格式确保
                      volume_from_radius()   体积估算

main.py               get_bbox()            裁剪加速
(Step 6)              (间接使用全部函数)
```

---

## 七、自检说明

`utils.py` 包含完整的 `if __name__ == "__main__"` 自检代码，覆盖所有公开函数：

| 测试组 | 测试内容 | 验证方式 |
|--------|---------|----------|
| [1] 坐标变换 | spacing 提取、voxel↔mm 往返一致性 | assert allclose |
| [2] HU处理 | clip 边界值、normalize [0,1] 范围 | assert 值域检查 |
| [3] 形态学 | erode 收缩量、dilate 膨胀量 | assert 体素计数 |
| [4] 几何计算 | 椭球中心/边界距离、体积公式 | assert 数值验证 |
| [5] 随机采样 | 采样在valid内、轴比例体积守恒 | assert 坐标/乘积检查 |
| [6] 弹性形变 | 位移场形状、体积变化 < 30% | assert 范围检查 |
| [7] 杂项 | bbox 索引范围 | assert 切片检查 |

**运行方式**: `python Step1/src/utils.py`
