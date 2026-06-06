# mask_generator.py — 详细解读

> **所属步骤**: Step 5 — Mask生成层  
> **文件路径**: `Step5/src/mask_generator.py`  
> **依赖**: `utils.py` (Step 1), `data_loader.py` (Step 2) — 仅 CTVolume  
> **被依赖**: `main.py` (Step 6)

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [完整管线](#二完整管线)
3. [函数详解](#三函数详解)
4. [体积变化分析](#四体积变化分析)
5. [自检说明](#五自检说明)

---

## 一、文件功能概述

`mask_generator.py` 是项目的**核心模块**，执行从椭球体到 `.nii.gz` 的完整 Mask 生成管线。

```
输入: center_zyx + radius_mm + CTVolume + config
  │
  ├── ① 椭球体                    纯椭球，边界锐利
  ├── ② 弹性形变                  边界自然不规则
  ├── ③ Salt-Noise               内部纹理扰动
  ├── ④ 高斯滤波                  边界平滑过渡
  ├── ⑤ 裁剪                     恢复 {0, 1} 二值
  │
  └── 输出: mask.nii.gz
```

### 论文依据

全部 5 步均来自 **DiffTumor (CVPR 2024) §F.1 (P22)** Hu et al. 管线的逐项描述:

> "ellipse generation, elastic deformation, salt-noise generation, Gaussian filtering, scaling, and clipping"

### 各步骤可独立开关

通过 config 中的 `enabled` 字段控制，方便调试和参数调优。

---

## 二、完整管线

```
create_mask(center_zyx, radius_mm, shape, spacing, shape_config, rng)
│
├── ① create_ellipsoid()
│     radius_mm → compute_radii_from_mm() → (rz, ry, rx)
│     → compute_ellipsoid_dist(shape, center, radii)
│     → (dist <= 1.0) → uint8 {0, 1}
│     论文: DiffTumor §3.3 (P5): "using ellipsoids"
│
├── ② apply_elastic()              [if elastic_deformation.enabled]
│     → 先裁剪到肿瘤周围区域 (crop + padding)
│     → generate_elastic_deformation_field(crop_shape, alpha, sigma)
│     → apply_deformation(crop_mask, field)
│     → 放回原体积
│     优化: 避免在全CT体积(如512³)上生成巨大位移场
│     论文: DiffTumor §F.1 (P22): "elastic deformation"
│
├── ③ apply_salt_noise()           [if salt_noise.enabled]
│     → 在 mask 内部随机翻转子集体素 (1→0)
│     → probability=0.02 → ~2% 体素翻转
│     论文: DiffTumor §F.1 (P22): "salt-noise generation"
│
├── ④ apply_gaussian_smoothing()   [if gaussian_filter.enabled]
│     → scipy.ndimage.gaussian_filter(mask, sigma_voxel)
│     → sigma_voxel = sigma_mm / spacing → 各向异性平滑
│     → 输出 float32 [0, 1]
│     论文: DiffTumor §F.1 (P22): "Gaussian filtering"
│
└── ⑤ apply_clipping()             [always]
     → (mask >= 0.5) → uint8 {0, 1}
     论文: DiffTumor §F.1 (P22): "clipping"
```

---

## 三、函数详解

### `compute_radii_from_mm(radius_mm, spacing, ratio_range, rng) → (rz, ry, rx)`

将单一半径值转为三个体素轴半径。

```
r_voxel = radius_mm / mean(spacing)
(rz, ry, rx) = r_voxel × random_axis_ratios()
```

### `create_ellipsoid(shape, center, radii) → mask`

基础椭球。使用 `utils.compute_ellipsoid_dist` 计算距离场，取 `dist ≤ 1` 为内部。

### `apply_elastic(mask, alpha, sigma, rng) → mask`

弹性形变包装器。内部调用 `utils.generate_elastic_deformation_field` + `utils.apply_deformation`。

### `apply_salt_noise(mask, prob, rng) → mask`

噪声添加。使用 `np.argwhere` 找到所有内部体素，随机选取 `prob` 比例翻转为 0。

### `apply_gaussian_smoothing(mask, sigma_mm, spacing) → mask`

高斯平滑。sigma 针对各轴各向异性：`sigma_voxel[i] = sigma_mm / spacing[i]`。

### `apply_clipping(mask, threshold=0.5) → mask`

阈值二值化。`>= 0.5 → 1, < 0.5 → 0`。

### `create_mask(...) → mask` — 主入口

管线编排器，按 config 开关依次执行各步骤。

### `mask_to_nifti(mask, affine, path) → path`

NIfTI 写入。自动创建父目录。

### `generate_one_mask(...) → (mask, metadata)` — 便捷入口

免去手动提取 shape/spacing，直接接受 CT 数组和完整 config。

---

## 四、体积变化分析

自检中 10mm 半径肿瘤的体积变化:

| 步骤 | 体积 | 变化 |
|------|------|------|
| ① 椭球体 | 1,745 | — |
| ② 弹性形变 | 1,540 | -11.7% |
| ③ Salt-Noise | 1,510 | -1.9% |
| ④ 高斯滤波 | — (float) | 边界模糊 |
| ⑤ 裁剪 | 308 | -79.6% vs ① |

**高斯滤波 + 裁剪导致体积大幅减小**。原因是平滑使边界体素值降低到 <0.5，被裁剪掉。

**这是否有问题？** 对于本项目（仅生成位置 mask），不需要精确体积。下游 DiffTumor 使用 mask 作为**位置条件**，不依赖 mask 的精确体积。如果需要控制输出体积，可调整 `sigma_mm` 参数（更小的 sigma 保留更多边界体素）。

---

## 五、自检说明

| 测试 | 内容 | 验证 |
|------|------|------|
| [1] radii_from_mm | radius=10mm, spacing=(2,1,1) | 三轴半径合理 |
| [2] create_ellipsoid | 基础椭球 | 体积>100, center=1 |
| [3] apply_elastic | alpha=10, sigma=3 | 体积变化<30% |
| [4] apply_salt_noise | prob=0.02 | 少量体素丢失 |
| [5] gaussian_smooth | sigma=1mm | 值域[0,1] |
| [6] apply_clipping | threshold=0.5 | 仅含{0,1} |
| [7] create_mask | 完整管线 | 输出合法mask |
| [8] 多样性 | 5个不同种子 | 5/5 体积不同 |
| [9] 开关测试 | 所有后处理关闭 | 纯椭球输出 |
| [10] mask_to_nifti | 保存+重载 | shape/affine一致 |

**运行方式**: `python Step5/src/mask_generator.py`
