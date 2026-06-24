"""
Tumor Mask Generator — 工具函数模块
=====================================

提供所有模块共用的纯函数工具，不涉及任何文件IO或业务逻辑。

模块依赖: 无（零依赖，全模块的基础）

函数分类:
    坐标变换:   voxel_to_mm, mm_to_voxel, get_spacing
    HU值处理:   clip_hu, normalize_hu
    形态学操作: erode_mask, dilate_mask
    几何计算:   compute_ellipsoid_dist, volume_from_radius, compute_valid_region
    随机采样:   random_sample_valid, random_axis_ratios
    弹性形变:   generate_elastic_deformation_field, apply_deformation

坐标约定: 所有坐标使用 (z, y, x) 顺序，与 nibabel 一致。
"""

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    gaussian_filter,
    map_coordinates,
    generate_binary_structure,
)
from typing import Tuple, Optional, List, Union


# ============================================================================
# 坐标变换
# ============================================================================

def get_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
    """
    从 4×4 affine 矩阵提取体素间距 (mm/voxel)。

    Args:
        affine: (4, 4) nibabel affine 矩阵

    Returns:
        (dz, dy, dx): 各轴体素间距，单位 mm/voxel

    Example:
        >>> aff = np.diag([1.0, 1.0, 2.0, 1.0])
        >>> get_spacing(aff)
        (2.0, 1.0, 1.0)
    """
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be (4,4), got {affine.shape}")

    # spacing = 各轴方向向量的 L2 范数
    dz = float(np.linalg.norm(affine[:3, 2]))
    dy = float(np.linalg.norm(affine[:3, 1]))
    dx = float(np.linalg.norm(affine[:3, 0]))
    return (dz, dy, dx)


def voxel_to_mm(voxel_coord: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    体素坐标 → 物理坐标 (mm)。

    Args:
        voxel_coord: (N, 3) 或 (3,) 体素坐标数组，顺序 (z, y, x)
        affine: (4, 4) nibabel affine 矩阵

    Returns:
        物理坐标 (mm)，形状与输入一致

    Example:
        >>> aff = np.eye(4)
        >>> aff[2, 2] = 2.0
        >>> voxel_to_mm(np.array([5, 10, 20]), aff)
        array([10., 10., 20.])
    """
    voxel_coord = np.atleast_2d(voxel_coord).astype(np.float64)
    if voxel_coord.shape[1] != 3:
        raise ValueError(f"voxel_coord must have shape (N,3) or (3,), got {voxel_coord.shape}")

    # 添加齐次坐标: (z, y, x) → (z, y, x, 1)
    ones = np.ones((voxel_coord.shape[0], 1), dtype=np.float64)
    homogeneous = np.column_stack([voxel_coord[:, 2], voxel_coord[:, 1],
                                    voxel_coord[:, 0], ones[:, 0]])  # → (x, y, z, 1)
    mm = homogeneous @ affine.T  # (N, 4)
    # 返回 (z, y, x) 顺序
    result = mm[:, :3][:, ::-1]  # 取前3列并反转为 (z, y, x)

    if result.shape[0] == 1:
        return result[0]
    return result


def mm_to_voxel(mm_coord: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    物理坐标 (mm) → 体素坐标。

    Args:
        mm_coord: (N, 3) 或 (3,) 物理坐标数组，顺序 (z, y, x)
        affine: (4, 4) nibabel affine 矩阵

    Returns:
        体素坐标，形状与输入一致

    Example:
        >>> aff = np.eye(4)
        >>> aff[2, 2] = 2.0
        >>> mm_to_voxel(np.array([10., 10., 20.]), aff)
        array([5., 10., 20.])
    """
    mm_coord = np.atleast_2d(mm_coord).astype(np.float64)
    if mm_coord.shape[1] != 3:
        raise ValueError(f"mm_coord must have shape (N,3) or (3,), got {mm_coord.shape}")

    inv_affine = np.linalg.inv(affine)
    # (z, y, x) → (x, y, z, 1)
    ones = np.ones((mm_coord.shape[0], 1), dtype=np.float64)
    homogeneous = np.column_stack([mm_coord[:, 2], mm_coord[:, 1],
                                    mm_coord[:, 0], ones[:, 0]])
    voxel = homogeneous @ inv_affine.T  # (N, 4)
    result = voxel[:, :3][:, ::-1]  # → (z, y, x)

    if result.shape[0] == 1:
        return result[0]
    return result


# ============================================================================
# HU值处理
# ============================================================================

def clip_hu(ct_array: np.ndarray,
            hu_min: float = -175,
            hu_max: float = 250) -> np.ndarray:
    """
    HU值裁剪到指定范围。

    论文依据: DiffTumor (CVPR 2024) §E.2 (P21):
        "intensity in each scan is truncated to the range [−175, 250]"

    Args:
        ct_array: 原始CT数组，任意形状
        hu_min: HU值下界，默认 -175
        hu_max: HU值上界，默认 250

    Returns:
        裁剪后的数组 (不修改原数组)

    Example:
        >>> data = np.array([-500, -175, 0, 100, 250, 500])
        >>> clip_hu(data)
        array([-175, -175,    0,  100,  250,  250])
    """
    return np.clip(ct_array, hu_min, hu_max).astype(ct_array.dtype)


def normalize_hu(ct_array: np.ndarray,
                 hu_min: float = -175,
                 hu_max: float = 250) -> np.ndarray:
    """
    HU值线性归一化到 [0, 1]。

    先裁剪到 [hu_min, hu_max]，再线性映射到 [0, 1]。

    Args:
        ct_array: 原始CT数组
        hu_min: 归一化下界对应的HU值
        hu_max: 归一化上界对应的HU值

    Returns:
        float32 数组，值域 [0, 1]

    Example:
        >>> data = np.array([-175, 0, 250], dtype=np.float32)
        >>> normalize_hu(data)
        array([0. , 0.4117647, 1. ], dtype=float32)
    """
    clipped = np.clip(ct_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


# ============================================================================
# 形态学操作
# ============================================================================

def _get_spherical_structure(radius_voxel: float) -> np.ndarray:
    """
    生成近似的球形结构元素。

    对于非整数半径，使用距离阈值近似。

    Args:
        radius_voxel: 球半径（体素单位），可以是浮点数

    Returns:
        3D 球形结构元素 (bool 数组)
    """
    r_int = max(1, int(np.ceil(radius_voxel)))
    size = 2 * r_int + 1
    Z, Y, X = np.ogrid[-r_int:r_int+1, -r_int:r_int+1, -r_int:r_int+1]
    dist = np.sqrt(Z**2 + Y**2 + X**2)
    return dist <= radius_voxel


def erode_mask(mask_3d: np.ndarray, radius_voxel: float) -> np.ndarray:
    """
    3D二值mask腐蚀，使用距离变换实现（O(N) 恒定时间）。

    用于计算有效采样区域：从器官mask边界向内收缩 margin 个体素，
    确保肿瘤不会超出器官边界。

    注意: 使用距离变换近似球形腐蚀，比 scipy binary_erosion 快数个数量级，
    特别是在大 margin 和大体积场景下。

    Args:
        mask_3d: (D, H, W) 二值数组
        radius_voxel: 腐蚀半径（体素单位），可以是浮点数

    Returns:
        腐蚀后的二值数组，dtype 与输入一致

    Example:
        >>> mask = np.zeros((10, 10, 10), dtype=np.uint8)
        >>> mask[2:8, 2:8, 2:8] = 1
        >>> eroded = erode_mask(mask, 2.0)
        >>> eroded.sum()  # 核心区域保留
        64
        >>> eroded[3, 3, 3]  # 距边界2体素，应保留
        1
        >>> eroded[2, 2, 2]  # 距边界<2体素，应被腐蚀
        0
    """
    if radius_voxel <= 0:
        return mask_3d.copy()

    from scipy.ndimage import distance_transform_edt
    # Distance from each foreground voxel to the nearest background voxel
    dist = distance_transform_edt(mask_3d.astype(bool))
    eroded = (dist > radius_voxel).astype(mask_3d.dtype)
    return eroded


def dilate_mask(mask_3d: np.ndarray, radius_voxel: float) -> np.ndarray:
    """
    3D二值mask膨胀，使用球形结构元素。

    Args:
        mask_3d: (D, H, W) 二值数组
        radius_voxel: 膨胀半径（体素单位）

    Returns:
        膨胀后的二值数组，dtype 与输入一致
    """
    if radius_voxel <= 0:
        return mask_3d.copy()

    structure = _get_spherical_structure(radius_voxel)
    dilated = binary_dilation(mask_3d.astype(bool), structure=structure)
    return dilated.astype(mask_3d.dtype)


# ============================================================================
# 几何计算
# ============================================================================

def compute_ellipsoid_dist(shape: Tuple[int, int, int],
                           center: Tuple[float, float, float],
                           radii: Tuple[float, float, float]) -> np.ndarray:
    """
    计算椭球距离场。

    距离场定义: dist = sqrt((z/r_z)² + (y/r_y)² + (x/r_x)²)
    椭球内部: dist <= 1

    论文依据: 派生自 DiffTumor §3.3 (P5): "using ellipsoids"

    Args:
        shape: CT体积形状 (D, H, W)
        center: 椭球中心体素坐标 (z, y, x)
        radii: 三轴半径（体素单位）(r_z, r_y, r_x)

    Returns:
        (D, H, W) float32 距离场数组

    Example:
        >>> dist = compute_ellipsoid_dist((5, 5, 5), (2, 2, 2), (2, 1, 1))
        >>> dist[2, 2, 2]  # 中心点
        0.0
        >>> dist[2, 2, 3]  # x方向距中心1体素，r_x=1 → dist≈1
        1.0
    """
    D, H, W = shape
    cz, cy, cx = center
    rz, ry, rx = radii

    if rz <= 0 or ry <= 0 or rx <= 0:
        raise ValueError(f"radii must be positive, got ({rz}, {ry}, {rx})")

    Z, Y, X = np.ogrid[:D, :H, :W]
    Z = Z.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X.astype(np.float64)

    dist = np.sqrt(
        ((Z - cz) / rz) ** 2 +
        ((Y - cy) / ry) ** 2 +
        ((X - cx) / rx) ** 2
    )
    return dist.astype(np.float32)


def volume_from_radius(radius_mm: float, spacing: Tuple[float, float, float]) -> int:
    """
    从等效半径估算椭球体积（体素数）。

    假设等半径球体: V = (4/3) * pi * r³
    体素数 = V / (spacing_z * spacing_y * spacing_x)

    Args:
        radius_mm: 肿瘤等效半径 (mm)
        spacing: 体素间距 (dz, dy, dx)，单位 mm/voxel

    Returns:
        估算体素数 (int)

    Example:
        >>> volume_from_radius(10.0, (1.0, 1.0, 1.0))
        4188
    """
    volume_mm3 = (4.0 / 3.0) * np.pi * (radius_mm ** 3)
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    return int(np.ceil(volume_mm3 / voxel_volume_mm3))


def _get_bbox_indices(mask_3d: np.ndarray):
    """
    Get bounding box indices of non-zero region in a 3D mask.

    Args:
        mask_3d: (D, H, W) binary array

    Returns:
        (z_min, z_max, y_min, y_max, x_min, x_max) — inclusive on both ends
        Returns None if mask is all zeros.
    """
    z_idx = np.any(mask_3d, axis=(1, 2))
    y_idx = np.any(mask_3d, axis=(0, 2))
    x_idx = np.any(mask_3d, axis=(0, 1))
    if not z_idx.any():
        return None
    zr = np.where(z_idx)[0]
    yr = np.where(y_idx)[0]
    xr = np.where(x_idx)[0]
    return (int(zr[0]), int(zr[-1]), int(yr[0]), int(yr[-1]), int(xr[0]), int(xr[-1]))


def compute_valid_region(organ_mask: np.ndarray,
                         margin_voxel: float) -> np.ndarray:
    """
    腐蚀器官mask得到可采样有效区域。

    valid_region = erode(organ_mask, margin_voxel)

    保证肿瘤中心即使取在 valid_region 的最外层体素上，
    整个肿瘤（含羽化边距和安全边距）也不会超出器官边界。

    优化: 先裁剪到器官bounding box再腐蚀，对大体积显著加速。

    Args:
        organ_mask: (D, H, W) 器官二值mask
        margin_voxel: 收缩边距（体素单位）
            margin = max_tumor_radius + feather + safety

    Returns:
        (D, H, W) uint8 有效区域二值mask

    Raises:
        ValueError: 如果有效区域为空

    Example:
        >>> mask = np.zeros((20, 20, 20), dtype=np.uint8)
        >>> mask[3:17, 3:17, 3:17] = 1
        >>> valid = compute_valid_region(mask, 3.0)
        >>> valid.sum() > 0
        True
        >>> valid[3, 3, 3]  # 距边界较近，应被排除
        0
    """
    # Crop to organ bounding box for efficiency
    bbox = _get_bbox_indices(organ_mask)
    if bbox is None:
        raise ValueError("Organ mask is empty — no valid region possible.")

    margin_int = int(np.ceil(margin_voxel))
    D, H, W = organ_mask.shape

    z0, z1, y0, y1, x0, x1 = bbox
    # Expand bbox to include margin padding
    z0_c = max(0, z0 - margin_int)
    z1_c = min(D - 1, z1 + margin_int)
    y0_c = max(0, y0 - margin_int)
    y1_c = min(H - 1, y1 + margin_int)
    x0_c = max(0, x0 - margin_int)
    x1_c = min(W - 1, x1 + margin_int)

    crop = organ_mask[z0_c:z1_c+1, y0_c:y1_c+1, x0_c:x1_c+1].copy()

    # Erode on cropped region
    crop_eroded = erode_mask(crop, margin_voxel)

    if crop_eroded.sum() == 0:
        raise ValueError(
            f"Effective region is empty after eroding by margin={margin_voxel:.1f} voxels. "
            f"Organ mask volume={organ_mask.sum()} voxels. "
            f"Consider reducing tumor radius, feather, or safety margin."
        )

    # Place back into full-size array
    valid = np.zeros_like(organ_mask)
    valid[z0_c:z1_c+1, y0_c:y1_c+1, x0_c:x1_c+1] = crop_eroded

    return valid


# ============================================================================
# 随机采样
# ============================================================================

def random_sample_valid(valid_mask: np.ndarray,
                        n: int = 1,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在有效区域中随机采样体素坐标。

    所有 valid_mask 中值为 1 的体素被等概率选取。

    Args:
        valid_mask: (D, H, W) 有效区域二值mask
        n: 采样数量，默认 1
        rng: numpy 随机数生成器，None 则使用默认全局状态

    Returns:
        (n, 3) int 体素坐标数组，每行 (z, y, x)

    Raises:
        ValueError: 如果 valid_mask 为空

    Example:
        >>> mask = np.zeros((10, 10, 10), dtype=np.uint8)
        >>> mask[5, 5, 5] = 1
        >>> coords = random_sample_valid(mask, n=1, rng=np.random.default_rng(42))
        >>> coords.shape
        (1, 3)
        >>> (coords[0] == [5, 5, 5]).all()
        True
    """
    if rng is None:
        rng = np.random.default_rng()

    indices = np.argwhere(valid_mask > 0)  # (N, 3)，每行 (z, y, x)

    if len(indices) == 0:
        raise ValueError("valid_mask is empty, cannot sample.")

    if n == 1:
        idx = rng.integers(0, len(indices))
        return indices[idx:idx+1]

    # 无放回采样
    n = min(n, len(indices))
    chosen = rng.choice(len(indices), size=n, replace=False)
    return indices[chosen]


def random_axis_ratios(ratio_range: Tuple[float, float] = (0.8, 1.2),
                       rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """
    生成随机轴比例，并保持体积守恒。

    各轴独立在 [ratio_min, ratio_max] 中采样，然后除以几何平均，
    使得 ratio_z * ratio_y * ratio_x = 1.0。

    体积守恒保证: 椭球体积 = 等半径球体体积
    V_ellipsoid = (4/3)*pi*(r*ratio_z)*(r*ratio_y)*(r*ratio_x)
                = (4/3)*pi*r³ * (ratio_z*ratio_y*ratio_x)
                = (4/3)*pi*r³  [因为 ratio_z*ratio_y*ratio_x = 1]

    Args:
        ratio_range: 各轴比例采样范围 (min, max)，默认 (0.8, 1.2)
        rng: numpy 随机数生成器

    Returns:
        (ratio_z, ratio_y, ratio_x): 三轴比例，乘积恒为 1.0

    Example:
        >>> ratios = random_axis_ratios(rng=np.random.default_rng(42))
        >>> np.isclose(ratios[0] * ratios[1] * ratios[2], 1.0)
        True
    """
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = ratio_range
    ratio_z = rng.uniform(lo, hi)
    ratio_y = rng.uniform(lo, hi)
    ratio_x = rng.uniform(lo, hi)

    # 体积守恒: 除以几何平均
    geo_mean = (ratio_z * ratio_y * ratio_x) ** (1.0 / 3.0)
    ratio_z /= geo_mean
    ratio_y /= geo_mean
    ratio_x /= geo_mean

    return (ratio_z, ratio_y, ratio_x)


# ============================================================================
# 弹性形变
# ============================================================================

def generate_elastic_deformation_field(
        shape: Tuple[int, int, int],
        alpha: float = 15.0,
        sigma: float = 3.0,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    生成弹性形变位移场。

    方法:
        ① 在 coarse grid 上生成标准正态随机位移
        ② 用 sigma 的高斯核平滑（实现低频变形）
        ③ 乘以 alpha 控制变形幅度

    论文依据: DiffTumor (CVPR 2024) §F.1 (P22): "elastic deformation"

    Args:
        shape: 3D体积形状 (D, H, W)
        alpha: 变形程度，值越大位移越大。默认 15
        sigma: 高斯滤波sigma（体素单位），控制位移场的平滑度。
               值越大变形越"宏观"，默认 3
        rng: numpy 随机数生成器

    Returns:
        (3, D, H, W) 位移场数组。
        field[0] = z方向位移, field[1] = y方向位移, field[2] = x方向位移

    Example:
        >>> field = generate_elastic_deformation_field((32, 32, 32), alpha=10, sigma=2)
        >>> field.shape
        (3, 32, 32, 32)
        >>> abs(field).max() < 50  # 位移应在合理范围内
        True
    """
    if rng is None:
        rng = np.random.default_rng()

    # 在粗网格上生成随机位移 → 降低计算量
    # 粗网格尺寸 = shape / sigma (取整)，确保低频
    grid_shape = tuple(max(3, int(s / sigma)) for s in shape)

    # 生成标准正态随机位移场
    field_coarse = rng.standard_normal((3,) + grid_shape).astype(np.float32)

    # 上采样到原始尺寸 + 高斯平滑
    zoom_factors = tuple(s / gs for s, gs in zip(shape, grid_shape))

    # 分别对每个轴做上采样+平滑
    field = np.zeros((3,) + shape, dtype=np.float32)
    for i in range(3):
        # 先上采样
        upsampled = _zoom_3d(field_coarse[i], zoom_factors)
        # 再高斯平滑
        smoothed = gaussian_filter(upsampled, sigma=sigma)
        # 标准化（保持零均值，控制幅度）
        std = float(np.std(smoothed))
        if std > 1e-8:
            smoothed = smoothed / std
        field[i] = smoothed * alpha

    return field


def apply_deformation(mask_3d: np.ndarray,
                      displacement_field: np.ndarray,
                      order: int = 1,
                      mode: str = 'nearest') -> np.ndarray:
    """
    对mask施加弹性形变位移场。

    使用 scipy.ndimage.map_coordinates 对mask进行坐标重映射。

    Args:
        mask_3d: (D, H, W) 二值/浮点mask数组
        displacement_field: (3, D, H, W) 位移场，由 generate_elastic_deformation_field 生成
        order: 插值阶数，1=线性插值，0=最近邻。默认 1
        mode: 边界处理模式，'nearest' 表示边界外使用最近的体素值

    Returns:
        变形后的mask，形状和dtype与输入一致

    Example:
        >>> mask = np.zeros((20, 20, 20), dtype=np.uint8)
        >>> mask[5:15, 5:15, 5:15] = 1
        >>> field = generate_elastic_deformation_field((20, 20, 20), alpha=2, sigma=5)
        >>> deformed = apply_deformation(mask, field)
        >>> deformed.shape == mask.shape
        True
        >>> abs(deformed.sum() - mask.sum()) / mask.sum() < 0.1  # 体积变化 < 10%
        True
    """
    D, H, W = mask_3d.shape

    if displacement_field.shape != (3, D, H, W):
        raise ValueError(
            f"displacement_field shape mismatch: "
            f"expected (3, {D}, {H}, {W}), got {displacement_field.shape}"
        )

    # 构建目标坐标网格
    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float64),
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing='ij'
    )

    # 施加位移: new_coord = original_coord + displacement
    Z_deformed = Z + displacement_field[0].astype(np.float64)
    Y_deformed = Y + displacement_field[1].astype(np.float64)
    X_deformed = X + displacement_field[2].astype(np.float64)

    # 坐标重映射
    coords = np.stack([Z_deformed, Y_deformed, X_deformed], axis=0)
    deformed = map_coordinates(
        mask_3d.astype(np.float64),
        coords,
        order=order,
        mode=mode
    )

    # 恢复二值 (阈值 0.5)
    if mask_3d.dtype == np.uint8 or np.issubdtype(mask_3d.dtype, np.integer):
        deformed = (deformed >= 0.5).astype(mask_3d.dtype)
    else:
        deformed = deformed.astype(mask_3d.dtype)

    return deformed


def _zoom_3d(array: np.ndarray, zoom_factors: Tuple[float, float, float]) -> np.ndarray:
    """
    3D数组上采样（低开销实现）。

    使用重复+线性插值的简化方式。对于低频位移场上采样足够精确。

    Args:
        array: (D, H, W) 输入数组
        zoom_factors: (fz, fy, fx) 各轴上采样因子

    Returns:
        上采样后的数组
    """
    from scipy.ndimage import zoom as scipy_zoom
    return scipy_zoom(array, zoom_factors, order=1)


# ============================================================================
# 杂项工具
# ============================================================================

def ensure_uint8(array: np.ndarray) -> np.ndarray:
    """
    确保数组为 uint8 类型，值域 {0, 1}。

    Args:
        array: 任意整数/浮点数组

    Returns:
        uint8 数组，值仅含 0 和 1
    """
    result = (array > 0).astype(np.uint8)
    return result


def get_bbox(mask_3d: np.ndarray) -> Tuple[slice, slice, slice]:
    """
    获取二值mask的bounding box。

    Args:
        mask_3d: (D, H, W) 二值数组

    Returns:
        (z_slice, y_slice, x_slice): 三个维度的切片对象

    Raises:
        ValueError: 如果mask全为零
    """
    if mask_3d.sum() == 0:
        raise ValueError("Cannot compute bbox of empty mask.")

    z_idx = np.any(mask_3d, axis=(1, 2))
    y_idx = np.any(mask_3d, axis=(0, 2))
    x_idx = np.any(mask_3d, axis=(0, 1))

    z_min, z_max = np.where(z_idx)[0][[0, -1]]
    y_min, y_max = np.where(y_idx)[0][[0, -1]]
    x_min, x_max = np.where(x_idx)[0][[0, -1]]

    return (
        slice(z_min, z_max + 1),
        slice(y_min, y_max + 1),
        slice(x_min, x_max + 1),
    )


# ============================================================================
# 模块自检
# ============================================================================

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("utils.py — 模块自检")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # 1. 坐标变换
    print("\n[1] 坐标变换")
    affine = np.eye(4)
    affine[0, 0] = 1.0  # x spacing
    affine[1, 1] = 1.0  # y spacing
    affine[2, 2] = 2.0  # z spacing
    spacing = get_spacing(affine)
    assert spacing == (2.0, 1.0, 1.0), f"spacing failed: {spacing}"
    print(f"    spacing: {spacing} ✓")

    v = np.array([5.0, 10.0, 20.0])
    mm = voxel_to_mm(v, affine)
    v_back = mm_to_voxel(mm, affine)
    assert np.allclose(v, v_back, atol=1e-6), f"roundtrip failed: {v} → {mm} → {v_back}"
    print(f"    roundtrip: {v} → {mm} → {v_back} ✓")

    # 2. HU值处理
    print("\n[2] HU值处理")
    data = np.array([-500, -175, 0, 100, 250, 500], dtype=np.int16)
    clipped = clip_hu(data)
    assert clipped[0] == -175 and clipped[-1] == 250, f"clip failed: {clipped}"
    print(f"    clip_hu: {data} → {clipped} ✓")

    normed = normalize_hu(data.astype(np.float32))
    assert 0.0 <= normed.min() <= normed.max() <= 1.0, "normalize failed"
    print(f"    normalize_hu: min={normed.min():.3f}, max={normed.max():.3f} ✓")

    # 3. 形态学操作
    print("\n[3] 形态学操作")
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[3:17, 3:17, 3:17] = 1
    eroded = erode_mask(mask, 3.0)
    assert eroded.sum() > 0, "erode emptied mask"
    assert eroded[3, 3, 3] == 0, "boundary should be eroded"
    assert eroded[10, 10, 10] == 1, "center should remain"
    print(f"    erode: {mask.sum()} → {eroded.sum()} voxels ✓")

    dilated = dilate_mask(eroded, 3.0)
    assert dilated.sum() >= eroded.sum(), "dilate should enlarge"
    print(f"    dilate: {eroded.sum()} → {dilated.sum()} voxels ✓")

    # 4. 几何计算
    print("\n[4] 几何计算")
    shape = (10, 10, 10)
    center = (5, 5, 5)
    radii = (3, 2, 2)
    dist = compute_ellipsoid_dist(shape, center, radii)
    assert abs(dist[5, 5, 5]) < 1e-6, f"center dist should be 0, got {dist[5,5,5]}"
    assert dist[5, 5, 7] > 0.9, f"edge dist should be ~1, got {dist[5,5,7]}"
    print(f"    ellipsoid_dist: center={dist[5,5,5]:.4f}, edge={dist[5,5,7]:.4f} ✓")

    vol = volume_from_radius(10.0, (1.0, 1.0, 1.0))
    expected_vol = int((4/3) * np.pi * 1000)
    assert abs(vol - expected_vol) < 10, f"volume mismatch: {vol} vs {expected_vol}"
    print(f"    volume(10mm, 1mm iso): {vol} voxels ✓")

    # 5. 随机采样
    print("\n[5] 随机采样")
    valid = np.zeros((10, 10, 10), dtype=np.uint8)
    valid[3:7, 3:7, 3:7] = 1
    coords = random_sample_valid(valid, n=3, rng=rng)
    assert coords.shape == (3, 3), f"shape mismatch: {coords.shape}"
    for i in range(3):
        z, y, x = coords[i]
        assert valid[z, y, x] == 1, f"sampled outside valid region: ({z},{y},{x})"
    print(f"    random_sample_valid: {coords} ✓")

    ratios = random_axis_ratios(rng=rng)
    prod = ratios[0] * ratios[1] * ratios[2]
    assert abs(prod - 1.0) < 1e-6, f"volume not conserved: {ratios} → prod={prod}"
    print(f"    axis_ratios: {ratios} → prod={prod:.6f} ✓")

    # 6. 弹性形变
    print("\n[6] 弹性形变")
    test_mask = np.zeros((32, 32, 32), dtype=np.uint8)
    test_mask[8:24, 8:24, 8:24] = 1
    original_sum = test_mask.sum()

    field = generate_elastic_deformation_field((32, 32, 32), alpha=10, sigma=3, rng=rng)
    assert field.shape == (3, 32, 32, 32), f"field shape: {field.shape}"
    print(f"    field shape: {field.shape} ✓")
    print(f"    field range: [{field.min():.2f}, {field.max():.2f}]")

    deformed = apply_deformation(test_mask, field)
    vol_change = abs(float(deformed.sum()) - float(original_sum)) / float(original_sum)
    assert vol_change < 0.3, f"volume change too large: {vol_change:.2%}"
    print(f"    volume change: {vol_change:.2%} ✓")

    # 7. 杂项
    print("\n[7] 杂项")
    bbox = get_bbox(test_mask)
    assert bbox[0].start == 8 and bbox[0].stop == 24, f"bbox z: {bbox[0]}"
    print(f"    bbox: z={bbox[0]}, y={bbox[1]}, x={bbox[2]} ✓")

    print("\n" + "=" * 60)
    print("全部自检通过 ✓")
    print("=" * 60)
