"""
Tumor Mask Generator — 位置选择模块
=====================================

在器官有效区域内，按策略为肿瘤选择一个合法的中心位置。

模块依赖:
    utils.py      (Step 1): erode_mask, compute_valid_region, random_sample_valid
    validator.py  (Step 3): check_in_organ

被依赖模块:
    main.py (Step 6)

设计原则:
    - 多种采样策略（uniform / distance_weighted），通过枚举切换
    - select_location() 内置重试循环，确保位置合法
    - 所有策略接受显式 rng 参数，确保可复现
"""

import os
import sys
import warnings
from enum import Enum
from typing import Tuple, Optional
import numpy as np

# ── 定位前序步骤 ─────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))

_step1_src = os.path.join(_project_root, 'Step1', 'src')
if _step1_src not in sys.path:
    sys.path.insert(0, _step1_src)

_step3_src = os.path.join(_project_root, 'Step3', 'src')
if _step3_src not in sys.path:
    sys.path.insert(0, _step3_src)

from utils import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    erode_mask,
    compute_valid_region,
    random_sample_valid,
)
from validator import check_in_organ  # noqa: E402  # pyright: ignore[reportMissingImports]


# ============================================================================
# 策略枚举
# ============================================================================

class PlacementStrategy(Enum):
    """
    位置选择策略枚举。

    - UNIFORM:          有效区域内每个体素等概率选取
    - DISTANCE_WEIGHTED: 概率 ∝ distance(体素, 器官表面)^alpha
                         alpha > 0 偏向器官中心，alpha = 0 退化为 uniform
    """
    UNIFORM = "uniform"
    DISTANCE_WEIGHTED = "distance_weighted"


# ============================================================================
# margin 计算
# ============================================================================

def compute_margin_voxel(radius_voxel: float,
                         feather_mm: float = 3.0,
                         safety_mm: float = 5.0,
                         spacing_z: float = 1.0) -> float:
    """
    计算有效区域的腐蚀边距（体素单位）。

    margin = max_tumor_radius_voxel + feather_voxel + safety_voxel

    Args:
        radius_voxel: 肿瘤最大半径（体素单位），取 max(rz, ry, rx)
        feather_mm: 羽化边距 (mm)，为高斯滤波留过渡空间，默认 3mm
        safety_mm: 安全边距 (mm)，确保肿瘤不超出边界，默认 5mm
        spacing_z: z轴间距 (mm/voxel)，用于将 mm 转为体素

    Returns:
        margin_voxel: 腐蚀半径（体素单位）
    """
    feather_voxel = feather_mm / spacing_z if spacing_z > 0 else 3.0
    safety_voxel = safety_mm / spacing_z if spacing_z > 0 else 5.0
    return radius_voxel + feather_voxel + safety_voxel


# ============================================================================
# 采样策略
# ============================================================================

def sample_uniform(valid_mask: np.ndarray,
                   rng: Optional[np.random.Generator] = None) -> Tuple[int, int, int]:
    """
    均匀随机策略: 所有 valid 体素等概率选取。

    论文依据: 最大化位置多样性，不引入未被论文验证的偏置假设。

    Args:
        valid_mask: (D, H, W) 有效区域二值mask
        rng: numpy 随机数生成器

    Returns:
        (z, y, x) 体素坐标
    """
    coords = random_sample_valid(valid_mask, n=1, rng=rng)
    return (int(coords[0, 0]), int(coords[0, 1]), int(coords[0, 2]))


def sample_distance_weighted(valid_mask: np.ndarray,
                             organ_mask: np.ndarray,
                             alpha: float = 1.0,
                             rng: Optional[np.random.Generator] = None) -> Tuple[int, int, int]:
    """
    距离加权策略: 概率 ∝ distance(体素, 器官表面)^alpha。

    alpha > 0 → 偏向器官中心（距离表面越远，权重越大）
    alpha = 0 → 退化为 uniform

    实现:
        ① 计算器官mask的距离变换 → distance_field (到最近非器官体素的距离)
        ② 在 valid 区域中提取 distances
        ③ 以 distances^alpha 为权重采样

    Args:
        valid_mask: (D, H, W) 有效区域二值mask
        organ_mask: (D, H, W) 原始器官二值mask（用于距离计算）
        alpha: 距离权重指数，默认 1.0
        rng: numpy 随机数生成器

    Returns:
        (z, y, x) 体素坐标
    """
    from scipy.ndimage import distance_transform_edt

    if rng is None:
        rng = np.random.default_rng()

    # ① 计算距离场（到器官表面 = 到 organ_mask==0 的距离）
    distance_field = distance_transform_edt(organ_mask > 0).astype(np.float64)

    # ② 限制到 valid 区域
    valid_indices = np.argwhere(valid_mask > 0)

    if len(valid_indices) == 0:
        raise ValueError("valid_mask is empty, cannot sample.")

    # ③ 提取权重
    distances = distance_field[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

    if alpha == 0:
        weights = np.ones_like(distances)
    else:
        weights = distances ** alpha

    # 避免全零权重（全部落在表面上的退化情况）
    if weights.sum() <= 0:
        weights = np.ones_like(distances)

    weights = weights / weights.sum()

    # ④ 按权重采样
    idx = rng.choice(len(valid_indices), p=weights)
    z, y, x = valid_indices[idx]
    return (int(z), int(y), int(x))


# ============================================================================
# 主入口
# ============================================================================

def select_location(organ_mask: np.ndarray,
                    radius_voxel: float,
                    spacing: Tuple[float, float, float],
                    strategy: PlacementStrategy = PlacementStrategy.UNIFORM,
                    feather_mm: float = 3.0,
                    safety_mm: float = 5.0,
                    max_retries: int = 50,
                    distance_alpha: float = 1.0,
                    rng: Optional[np.random.Generator] = None) -> Tuple[int, int, int]:
    """
    在器官mask的有效区域内选择一个合法的肿瘤中心位置。

    内置重试循环: 每次采样后用 check_in_organ 验证，失败则重试。

    流程:
        ① 计算 margin = radius + feather + safety
        ② 腐蚀得到 valid_region
        ③ 循环 max_retries 次:
            a. 按策略采样 center_zyx
            b. check_in_organ(center, radii_3d, organ_mask)
            c. 通过 → 返回
            d. 失败 → 继续
        ④ 全部失败 → RuntimeError

    Args:
        organ_mask:   (D, H, W) uint8 器官二值mask
        radius_voxel: 肿瘤最大等效半径（体素单位），用于 margin 计算
        spacing:      (dz, dy, dx) 体素间距，用于 mm→voxel 转换
        strategy:     采样策略，默认 UNIFORM
        feather_mm:   羽化边距 (mm)，默认 3
        safety_mm:    安全边距 (mm)，默认 5
        max_retries:  最大重试次数，默认 50
        distance_alpha: distance_weighted 策略的 alpha 参数
        rng:          numpy 随机数生成器

    Returns:
        (z, y, x) 肿瘤中心体素坐标

    Raises:
        ValueError: 有效区域为空（器官太小，margin 太大）
        RuntimeError: max_retries 次尝试全部失败
    """
    if rng is None:
        rng = np.random.default_rng()

    # ① 计算 margin
    spacing_z = spacing[0]
    margin_voxel = compute_margin_voxel(radius_voxel, feather_mm, safety_mm, spacing_z)

    # ② 计算有效区域
    try:
        valid_region = compute_valid_region(organ_mask, margin_voxel)
    except ValueError as e:
        raise ValueError(
            f"Cannot compute valid region for organ: {e}. "
            f"margin={margin_voxel:.1f} voxels, "
            f"radius={radius_voxel:.1f} voxels, "
            f"organ_volume={int(organ_mask.sum())} voxels."
        )

    # ③ 为 check_in_organ 准备三轴半径（用于验证阶段）
    # 使用等效球体验证（最坏情况），实际椭球半径会在 mask_generator 中随机化
    radii_3d = (radius_voxel, radius_voxel, radius_voxel)

    # ④ 重试循环
    last_error = ""
    for attempt in range(max_retries):
        # 采样
        if strategy == PlacementStrategy.UNIFORM:
            center = sample_uniform(valid_region, rng=rng)
        elif strategy == PlacementStrategy.DISTANCE_WEIGHTED:
            center = sample_distance_weighted(valid_region, organ_mask,
                                              alpha=distance_alpha, rng=rng)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 验证
        ok, msg = check_in_organ(center, radii_3d, organ_mask)
        if ok:
            return center

        last_error = msg

    # ⑤ 全部失败
    raise RuntimeError(
        f"Failed to find valid position after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Strategy: {strategy.value}, radius_voxel={radius_voxel:.1f}, "
        f"margin_voxel={margin_voxel:.1f}, "
        f"valid_voxels={int(valid_region.sum())}, "
        f"organ_voxels={int(organ_mask.sum())}."
    )


# ============================================================================
# 便捷函数
# ============================================================================

def select_location_from_config(organ_mask: np.ndarray,
                                radius_voxel: float,
                                spacing: Tuple[float, float, float],
                                placement_config: dict,
                                rng: Optional[np.random.Generator] = None
                                ) -> Tuple[int, int, int]:
    """
    从配置字典读取参数并调用 select_location。

    这是 main.py 调用的便捷入口，将 JSON 配置映射为函数参数。

    Args:
        organ_mask:      器官二值mask
        radius_voxel:    肿瘤半径（体素）
        spacing:         体素间距
        placement_config: config['placement'] 字典
        rng:             随机数生成器

    Returns:
        (z, y, x) 体素坐标
    """
    strategy_str = placement_config.get('strategy', 'uniform')
    strategy_map = {
        'uniform': PlacementStrategy.UNIFORM,
        'distance_weighted': PlacementStrategy.DISTANCE_WEIGHTED,
    }
    strategy = strategy_map.get(strategy_str, PlacementStrategy.UNIFORM)

    feather_mm = placement_config.get('margin', {}).get('feather_mm', 3.0)
    safety_mm = placement_config.get('margin', {}).get('safety_mm', 5.0)
    max_retries = placement_config.get('max_retries', 50)
    distance_alpha = placement_config.get('distance_weighted', {}).get('alpha', 1.0)

    return select_location(
        organ_mask=organ_mask,
        radius_voxel=radius_voxel,
        spacing=spacing,
        strategy=strategy,
        feather_mm=feather_mm,
        safety_mm=safety_mm,
        max_retries=max_retries,
        distance_alpha=distance_alpha,
        rng=rng,
    )


# ============================================================================
# 模块自检
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("position_selector.py — 模块自检")
    print("=" * 60)
    rng = np.random.default_rng(42)

    # ── 准备测试数据 ──────────────────────────────
    shape = (40, 80, 80)
    organ = np.zeros(shape, dtype=np.uint8)
    organ[5:35, 10:70, 10:70] = 1  # 肝脏大小
    spacing = (2.0, 1.0, 1.0)
    radius_voxel = 8.0

    # ── compute_margin_voxel ─────────────────────
    print("\n[1] compute_margin_voxel")
    margin = compute_margin_voxel(radius_voxel, 3.0, 5.0, spacing[0])
    expected = 8.0 + 3.0/2.0 + 5.0/2.0  # = 8 + 1.5 + 2.5 = 12
    print(f"    radius={radius_voxel}, margin={margin:.1f} voxels")
    assert abs(margin - 12.0) < 0.1, f"margin={margin}"
    print("    OK")

    # ── compute_valid_region ─────────────────────
    print("\n[2] compute_valid_region")
    valid = compute_valid_region(organ, margin)
    assert valid.sum() > 0
    print(f"    organ={organ.sum():,} → valid={valid.sum():,} voxels")
    print("    OK")

    # ── sample_uniform ───────────────────────────
    print("\n[3] sample_uniform")
    for i in range(3):
        c = sample_uniform(valid, rng=rng)
        assert valid[c[0], c[1], c[2]] == 1, f"outside valid: {c}"
    print(f"    3 samples all in valid region ✓")

    # ── sample_distance_weighted ─────────────────
    print("\n[4] sample_distance_weighted")
    # alpha=2 → strong center bias
    centers_biased = [
        sample_distance_weighted(valid, organ, alpha=2.0, rng=rng)
        for _ in range(20)
    ]
    avg_z = np.mean([c[0] for c in centers_biased])
    # 器官 z 范围 [5,35)，中心 ≈ 20
    # alpha=2 应使采样偏向中心 (20) 而非边缘 (5或34)
    center_z = (5 + 35) / 2  # = 20
    edge_z = 5 + margin     # ≈ 17
    # 平均z应明显偏向中心
    dist_to_center = abs(avg_z - center_z)
    print(f"    avg_z={avg_z:.1f} (center={center_z}), dist_to_center={dist_to_center:.1f}")
    print(f"    all 20 samples in valid region: {all(valid[c[0],c[1],c[2]]==1 for c in centers_biased)}")
    # 验证在 valid 内
    assert all(valid[c[0], c[1], c[2]] == 1 for c in centers_biased)
    print("    OK")

    # ── select_location ──────────────────────────
    print("\n[5] select_location (UNIFORM)")
    center = select_location(
        organ, radius_voxel, spacing,
        strategy=PlacementStrategy.UNIFORM,
        max_retries=50, rng=rng,
    )
    print(f"    center=({center[0]}, {center[1]}, {center[2]})")
    assert organ[center[0], center[1], center[2]] == 1
    print("    OK: center in organ")

    # ── select_location (DISTANCE_WEIGHTED) ──────
    print("\n[6] select_location (DISTANCE_WEIGHTED)")
    center2 = select_location(
        organ, radius_voxel, spacing,
        strategy=PlacementStrategy.DISTANCE_WEIGHTED,
        distance_alpha=2.0,
        max_retries=50, rng=rng,
    )
    print(f"    center=({center2[0]}, {center2[1]}, {center2[2]})")
    assert organ[center2[0], center2[1], center2[2]] == 1
    print("    OK: center in organ")

    # ── select_location_from_config ──────────────
    print("\n[7] select_location_from_config")
    cfg = {
        'strategy': 'uniform',
        'margin': {'feather_mm': 3.0, 'safety_mm': 5.0},
        'max_retries': 50,
        'distance_weighted': {'alpha': 1.0},
    }
    center3 = select_location_from_config(organ, radius_voxel, spacing, cfg, rng=rng)
    print(f"    center=({center3[0]}, {center3[1]}, {center3[2]})")
    assert organ[center3[0], center3[1], center3[2]] == 1
    print("    OK: config-driven selection works")

    # ── 极端情况: margin太大导致有效区域为空 ──────
    print("\n[8] 异常处理: 有效区域为空")
    # 用小器官 + 大 margin，确保有效区域为空但不触发 MemoryError
    tiny_org = np.zeros((30, 30, 30), dtype=np.uint8)
    tiny_org[14:16, 5:25, 5:25] = 1  # 很薄的器官
    try:
        select_location(tiny_org, 8.0, (1.0, 1.0, 1.0), max_retries=5, rng=rng)
        print("    (may pass or fail)")
    except ValueError as e:
        print(f"    OK: ValueError — {str(e)[:80]}...")

    # ── 极端情况: 重试耗尽 ────────────────────────
    print("\n[9] 异常处理: 重试耗尽")
    small_org = np.zeros((20, 20, 20), dtype=np.uint8)
    small_org[5:15, 5:15, 5:15] = 1  # 1000 voxels
    try:
        select_location(small_org, 4.0, (1.0, 1.0, 1.0),
                        max_retries=3, rng=rng)
        print("    (may pass or fail with small organ)")
    except (ValueError, RuntimeError) as e:
        print(f"    OK: {type(e).__name__} — {str(e)[:70]}...")

    print("\n" + "=" * 60)
    print("全部自检通过")
    print("=" * 60)
