"""
Tumor Mask Generator — 校验模块
================================

提供位置选择和 mask 生成后的校验函数，确保生成结果满足所有约束条件。

模块依赖:
    utils.py (Step 1): compute_ellipsoid_dist, erode_mask

被依赖模块:
    position_selector.py (Step 4)
    mask_generator.py    (Step 5)
    main.py              (Step 6)

设计原则:
    - 所有 check_*() 返回 (is_valid: bool, detail) 元组
    - 校验失败不抛异常，由调用方决定重试或放弃
    - validate_sample() 一站式执行所有必要检查
"""

import os
import sys
from typing import Tuple, List, Optional, Dict

import numpy as np

# ── 定位 Step1/src ──────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))  # Mask/
_step1_src = os.path.join(_project_root, 'Step1', 'src')
if _step1_src not in sys.path:
    sys.path.insert(0, _step1_src)

from utils import compute_ellipsoid_dist  # noqa: E402  # pyright: ignore[reportMissingImports]


# ============================================================================
# 单项校验
# ============================================================================

def check_organ_volume(organ_mask: np.ndarray,
                       min_voxels: int = 100) -> Tuple[bool, int]:
    """
    检查器官 mask 的体积是否足够放置肿瘤。

    器官体积过小（如 min_voxels < 100）时，即使最小的肿瘤也可能超出边界。

    Args:
        organ_mask: (D, H, W) uint8 器官二值 mask
        min_voxels: 最小体素数阈值，默认 100

    Returns:
        (is_valid, volume_voxels):
            is_valid: volume >= min_voxels
            volume_voxels: 器官总体素计数

    Example:
        >>> mask = np.zeros((20,20,20), dtype=np.uint8); mask[5:15,5:15,5:15] = 1
        >>> check_organ_volume(mask, min_voxels=100)
        (True, 1000)
    """
    volume = int(organ_mask.sum())

    if volume == 0:
        return (False, 0)

    is_valid = volume >= min_voxels
    return (is_valid, volume)


def check_size_range(radius_mm: float,
                     size_category_config: Dict) -> Tuple[bool, Tuple[float, float]]:
    """
    检查肿瘤半径是否在指定 size_category 的范围内。

    论文依据: Scaling Tumor (ICCV 2025) §3.2 (P5):
        tiny(r≤5), small(5<r≤10), medium(10<r≤20), large(r>20)

    Args:
        radius_mm: 肿瘤等效半径 (mm)
        size_category_config: 单个category配置，如 {"r_min_mm": 5, "r_max_mm": 10, "weight": 4}

    Returns:
        (is_valid, expected_range):
            is_valid: r_min < r <= r_max (注意左开右闭，除tiny外)
            expected_range: (r_min_mm, r_max_mm)

    Example:
        >>> cfg = {"r_min_mm": 5, "r_max_mm": 10}
        >>> check_size_range(7.5, cfg)
        (True, (5, 10))
        >>> check_size_range(3.0, cfg)
        (False, (5, 10))
    """
    r_min = size_category_config['r_min_mm']
    r_max = size_category_config['r_max_mm']

    # tiny 是闭区间 [r_min, r_max]，其他是左开右闭 (r_min, r_max]
    is_valid = (r_min < radius_mm <= r_max) or (r_min <= radius_mm <= r_max and r_min <= 1)

    return (is_valid, (r_min, r_max))


def check_in_organ(center_zyx: Tuple[float, float, float],
                   radius_voxel: Tuple[float, float, float],
                   organ_mask: np.ndarray) -> Tuple[bool, str]:
    """
    检查以 center 为中心、radii 为半径的椭球体是否完全在器官 mask 内。

    核心算法:
        ① 计算椭球体 bounding box (center ± max_radius)，裁剪计算范围
        ② 在 bbox 内计算椭球距离场
        ③ 检查所有 dist <= 1 的体素是否 organ_mask == 1
        ④ 如果存在 organ_mask == 0 的体素 → 肿瘤超出器官边界

    Args:
        center_zyx: 椭球中心体素坐标 (z, y, x)
        radius_voxel: 三轴半径（体素）(rz, ry, rx)
        organ_mask: (D, H, W) uint8 器官二值 mask

    Returns:
        (is_valid, message):
            is_valid=True:  肿瘤完全在器官内
            is_valid=False: message 包含超出比例和位置信息
    """
    shape = organ_mask.shape
    cz, cy, cx = center_zyx
    rz, ry, rx = radius_voxel

    # ① 检查中心是否在器官内
    zi, yi, xi = int(round(cz)), int(round(cy)), int(round(cx))

    # 边界检查
    if not (0 <= zi < shape[0] and 0 <= yi < shape[1] and 0 <= xi < shape[2]):
        return (False, f"Center ({cz:.1f}, {cy:.1f}, {cx:.1f}) is outside volume bounds {shape}")

    if organ_mask[zi, yi, xi] == 0:
        return (False, f"Center ({zi}, {yi}, {xi}) is outside organ mask")

    # ② 计算椭球 bounding box
    max_r = int(np.ceil(max(rz, ry, rx))) + 1
    z_min = max(0, zi - max_r)
    z_max = min(shape[0], zi + max_r + 1)
    y_min = max(0, yi - max_r)
    y_max = min(shape[1], yi + max_r + 1)
    x_min = max(0, xi - max_r)
    x_max = min(shape[2], xi + max_r + 1)

    bbox_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    bbox_center = (cz - z_min, cy - y_min, cx - x_min)

    # ③ 在 bbox 内计算距离场
    dist = compute_ellipsoid_dist(bbox_shape, bbox_center, (rz, ry, rx))

    # ④ 检查椭球内体素
    tumor_voxels = (dist <= 1.0)
    organ_bbox = organ_mask[z_min:z_max, y_min:y_max, x_min:x_max]

    # 肿瘤内的体素在 organ 中非零的比例
    tumor_count = int(tumor_voxels.sum())
    if tumor_count == 0:
        return (False, "Tumor volume is zero (radii too small for this resolution)")

    covered = int((tumor_voxels & (organ_bbox > 0)).sum())
    ratio = covered / tumor_count

    if ratio < 1.0:
        overflow = tumor_count - covered
        return (False, f"Tumor exceeds organ boundary: {overflow}/{tumor_count} voxels "
                       f"({(1-ratio)*100:.1f}%) outside. Center=({cz:.0f},{cy:.0f},{cx:.0f}), "
                       f"radii=({rz:.1f},{ry:.1f},{rx:.1f})")

    return (True, f"Tumor fully inside organ ({tumor_count} voxels)")


def check_mask_nonzero(mask_3d: np.ndarray) -> Tuple[bool, int]:
    """
    检查生成的 mask 是否非空。

    Args:
        mask_3d: (D, H, W) 二值 mask 数组

    Returns:
        (is_valid, nonzero_count):
            is_valid: nonzero_count > 0
    """
    count = int(mask_3d.sum())
    return (count > 0, count)


def check_not_overlapping(new_mask: np.ndarray,
                          existing_masks: Optional[List[np.ndarray]] = None,
                          overlap_threshold: float = 0.0) -> Tuple[bool, float]:
    """
    检查新 mask 不与已有 mask 重叠。

    对于多肿瘤场景（同一器官生成多个肿瘤），确保肿瘤之间不重叠。

    Args:
        new_mask: (D, H, W) 新生成的肿瘤二值 mask
        existing_masks: 已有肿瘤 mask 列表，每个 (D, H, W)
        overlap_threshold: 允许的最大重叠比例，默认 0.0（不允许任何重叠）

    Returns:
        (is_valid, max_overlap_ratio):
            is_valid: 所有重叠比例 <= threshold
            max_overlap_ratio: 最大重叠比例 (重叠体素数 / new_mask体素数)
    """
    if existing_masks is None or len(existing_masks) == 0:
        return (True, 0.0)

    new_count = int(new_mask.sum())
    if new_count == 0:
        return (True, 0.0)

    max_overlap = 0.0
    for i, existing in enumerate(existing_masks):
        overlap = int((new_mask & existing).sum())
        ratio = overlap / new_count
        max_overlap = max(max_overlap, ratio)

        if ratio > overlap_threshold:
            return (False, max_overlap)

    return (True, max_overlap)


# ============================================================================
# 一站式校验
# ============================================================================

def validate_sample(center_zyx: Tuple[float, float, float],
                    radius_voxel: Tuple[float, float, float],
                    radius_mm: float,
                    mask_3d: np.ndarray,
                    organ_mask: np.ndarray,
                    size_category_config: Dict,
                    existing_masks: Optional[List[np.ndarray]] = None,
                    min_organ_volume: int = 100) -> Dict:
    """
    一站式校验: 依次执行所有必要检查，返回完整校验报告。

    校验流程:
       ① check_organ_volume      → 器官足够大？
       ② check_size_range        → 半径在分类范围内？
       ③ check_mask_nonzero      → mask非空？
       ④ check_in_organ          → 肿瘤完全在器官内？
       ⑤ check_not_overlapping   → 不与已有mask重叠？

    Args:
        center_zyx:        肿瘤中心体素坐标 (z, y, x)
        radius_voxel:      三轴半径（体素）(rz, ry, rx)
        radius_mm:         肿瘤等效半径 (mm)
        mask_3d:           生成的肿瘤 mask (D, H, W) uint8
        organ_mask:        器官二值 mask (D, H, W) uint8
        size_category_config: 当前 size_category 的配置 (含 r_min_mm/r_max_mm)
        existing_masks:    已有肿瘤 mask 列表（多肿瘤场景）
        min_organ_volume:  器官最小体素体积阈值

    Returns:
        {
            'passed': bool,                       # 全部检查通过
            'checks': {
                'organ_volume':   (bool, detail),
                'size_range':     (bool, detail),
                'mask_nonzero':   (bool, detail),
                'in_organ':       (bool, detail),
                'not_overlapping':(bool, detail),
            },
            'errors': [str],                       # 失败项的摘要
            'warnings': [str],                     # 警告信息
        }
    """
    results = {}
    errors_list = []
    warnings_list = []

    # ① organ_volume
    ok, vol = check_organ_volume(organ_mask, min_organ_volume)
    results['organ_volume'] = (ok, f"volume={vol} voxels")
    if not ok:
        errors_list.append(f"Organ too small: {vol} < {min_organ_volume} voxels")

    # ② size_range
    ok, (r_min, r_max) = check_size_range(radius_mm, size_category_config)
    results['size_range'] = (ok, f"radius={radius_mm:.1f}mm, range=({r_min},{r_max}]mm")
    if not ok:
        errors_list.append(f"Radius {radius_mm:.1f}mm outside ({r_min},{r_max}]mm")

    # ③ mask_nonzero
    ok, count = check_mask_nonzero(mask_3d)
    results['mask_nonzero'] = (ok, f"{count} nonzero voxels")
    if not ok:
        errors_list.append("Generated mask is empty (all zeros)")

    # ④ in_organ
    ok, msg = check_in_organ(center_zyx, radius_voxel, organ_mask)
    results['in_organ'] = (ok, msg)
    if not ok:
        errors_list.append(f"Tumor not fully inside organ: {msg}")

    # ⑤ not_overlapping
    ok, overlap = check_not_overlapping(mask_3d, existing_masks)
    results['not_overlapping'] = (ok, f"max_overlap={overlap:.4f}")
    if not ok:
        errors_list.append(f"Tumor overlaps with existing mask (ratio={overlap:.4f})")

    passed = len(errors_list) == 0

    # 如果器官体积勉强够但 margin 很小，添加警告
    if vol > 0:
        tumor_vol_estimate = int(mask_3d.sum())
        if tumor_vol_estimate > 0 and tumor_vol_estimate > vol * 0.5:
            warnings_list.append(
                f"Tumor volume ({tumor_vol_estimate}) is >50% of organ volume ({vol}). "
                f"Consider reducing tumor size."
            )

    return {
        'passed': passed,
        'checks': results,
        'errors': errors_list,
        'warnings': warnings_list,
    }


# ============================================================================
# 模块自检
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("validator.py — 模块自检")
    print("=" * 60)
    rng = np.random.default_rng(42)

    # ── 准备测试数据 ──────────────────────────────
    shape = (40, 80, 80)
    organ = np.zeros(shape, dtype=np.uint8)
    organ[5:35, 10:70, 10:70] = 1  # 肝脏大小

    center = (20, 40, 40)
    radii = (10, 10, 10)

    # 一个在器官内的椭球 mask
    from utils import compute_ellipsoid_dist as _ced  # pyright: ignore[reportMissingImports]
    dist_field = _ced(shape, center, radii)
    tumor_mask = (dist_field <= 1.0).astype(np.uint8)

    # ── check_organ_volume ───────────────────────
    print("\n[1] check_organ_volume")
    ok, vol = check_organ_volume(organ, min_voxels=100)
    assert ok and vol > 10000, f"vol={vol}"
    print(f"    OK: volume={vol:,} voxels ✓")

    ok2, vol2 = check_organ_volume(np.zeros((5,5,5), dtype=np.uint8))
    assert not ok2
    print(f"    OK: empty organ rejected ✓")

    # ── check_size_range ─────────────────────────
    print("\n[2] check_size_range")
    cfg_small = {"r_min_mm": 5, "r_max_mm": 10}
    ok, rng = check_size_range(7.5, cfg_small)
    assert ok
    print(f"    OK: 7.5mm in (5,10] ✓")

    ok, rng = check_size_range(3.0, cfg_small)
    assert not ok
    print(f"    OK: 3.0mm outside (5,10] ✓")

    # ── check_in_organ ───────────────────────────
    print("\n[3] check_in_organ")
    ok, msg = check_in_organ(center, radii, organ)
    assert ok, msg
    print(f"    OK: {msg} ✓")

    # 边界位置（确保超出检测工作）
    edge_center = (5, 11, 11)  # 非常靠近器官边界
    ok, msg = check_in_organ(edge_center, (5, 5, 5), organ)
    # 此位置可能通过也可能失败，取决于具体位置
    print(f"    edge_center({edge_center}): {'PASS' if ok else 'FAIL'} — {msg}")

    # 明确超出
    out_center = (0, 5, 5)
    ok, msg = check_in_organ(out_center, radii, organ)
    assert not ok, f"Should fail but passed: {msg}"
    print(f"    OK: center outside correctly rejected ✓")

    # ── check_mask_nonzero ───────────────────────
    print("\n[4] check_mask_nonzero")
    ok, cnt = check_mask_nonzero(tumor_mask)
    assert ok and cnt > 0
    print(f"    OK: nonzero_count={cnt:,} ✓")

    ok, cnt = check_mask_nonzero(np.zeros(shape, dtype=np.uint8))
    assert not ok
    print(f"    OK: empty mask rejected ✓")

    # ── check_not_overlapping ────────────────────
    print("\n[5] check_not_overlapping")
    # 两个不相交的 mask
    mask_a = np.zeros(shape, dtype=np.uint8)
    mask_a[5:15, 10:20, 10:20] = 1
    mask_b = np.zeros(shape, dtype=np.uint8)
    mask_b[25:35, 50:60, 50:60] = 1
    ok, ov = check_not_overlapping(mask_b, [mask_a])
    assert ok and ov == 0.0
    print(f"    OK: no overlap ✓")

    # 重叠的 mask
    mask_c = np.zeros(shape, dtype=np.uint8)
    mask_c[10:20, 15:25, 15:25] = 1  # 与 mask_a 部分重叠
    ok, ov = check_not_overlapping(mask_c, [mask_a])
    assert not ok
    print(f"    OK: overlap detected (ratio={ov:.4f}) ✓")

    # ── validate_sample ──────────────────────────
    print("\n[6] validate_sample (一站式)")
    result = validate_sample(
        center_zyx=center,
        radius_voxel=radii,
        radius_mm=10.0,
        mask_3d=tumor_mask,
        organ_mask=organ,
        size_category_config={"r_min_mm": 5, "r_max_mm": 20},
    )
    print(f"    passed: {result['passed']}")
    for check_name, (ok, detail) in result['checks'].items():
        print(f"      {check_name}: {'PASS' if ok else 'FAIL'} — {detail}")
    assert result['passed'], f"validate_sample failed: {result['errors']}"
    print(f"    OK: all checks passed ✓")

    # 失败的 validate_sample
    bad_result = validate_sample(
        center_zyx=(0, 5, 5),  # 在器官外
        radius_voxel=radii,
        radius_mm=50.0,          # 超出 large 范围
        mask_3d=np.zeros(shape, dtype=np.uint8),  # 空mask
        organ_mask=organ,
        size_category_config={"r_min_mm": 5, "r_max_mm": 10},
    )
    assert not bad_result['passed']
    print(f"\n    Negative test (expected failures):")
    for check_name, (ok, detail) in bad_result['checks'].items():
        print(f"      {check_name}: {'PASS' if ok else 'FAIL'} — {detail}")
    print(f"    errors: {bad_result['errors']}")
    print(f"    OK: correctly identified {len(bad_result['errors'])} failures ✓")

    print("\n" + "=" * 60)
    print("全部自检通过")
    print("=" * 60)
