"""
Tumor Mask Generator — 主入口
===============================

批量生成肿瘤位置Mask：6类器官 × 每类20张 = 120个 .nii.gz 文件。

依赖: 所有 Step0~Step5 模块
    utils.py              (Step 1) — get_spacing
    data_loader.py        (Step 2) — CTVolume, OrganMask, load_ct, load_organ_mask, build_manifest
    validator.py          (Step 3) — validate_sample
    position_selector.py  (Step 4) — select_location_from_config
    mask_generator.py     (Step 5) — create_mask, mask_to_nifti

用法:
    python main.py                          # 使用默认 config
    python main.py --config path/to.json    # 指定配置
    python main.py --dry-run                # 预览不生成
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np

# ── 定位所有前序步骤 ──────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))

for step in ['Step0', 'Step1', 'Step2', 'Step3', 'Step4', 'Step5']:
    step_src = os.path.join(_project_root, step, 'src')
    if step_src not in sys.path:
        sys.path.insert(0, step_src)

from utils import get_spacing  # noqa: E402  # pyright: ignore[reportMissingImports]
from data_loader import (  # noqa: E402
    load_ct, load_organ_mask, build_manifest,
    save_manifest_csv, validate_compatibility,
)
from validator import validate_sample  # noqa: E402
from position_selector import select_location_from_config  # noqa: E402
from mask_generator import create_mask, mask_to_nifti  # noqa: E402


# ============================================================================
# 配置加载
# ============================================================================

def load_config(config_path: str = "config/generation_config.json") -> dict:
    """
    加载 JSON 配置文件。

    搜索顺序:
        ① 给定路径
        ② Step0/config/generation_config.json (相对于项目根目录)

    Args:
        config_path: JSON 配置文件路径

    Returns:
        配置字典
    """
    # 如果给定路径不存在，尝试 Step0/
    if not os.path.exists(config_path):
        alt_path = os.path.join(_project_root, 'Step0', 'config', 'generation_config.json')
        if os.path.exists(alt_path):
            config_path = alt_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Please create it or specify --config path."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"Config loaded: {config_path}")
    return config


# ============================================================================
# 尺寸采样
# ============================================================================

def sample_size_category(size_config: dict,
                         rng: Optional[np.random.Generator] = None) -> str:
    """
    按 4:2:1 权重采样 size_category。

    论文依据: Scaling Tumor (ICCV 2025) §4.1 (P7):
        "ratio of 4:2:1 for small, medium, and large tumors"

    只从 weight > 0 的分类中采样。tiny (weight=0) 不参与自动采样。

    Args:
        size_config: config['size_categories']
        rng: 随机数生成器

    Returns:
        分类名称: "small" / "medium" / "large"
    """
    if rng is None:
        rng = np.random.default_rng()

    cats = size_config['categories']
    # 只取 weight > 0 的分类
    active = [(name, cfg['weight']) for name, cfg in cats.items() if cfg['weight'] > 0]

    if not active:
        raise ValueError("No size categories with weight > 0")

    names, weights = zip(*active)
    total = sum(weights)
    probs = [w / total for w in weights]

    chosen = rng.choice(names, p=probs)
    return str(chosen)


def sample_radius(size_category_name: str,
                  size_config: dict,
                  rng: Optional[np.random.Generator] = None) -> float:
    """
    在指定 size_category 的半径范围内 uniform 采样。

    Args:
        size_category_name: 分类名称
        size_config: config['size_categories']
        rng: 随机数生成器

    Returns:
        肿瘤等效半径 (mm)
    """
    if rng is None:
        rng = np.random.default_rng()

    cat = size_config['categories'][size_category_name]
    r_min = cat['r_min_mm']
    r_max = cat['r_max_mm']

    return float(rng.uniform(r_min, r_max))


# ============================================================================
# 单样本生成
# ============================================================================

def generate_one(ct_path: str,
                 organ_mask_path: str,
                 organ_type: str,
                 organ_label: str,
                 sample_id: str,
                 config: dict,
                 rng: Optional[np.random.Generator] = None) -> dict:
    """
    为单个样本生成一张肿瘤mask的完整流程。

    流程:
        ① 加载 CT + 器官mask
        ② 采样尺寸分类 + 半径
        ③ 选择肿瘤位置
        ④ 生成 mask
        ⑤ 校验
        ⑥ 保存 .nii.gz
        ⑦ 返回元数据

    Args:
        ct_path: CT .nii.gz 路径
        organ_mask_path: 器官mask .nii.gz 路径
        organ_type: 肿瘤类型名称
        organ_label: 器官标签文件名
        sample_id: 样本ID (如 "BDMAP_00000001")
        config: 完整配置
        rng: 随机数生成器

    Returns:
        metadata dict:
            success, organ_type, sample_id, size_category, radius_mm,
            center_zyx, output_path, validation, error, ...
    """
    if rng is None:
        rng = np.random.default_rng()

    metadata = {
        'organ_type': organ_type,
        'sample_id': sample_id,
        'success': False,
    }

    try:
        # ── ① 加载 ─────────────────────────────────────
        ct = load_ct(ct_path,
                     hu_min=config['preprocessing']['hu_min'],
                     hu_max=config['preprocessing']['hu_max'])
        organ_mask = load_organ_mask(organ_mask_path, organ_type, organ_label)
        validate_compatibility(ct, organ_mask)

        metadata['ct_shape'] = ct.shape
        metadata['spacing'] = ct.spacing
        metadata['organ_volume'] = int(organ_mask.array.sum())

        # ── ② Pre-compute distance transform (once for all sizes) ──
        from scipy.ndimage import distance_transform_edt
        mean_spacing = float(np.mean(ct.spacing))

        # Crop organ to bbox for efficiency
        z_idx = np.any(organ_mask.array, axis=(1, 2))
        y_idx = np.any(organ_mask.array, axis=(0, 2))
        x_idx = np.any(organ_mask.array, axis=(0, 1))
        if not z_idx.any():
            raise RuntimeError("Empty organ mask")
        zr = np.where(z_idx)[0]; yr = np.where(y_idx)[0]; xr = np.where(x_idx)[0]
        pad = 20
        z0, z1 = max(0, zr[0]-pad), min(organ_mask.array.shape[0], zr[-1]+pad+1)
        y0, y1 = max(0, yr[0]-pad), min(organ_mask.array.shape[1], yr[-1]+pad+1)
        x0, x1 = max(0, xr[0]-pad), min(organ_mask.array.shape[2], xr[-1]+pad+1)
        organ_crop = organ_mask.array[z0:z1, y0:y1, x0:x1].astype(bool)

        # Distance from each interior voxel to nearest boundary
        dist_map = distance_transform_edt(organ_crop)
        max_dist = float(dist_map.max())

        # ── ③ Check which sizes can fit ─────────────────────
        # Required margin = radius_voxel + feather_vox + safety_vox
        # The tumor can fit if max_dist > required_margin
        size_cat = sample_size_category(config['size_categories'], rng)
        radius_mm = sample_radius(size_cat, config['size_categories'], rng)
        metadata['size_category'] = size_cat
        metadata['radius_mm'] = round(radius_mm, 2)

        # Determine viable sizes (can fit in this organ)
        spacing_z = ct.spacing[0]
        viable_sizes = []
        for cat in ['large', 'medium', 'small', 'tiny']:
            cat_cfg = config['size_categories']['categories'].get(cat)
            if cat_cfg is None:
                continue
            margin_cfg = config['placement'].get('margin', {})
            f_mm = 0.5 if cat == 'tiny' else margin_cfg.get('feather_mm', 1)
            s_mm = 0.5 if cat == 'tiny' else margin_cfg.get('safety_mm', 2)
            # Use max radius for conservative check
            r_max_mm = cat_cfg['r_max_mm']
            r_vox_max = r_max_mm / mean_spacing if mean_spacing > 0 else r_max_mm
            margin_vox = r_vox_max + f_mm / spacing_z + s_mm / spacing_z
            if max_dist > margin_vox:
                viable_sizes.append(cat)

        if 'tiny' not in viable_sizes:
            viable_sizes.append('tiny')  # Always try tiny as last resort

        # ── ④ Filter fallback order to viable sizes ─────────
        fallback_order = ['large', 'medium', 'small', 'tiny']
        try_idx = fallback_order.index(size_cat) if size_cat in fallback_order else 0
        fallback_cats = [c for c in fallback_order[try_idx:] if c in viable_sizes]

        center_zyx = None
        for attempt_cat in fallback_cats:
            cat_cfg = config['size_categories']['categories'].get(attempt_cat)
            if cat_cfg is None:
                continue
            if attempt_cat != 'tiny' and cat_cfg.get('weight', 0) <= 0:
                continue
            if attempt_cat == size_cat:
                r_mm = radius_mm
            else:
                r_mm = sample_radius(attempt_cat, config['size_categories'], rng)
            r_vox = r_mm / mean_spacing if mean_spacing > 0 else r_mm

            # Quick check: can this specific radius fit?
            f_mm = 0.5 if attempt_cat == 'tiny' else config['placement']['margin'].get('feather_mm', 1)
            s_mm = 0.5 if attempt_cat == 'tiny' else config['placement']['margin'].get('safety_mm', 2)
            margin_vox = r_vox + f_mm / spacing_z + s_mm / spacing_z
            if max_dist <= margin_vox:
                continue  # This radius won't fit

            temp_placement = dict(config['placement'])
            if attempt_cat == 'tiny':
                temp_placement['margin'] = {'feather_mm': 0.5, 'safety_mm': 0.5}

            try:
                center_zyx = select_location_from_config(
                    organ_mask=organ_mask.array,
                    radius_voxel=r_vox,
                    spacing=ct.spacing,
                    placement_config=temp_placement,
                    rng=rng,
                )
                metadata['size_category'] = attempt_cat
                metadata['radius_mm'] = round(r_mm, 2)
                radius_voxel = r_vox
                radius_mm = r_mm
                break
            except (ValueError, RuntimeError):
                continue

        if center_zyx is None:
            raise RuntimeError(
                f"Cannot compute valid region for organ: "
                f"max_dist={max_dist:.1f}, organ_vol={int(organ_mask.array.sum())}. "
                f"Viable sizes: {viable_sizes}"
            )

        metadata['center_zyx'] = center_zyx

        # ── ④ 生成 mask ────────────────────────────────
        # For tiny tumors, disable elastic deformation to avoid boundary overflow
        shape_cfg = dict(config['shape'])
        if radius_mm < 5:
            shape_cfg['elastic_deformation'] = {'enabled': False}
            shape_cfg['gaussian_filter'] = {'enabled': True, 'sigma_mm': 0.3}

        mask_3d = create_mask(
            center_zyx=center_zyx,
            radius_mm=radius_mm,
            shape=ct.shape,
            spacing=ct.spacing,
            shape_config=shape_cfg,
            rng=rng,
        )
        metadata['mask_volume_voxels'] = int(mask_3d.sum())

        # Reject immediately if generated mask is empty
        if metadata['mask_volume_voxels'] == 0:
            raise RuntimeError(
                f"Created mask has zero volume. radius_mm={radius_mm:.1f}, "
                f"center={center_zyx}, shape={ct.shape}"
            )

        # ── ⑤ 校验 ─────────────────────────────────────
        # 使用等半径球体验证（保守策略）
        radii_3d = (radius_voxel, radius_voxel, radius_voxel)

        validation = validate_sample(
            center_zyx=center_zyx,
            radius_voxel=radii_3d,
            radius_mm=radius_mm,
            mask_3d=mask_3d,
            organ_mask=organ_mask.array,
            size_category_config=config['size_categories']['categories'][size_cat],
        )
        metadata['validation_passed'] = validation['passed']
        if not validation['passed']:
            metadata['validation_errors'] = validation['errors']

        # ── ⑥ Clip to organ boundary ───────────────────────────────
        # Ensure tumor is fully inside organ (fix for elastic deformation overflow)
        mask_3d = ((mask_3d > 0) & (organ_mask.array > 0)).astype(np.uint8)
        final_vol = int(mask_3d.sum())
        metadata['final_volume_voxels'] = final_vol

        # Reject if volume is too small
        original_vol = metadata['mask_volume_voxels']
        if final_vol < 3:
            raise RuntimeError(
                f"Final tumor volume too small: {final_vol} voxels. "
                f"Original={original_vol}. Organ may be too small for this tumor size."
            )

        # Reject if too much was clipped (more than 20% lost)
        clip_loss = (original_vol - final_vol) / max(1, original_vol)
        if clip_loss > 0.20:
            raise RuntimeError(
                f"Too much clipped ({clip_loss:.1%} lost). Original={original_vol}, final={final_vol}. "
                f"Tumor position may be at organ edge."
            )
        metadata['clip_loss_pct'] = round(clip_loss * 100, 1)

        # ── ⑦ 保存 ─────────────────────────────────────
        naming = config['output']['naming_pattern']
        filename = naming.format(organ_type=organ_type, sample_id=sample_id)
        output_dir = os.path.join(
            _project_root,
            config['project']['output_dir'],
            organ_type,
        )
        output_path = os.path.join(output_dir, filename)

        saved = mask_to_nifti(mask_3d, ct.affine, output_path)
        metadata['output_path'] = saved
        metadata['success'] = True

    except Exception as e:
        metadata['success'] = False
        metadata['error'] = str(e)
        metadata['error_type'] = type(e).__name__

    return metadata


# ============================================================================
# 批量生成
# ============================================================================

def generate_batch(config: dict,
                   rng_seed: int = 42) -> List[dict]:
    """
    批量生成主循环: 6类器官 × 每类20张 = 最多120个mask。

    Args:
        config: 完整配置字典
        rng_seed: 随机种子（确保可复现）

    Returns:
        results: metadata 列表
    """
    rng = np.random.default_rng(rng_seed)
    results = []

    organs = config['organs']
    total_target = sum(o['count'] for o in organs)

    # ── 构建 manifest ──────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1/4: 构建样本索引")
    print("=" * 60)

    ct_dir = os.path.join(_project_root, config['data']['ct_dir'])
    label_dir = os.path.join(_project_root, config['data']['organ_label_dir'])

    try:
        manifest = build_manifest(ct_dir, label_dir, organs)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Running in DRY-RUN / synthetic mode.\n")
        manifest = _build_synthetic_manifest(organs)

    # 只保留存在的样本
    available = [m for m in manifest if m['exists']]

    if not available:
        print("  No real samples available. Using fully synthetic data.\n")
        available = _build_synthetic_manifest(organs)
        for m in available:
            m['exists'] = True

    print(f"  {len(available)} available, target={total_target}")

    # ── 按器官分组 + 分配 ──────────────────────────
    # 简单策略: 如果可用样本 >= 目标数，随机选取；否则允许复用
    from collections import defaultdict
    by_organ = defaultdict(list)
    for m in available:
        by_organ[m['organ_type']].append(m)

    # ── 批量循环 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2/4: 生成Mask")
    print("=" * 60)

    generated = 0
    start_time = time.time()

    for organ_cfg in organs:
        organ_type = organ_cfg['name']
        target_count = organ_cfg['count']
        organ_samples = by_organ.get(organ_type, [])

        if not organ_samples:
            print(f"\n  {organ_type}: 0 samples — SKIPPED")
            continue

        print(f"\n  {organ_type} (target: {target_count} successful):")

        success_count = 0
        attempt = 0
        max_attempts = target_count * 5  # Allow up to 5x attempts to reach target

        while success_count < target_count and attempt < max_attempts:
            # 循环选取样本
            sample = organ_samples[attempt % len(organ_samples)]
            sample_id = f"{sample['sample_id']}_t{success_count:02d}"

            seed = rng_seed + generated + attempt
            local_rng = np.random.default_rng(seed)

            meta = generate_one(
                ct_path=sample['ct_path'],
                organ_mask_path=sample['organ_mask_path'],
                organ_type=organ_type,
                organ_label=sample['organ_label'],
                sample_id=sample_id,
                config=config,
                rng=local_rng,
            )

            results.append(meta)
            generated += 1
            attempt += 1

            if meta['success']:
                success_count += 1
                status = "OK"
                print(f"    [{success_count:2d}/{target_count}] {meta.get('size_category','?'):6s} "
                      f"r={meta.get('radius_mm',0):.0f}mm  "
                      f"center={meta.get('center_zyx',('?',))}  {status}")
            else:
                # Failed but continue trying
                err_short = meta.get('error', '?')[:50]
                if attempt % 10 == 0:  # Print every 10 failures
                    print(f"    ... attempt {attempt}, {success_count} successful so far")

        if success_count < target_count:
            print(f"    WARNING: Only {success_count}/{target_count} successful after {attempt} attempts")
        else:
            print(f"    Done: {success_count} successful masks")

    elapsed = time.time() - start_time

    # ── 统计 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3/4: 统计汇总")
    print("=" * 60)
    stats = compute_statistics(results, config)
    print(f"  成功: {stats['success_count']}/{stats['total']}")
    print(f"  失败: {stats['failure_count']}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/stats['total']:.1f}s/sample)" if stats['total'] > 0 else "")
    print(f"  尺寸分布: {stats['size_distribution']}")
    if stats['failure_count'] > 0:
        print(f"  失败详情:")
        for r in results:
            if not r['success']:
                print(f"    - {r['organ_type']}/{r['sample_id']}: {r.get('error','?')[:80]}")

    # ── 保存日志 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4/4: 保存日志")
    print("=" * 60)

    log_cfg = config['logging']
    log_path = os.path.join(_project_root, log_cfg['log_file'])
    stats_path = os.path.join(_project_root, log_cfg['stats_file'])

    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

    # 将不可序列化的字段转换
    serializable_results = []
    for r in results:
        sr = dict(r)
        if 'center_zyx' in sr and sr['center_zyx'] is not None:
            sr['center_zyx'] = list(sr['center_zyx'])
        if 'ct_shape' in sr and sr['ct_shape'] is not None:
            sr['ct_shape'] = list(sr['ct_shape'])
        if 'spacing' in sr and sr['spacing'] is not None:
            sr['spacing'] = list(sr['spacing'])
        serializable_results.append(sr)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  日志: {log_path}")

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"  统计: {stats_path}")

    return results


def _build_synthetic_manifest(organs: List[dict]) -> List[dict]:
    """构建合成 manifest（无真实数据时的回退方案）。"""
    manifest = []
    for organ_cfg in organs:
        for i in range(organ_cfg['count']):
            manifest.append({
                'sample_id': f'SYNTHETIC_{i:03d}',
                'ct_path': '',       # 需要真实CT文件
                'organ_type': organ_cfg['name'],
                'organ_label': organ_cfg['organ_label_file'],
                'organ_mask_path': '',
                'exists': False,
            })
    return manifest


# ============================================================================
# 统计
# ============================================================================

def compute_statistics(results: List[dict],
                       config: dict) -> dict:
    """
    汇总统计: 成功率、尺寸分布、器官分布。

    Args:
        results: generate_batch 的返回列表
        config: 完整配置

    Returns:
        统计 dict
    """
    total = len(results)
    success = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]

    # 尺寸分布
    size_dist = {}
    for r in success:
        cat = r.get('size_category', 'unknown')
        size_dist[cat] = size_dist.get(cat, 0) + 1

    # 器官分布
    organ_dist = {}
    for r in success:
        org = r.get('organ_type', 'unknown')
        organ_dist[org] = organ_dist.get(org, 0) + 1

    # 目标分布 (来自 config)
    target_weights = {
        name: cfg['weight']
        for name, cfg in config['size_categories']['categories'].items()
        if cfg['weight'] > 0
    }
    total_weight = sum(target_weights.values())
    target_dist = {
        name: round(w / total_weight * len(success)) if total_weight > 0 else 0
        for name, w in target_weights.items()
    }

    return {
        'total': total,
        'success_count': len(success),
        'failure_count': len(failures),
        'success_rate': len(success) / total if total > 0 else 0,
        'size_distribution': size_dist,
        'target_distribution': target_dist,
        'organ_distribution': organ_dist,
        'failure_details': [
            {'organ_type': r.get('organ_type'), 'sample_id': r.get('sample_id'),
             'error': r.get('error', 'unknown')}
            for r in failures[:10]  # 只保留前10条
        ],
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description='Tumor Mask Generator — 批量生成肿瘤位置Mask',
    )
    parser.add_argument('--config', type=str, default=None,
                        help='JSON配置文件路径 (默认: Step0/config/generation_config.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式: 只打印计划，不实际生成')
    parser.add_argument('--organ', type=str, default=None,
                        help='只生成指定器官类型 (如 liver_lesion)')

    args = parser.parse_args()

    print("=" * 60)
    print("Tumor Mask Generator v1.0.0")
    print("=" * 60)

    # 加载配置
    config_path = args.config or os.path.join(
        _project_root, 'Step0', 'config', 'generation_config.json'
    )
    config = load_config(config_path)

    # 预览模式
    if args.dry_run:
        organs = config['organs']
        if args.organ:
            organs = [o for o in organs if o['name'] == args.organ]
        print(f"\n[Dry Run] Would generate:")
        for o in organs:
            print(f"  {o['name']}: {o['count']} masks")
        print(f"  Total: {sum(o['count'] for o in organs)} masks")
        return

    # 限制单器官
    if args.organ:
        config = dict(config)  # shallow copy
        config['organs'] = [o for o in config['organs'] if o['name'] == args.organ]
        if not config['organs']:
            print(f"ERROR: Unknown organ type '{args.organ}'")
            print(f"  Available: {[o['name'] for o in load_config(config_path)['organs']]}")
            sys.exit(1)

    # 执行批量生成
    results = generate_batch(config, rng_seed=args.seed)

    # 最终摘要
    success = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"完成: {success}/{total} 成功")
    print(f"{'='*60}")

    if success < total:
        sys.exit(1)


# ============================================================================
# 模块自检
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("main.py — 模块自检 (合成数据模式)")
    print("=" * 60)

    # ── 构建最小合成配置 ──────────────────────────
    import tempfile
    import nibabel as nib

    tmpdir = tempfile.mkdtemp(prefix="main_test_")
    print(f"\n临时目录: {tmpdir}")

    # 合成数据 — 足够大的体积确保 margin 不会清空器官
    shape = (80, 80, 80)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0

    # CT
    ct_data = np.random.default_rng(0).integers(-175, 250, shape, dtype=np.int16)
    ct_dir = os.path.join(tmpdir, 'ct', 'BDMAP_TEST01')
    os.makedirs(ct_dir, exist_ok=True)
    ct_path = os.path.join(ct_dir, 'ct.nii.gz')
    nib.save(nib.Nifti1Image(ct_data, affine), ct_path)

    # 器官mask — 大范围确保经得起 margin 腐蚀
    organ_data = np.zeros(shape, dtype=np.uint8)
    organ_data[10:70, 10:70, 10:70] = 1  # 216,000 voxels
    label_dir = os.path.join(tmpdir, 'labels', 'BDMAP_TEST01', 'segmentations')
    os.makedirs(label_dir, exist_ok=True)
    organ_path = os.path.join(label_dir, 'liver.nii.gz')
    nib.save(nib.Nifti1Image(organ_data, affine), organ_path)

    # 最小配置
    mini_config = {
        'project': {'output_dir': os.path.join(tmpdir, 'output')},
        'data': {
            'ct_dir': os.path.join(tmpdir, 'ct'),
            'organ_label_dir': os.path.join(tmpdir, 'labels'),
            'manifest_path': os.path.join(tmpdir, 'manifest.csv'),
        },
        'organs': [
            {'name': 'liver_lesion', 'class_id': 27, 'organ_label_file': 'liver.nii.gz',
             'organ_name': 'liver', 'count': 3},
        ],
        'size_categories': {
            'categories': {
                'tiny': {'r_min_mm': 1, 'r_max_mm': 5, 'weight': 0},
                'small': {'r_min_mm': 5, 'r_max_mm': 10, 'weight': 4},
                'medium': {'r_min_mm': 10, 'r_max_mm': 20, 'weight': 2},
                'large': {'r_min_mm': 20, 'r_max_mm': 50, 'weight': 1},
            }
        },
        'shape': {
            'axis_ratio_range': [0.8, 1.2],
            'elastic_deformation': {'enabled': True, 'alpha': 10, 'sigma': 2},
            'salt_noise': {'enabled': True, 'probability': 0.02},
            'gaussian_filter': {'enabled': True, 'sigma_mm': 0.5},
            'scaling_clipping': {'enabled': True},
        },
        'placement': {
            'strategy': 'uniform',
            'margin': {'feather_mm': 2, 'safety_mm': 3},
            'max_retries': 20,
            'distance_weighted': {'alpha': 1.0},
        },
        'preprocessing': {'hu_min': -175, 'hu_max': 250},
        'output': {
            'format': 'nifti', 'dtype': 'uint8', 'value_range': [0, 1],
            'compress': True, 'naming_pattern': '{organ_type}_{sample_id}.nii.gz',
        },
        'logging': {
            'log_file': os.path.join(tmpdir, 'output', 'generation_log.json'),
            'stats_file': os.path.join(tmpdir, 'output', 'statistics.json'),
            'verbose': True,
        },
    }

    # ── sample_size_category ───────────────────────
    print("\n[1] sample_size_category")
    rng = np.random.default_rng(42)
    cats = [sample_size_category(mini_config['size_categories'], rng) for _ in range(20)]
    from collections import Counter
    dist = Counter(cats)
    print(f"    20 samples: {dict(dist)}")
    # 应该有 small/medium/large，不应有 tiny
    assert 'tiny' not in dist
    print("    OK")

    # ── sample_radius ──────────────────────────────
    print("\n[2] sample_radius")
    for cat in ['small', 'medium', 'large']:
        r = sample_radius(cat, mini_config['size_categories'], rng)
        cfg = mini_config['size_categories']['categories'][cat]
        assert cfg['r_min_mm'] < r <= cfg['r_max_mm'], f"{cat}: {r} outside range"
        print(f"    {cat}: r={r:.1f}mm (range [{cfg['r_min_mm']},{cfg['r_max_mm']}])")
    print("    OK")

    # ── generate_one ───────────────────────────────
    print("\n[3] generate_one (单样本)")
    meta = generate_one(
        ct_path=ct_path,
        organ_mask_path=organ_path,
        organ_type='liver_lesion',
        organ_label='liver.nii.gz',
        sample_id='TEST_001',
        config=mini_config,
        rng=rng,
    )
    print(f"    success: {meta['success']}")
    print(f"    size_category: {meta.get('size_category')}")
    print(f"    radius_mm: {meta.get('radius_mm')}")
    print(f"    center_zyx: {meta.get('center_zyx')}")
    print(f"    mask_volume_voxels: {meta.get('mask_volume_voxels')}")
    if meta['success']:
        assert os.path.exists(meta['output_path']), f"output not found: {meta['output_path']}"
        # 验证输出文件
        reloaded = nib.load(meta['output_path'])
        assert reloaded.get_fdata().shape == shape
        print(f"    output: {meta['output_path']} ({os.path.getsize(meta['output_path']):,}B)")
    else:
        print(f"    error: {meta.get('error')}")
    print("    OK")

    # ── generate_batch ─────────────────────────────
    print("\n[4] generate_batch (批量)")
    results = generate_batch(mini_config, rng_seed=42)
    success = sum(1 for r in results if r['success'])
    print(f"\n    结果: {success}/{len(results)} 成功")
    stats = compute_statistics(results, mini_config)
    print(f"    成功率: {stats['success_rate']:.0%}")
    print(f"    尺寸分布: {stats['size_distribution']}")
    assert success >= 1, f"All failed: {[r.get('error') for r in results]}"
    print("    OK")

    # ── compute_statistics ─────────────────────────
    print("\n[5] compute_statistics")
    print(f"    total={stats['total']}, success={stats['success_count']}")
    print(f"    size_dist={stats['size_distribution']}")
    print(f"    target_dist={stats['target_distribution']}")
    print("    OK")

    # 清理
    import shutil
    shutil.rmtree(tmpdir)
    print(f"\n临时目录已清理")

    print("\n" + "=" * 60)
    print("全部自检通过")
    print("=" * 60)
