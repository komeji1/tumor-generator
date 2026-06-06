"""
Tumor Mask Generator — Mask生成模块
=====================================

执行完整的肿瘤Mask生成管线，从椭球体到最终 .nii.gz 文件。

管线步骤 (按论文顺序):
    ① 创建基础椭球体          — create_ellipsoid()
    ② 弹性形变                — apply_elastic_deformation()
    ③ Salt-Noise 噪声添加     — apply_salt_noise()
    ④ 高斯滤波                — apply_gaussian_smoothing()
    ⑤ 裁剪输出                — scaling & clipping → {0, 1}
    ⑥ 保存为 .nii.gz          — mask_to_nifti()

论文依据:
    DiffTumor (CVPR 2024) §3.3 (P5): "using ellipsoids"
    DiffTumor §F.1 (P22): "ellipse generation, elastic deformation,
        salt-noise generation, Gaussian filtering, scaling, and clipping"

模块依赖:
    utils.py (Step 1): compute_ellipsoid_dist, random_axis_ratios,
        generate_elastic_deformation_field, apply_deformation, ensure_uint8
    data_loader.py (Step 2): CTVolume (for shape/spacing/affine)

被依赖模块:
    main.py (Step 6)
"""

import os
import sys
from typing import Tuple, Optional
import numpy as np
import nibabel as nib

# ── 定位前序步骤 ─────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))

_step1_src = os.path.join(_project_root, 'Step1', 'src')
if _step1_src not in sys.path:
    sys.path.insert(0, _step1_src)

from utils import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    compute_ellipsoid_dist,
    random_axis_ratios,
    generate_elastic_deformation_field,
    apply_deformation,
    ensure_uint8,
    volume_from_radius,
    get_spacing,
)


# ============================================================================
# 半径计算
# ============================================================================

def compute_radii_from_mm(radius_mm: float,
                          spacing: Tuple[float, float, float],
                          axis_ratio_range: Tuple[float, float] = (0.8, 1.2),
                          rng: Optional[np.random.Generator] = None
                          ) -> Tuple[float, float, float]:
    """
    将等效半径(mm)转为三轴体素半径。

    ① 等效半径转为体素单位（取平均 spacing）
    ② 生成随机轴比例（体积守恒）
    ③ 计算各轴体素半径

    Args:
        radius_mm: 肿瘤等效半径 (mm)
        spacing: (dz, dy, dx) 体素间距 mm/voxel
        axis_ratio_range: 各轴比例随机范围
        rng: 随机数生成器

    Returns:
        (rz, ry, rx): 三轴半径（体素单位）
    """
    # 平均体素边长
    mean_spacing = float(np.mean(spacing))
    r_voxel = radius_mm / mean_spacing if mean_spacing > 0 else radius_mm

    # 随机轴比例
    ratio_z, ratio_y, ratio_x = random_axis_ratios(axis_ratio_range, rng=rng)

    rz = r_voxel * ratio_z
    ry = r_voxel * ratio_y
    rx = r_voxel * ratio_x

    return (rz, ry, rx)


# ============================================================================
# 管线步骤 ①: 基础椭球体
# ============================================================================

def create_ellipsoid(shape: Tuple[int, int, int],
                     center_zyx: Tuple[float, float, float],
                     radii_voxel: Tuple[float, float, float]) -> np.ndarray:
    """
    创建基础椭球体 mask（无变形）。

    论文依据: DiffTumor §3.3 (P5): "using ellipsoids"

    Args:
        shape: CT体积形状 (D, H, W)
        center_zyx: 椭球中心体素坐标 (z, y, x)
        radii_voxel: 三轴半径 (rz, ry, rx)，体素单位

    Returns:
        (D, H, W) uint8 二值mask，值域 {0, 1}
    """
    dist = compute_ellipsoid_dist(shape, center_zyx, radii_voxel)
    mask = (dist <= 1.0).astype(np.uint8)
    return mask


# ============================================================================
# 管线步骤 ②: 弹性形变
# ============================================================================

def apply_elastic(mask_3d: np.ndarray,
                  alpha: float = 15.0,
                  sigma: float = 3.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    对 mask 施加弹性形变，使边界自然不规则。

    论文依据: DiffTumor §F.1 (P22): "elastic deformation"

    Args:
        mask_3d: (D, H, W) uint8 二值mask
        alpha: 变形幅度，默认 15
        sigma: 平滑度（体素），默认 3
        rng: 随机数生成器

    Returns:
        (D, H, W) uint8 变形后的二值mask
    """
    field = generate_elastic_deformation_field(mask_3d.shape, alpha, sigma, rng=rng)
    deformed = apply_deformation(mask_3d, field, order=1, mode='nearest')
    return ensure_uint8(deformed)


# ============================================================================
# 管线步骤 ③: Salt-Noise
# ============================================================================

def apply_salt_noise(mask_3d: np.ndarray,
                     probability: float = 0.02,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在 mask 内部随机翻转少量体素，模拟内部纹理不规则。

    论文依据: DiffTumor §F.1 (P22): "salt-noise generation"

    Args:
        mask_3d: (D, H, W) uint8 二值mask
        probability: 每个内部体素被翻转的概率，默认 0.02
        rng: 随机数生成器

    Returns:
        (D, H, W) uint8 带噪声的二值mask
    """
    if rng is None:
        rng = np.random.default_rng()

    result = mask_3d.copy()
    interior = (result == 1)
    interior_indices = np.argwhere(interior)

    if len(interior_indices) == 0:
        return result

    n_flip = max(1, int(len(interior_indices) * probability))
    flip_idx = rng.choice(len(interior_indices), size=n_flip, replace=False)

    for idx in flip_idx:
        z, y, x = interior_indices[idx]
        result[z, y, x] = 0

    return result


# ============================================================================
# 管线步骤 ④: 高斯滤波
# ============================================================================

def apply_gaussian_smoothing(mask_3d: np.ndarray,
                             sigma_mm: float = 1.0,
                             spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                             ) -> np.ndarray:
    """
    对 mask 边界做高斯平滑，使肿瘤边界不锐利。

    论文依据: DiffTumor §F.1 (P22): "Gaussian filtering"

    Args:
        mask_3d: (D, H, W) 二值mask
        sigma_mm: 高斯核 sigma (mm)，默认 1.0mm
        spacing: 体素间距，用于 mm→voxel 转换

    Returns:
        (D, H, W) float32 平滑后的mask（值域 [0, 1]）
    """
    from scipy.ndimage import gaussian_filter

    # 各轴 sigma（体素单位）
    sigma_voxel = tuple(sigma_mm / s if s > 0 else 1.0 for s in spacing)

    smoothed = gaussian_filter(mask_3d.astype(np.float32), sigma=sigma_voxel)
    return smoothed


# ============================================================================
# 管线步骤 ⑤: 裁剪
# ============================================================================

def apply_clipping(mask_3d: np.ndarray,
                   threshold: float = 0.5) -> np.ndarray:
    """
    将平滑后的 mask 裁剪到 {0, 1}。

    论文依据: DiffTumor §F.1 (P22): "clipping"

    Args:
        mask_3d: 浮点mask
        threshold: 二值化阈值，默认 0.5

    Returns:
        uint8 二值mask
    """
    return (mask_3d >= threshold).astype(np.uint8)


# ============================================================================
# 主入口: 完整管线
# ============================================================================

def create_mask(center_zyx: Tuple[float, float, float],
                radius_mm: float,
                shape: Tuple[int, int, int],
                spacing: Tuple[float, float, float],
                shape_config: dict,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    执行完整的 Mask 生成管线。

    管线流程（严格按论文顺序）:
        ① create_ellipsoid()           — 基础椭球
        ② apply_elastic()             — 弹性形变 (if enabled)
        ③ apply_salt_noise()          — 噪声添加 (if enabled)
        ④ apply_gaussian_smoothing()  — 高斯平滑 (if enabled)
        ⑤ apply_clipping()            — 二值化裁剪

    优化: 弹性形变仅在肿瘤周围的裁剪区域内计算，避免在完整CT体积上
    生成巨大位移场（对 512³ 体积可节省 ~4000x 内存和时间）。

    Args:
        center_zyx: 肿瘤中心体素坐标 (z, y, x)
        radius_mm: 肿瘤等效半径 (mm)
        shape: CT体积形状 (D, H, W)
        spacing: 体素间距 (dz, dy, dx)
        shape_config: config['shape'] 字典，控制各步骤参数和开关
        rng: 随机数生成器

    Returns:
        (D, H, W) uint8 二值 mask，值域 {0, 1}
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── ① 基础椭球体 ────────────────────────────────
    axis_range = tuple(shape_config.get('axis_ratio_range', [0.8, 1.2]))
    radii_voxel = compute_radii_from_mm(radius_mm, spacing, axis_range, rng=rng)

    elastic_cfg = shape_config.get('elastic_deformation', {})
    use_elastic = elastic_cfg.get('enabled', True)

    if use_elastic:
        # ── Crop around tumor for efficient elastic deformation ──
        alpha = elastic_cfg.get('alpha', 15.0)
        sigma = elastic_cfg.get('sigma', 3.0)

        # Padding: max displacement (~alpha) + smoothing kernel (~3*sigma) + safety
        max_r_vox = max(radii_voxel)
        pad = int(np.ceil(max_r_vox + alpha + 3 * sigma + 10))

        cz, cy, cx = [int(round(c)) for c in center_zyx]
        D, H, W = shape

        z0 = max(0, cz - pad)
        z1 = min(D, cz + pad + 1)
        y0 = max(0, cy - pad)
        y1 = min(H, cy + pad + 1)
        x0 = max(0, cx - pad)
        x1 = min(W, cx + pad + 1)

        crop_shape = (z1 - z0, y1 - y0, x1 - x0)
        crop_center = (cz - z0, cy - y0, cx - x0)

        # Create ellipsoid on cropped region only
        crop_mask = create_ellipsoid(crop_shape, crop_center, radii_voxel)

        # Elastic deformation on cropped region
        crop_mask = _apply_elastic_cropped(crop_mask, alpha, sigma, rng=rng)

        # Salt noise on cropped region
        noise_cfg = shape_config.get('salt_noise', {})
        if noise_cfg.get('enabled', True):
            crop_mask = apply_salt_noise(
                crop_mask,
                probability=noise_cfg.get('probability', 0.02),
                rng=rng,
            )

        # Place back into full volume
        mask = np.zeros(shape, dtype=np.uint8)
        mask[z0:z1, y0:y1, x0:x1] = crop_mask

        # Gaussian smoothing on full volume (operates only near non-zero region)
        gauss_cfg = shape_config.get('gaussian_filter', {})
        if gauss_cfg.get('enabled', True):
            mask = apply_gaussian_smoothing(
                mask,
                sigma_mm=gauss_cfg.get('sigma_mm', 1.0),
                spacing=spacing,
            )
            mask = apply_clipping(mask)
    else:
        # ── Full volume (no elastic) ─────────────────
        mask = create_ellipsoid(shape, center_zyx, radii_voxel)

        noise_cfg = shape_config.get('salt_noise', {})
        if noise_cfg.get('enabled', True):
            mask = apply_salt_noise(
                mask,
                probability=noise_cfg.get('probability', 0.02),
                rng=rng,
            )

        gauss_cfg = shape_config.get('gaussian_filter', {})
        if gauss_cfg.get('enabled', True):
            mask = apply_gaussian_smoothing(
                mask,
                sigma_mm=gauss_cfg.get('sigma_mm', 1.0),
                spacing=spacing,
            )
            mask = apply_clipping(mask)
        else:
            mask = ensure_uint8(mask)

    # ── ⑤ 最终裁剪 ──────────────────────────────────
    clip_cfg = shape_config.get('scaling_clipping', {})
    if clip_cfg.get('enabled', True):
        mask = apply_clipping(mask)

    return ensure_uint8(mask)


def _apply_elastic_cropped(mask_3d: np.ndarray,
                           alpha: float = 15.0,
                           sigma: float = 3.0,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply elastic deformation on a cropped mask region."""
    field = generate_elastic_deformation_field(mask_3d.shape, alpha, sigma, rng=rng)
    deformed = apply_deformation(mask_3d, field, order=1, mode='nearest')
    return ensure_uint8(deformed)


# ============================================================================
# NIfTI 输出
# ============================================================================

def mask_to_nifti(mask_3d: np.ndarray,
                  affine: np.ndarray,
                  output_path: str) -> str:
    """
    将 3D mask 数组保存为 .nii.gz 文件。

    Args:
        mask_3d: (D, H, W) uint8 二值mask
        affine: 4×4 仿射矩阵（使用对应 CT 的 affine）
        output_path: 输出文件路径（.nii.gz）

    Returns:
        保存的绝对路径

    Note:
        自动创建父目录（如果不存在）。
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    img = nib.Nifti1Image(mask_3d.astype(np.uint8), affine)
    nib.save(img, output_path)

    return os.path.abspath(output_path)


# ============================================================================
# 便捷函数
# ============================================================================

def generate_one_mask(center_zyx: Tuple[float, float, float],
                      radius_mm: float,
                      ct_array: np.ndarray,
                      ct_affine: np.ndarray,
                      config: dict,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[np.ndarray, dict]:
    """
    一站式: 从 CT 数据和配置生成一张 mask + 元数据。

    这是 main.py 调用的便捷入口。

    Args:
        center_zyx: 肿瘤中心
        radius_mm: 等效半径
        ct_array: CT 数据数组（用于提取 shape）
        ct_affine: CT affine 矩阵
        config: 完整配置 dict
        rng: 随机数生成器

    Returns:
        (mask_3d, metadata):
            mask_3d: (D, H, W) uint8
            metadata: 包含 radii/volume 等信息的 dict
    """
    shape = ct_array.shape
    spacing = get_spacing(ct_affine)
    shape_config = config['shape']

    mask = create_mask(center_zyx, radius_mm, shape, spacing, shape_config, rng=rng)

    vol = volume_from_radius(radius_mm, spacing)

    metadata = {
        'center_zyx': center_zyx,
        'radius_mm': radius_mm,
        'estimated_volume_voxels': vol,
        'actual_volume_voxels': int(mask.sum()),
        'shape': shape,
        'spacing': spacing,
    }

    return mask, metadata


# ============================================================================
# 模块自检
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("mask_generator.py — 模块自检")
    print("=" * 60)
    rng = np.random.default_rng(42)

    shape = (40, 80, 80)
    center = (20, 40, 40)
    radius_mm = 10.0
    spacing = (2.0, 1.0, 1.0)
    affine = np.eye(4)
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 2.0

    # ── compute_radii_from_mm ──────────────────────
    print("\n[1] compute_radii_from_mm")
    radii = compute_radii_from_mm(radius_mm, spacing, rng=rng)
    prod = (radii[0] * radii[1] * radii[2]) / ((radius_mm / 1.33) ** 3)
    print(f"    radius_mm={radius_mm}, spacing={spacing}")
    print(f"    radii_voxel=({radii[0]:.1f}, {radii[1]:.1f}, {radii[2]:.1f})")
    assert all(r > 0 for r in radii)
    print("    OK")

    # ── create_ellipsoid ───────────────────────────
    print("\n[2] create_ellipsoid")
    ellipsoid = create_ellipsoid(shape, center, radii)
    vol = int(ellipsoid.sum())
    assert vol > 100, f"volume too small: {vol}"
    assert ellipsoid.dtype == np.uint8
    assert ellipsoid[center[0], center[1], center[2]] == 1
    print(f"    shape={shape}, center={center}")
    print(f"    volume={vol:,} voxels, center={ellipsoid[center[0],center[1],center[2]]}")
    print("    OK")

    # ── apply_elastic ──────────────────────────────
    print("\n[3] apply_elastic")
    deformed = apply_elastic(ellipsoid, alpha=10, sigma=3, rng=rng)
    vol_change = abs(float(deformed.sum()) - float(ellipsoid.sum())) / float(ellipsoid.sum())
    assert deformed.dtype == np.uint8
    assert vol_change < 0.3, f"vol_change={vol_change:.1%}"
    print(f"    original={ellipsoid.sum():,}, deformed={deformed.sum():,}")
    print(f"    vol_change={vol_change:.1%}")
    print("    OK")

    # ── apply_salt_noise ───────────────────────────
    print("\n[4] apply_salt_noise")
    noisy = apply_salt_noise(deformed, probability=0.02, rng=rng)
    assert noisy.dtype == np.uint8
    loss = int(deformed.sum()) - int(noisy.sum())
    print(f"    before={deformed.sum():,}, after={noisy.sum():,}, lost={loss}")
    print("    OK")

    # ── apply_gaussian_smoothing ───────────────────
    print("\n[5] apply_gaussian_smoothing")
    smoothed = apply_gaussian_smoothing(noisy, sigma_mm=1.0, spacing=spacing)
    assert smoothed.dtype == np.float32
    assert 0.0 <= smoothed.min() <= smoothed.max() <= 1.0
    print(f"    range=[{smoothed.min():.3f}, {smoothed.max():.3f}]")
    print("    OK")

    # ── apply_clipping ─────────────────────────────
    print("\n[6] apply_clipping")
    binary = apply_clipping(smoothed)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 1})
    print(f"    volume={binary.sum():,}, unique_values={np.unique(binary).tolist()}")
    print("    OK")

    # ── create_mask (完整管线) ──────────────────────
    print("\n[7] create_mask (完整管线)")
    shape_cfg = {
        'method': 'ellipsoid',
        'axis_ratio_range': [0.8, 1.2],
        'elastic_deformation': {'enabled': True, 'alpha': 15, 'sigma': 3},
        'salt_noise': {'enabled': True, 'probability': 0.02},
        'gaussian_filter': {'enabled': True, 'sigma_mm': 1.0},
        'scaling_clipping': {'enabled': True},
    }
    mask = create_mask(center, radius_mm, shape, spacing, shape_cfg, rng=rng)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    print(f"    shape={mask.shape}, volume={mask.sum():,}")
    print(f"    unique_values={np.unique(mask).tolist()}")
    print("    OK")

    # ── 用不同随机种子生成5个，验证多样性 ──────────
    print("\n[8] 多样性验证 (5 samples)")
    volumes = []
    for i in range(5):
        rng_i = np.random.default_rng(i * 100)
        m = create_mask(center, radius_mm, shape, spacing, shape_cfg, rng=rng_i)
        volumes.append(int(m.sum()))
        print(f"    seed={i*100:3d}: volume={volumes[-1]:,}")
    # 体积应该有变化（弹性形变 + 轴比例随机）
    unique_vols = len(set(volumes))
    print(f"    unique volumes: {unique_vols}/5")
    assert unique_vols >= 2, "All volumes identical — no diversity"
    print("    OK")

    # ── 管线步骤可关闭 ──────────────────────────────
    print("\n[9] 管线开关测试")
    cfg_minimal = {
        'method': 'ellipsoid',
        'elastic_deformation': {'enabled': False},
        'salt_noise': {'enabled': False},
        'gaussian_filter': {'enabled': False},
        'scaling_clipping': {'enabled': True},
    }
    mask_minimal = create_mask(center, radius_mm, shape, spacing, cfg_minimal, rng=rng)
    vol_min = int(mask_minimal.sum())
    print(f"    all disabled: volume={vol_min:,} (pure ellipsoid)")
    assert mask_minimal.sum() > 0
    print("    OK")

    # ── mask_to_nifti ──────────────────────────────
    print("\n[10] mask_to_nifti")
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="mask_test_")
    output_path = os.path.join(tmpdir, 'test_mask.nii.gz')
    saved_path = mask_to_nifti(mask, affine, output_path)
    assert os.path.exists(saved_path)
    # 验证可重新加载
    reloaded = nib.load(saved_path)
    reloaded_data = reloaded.get_fdata()
    assert reloaded_data.shape == mask.shape
    assert np.allclose(reloaded.affine, affine)
    print(f"    saved: {saved_path} ({os.path.getsize(saved_path):,}B)")
    print(f"    reloaded shape={reloaded_data.shape}, values={np.unique(reloaded_data).tolist()}")
    import shutil; shutil.rmtree(tmpdir)
    print("    OK")

    print("\n" + "=" * 60)
    print("全部自检通过")
    print("=" * 60)
