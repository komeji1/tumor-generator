"""
Tumor Mask Generator — 数据加载模块
=====================================

负责加载 CT 影像 (.nii.gz) 和器官分割 mask，提供统一的数据结构。

模块依赖:
    sys.path 中需包含 Step1/src
    from utils import clip_hu, ensure_uint8, get_spacing

数据结构:
    CTVolume    — CT 扫描的标准化容器
    OrganMask   — 器官分割 mask 的标准化容器

被依赖模块:
    position_selector.py (Step 4)
    mask_generator.py    (Step 5)
    main.py              (Step 6)
"""

import os
import sys
import json
import csv
import warnings
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np
import nibabel as nib

# ── 根据运行时位置导入 utils ──────────────────────────────────
# 本项目按 StepN/src/ 分步存放，运行时通过 sys.path 动态定位依赖。
# IDE 可能报 "无法解析导入" — 这是静态分析的限制，不影响运行。
# 验证方式: python Step2/src/data_loader.py (9/9 自检通过)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))  # Mask/
_step1_src = os.path.join(_project_root, 'Step1', 'src')
if _step1_src not in sys.path:
    sys.path.insert(0, _step1_src)

from utils import clip_hu, ensure_uint8, get_spacing  # noqa: E402  # pyright: ignore[reportMissingImports]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CTVolume:
    """
    CT 扫描数据的标准化容器。

    Attributes:
        array:    (D, H, W) uint16/int16 数组，HU值（已裁剪到 [hu_min, hu_max]）
        affine:   (4, 4) float 仿射矩阵，体素坐标→物理坐标
        spacing:  (dz, dy, dx) 体素间距，单位 mm/voxel
        shape:    (D, H, W) 体积形状
        path:     原始 .nii.gz 文件路径
    """
    array: np.ndarray
    affine: np.ndarray
    spacing: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    path: str = ""

    def __repr__(self) -> str:
        return (f"CTVolume(shape={self.shape}, "
                f"spacing=({self.spacing[0]:.2f},{self.spacing[1]:.2f},{self.spacing[2]:.2f}), "
                f"dtype={self.array.dtype}, "
                f"path='{os.path.basename(self.path)}')")


@dataclass
class OrganMask:
    """
    器官分割 mask 的标准化容器。

    Attributes:
        array:      (D, H, W) uint8 二值数组，值域 {0, 1}
        affine:     (4, 4) float 仿射矩阵（应与对应 CT 一致）
        organ_type: 器官名称，如 "liver_lesion"
        organ_label:器官标签文件名，如 "liver.nii.gz"
        path:       原始 .nii.gz 文件路径
    """
    array: np.ndarray
    affine: np.ndarray
    organ_type: str
    organ_label: str = ""
    path: str = ""

    def __repr__(self) -> str:
        vol = int(self.array.sum())
        return (f"OrganMask(type={self.organ_type}, "
                f"shape={self.array.shape}, "
                f"volume={vol:,} voxels, "
                f"path='{os.path.basename(self.path)}')")


@dataclass
class Sample:
    """
    一个完整的训练/生成样本。

    封装了一对 CT + 器官 mask，供 main.py 的批量循环使用。
    """
    ct: CTVolume
    organ_mask: OrganMask
    sample_id: str = ""

    def __repr__(self) -> str:
        return (f"Sample(id={self.sample_id}, "
                f"organ={self.organ_mask.organ_type}, "
                f"ct_shape={self.ct.shape})")


# ============================================================================
# 文件加载
# ============================================================================

def load_ct(ct_path: str,
            hu_min: float = -175,
            hu_max: float = 250) -> CTVolume:
    """
    加载 CT .nii.gz 文件并返回 CTVolume 对象。

    自动执行:
        ① 读取 .nii.gz → numpy 数组
        ② 提取 affine 矩阵
        ③ 计算 spacing
        ④ HU 值裁剪到 [hu_min, hu_max]

    论文依据: DiffTumor (CVPR 2024) §E.2 (P21) — HU 裁剪范围

    Args:
        ct_path: CT .nii.gz 文件路径
        hu_min:  HU 下界，默认 -175
        hu_max:  HU 上界，默认 250

    Returns:
        CTVolume 对象

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 数据维度不是 3D
    """
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    try:
        img = nib.load(ct_path)
    except Exception as e:
        raise IOError(f"Failed to load CT file: {ct_path}") from e

    data = img.get_fdata()
    affine = img.affine.copy()

    # 确保 3D
    if data.ndim == 4:
        # 4D: 取第一个时间点/volume
        data = data[..., 0]
    elif data.ndim == 5:
        data = data[..., 0, 0]

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D CT volume, got shape={data.shape} (ndim={data.ndim}). "
            f"File: {ct_path}"
        )

    # HU 裁剪
    data = clip_hu(data, hu_min, hu_max)

    # spacing
    spacing = get_spacing(affine)

    shape = tuple(int(s) for s in data.shape)

    return CTVolume(
        array=data,
        affine=affine,
        spacing=spacing,
        shape=shape,
        path=ct_path,
    )


def load_organ_mask(mask_path: str,
                    organ_type: str,
                    organ_label: str = "") -> OrganMask:
    """
    加载器官分割 mask .nii.gz 文件并返回 OrganMask 对象。

    自动执行:
        ① 读取 .nii.gz → numpy 数组
        ② 二值化: > 0 → 1，确保值域 {0, 1}
        ③ 验证 mask 非空
        ④ 提取 affine

    注意: AbdomenAtlas2.0 的器官分割mask可能包含多个器官类别
    （如 liver.nii.gz 可能同时包含 liver 和其他 label 值）。
    函数会自动将所有非零值转为1，不区分子类别。

    Args:
        mask_path: 器官分割 .nii.gz 文件路径
        organ_type: 肿瘤类型名称，如 "liver_lesion"
        organ_label: 器官标签文件名，如 "liver.nii.gz"（用于日志）

    Returns:
        OrganMask 对象

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: mask 全为零（空mask）
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Organ mask file not found: {mask_path}")

    try:
        img = nib.load(mask_path)
    except Exception as e:
        raise IOError(f"Failed to load organ mask: {mask_path}") from e

    data = img.get_fdata()
    affine = img.affine.copy()

    # 确保 3D
    if data.ndim > 3:
        data = data.squeeze()
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D organ mask, got shape={data.shape}. File: {mask_path}"
        )

    # 二值化: 任何非零 → 1
    binary = ensure_uint8(data)

    if binary.sum() == 0:
        raise ValueError(
            f"Organ mask is empty (all zeros). "
            f"Organ type: {organ_type}, file: {mask_path}"
        )

    return OrganMask(
        array=binary,
        affine=affine,
        organ_type=organ_type,
        organ_label=organ_label,
        path=mask_path,
    )


def load_sample(ct_path: str,
                organ_mask_path: str,
                organ_type: str,
                organ_label: str = "",
                hu_min: float = -175,
                hu_max: float = 250) -> Sample:
    """
    一次加载 CT + 器官mask 并打包为 Sample。

    等价于: load_ct() + load_organ_mask() + validate_compatibility()

    Args:
        ct_path:          CT .nii.gz 路径
        organ_mask_path:  器官分割 .nii.gz 路径
        organ_type:       肿瘤类型名称
        organ_label:      器官标签文件名
        hu_min, hu_max:   HU 裁剪范围

    Returns:
        Sample 对象 (包含 ct 和 organ_mask)

    Raises:
        FileNotFoundError, ValueError — 与 load_ct/load_organ_mask 一致
    """
    ct = load_ct(ct_path, hu_min, hu_max)
    organ_mask = load_organ_mask(organ_mask_path, organ_type, organ_label)

    # 自动校验
    validate_compatibility(ct, organ_mask)

    return Sample(ct=ct, organ_mask=organ_mask)


# ============================================================================
# 校验
# ============================================================================

def validate_compatibility(ct: CTVolume,
                           organ_mask: OrganMask,
                           strict_spacing: bool = False) -> None:
    """
    验证 CT 与器官 mask 的兼容性。

    检查项:
        ① shape 是否一致
        ② 器官 mask 是否至少部分在 CT 的有效范围内
        ③ (可选) spacing 是否一致

    Args:
        ct:           CTVolume 对象
        organ_mask:   OrganMask 对象
        strict_spacing: 是否严格要求 spacing 一致。默认 False（仅警告）

    Raises:
        ValueError: shape 不一致，且无法通过裁剪/填充对齐
    """
    ct_shape = ct.shape
    mask_shape = organ_mask.array.shape

    # shape 检查
    if ct_shape != mask_shape:
        raise ValueError(
            f"Shape mismatch: CT {ct_shape} vs OrganMask {mask_shape}. "
            f"CT: {ct.path}, Mask: {organ_mask.path}. "
            f"Ensure CT and organ mask are from the same scan."
        )

    # 器官是否在 CT 视野内（至少有一些非零体素）
    mask_sum = int(organ_mask.array.sum())
    if mask_sum == 0:
        raise ValueError(f"Organ mask is empty: {organ_mask.path}")

    # spacing 检查
    ct_spacing = ct.spacing
    mask_spacing = get_spacing(organ_mask.affine)

    spacing_diff = tuple(abs(a - b) for a, b in zip(ct_spacing, mask_spacing))
    max_diff = max(spacing_diff)

    if max_diff > 0.5:  # >0.5mm 差异
        msg = (
            f"Spacing differs by {max_diff:.2f} mm: "
            f"CT={ct_spacing}, Mask={mask_spacing}. "
            f"This may affect tumor size accuracy."
        )
        if strict_spacing:
            raise ValueError(msg)
        else:
            warnings.warn(msg)

    # affine 方向检查（行列式符号应一致）
    ct_det = np.linalg.det(ct.affine[:3, :3])
    mask_det = np.linalg.det(organ_mask.affine[:3, :3])
    if np.sign(ct_det) != np.sign(mask_det):
        warnings.warn(
            f"Affine orientation differs between CT and mask. "
            f"CT det={ct_det:.2f}, Mask det={mask_det:.2f}. "
            f"This could mean left/right or anterior/posterior flip."
        )


# ============================================================================
# 辅助工具
# ============================================================================

def get_organ_bbox(organ_mask: OrganMask) -> Tuple[slice, slice, slice]:
    """
    获取器官mask在各维度的 bounding box 切片。

    Args:
        organ_mask: OrganMask 对象

    Returns:
        (z_slice, y_slice, x_slice): 三个维度的 slice 对象

    Raises:
        ValueError: mask 全为零
    """
    from utils import get_bbox  # pyright: ignore[reportMissingImports]
    return get_bbox(organ_mask.array)


def build_manifest(ct_dir: str,
                   organ_label_dir: str,
                   organ_config: List[Dict]) -> List[Dict]:
    """
    扫描数据目录，构建 sample 索引列表。

    从 organ_label_dir 读取可用的样本列表，匹配 CT 目录中对应的扫描。

    预期目录结构 (AbdomenAtlas2.0):
        ct_dir/
          BDMAP_00000001/ct.nii.gz
          BDMAP_00000002/ct.nii.gz
          ...
        organ_label_dir/
          BDMAP_00000001/segmentations/liver.nii.gz
          BDMAP_00000001/segmentations/pancreas.nii.gz
          ...

    Args:
        ct_dir:           CT 扫描根目录
        organ_label_dir:  器官标签根目录
        organ_config:     config['organs'] 列表，每项含 name/organ_label_file

    Returns:
        manifest: [
            {
                'sample_id':     'BDMAP_00000001',
                'ct_path':       'data/ct/BDMAP_00000001/ct.nii.gz',
                'organ_type':    'liver_lesion',
                'organ_label':   'liver.nii.gz',
                'organ_mask_path': 'data/organ_labels/BDMAP_00000001/segmentations/liver.nii.gz',
                'exists':        True/False,
            },
            ...
        ]

    Note:
        此函数只构建索引，不加载任何 .nii.gz 文件。
        加载在 main.py 的批量循环中进行。
    """
    manifest = []

    # 获取可用样本ID（从 organ_label_dir 的子目录名推断）
    if os.path.isdir(organ_label_dir):
        available_ids = sorted([
            d for d in os.listdir(organ_label_dir)
            if os.path.isdir(os.path.join(organ_label_dir, d))
        ])
    else:
        available_ids = []

    if not available_ids:
        # 尝试从 ct_dir 获取
        if os.path.isdir(ct_dir):
            available_ids = sorted([
                d for d in os.listdir(ct_dir)
                if os.path.isdir(os.path.join(ct_dir, d))
            ])

    if not available_ids:
        raise FileNotFoundError(
            f"No sample directories found in {organ_label_dir} or {ct_dir}. "
            f"Please check data paths in generation_config.json."
        )

    for sample_id in available_ids:
        ct_path = os.path.join(ct_dir, sample_id, 'ct.nii.gz')

        for organ_cfg in organ_config:
            organ_type = organ_cfg['name']
            organ_label = organ_cfg['organ_label_file']

            # 路径: organ_label_dir/<sample_id>/segmentations/<organ_label>
            organ_mask_path = os.path.join(
                organ_label_dir, sample_id, 'segmentations', organ_label
            )

            # 也尝试无 segmentations 子目录的情况
            if not os.path.exists(organ_mask_path):
                organ_mask_path = os.path.join(
                    organ_label_dir, sample_id, organ_label
                )

            exists = (
                os.path.exists(ct_path) and
                os.path.exists(organ_mask_path)
            )

            manifest.append({
                'sample_id': sample_id,
                'ct_path': ct_path,
                'organ_type': organ_type,
                'organ_label': organ_label,
                'organ_mask_path': organ_mask_path,
                'exists': exists,
            })

    existing = sum(1 for m in manifest if m['exists'])
    total = len(manifest)
    print(f"build_manifest: {existing}/{total} samples available "
          f"({len(available_ids)} scans × {len(organ_config)} organs)")

    return manifest


def save_manifest_csv(manifest: List[Dict], csv_path: str) -> None:
    """
    将 manifest 保存为 CSV 文件。

    Args:
        manifest:  build_manifest() 的输出
        csv_path:  输出 CSV 路径
    """
    fieldnames = ['sample_id', 'organ_type', 'organ_label',
                  'ct_path', 'organ_mask_path', 'exists']

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Manifest saved: {csv_path} ({len(manifest)} entries)")


def load_manifest_csv(csv_path: str, only_existing: bool = True) -> List[Dict]:
    """
    从 CSV 文件读取 manifest。

    Args:
        csv_path:       CSV 文件路径
        only_existing:  是否仅返回 exists=True 的条目

    Returns:
        manifest 列表
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    manifest = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['exists'] = row['exists'] == 'True'
            manifest.append(row)

    if only_existing:
        manifest = [m for m in manifest if m['exists']]

    return manifest


# ============================================================================
# 模块自检
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("data_loader.py — 模块自检")
    print("=" * 60)

    # 使用合成数据测试（不依赖实际 .nii.gz 文件）
    import tempfile
    rng = np.random.default_rng(42)

    tmpdir = tempfile.mkdtemp(prefix="mask_test_")
    print(f"\n临时目录: {tmpdir}")

    # ── 创建合成 CT ──────────────────────────────
    ct_data = rng.integers(-175, 250, (30, 64, 64), dtype=np.int16)
    ct_affine = np.eye(4)
    ct_affine[0, 0] = 1.0   # x: 1.0 mm
    ct_affine[1, 1] = 1.0   # y: 1.0 mm
    ct_affine[2, 2] = 2.0   # z: 2.0 mm
    ct_path = os.path.join(tmpdir, 'test_ct.nii.gz')
    nib.save(nib.Nifti1Image(ct_data, ct_affine), ct_path)
    print(f"\n[1] 创建合成CT: {ct_path}")
    print(f"    shape={ct_data.shape}, dtype={ct_data.dtype}")

    # ── 创建合成器官mask ─────────────────────────
    organ_data = np.zeros((30, 64, 64), dtype=np.uint8)
    organ_data[5:25, 10:54, 10:54] = 1  # 肝脏大小的椭球区域
    organ_path = os.path.join(tmpdir, 'test_liver.nii.gz')
    nib.save(nib.Nifti1Image(organ_data, ct_affine), organ_path)
    print(f"\n[2] 创建合成器官mask: {organ_path}")
    print(f"    shape={organ_data.shape}, volume={organ_data.sum():,} voxels")

    # ── load_ct ─────────────────────────────────
    print("\n[3] load_ct")
    ct = load_ct(ct_path)
    print(f"    {ct}")
    assert ct.shape == (30, 64, 64), f"shape mismatch: {ct.shape}"
    assert ct.spacing == (2.0, 1.0, 1.0), f"spacing mismatch: {ct.spacing}"
    assert ct.array.min() >= -175, f"HU below min: {ct.array.min()}"
    assert ct.array.max() <= 250, f"HU above max: {ct.array.max()}"
    print("    OK")

    # ── load_organ_mask ─────────────────────────
    print("\n[4] load_organ_mask")
    mask = load_organ_mask(organ_path, 'liver_lesion', 'liver.nii.gz')
    print(f"    {mask}")
    assert mask.array.dtype == np.uint8, f"dtype: {mask.array.dtype}"
    assert mask.array.max() <= 1, f"max value: {mask.array.max()}"
    assert mask.array.sum() == organ_data.sum(), f"volume mismatch"
    print("    OK")

    # ── load_sample ─────────────────────────────
    print("\n[5] load_sample")
    sample = load_sample(ct_path, organ_path, 'liver_lesion', 'liver.nii.gz')
    print(f"    {sample}")
    assert sample.ct is not None
    assert sample.organ_mask is not None
    print("    OK")

    # ── validate_compatibility ──────────────────
    print("\n[6] validate_compatibility")
    # 正常情况
    validate_compatibility(ct, mask)
    print("    OK: matching CT + mask")

    # shape不匹配
    bad_data = np.zeros((20, 64, 64), dtype=np.uint8)
    bad_data[5:15, 10:54, 10:54] = 1  # 非空
    bad_path = os.path.join(tmpdir, 'bad_shape_mask.nii.gz')
    nib.save(nib.Nifti1Image(bad_data, ct_affine), bad_path)
    bad_mask = load_organ_mask(bad_path, 'liver_lesion')
    try:
        validate_compatibility(ct, bad_mask)
        print("    ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"    OK: caught shape mismatch: CT{ct.shape} vs Mask{bad_mask.array.shape}")

    # ── load_ct with missing file ───────────────
    print("\n[7] FileNotFoundError handling")
    try:
        load_ct(os.path.join(tmpdir, 'nonexistent.nii.gz'))
        print("    ERROR: should have raised")
    except FileNotFoundError as e:
        print(f"    OK: {str(e)[:60]}...")

    # ── load_organ_mask with empty mask ──────────
    print("\n[8] Empty mask detection")
    empty_data = np.zeros((30, 64, 64), dtype=np.uint8)
    empty_path = os.path.join(tmpdir, 'empty_mask.nii.gz')
    nib.save(nib.Nifti1Image(empty_data, ct_affine), empty_path)
    try:
        load_organ_mask(empty_path, 'liver_lesion')
        print("    ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"    OK: {str(e)[:60]}...")

    # ── build_manifest (模拟) ────────────────────
    print("\n[9] build_manifest (模拟)")
    # 创建模拟目录结构
    ct_dir = os.path.join(tmpdir, 'ct')
    label_dir = os.path.join(tmpdir, 'labels')
    for sid in ['BDMAP_00000001', 'BDMAP_00000002']:
        os.makedirs(os.path.join(ct_dir, sid), exist_ok=True)
        seg_dir = os.path.join(label_dir, sid, 'segmentations')
        os.makedirs(seg_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(ct_data, ct_affine),
                 os.path.join(ct_dir, sid, 'ct.nii.gz'))
        nib.save(nib.Nifti1Image(organ_data, ct_affine),
                 os.path.join(seg_dir, 'liver.nii.gz'))

    organ_cfg = [
        {'name': 'liver_lesion', 'organ_label_file': 'liver.nii.gz'},
        {'name': 'pancreatic_lesion', 'organ_label_file': 'pancreas.nii.gz'},
    ]
    manifest = build_manifest(ct_dir, label_dir, organ_cfg)
    existing = sum(1 for m in manifest if m['exists'])
    print(f"    Total: {len(manifest)}, Existing: {existing}")
    assert existing >= 2, f"Expected >=2 existing, got {existing}"

    # 保存/加载 CSV
    csv_path = os.path.join(tmpdir, 'manifest.csv')
    save_manifest_csv(manifest, csv_path)
    loaded = load_manifest_csv(csv_path)
    assert len(loaded) == existing
    print(f"    CSV roundtrip: {len(loaded)} entries OK")

    # ── 清理 ────────────────────────────────────
    import shutil
    shutil.rmtree(tmpdir)
    print(f"\n临时目录已清理: {tmpdir}")

    print("\n" + "=" * 60)
    print("全部自检通过")
    print("=" * 60)
