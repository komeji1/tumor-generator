"""
共享工具函数: NIfTI IO, HU转换, 坐标映射
"""
import nibabel as nib
import numpy as np
import torch
from typing import Tuple


HU_MIN, HU_MAX = -175, 250


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载 .nii.gz, 返回 (data, affine)"""
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine.copy()


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str, dtype=np.float32):
    """保存 .nii.gz, 自动处理轴序"""
    nib.save(nib.Nifti1Image(data.astype(dtype), affine), path)


def normalize_hu(ct_array: np.ndarray) -> np.ndarray:
    """HU [-175,250] → [0,1]"""
    clipped = np.clip(ct_array, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def denormalize_hu(norm_array: np.ndarray) -> np.ndarray:
    """[0,1] → HU"""
    return norm_array * (HU_MAX - HU_MIN) + HU_MIN


def compute_tumor_radius_voxel(mask: np.ndarray, spacing: Tuple[float, ...]) -> float:
    """从mask体素数和spacing估算等效球体半径 (mm)"""
    voxel_volume_mm3 = np.prod(spacing)
    n_voxels = int(mask.sum())
    volume_mm3 = n_voxels * voxel_volume_mm3
    radius_mm = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
    return radius_mm
