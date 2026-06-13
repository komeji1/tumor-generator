"""
Step 1.1: CT 预处理器

依据: DiffTumor 源码 monai_trainer.py L240-245 + §E.2 (P21)
      "intensity truncated to the range [-175, 250]"
      "isotropic spacing 1.0x1.0x1.0mm3" + "orientation to RAS"

使用 SimpleITK 直接操作 (避免 MONAI 版本兼容性问题)

功能:
  1. 加载 CT / organ_mask / tumor_mask (NIfTI .nii.gz)
  2. 重采样 → 1.0 mm3 各向同性 (CT: linear, mask: nearest)
  3. HU裁剪 → [-175, 250] + 归一化 → [0, 1]
  4. 提取器官HU统计 (mu, sigma)
  5. 转换 → PyTorch Tensor + 维度重排 (D,H,W) → (1,D,H,W)
"""

import SimpleITK as sitk
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Dict

# ---------------------------------------------
# 配置 (全部出自 DiffTumor 源码)
# ---------------------------------------------
HU_MIN = -175
HU_MAX = 250
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm3  isotropic


@dataclass
class PreprocessResult:
    """预处理输出容器"""
    ct_tensor: torch.Tensor          # (1, 1, D, H, W), [0, 1]
    organ_mask_tensor: torch.Tensor   # (1, 1, D, H, W), bool
    tumor_mask_tensor: torch.Tensor   # (1, 1, D, H, W), bool
    hu_stats: Dict[str, float]        # {"mean": float, "std": float}
    affine_original: np.ndarray       # (4,4) 原始 affine
    original_shape: Tuple[int, int, int]  # (D, H, W)
    organ_name: str
    crop_coords: Tuple[int, int, int, int, int, int] = None  # (z0,z1,y0,y1,x0,x1) of last crop

    def crop_around_tumor(self, patch_size: int = 96) -> "PreprocessResult":
        """裁剪肿瘤周围固定大小patch → 适配UNet的24³潜在空间"""
        B, C, D, H, W = self.ct_tensor.shape
        tumor_nz = torch.nonzero(self.tumor_mask_tensor[0, 0])
        if tumor_nz.numel() == 0:
            raise ValueError("Tumor mask empty")
        ctr = tumor_nz.float().mean(dim=0).long()
        half = patch_size // 2
        z0, z1 = max(0, ctr[0]-half), min(D, ctr[0]+half)
        y0, y1 = max(0, ctr[1]-half), min(H, ctr[1]+half)
        x0, x1 = max(0, ctr[2]-half), min(W, ctr[2]+half)
        ct  = self.ct_tensor[:,:,z0:z1,y0:y1,x0:x1]
        om  = self.organ_mask_tensor[:,:,z0:z1,y0:y1,x0:x1]
        tm  = self.tumor_mask_tensor[:,:,z0:z1,y0:y1,x0:x1]
        dz, dh, dw = patch_size - ct.shape[2], patch_size - ct.shape[3], patch_size - ct.shape[4]
        if dz > 0 or dh > 0 or dw > 0:
            from torch.nn.functional import pad
            ct = pad(ct, [0, dw, 0, dh, 0, dz])
            om = pad(om, [0, dw, 0, dh, 0, dz])
            tm = pad(tm, [0, dw, 0, dh, 0, dz])
        return PreprocessResult(ct, om, tm, self.hu_stats,
                                self.affine_original, (patch_size,)*3, self.organ_name,
                                crop_coords=(z0.item(), z1.item(), y0.item(), y1.item(), x0.item(), x1.item()))


class CTPreprocessor:
    """CT + masks 统一预处理管线 (SimpleITK实现)"""

    def __init__(self, device: str = "cpu"):
        self.device = device

    # -----------------------------------------
    # 公开接口
    # -----------------------------------------
    def process(
        self,
        ct_path: str,
        organ_mask_path: str,
        tumor_mask_path: str,
        organ_name: str,
    ) -> PreprocessResult:
        """
        统一预处理: CT + organ_mask + tumor_mask → 标准 Tensor

        Args:
            ct_path:          CT .nii.gz
            organ_mask_path:  器官mask .nii.gz
            tumor_mask_path:  肿瘤mask .nii.gz
            organ_name:       器官名
        """
        # -- 加载原始数据 (SimpleITK) --
        ct_sitk = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        organ_sitk = sitk.ReadImage(organ_mask_path, sitk.sitkUInt8)
        tumor_sitk = sitk.ReadImage(tumor_mask_path, sitk.sitkUInt8)

        # 保存原始参数
        original_shape = tuple(ct_sitk.GetSize())[::-1]  # (D,H,W)
        affine_original = self._get_affine(ct_sitk)

        # -- Step 1: 提取 HU 统计 (原始空间, 归一化前) --
        ct_arr_raw = sitk.GetArrayFromImage(ct_sitk)          # (D,H,W)
        organ_arr_raw = sitk.GetArrayFromImage(organ_sitk)    # (D,H,W)
        organ_bool_raw = organ_arr_raw > 0                    # 类别标签→二值
        hu_stats = self._compute_hu_stats(ct_arr_raw, organ_bool_raw)

        # -- Step 2: 重采样到 1mm³ 各向同性 --
        ct_sitk = self._resample_image(ct_sitk, TARGET_SPACING, sitk.sitkLinear)
        organ_sitk = self._resample_image(organ_sitk, TARGET_SPACING, sitk.sitkNearestNeighbor)
        tumor_sitk = self._resample_image(tumor_sitk, TARGET_SPACING, sitk.sitkNearestNeighbor)

        # -- Step 3: CT HU 裁剪 + 归一化 --
        ct_arr = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)  # (D,H,W)
        ct_arr = np.clip(ct_arr, HU_MIN, HU_MAX)
        ct_arr = (ct_arr - HU_MIN) / (HU_MAX - HU_MIN)  # → [0, 1]

        # -- Step 4: mask 提取 --
        organ_arr = sitk.GetArrayFromImage(organ_sitk).astype(bool)    # (D,H,W)
        tumor_arr = sitk.GetArrayFromImage(tumor_sitk).astype(bool)

        # -- Step 5: 转为 Tensor (添加 C 维度) --
        ct_tensor = torch.from_numpy(ct_arr).float().unsqueeze(0).unsqueeze(0)       # (1,1,D,H,W)
        organ_mask_tensor = torch.from_numpy(organ_arr).bool().unsqueeze(0).unsqueeze(0)
        tumor_mask_tensor = torch.from_numpy(tumor_arr).bool().unsqueeze(0).unsqueeze(0)

        ct_tensor = ct_tensor.to(self.device)
        organ_mask_tensor = organ_mask_tensor.to(self.device)
        tumor_mask_tensor = tumor_mask_tensor.to(self.device)

        return PreprocessResult(
            ct_tensor=ct_tensor,
            organ_mask_tensor=organ_mask_tensor,
            tumor_mask_tensor=tumor_mask_tensor,
            hu_stats=hu_stats,
            affine_original=affine_original,
            original_shape=original_shape,
            organ_name=organ_name,
        )

    # -----------------------------------------
    # 内部方法
    # -----------------------------------------
    def _resample_image(
        self,
        image: sitk.Image,
        target_spacing: Tuple[float, float, float],
        interpolator,
    ) -> sitk.Image:
        """将图像重采样到指定 spacing"""
        orig_spacing = np.array(image.GetSpacing())
        orig_size = np.array(image.GetSize())
        target_spacing = np.array(target_spacing)

        new_size = (orig_size * orig_spacing / target_spacing).astype(np.int32).tolist()

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing.tolist())
        resampler.SetSize(new_size)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        return resampler.Execute(image)

    @staticmethod
    def _get_affine(image: sitk.Image) -> np.ndarray:
        """从 SimpleITK 图像提取 4x4 affine 矩阵"""
        direction = np.array(image.GetDirection()).reshape(3, 3)
        spacing = np.array(image.GetSpacing())
        origin = np.array(image.GetOrigin())

        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        return affine

    @staticmethod
    def _compute_hu_stats(
        ct_volume: np.ndarray, organ_mask: np.ndarray
    ) -> Dict[str, float]:
        """计算器官区域 HU 均值和标准差 (D,H,W 布局)"""
        organ_voxels = ct_volume[organ_mask]
        if len(organ_voxels) == 0:
            return {"mean": 0.0, "std": 1.0}
        return {
            "mean": float(organ_voxels.mean()),
            "std": float(organ_voxels.std()),
        }
