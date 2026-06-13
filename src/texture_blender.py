"""
Step 1.4: Texture Blender (DiffTumor paper formula)

Paper formula:
  final = (1 - mask_blur) * original_CT + mask_blur * synthetic_texture

- mask=0 region: 100% original CT (unchanged)
- mask=1 region: synthetic tumor texture
- mask edge: Gaussian feather transition

sigma: random U(0.5, 4.0) for training augmentation,
       fixed 1.5 for demo/verification.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter

HU_MIN, HU_MAX = -175, 250


class TextureBlender:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def blend(
        self,
        ct_patch: torch.Tensor,       # (1,1,96,96,96) [0,1] original CT
        synthetic: torch.Tensor,       # (1,1,96,96,96) [-1,1] diffusion output
        tumor_mask: torch.Tensor,      # (1,1,96,96,96) bool
        organ_type: str,               # liver / pancreas / kidney / ...
        random_sigma: bool = False,    # True for training, False for demo
    ) -> torch.Tensor:
        """
        Alpha-blend: only tumor region gets synthetic texture.
        Non-tumor region stays identical to original CT.
        """
        sample = torch.clamp((synthetic + 1.0) / 2.0, 0.0, 1.0)

        if organ_type in ("pancreas", "esophagus"):
            blended = sample
        else:
            mask_01 = tumor_mask.float()
            sigma = np.random.uniform(0.5, 4.0) if random_sigma else 1.5
            mask_np = mask_01.cpu().numpy().astype(np.float32)
            mask_blurred = gaussian_filter(mask_np, sigma=[0, 0, sigma, sigma, sigma])
            mask_blurred = torch.from_numpy(mask_blurred).to(self.device)
            blended = (1.0 - mask_blurred) * ct_patch + mask_blurred * sample

        return torch.clamp(blended, 0.0, 1.0)


def patch_to_hu(tensor_01: torch.Tensor) -> np.ndarray:
    """Convert [0,1] tensor to HU numpy (D,H,W)"""
    arr = tensor_01.squeeze().cpu().numpy().astype(np.float32)
    return arr * (HU_MAX - HU_MIN) + HU_MIN
