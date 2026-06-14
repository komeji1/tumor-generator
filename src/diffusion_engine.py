"""
Step 1.3: 扩散推理引擎

依据: DiffTumor 源码 utils.py synt_model_prepare() + synthesize_*()

双分支推理:
  分支A (early):  DDPM T=4  步采样 → tester.ema_model.sample()
  分支B (noearly): DDIM S=50 步采样 → DDIMSampler.sample() → VQGAN decode

功能:
  1. 根据 (organ, phase) 加载对应的 UNet + GaussianDiffusion 权重
  2. 执行扩散逆向采样
  3. VQGAN Decode → 生成合成肿瘤纹理
"""

import sys, os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DIFFTUMOR_REPO_DIR

# 添加 DiffTumor 源码路径
sys.path.insert(0, DIFFTUMOR_REPO_DIR)

from TumorGeneration.ldm.ddpm import Unet3D, GaussianDiffusion, Tester
from TumorGeneration.ldm.ddpm.ddim import DDIMSampler
from TumorGeneration.ldm.vq_gan_3d.model.vqgan import VQGAN

# Tester.load() 内部 torch.load 无 weights_only 参数 (PyTorch>=2.6 兼容)
# 加载权重文件并手动注入
def _load_tester_weights(tester, weight_path, map_location):
    data = torch.load(weight_path, map_location=map_location, weights_only=False)
    tester.model.load_state_dict(data["model"])
    tester.ema_model.load_state_dict(data["ema"])


class DiffusionEngine:
    """
    扩散模型推理引擎。

    器官→权重映射:
      liver      → liver_early.pt / liver_noearly.pt
      pancreas   → pancreas_early.pt / pancreas_noearly.pt
      kidney     → kidney_early.pt / kidney_noearly.pt
      colon      → liver_early.pt (零样本, 仅 early)
      esophagus  → liver_early.pt (零样本, 仅 early)
      uterus     → liver_early.pt (零样本, 仅 early)
    """

    # 权重路由表: organ → (early_source, noearly_source)
    WEIGHT_MAP = {
        "liver":       ("liver_early.pt",    "liver_noearly.pt"),
        "pancreas":    ("pancreas_early.pt", "pancreas_noearly.pt"),
        "kidney":      ("kidney_early.pt",   "kidney_noearly.pt"),
        "colon":       ("liver_early.pt",    "liver_early.pt"),      # 零样本
        "esophagus":   ("liver_early.pt",    "liver_early.pt"),      # 零样本
        "uterus":      ("liver_early.pt",    "liver_early.pt"),      # 零样本
    }

    def __init__(
        self,
        vqgan_ckpt_path: str,
        diffusion_ckpt_dir: str,
        organ: str,
        phase: str = "early",       # "early" | "noearly"
        device: str = "cpu",
    ):
        self.organ = organ
        self.phase = phase
        self.device = device

        # ── 加载 VQGAN ──
        ckpt = torch.load(vqgan_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["hyper_parameters"]["cfg"]
        self.vqgan = VQGAN(cfg).to(device)
        self.vqgan.load_state_dict(ckpt["state_dict"], strict=False)
        self.vqgan.eval()

        # ── 确定权重文件 ──
        if phase not in ("early", "noearly"):
            raise ValueError(f"phase must be 'early' or 'noearly', got {phase}")
        weight_idx = 0 if phase == "early" else 1
        weight_file = self.WEIGHT_MAP[organ][weight_idx]
        weight_path = os.path.join(diffusion_ckpt_dir, weight_file)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight not found: {weight_path}")

        # ── 构建 UNet + Diffusion ──
        # 参数与 DiffTumor 训练配置完全一致 (config/model/ddpm.yaml)
        # dim=24, channels=17 (8 masked_feat + 9 cond = 17)
        self.unet = Unet3D(
            dim=24,
            dim_mults=(1, 2, 4, 8),
            channels=17,
            out_dim=8,
        ).to(device)

        if phase == "early":
            # 早期: DDPM, T=4
            # pass vqgan_ckpt=None 避免内部 PyTorch Lightning load_from_checkpoint
            self.diffusion = GaussianDiffusion(
                self.unet,
                vqgan_ckpt=None,
                image_size=24,
                num_frames=24,
                channels=8,
                timesteps=4,
                loss_type="l1",
                device=device,
            ).to(device)
            self.diffusion.vqgan = self.vqgan   # 手动注入我们加载的 VQGAN
            self.tester = Tester(self.diffusion)
            _load_tester_weights(self.tester, weight_path, device)
            self.tester.ema_model.eval()
            self.ddim_sampler = None
        else:
            # 中晚期: DDIM S=50
            self.diffusion = GaussianDiffusion(
                self.unet,
                vqgan_ckpt=None,
                image_size=24,
                num_frames=24,
                channels=8,
                timesteps=200,
                loss_type="l1",
                device=device,
            ).to(device)
            self.diffusion.vqgan = self.vqgan
            noearly_ckpt = torch.load(weight_path, map_location=device, weights_only=False)
            self.diffusion.load_state_dict(noearly_ckpt["ema"])
            self.diffusion.eval()
            self.ddim_sampler = DDIMSampler(self.diffusion, schedule="cosine")
            self.tester = None

    def generate(self, cond: torch.Tensor, seed: int = None, eta: float = 0.0) -> torch.Tensor:
        """
        执行扩散采样 + VQGAN解码。

        Args:
            cond: (B, 9, D_lat, H_lat, W_lat) 条件向量
            seed: 随机种子 (None=每次不同, 指定=可复现)
            eta: DDIM 随机性 (0=确定性/论文默认, 1=最大随机性, 仅 noearly 有效)

        Returns:
            synthetic: (B, 1, D, H, W) 合成肿瘤纹理, 值域 [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
        cond = cond.to(self.device)
        batch_size = cond.shape[0]

        with torch.no_grad():
            if self.phase == "early":
                # ── 分支A: DDPM T=4 ──
                sample_latent = self.tester.ema_model.sample(
                    cond=cond, batch_size=batch_size
                )  # 内部自动 VQGAN decode
            else:
                # ── 分支B: DDIM S=50 ──
                shape = cond[:, :8].shape[1:]  # (8, D_lat, H_lat, W_lat)
                samples_ddim, _ = self.ddim_sampler.sample(
                    S=50,
                    conditioning=cond,
                    batch_size=batch_size,
                    shape=shape,
                    eta=eta,
                    verbose=False,
                )
                # 反归一化 + VQGAN decode
                samples_ddim = (
                    (samples_ddim + 1.0) / 2.0
                ) * (
                    self.vqgan.codebook.embeddings.max()
                    - self.vqgan.codebook.embeddings.min()
                ) + self.vqgan.codebook.embeddings.min()
                sample_latent = self.vqgan.decode(samples_ddim, quantize=True)

        # 还原轴顺序: (B,C,W,D,H) → (B,C,D,H,W) 与CT tensor对齐
        sample_latent = sample_latent.permute(0, 1, -2, -1, -3)
        return sample_latent  # (B, 1, D, H, W) 值域 [-1, 1]
