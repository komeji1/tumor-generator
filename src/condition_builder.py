"""
Step 1.2: 条件构造器

依据: DiffTumor 源码 diffusion.py forward() L838-860 + utils.py synthesize_*()

功能: 将预处理后的 CT + tumor_mask 转化为扩散模型的条件向量 cond

管线 (精确到维度):
  ① volume = ct_tensor * 2.0 - 1.0          [-1, 1]
  ② mask   = tumor_mask * 2.0 - 1.0          {-1, 1}
  ③ mask_  = 1 - tumor_mask                  {0, 1}
  ④ masked_volume = (volume * mask_).detach()
  ⑤ permute: (B,1,D,H,W) → (B,1,W/?,D/?,H/?)  [DiffTumor轴约定]
  ⑥ VQGAN encode → masked_feat (B,8,D/4,H/4,W/4)
  ⑦ 归一化: (feat-emb_min)/(emb_max-emb_min)*2-1
  ⑧ mask下采样: interpolate(mask, feat_size)
  ⑨ cond = cat([masked_feat, cc], dim=1) → (B,9,D/4,H/4,W/4)
"""

import torch
import torch.nn.functional as F
import sys
import os

# 确保能找到我们的 vqgan 模块
sys.path.insert(0, os.path.dirname(__file__))
from vqgan.vqgan import VQGAN


class ConditionBuilder:
    """
    将预处理后的 CT + tumor_mask 转化为扩散模型条件。

    管线和 DiffTumor 源码 utils.py synthesize_*() 完全一致。
    """

    def __init__(
        self,
        vqgan_ckpt_path: str,
        device: str = "cpu",
    ):
        self.device = device

        # 加载冻结的 VQGAN (eval模式, 不训练)
        # PyTorch >=2.6 默认 weights_only=True 阻止 omegaconf 对象, 手动 load
        checkpoint = torch.load(vqgan_ckpt_path, map_location=device, weights_only=False)
        cfg = checkpoint["hyper_parameters"]["cfg"]
        self.vqgan = VQGAN(cfg)
        self.vqgan.load_state_dict(checkpoint["state_dict"], strict=False)
        self.vqgan.eval()
        self.vqgan.to(device)

        # 缓存 codebook 极值 (用于 Step ⑦)
        emb = self.vqgan.codebook.embeddings  # (16384, 8)
        self.emb_min = emb.min().detach()
        self.emb_max = emb.max().detach()

    def build(self, ct_tensor: torch.Tensor, tumor_mask: torch.Tensor) -> torch.Tensor:
        """
        构建扩散模型条件向量。

        Args:
            ct_tensor:     (B,1,D,H,W), 值域 [0,1] (来自 ct_preprocessor)
            tumor_mask:    (B,1,D,H,W), bool/{0,1}

        Returns:
            cond:          (B,9,D/4,H/4,W/4), [-1,1]
                           [8通道: 被mask遮挡的健康CT潜在特征
                            1通道: 下采样到潜在空间的tumor_mask]
        """
        ct_tensor = ct_tensor.to(self.device)
        tumor_mask = tumor_mask.to(self.device)

        # ── Step ①②③: 值域映射 ──
        volume = ct_tensor * 2.0 - 1.0            # [0,1] → [-1,1]
        mask = tumor_mask.float() * 2.0 - 1.0     # {0,1} → {-1,1}
        mask_ = 1.0 - tumor_mask.float()           # 反mask: {0,1}
        masked_volume = (volume * mask_).detach()   # mask区域置0

        # ── Step ⑤: 维度重排 (DiffTumor轴约定) ──
        # (B,C,D,H,W) → (B,C,W,D,H)
        # 源码: permute(0,1,-1,-3,-2) 即保持B,C, 三个空间轴循环移位
        volume_p = volume.permute(0, 1, -1, -3, -2)
        masked_volume_p = masked_volume.permute(0, 1, -1, -3, -2)
        mask_p = mask.permute(0, 1, -1, -3, -2)

        # ── Step ⑥: VQGAN Encoder ──
        with torch.no_grad():
            masked_feat = self.vqgan.encode(
                masked_volume_p,
                quantize=False,
                include_embeddings=True,
            )  # (B,8,D/4,H/4,W/4) — shape depends on encoder

        # ── Step ⑦: 潜在空间归一化 → [-1, 1] ──
        masked_feat = (
            (masked_feat - self.emb_min) / (self.emb_max - self.emb_min)
        ) * 2.0 - 1.0

        # ── Step ⑧: mask 下采样到潜在空间尺寸 ──
        cc = F.interpolate(mask_p, size=masked_feat.shape[-3:])

        # ── Step ⑨: 拼接条件 ──
        cond = torch.cat([masked_feat, cc], dim=1)  # (B, 9, D/4, H/4, W/4)

        return cond
