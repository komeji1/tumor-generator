"""
肿瘤纹理注入适配器 — 将 DiffTumor 肿瘤纹理生成能力对接到 MAISI 合成 CT 管线。

三阶段管线:
  Phase 1: MAISI 基础 CT 生成 (整张 CT, 含粗略肿瘤区域)
  Phase 2: DiffTumor 肿瘤纹理生成 (96³ patch, 1mm³ 各向同性)
  Phase 3: 纹理融合嵌入 (将合成肿瘤纹理嵌入回完整 CT)

设计依据:
  - DiffTumor (CVPR 2024) 条件编码: cond = concat([z_healthy, mask_downsampled])
  - MAISI (NV-Generate-CTMR) 132 类分割标签体系
  - tumor-generator (komeji1) 提示词 JSON 格式

依赖:
  - MAISI 自身模块 (scripts.sample, scripts.utils_infer, scripts.utils)
  - DiffTumor 源码 (TumorGeneration.ldm.ddpm, TumorGeneration.ldm.vq_gan_3d)
  - DiffTumor 预训练权重 (VQGAN + 各器官扩散模型)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("maisi.tumor_adapter")


# ═══════════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class TumorTask:
    """单个肿瘤生成任务 (来自 JSON 提示词)"""
    organ: str                           # liver / pancreas / kidney / ...
    size_category: str = "small"         # tiny / small / medium / large
    phase: Optional[str] = None          # early / noearly / None(自动)
    modality: str = "ct"                 # ct (目前仅支持 ct)
    output: str = "both"                 # full_ct / patch_96 / both
    eta: float = 0.0                     # DDIM 随机性 (仅 noearly)
    mask_file: Optional[str] = None      # 直接指定外部 mask 文件
    repeat: int = 1                      # 重复次数
    output_name: Optional[str] = None    # 自定义输出文件名
    # 高级字段
    radius_mm: Optional[float] = None    # 精确半径筛选
    position: Optional[list] = None      # 归一化位置 [0,1]³

    # MAISI 参数 (由 TumorConfigAdapter 填充)
    output_size: Tuple[int, ...] = (256, 256, 128)
    spacing: Tuple[float, ...] = (1.7, 1.7, 2.0)


@dataclass
class TumorPipelineResult:
    """管线运行结果"""
    status: str = "ok"                   # ok / skip / fail
    organ: str = ""
    phase: str = ""
    output_paths: List[str] = field(default_factory=list)
    tumor_hu_mean: float = 0.0
    tumor_hu_std: float = 0.0
    time_s: float = 0.0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
#  配置适配器: 肿瘤提示词 → MAISI 配置
# ═══════════════════════════════════════════════════════════════

# 器官 → MAISI 解剖映射 (与 label_dict_ctmr.json 对齐)
ORGAN_TO_MAISI = {
    "liver": {
        "anatomy_list": ["liver", "hepatic tumor"],
        "body_region": ["chest", "abdomen"],
        "tumor_label": 26,
        "organ_label": 1,
        "organ_type": "liver_lesion",
    },
    "pancreas": {
        "anatomy_list": ["pancreas", "pancreatic tumor"],
        "body_region": ["abdomen"],
        "tumor_label": 24,
        "organ_label": 4,
        "organ_type": "pancreatic_lesion",
    },
    "kidney": {
        "anatomy_list": ["kidney", "left kidney cyst"],
        "body_region": ["abdomen"],
        "tumor_label": 116,
        "organ_label": 14,
        "organ_type": "kidney_lesion",
    },
    "colon": {
        "anatomy_list": ["colon", "colon cancer primaries"],
        "body_region": ["abdomen", "pelvis"],
        "tumor_label": 27,
        "organ_label": 62,
        "organ_type": "colon_lesion",
    },
    "lung": {
        "anatomy_list": ["lung", "lung tumor"],
        "body_region": ["chest"],
        "tumor_label": 23,
        "organ_label": 20,
        "organ_type": "lung_lesion",
    },
    "bone": {
        "anatomy_list": ["bone", "bone lesion"],
        "body_region": ["chest", "abdomen"],
        "tumor_label": 128,
        "organ_label": 21,
        "organ_type": "bone_lesion",
    },
    "esophagus": {
        "anatomy_list": ["esophagus"],
        "body_region": ["chest"],
        "tumor_label": 0,
        "organ_label": 11,
        "organ_type": "esophagus_tumor",
    },
    "uterus": {
        "anatomy_list": ["uterocervix"],
        "body_region": ["pelvis"],
        "tumor_label": 0,
        "organ_label": 161,
        "organ_type": "endometrioma_tumor",
    },
}

# DiffTumor 权重路由
WEIGHT_MAP = {
    "liver":     ("liver_early.pt",    "liver_noearly.pt"),
    "pancreas":  ("pancreas_early.pt", "pancreas_noearly.pt"),
    "kidney":    ("kidney_early.pt",   "kidney_noearly.pt"),
    "colon":     ("colon_early.pt",    "colon_early.pt"),
    "esophagus": ("liver_early.pt",    "liver_early.pt"),
    "uterus":    ("liver_early.pt",    "liver_early.pt"),
    "lung":      ("liver_early.pt",    "liver_early.pt"),
    "bone":      ("liver_early.pt",    "liver_early.pt"),
}

# 尺寸档位 → phase 映射
SIZE_PHASE = {
    "tiny":   "early",
    "small":  "early",
    "medium": "noearly",
    "large":  "noearly",
}


class TumorConfigAdapter:
    """将肿瘤 JSON 提示词翻译为 MAISI 推理配置。"""

    def __init__(self, pipeline_config_path: str = None):
        """
        Args:
            pipeline_config_path: config_tumor_pipeline.json 路径
        """
        if pipeline_config_path is None:
            pipeline_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "config_tumor_pipeline.json"
            )
        self.config = self._load_config(pipeline_config_path)

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def resolve_phase(self, organ: str, size_category: str, phase: Optional[str] = None) -> str:
        """确定扩散采样策略 (early/noearly)。

        规则:
          1. 用户显式指定 → 使用用户指定
          2. 器官无 noearly 权重 → 强制 early
          3. 按尺寸自动选择: tiny/small → early, medium/large → noearly
        """
        if phase is not None:
            resolved = phase
        else:
            resolved = SIZE_PHASE.get(size_category, "early")

        # 器官无 noearly 权重时降级为 early
        early_wt, noearly_wt = WEIGHT_MAP.get(organ, ("liver_early.pt", "liver_early.pt"))
        if resolved == "noearly" and early_wt == noearly_wt:
            logger.info(f"  {organ} 无 noearly 权重, 降级为 early")
            resolved = "early"

        return resolved

    def task_to_maisi_params(self, task: TumorTask) -> dict:
        """将 TumorTask 翻译为 MAISI LDMSampler 所需的参数。

        Returns:
            dict: 包含 body_region, anatomy_list, controllable_anatomy_size 等键
        """
        organ_info = ORGAN_TO_MAISI.get(task.organ)
        if organ_info is None:
            raise ValueError(
                f"不支持的器官: {task.organ}。"
                f"支持: {list(ORGAN_TO_MAISI.keys())}"
            )

        phase = self.resolve_phase(task.organ, task.size_category, task.phase)

        # 构建 controllable_anatomy_size
        # MAISI 的 anatomy_size 是一个 10 维向量:
        # [gallbladder, liver, stomach, pancreas, colon,
        #  lung_tumor, pancreatic_tumor, hepatic_tumor, colon_cancer, bone_lesion]
        # 值域: -1=不控制, 0~1=相对大小
        # 我们需要设置目标肿瘤的大小
        controllable_anatomy_size = self._build_anatomy_size(task.organ, task.size_category)

        return {
            "body_region": organ_info["body_region"],
            "anatomy_list": organ_info["anatomy_list"],
            "output_size": list(task.output_size),
            "spacing": list(task.spacing),
            "controllable_anatomy_size": controllable_anatomy_size,
            "modality": 1 if task.modality == "ct" else 8,
            "num_inference_steps": self.config.get("maisi", {}).get("num_inference_steps", 30),
            "phase": phase,
            "organ_info": organ_info,
        }

    @staticmethod
    def _build_anatomy_size(organ: str, size_category: str) -> list:
        """构建 MAISI controllable_anatomy_size 参数。

        MAISI 10 维向量的索引:
          0=gallbladder, 1=liver, 2=stomach, 3=pancreas, 4=colon,
          5=lung_tumor, 6=pancreatic_tumor, 7=hepatic_tumor,
          8=colon_cancer_primaries, 9=bone_lesion

        值: -1=不控制, 0~1=相对大小 (0=最小, 1=最大)
        """
        # 默认全部不控制
        anatomy_size = [(-1, -1)] * 10  # (name, value) — value=-1 表示不控制

        # 器官名称 → anatomy_size 索引
        organ_idx = {
            "gallbladder": 0, "liver": 1, "stomach": 2, "pancreas": 3, "colon": 4,
            "lung_tumor": 5, "pancreatic_tumor": 6, "hepatic_tumor": 7,
            "colon_cancer_primaries": 8, "bone_lesion": 9,
        }

        # 尺寸档位 → 归一化大小
        size_scales = {
            "tiny": 0.1, "small": 0.3, "medium": 0.6, "large": 0.9,
        }
        scale = size_scales.get(size_category, 0.3)

        # 器官 → tumor 索引映射
        organ_to_tumor_idx = {
            "liver": 7,        # hepatic_tumor
            "pancreas": 6,     # pancreatic_tumor
            "kidney": 5,       # lung_tumor (复用, MAISI 无肾脏肿瘤专用索引)
            "colon": 8,        # colon_cancer_primaries
            "lung": 5,         # lung_tumor
            "bone": 9,         # bone_lesion
            "esophagus": 7,    # 复用 hepatic_tumor (零样本)
            "uterus": 7,       # 复用 hepatic_tumor (零样本)
        }

        tumor_idx = organ_to_tumor_idx.get(organ)
        if tumor_idx is not None:
            # 构建 MAISI 格式的 controllable_anatomy_size
            # 格式: [(organ_name, scale), ...]
            tumor_names = [
                "gallbladder", "liver", "stomach", "pancreas", "colon",
                "lung tumor", "pancreatic tumor", "hepatic tumor",
                "colon cancer primaries", "bone lesion",
            ]
            return [(tumor_names[tumor_idx], scale)]

        return []


# ═══════════════════════════════════════════════════════════════
#  肿瘤纹理注入器 (DiffTumor 核心管线移植)
# ═══════════════════════════════════════════════════════════════

# DiffTumor HU 参数
HU_MIN = -175
HU_MAX = 250
TARGET_SPACING = (1.0, 1.0, 1.0)


def _load_diffumor_modules(diffumor_repo_dir: str):
    """延迟加载 DiffTumor 模块, 返回 (Unet3D, GaussianDiffusion, Tester, DDIMSampler, VQGAN)

    注意: TumorGeneration/__init__.py 会导入 elasticdeform (我们不需要),
    所以我们通过 importlib 方式直接加载子包, 跳过 TumorGeneration 的 __init__。
    ddpm/ddim.py 使用相对导入, 必须作为 ddpm.ddim 导入。
    """
    import importlib

    tg_dir = os.path.join(diffumor_repo_dir, "TumorGeneration")
    if not os.path.isdir(tg_dir):
        raise FileNotFoundError(
            f"DiffTumor TumorGeneration 目录未找到: {tg_dir}\n"
            f"请确保 tumor_paths.json 中 diffumor_repo_dir 指向 STEP3.SegmentationModel 子目录"
        )

    # 确保 STEP3 路径在 sys.path 中
    if diffumor_repo_dir not in sys.path:
        sys.path.insert(0, diffumor_repo_dir)

    try:
        # 先注册 TumorGeneration 为包, 但阻止其 __init__.py 的自动执行
        # 通过手动创建模块对象来实现
        import types
        if "TumorGeneration" not in sys.modules:
            tg_mod = types.ModuleType("TumorGeneration")
            tg_mod.__path__ = [os.path.join(diffumor_repo_dir, "TumorGeneration")]
            tg_mod.__package__ = "TumorGeneration"
            sys.modules["TumorGeneration"] = tg_mod

        # 注册 ldm 子包
        ldm_mod = types.ModuleType("TumorGeneration.ldm")
        ldm_mod.__path__ = [os.path.join(tg_dir, "ldm")]
        ldm_mod.__package__ = "TumorGeneration.ldm"
        sys.modules["TumorGeneration.ldm"] = ldm_mod

        # 导入 ddpm 包 (含相对导入的子模块)
        from TumorGeneration.ldm.ddpm import Unet3D, GaussianDiffusion, Tester
        from TumorGeneration.ldm.ddpm.ddim import DDIMSampler
        from TumorGeneration.ldm.vq_gan_3d.model.vqgan import VQGAN
        return Unet3D, GaussianDiffusion, Tester, DDIMSampler, VQGAN
    except ImportError as e:
        raise ImportError(
            f"无法导入 DiffTumor 模块: {e}\n"
            f"请确保 tumor_paths.json 中 diffumor_repo_dir 指向 DiffTumor 仓库的 "
            f"STEP3.SegmentationModel 子目录。\n"
            f"获取: git clone https://github.com/MrGiovanni/DiffTumor\n"
            f"可能缺少依赖: pip install torchvision opencv-python-headless"
        ) from e


class TumorTextureInjector:
    """DiffTumor 肿瘤纹理生成 + 融合嵌入核心管线。

    从 tumor-generator 项目移植, 适配 MAISI 生成的 CT 作为输入。

    管线:
      ① 从 MAISI 生成的 CT + mask 中裁剪肿瘤区域
      ② CTPreprocessor: 重采样到 1mm³, HU 裁剪归一化
      ③ ConditionBuilder: VQGAN 编码 → 9 通道条件向量
      ④ DiffusionEngine: DDPM/DDIM 采样 → 合成肿瘤纹理
      ⑤ TextureBlender: Gaussian alpha 融合
      ⑥ Resample 回原生空间 + 软器官边界嵌入
    """

    def __init__(
        self,
        vqgan_ckpt_path: str,
        diffusion_ckpt_dir: str,
        diffumor_repo_dir: str,
        device: str = "cpu",
    ):
        self.device = device
        self.vqgan_ckpt_path = vqgan_ckpt_path
        self.diffusion_ckpt_dir = diffusion_ckpt_dir
        self.diffumor_repo_dir = diffumor_repo_dir

        # 延迟加载 DiffTumor 模块
        self._Unet3D, self._GaussianDiffusion, self._Tester, self._DDIMSampler, self._VQGAN = \
            _load_diffumor_modules(diffumor_repo_dir)

        # 加载 VQGAN (所有器官共用)
        self.vqgan = self._load_vqgan(vqgan_ckpt_path, device)
        self.emb_min = self.vqgan.codebook.embeddings.min().detach()
        self.emb_max = self.vqgan.codebook.embeddings.max().detach()

        # 缓存已加载的扩散引擎 (避免重复加载同一权重)
        self._engine_cache: Dict[str, dict] = {}

    def _load_vqgan(self, ckpt_path: str, device: str):
        """加载 VQGAN 自编码器"""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["hyper_parameters"]["cfg"]
        vqgan = self._VQGAN(cfg).to(device)
        vqgan.load_state_dict(ckpt["state_dict"], strict=False)
        vqgan.eval()
        return vqgan

    def _get_diffusion_engine(self, organ: str, phase: str) -> dict:
        """获取或创建扩散引擎 (带缓存)"""
        cache_key = f"{organ}_{phase}"
        if cache_key in self._engine_cache:
            return self._engine_cache[cache_key]

        early_wt, noearly_wt = WEIGHT_MAP[organ]
        weight_file = early_wt if phase == "early" else noearly_wt
        weight_path = os.path.join(self.diffusion_ckpt_dir, weight_file)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"DiffTumor 权重未找到: {weight_path}")

        # 构建 UNet (参数与 DiffTumor 训练配置一致: dim=24, channels=17)
        unet = self._Unet3D(
            dim=24,
            dim_mults=(1, 2, 4, 8),
            channels=17,
            out_dim=8,
        ).to(self.device)

        if phase == "early":
            # DDPM, T=4
            diffusion = self._GaussianDiffusion(
                unet,
                vqgan_ckpt=None,
                image_size=24,
                num_frames=24,
                channels=8,
                timesteps=4,
                loss_type="l1",
                device=self.device,
            ).to(self.device)
            diffusion.vqgan = self.vqgan
            tester = self._Tester(diffusion)
            data = torch.load(weight_path, map_location=self.device, weights_only=False)
            tester.model.load_state_dict(data["model"])
            tester.ema_model.load_state_dict(data["ema"])
            tester.ema_model.eval()

            engine = {"type": "early", "tester": tester, "ddim_sampler": None, "diffusion": diffusion}
        else:
            # DDIM, S=50 (基于 T=200)
            diffusion = self._GaussianDiffusion(
                unet,
                vqgan_ckpt=None,
                image_size=24,
                num_frames=24,
                channels=8,
                timesteps=200,
                loss_type="l1",
                device=self.device,
            ).to(self.device)
            diffusion.vqgan = self.vqgan
            ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
            diffusion.load_state_dict(ckpt["ema"])
            diffusion.eval()

            ddim_sampler = self._DDIMSampler(diffusion, schedule="cosine")
            engine = {"type": "noearly", "tester": None, "ddim_sampler": ddim_sampler, "diffusion": diffusion}

        self._engine_cache[cache_key] = engine
        return engine

    def build_condition(
        self,
        ct_tensor: torch.Tensor,     # (1,1,D,H,W) [0,1]
        tumor_mask: torch.Tensor,     # (1,1,D,H,W) bool/{0,1}
    ) -> torch.Tensor:
        """构建扩散模型条件向量 (与 DiffTumor 源码一致)。

        管线:
          ① volume = ct * 2.0 - 1.0        [-1, 1]
          ② mask = tumor_mask * 2.0 - 1.0   {-1, 1}
          ③ mask_ = 1 - tumor_mask           {0, 1}
          ④ masked_volume = volume * mask_
          ⑤ permute: (B,1,D,H,W) → (B,1,W,D,H)  [DiffTumor 轴约定]
          ⑥ VQGAN encode → masked_feat (B,8,D/4,H/4,W/4)
          ⑦ 归一化: (feat-emb_min)/(emb_max-emb_min)*2-1
          ⑧ mask 下采样
          ⑨ cond = cat([masked_feat, cc], dim=1) → (B,9,D/4,H/4,W/4)
        """
        ct_tensor = ct_tensor.to(self.device)
        tumor_mask = tumor_mask.to(self.device)

        # Step ①②③
        volume = ct_tensor * 2.0 - 1.0
        mask = tumor_mask.float() * 2.0 - 1.0
        mask_ = 1.0 - tumor_mask.float()
        masked_volume = (volume * mask_).detach()

        # Step ⑤: 轴重排 (B,C,D,H,W) → (B,C,W,D,H)
        masked_volume_p = masked_volume.permute(0, 1, -1, -3, -2)
        mask_p = mask.permute(0, 1, -1, -3, -2)

        # Step ⑥: VQGAN 编码
        with torch.no_grad():
            masked_feat = self.vqgan.encode(
                masked_volume_p,
                quantize=False,
                include_embeddings=True,
            )

        # Step ⑦: 潜在空间归一化 → [-1, 1]
        masked_feat = (
            (masked_feat - self.emb_min) / (self.emb_max - self.emb_min)
        ) * 2.0 - 1.0

        # Step ⑧: mask 下采样
        cc = torch.nn.functional.interpolate(mask_p, size=masked_feat.shape[-3:])

        # Step ⑨: 拼接
        cond = torch.cat([masked_feat, cc], dim=1)  # (B, 9, D/4, H/4, W/4)
        return cond

    def generate_texture(
        self,
        cond: torch.Tensor,
        organ: str,
        phase: str,
        seed: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """执行扩散采样 + VQGAN 解码 → 合成肿瘤纹理。

        Args:
            cond: (B, 9, D_lat, H_lat, W_lat) 条件向量
            organ: 器官名 (用于权重路由)
            phase: early / noearly
            seed: 随机种子
            eta: DDIM 随机性 (0=确定性, 1=最大随机, 仅 noearly)

        Returns:
            synthetic: (B, 1, D, H, W) 合成肿瘤纹理, 值域 [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
        cond = cond.to(self.device)
        batch_size = cond.shape[0]

        engine = self._get_diffusion_engine(organ, phase)

        with torch.no_grad():
            if engine["type"] == "early":
                # DDPM T=4
                sample_latent = engine["tester"].ema_model.sample(
                    cond=cond, batch_size=batch_size
                )
            else:
                # DDIM S=50
                shape = cond[:, :8].shape[1:]
                samples_ddim, _ = engine["ddim_sampler"].sample(
                    S=50,
                    conditioning=cond,
                    batch_size=batch_size,
                    shape=shape,
                    eta=eta,
                    verbose=False,
                )
                # 反归一化 + VQGAN 解码
                samples_ddim = (
                    (samples_ddim + 1.0) / 2.0
                ) * (
                    self.vqgan.codebook.embeddings.max()
                    - self.vqgan.codebook.embeddings.min()
                ) + self.vqgan.codebook.embeddings.min()
                sample_latent = self.vqgan.decode(samples_ddim, quantize=True)

        # 还原轴顺序: (B,C,W,D,H) → (B,C,D,H,W)
        sample_latent = sample_latent.permute(0, 1, -2, -1, -3)
        return sample_latent  # (B, 1, D, H, W), [-1, 1]

    @staticmethod
    def blend_texture(
        ct_patch: torch.Tensor,      # (1,1,96,96,96) [0,1] 原始 CT
        synthetic: torch.Tensor,      # (1,1,96,96,96) [-1,1] 扩散输出
        tumor_mask: torch.Tensor,     # (1,1,96,96,96) bool
        organ_type: str,
        sigma: float = None,
    ) -> torch.Tensor:
        """纹理融合 (DiffTumor 原版逻辑, 见 TumorGeneration/utils.py)。

        liver/kidney: final = (1-mask_blur)*orig + mask_blur*synthetic, sigma~U(0,4)
        pancreas: 直接替换
        """
        from scipy.ndimage import gaussian_filter

        sample = torch.clamp((synthetic + 1.0) / 2.0, 0.0, 1.0)

        # pancreas/esophagus: 直接替换 (原版做法)
        if organ_type in ("pancreas", "esophagus"):
            return torch.clamp(sample, 0.0, 1.0)

        # liver/kidney: Gaussian alpha 融合, sigma 随机 0~4 (原版: np.random.uniform(0, 4))
        if sigma is None:
            sigma = float(np.random.uniform(0, 4))

        mask_01 = tumor_mask.float()
        mask_np = mask_01.cpu().numpy().astype(np.float32)
        mask_blurred = gaussian_filter(mask_np, sigma=[0, 0, sigma, sigma, sigma])
        mask_blurred = torch.from_numpy(mask_blurred).to(ct_patch.device)

        volume_ = torch.clamp(ct_patch, 0.0, 1.0)
        blended = (1.0 - mask_blurred) * volume_ + mask_blurred * sample
        return torch.clamp(blended, 0.0, 1.0)

    def inject_tumor(
        self,
        full_ct: np.ndarray,           # (D,H,W) HU 值, maisi 生成
        full_mask: np.ndarray,          # (D,H,W) 132-class 整数标签
        organ: str,
        phase: str,
        spacing: Tuple[float, ...],
        affine: np.ndarray,
        output_mode: str = "both",
        eta: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """将 DiffTumor 合成肿瘤纹理嵌入到 MAISI 生成的完整 CT 中。

        Args:
            full_ct: maisi 生成的完整 CT (D,H,W), HU 值
            full_mask: maisi 生成的分割 mask (D,H,W), 132-class
            organ: 器官名
            phase: early / noearly
            spacing: 体素间距 (3,)
            affine: 4×4 仿射矩阵
            output_mode: full_ct / patch_96 / both
            eta: DDIM 随机性

        Returns:
            (final_ct, tumor_mask_native, meta)
        """
        import SimpleITK as sitk
        import nibabel as nib
        from scipy.ndimage import gaussian_filter as gf

        organ_info = ORGAN_TO_MAISI[organ]
        tumor_label = organ_info["tumor_label"]
        organ_label = organ_info["organ_label"]
        organ_type = organ_info["organ_type"]

        # 如果此器官没有肿瘤标签 (esophagus/uterus 零样本), 创建一个合成肿瘤区域
        # 如果 MAISI mask 缺少肿瘤标签, 同样在器官内创建合成肿瘤
        if tumor_label == 0:
            organ_bool = (full_mask == organ_label)
            if organ_bool.sum() < 10:
                logger.warning(f"器官 {organ} 体素过少 ({organ_bool.sum()}), 跳过肿瘤注入")
                return full_ct, np.zeros_like(full_ct, dtype=np.uint8), {"status": "skip"}
            tumor_mask = self._create_synthetic_tumor_mask(organ_bool, spacing, phase)
        else:
            tumor_mask_from_label = (full_mask == tumor_label)
            if tumor_mask_from_label.sum() < 10:
                # MAISI mask 缺少肿瘤标签 — 在器官内部创建合成肿瘤区域
                organ_bool = (full_mask == organ_label)
                if organ_bool.sum() < 10:
                    logger.warning(f"器官 {organ} 体素过少 ({organ_bool.sum()}), 跳过肿瘤注入")
                    return full_ct, np.zeros_like(full_ct, dtype=np.uint8), {"status": "skip"}
                logger.warning(
                    f"MAISI mask 缺少肿瘤标签 {tumor_label} (体素={tumor_mask_from_label.sum()}), "
                    f"在器官 {organ} (体素={organ_bool.sum()}) 内创建合成肿瘤"
                )
                tumor_mask = self._create_synthetic_tumor_mask(organ_bool, spacing, phase)
            else:
                tumor_mask = tumor_mask_from_label

        logger.info(f"tumor_mask vox={tumor_mask.sum()}, organ_label={organ_label}")

        organ_mask = (full_mask == organ_label)
        # kidney: 合并左右肾
        if organ == "kidney":
            organ_mask = organ_mask | (full_mask == 5) | (full_mask == 14)

        if tumor_mask.sum() < 10:
            logger.warning(f"肿瘤 mask 体素过少 ({tumor_mask.sum()}), 跳过肿瘤注入")
            return full_ct, np.zeros_like(full_ct, dtype=np.uint8), {"status": "skip"}

        # ── Step 1: 裁剪肿瘤区域 (原生空间) ──
        t_idx = np.argwhere(tumor_mask)
        ctr = t_idx.mean(axis=0).astype(int)
        logger.debug(f"tumor center={ctr}, spacing={spacing}")
        half_phys = 48.0  # 96mm / 2
        half = [int(np.ceil(half_phys / s)) for s in spacing]
        x0, x1 = max(0, ctr[0]-half[0]), min(full_ct.shape[0], ctr[0]+half[0])
        y0, y1 = max(0, ctr[1]-half[1]), min(full_ct.shape[1], ctr[1]+half[1])
        z0, z1 = max(0, ctr[2]-half[2]), min(full_ct.shape[2], ctr[2]+half[2])

        ct_crop = full_ct[x0:x1, y0:y1, z0:z1].copy()
        tm_crop = tumor_mask[x0:x1, y0:y1, z0:z1].copy()
        og_crop = organ_mask[x0:x1, y0:y1, z0:z1].copy()
        orig_crop_shape = ct_crop.shape

        # 如果物理尺寸 < 96mm, 进行 padding
        for i, s in enumerate(spacing):
            current = ct_crop.shape[i] * s
            if current < 96.0:
                pad_voxels = int(np.ceil((96.0 - current) / s))
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[i] = (0, pad_voxels)
                ct_crop = np.pad(ct_crop, pad_width, mode='constant', constant_values=-1000)
                tm_crop = np.pad(tm_crop, pad_width, mode='constant', constant_values=0)
                og_crop = np.pad(og_crop, pad_width, mode='constant', constant_values=0)

        # ── Step 2: 保存临时文件 → 预处理 → 1mm³ 96³ ──
        tmp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "_tmp_tumor"
        )
        os.makedirs(tmp_dir, exist_ok=True)

        real_aff = np.diag(list(spacing) + [1.0])
        for name, arr in [("ct", ct_crop), ("org", og_crop.astype(np.int16)),
                           ("tm", tm_crop.astype(np.int16))]:
            nib.save(
                nib.Nifti1Image(arr.astype(np.float32), real_aff),
                os.path.join(tmp_dir, f"inject_{name}.nii.gz")
            )

        # 预处理: 重采样 + HU 裁剪归一化
        ct_t, tm_t, og_t = self._preprocess_crop(
            os.path.join(tmp_dir, "inject_ct.nii.gz"),
            os.path.join(tmp_dir, "inject_org.nii.gz"),
            os.path.join(tmp_dir, "inject_tm.nii.gz"),
        )

        # 居中裁剪到 96³
        d, h, w = ct_t.shape[2:]
        ct_t = ct_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]
        tm_t = tm_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]

        # ── Step 3: DiffTumor 纹理生成 ──
        cond = self.build_condition(ct_t, tm_t)
        logger.debug(f"condition shape={cond.shape}")
        synthetic = self.generate_texture(cond, organ, phase, eta=eta)

        # ── Step 4: 融合 ──
        blended = self.blend_texture(ct_t, synthetic, tm_t, organ_type)

        # 转回 HU
        blended_hu = blended.squeeze().cpu().numpy() * (HU_MAX - HU_MIN) + HU_MIN

        # ── Step 5: 重采样回原生空间 + 嵌入 ──
        blended_sitk = sitk.GetImageFromArray(blended_hu.transpose(2, 1, 0))
        blended_sitk.SetSpacing((1.0, 1.0, 1.0))
        native_crop_sitk = sitk.GetImageFromArray(ct_crop.transpose(2, 1, 0))
        native_crop_sitk.SetSpacing([float(s) for s in spacing])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(native_crop_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        blended_native = sitk.GetArrayFromImage(
            resampler.Execute(blended_sitk)
        ).transpose(2, 1, 0)

        # 修剪 padding
        if blended_native.shape != orig_crop_shape:
            blended_native = blended_native[:orig_crop_shape[0],
                                            :orig_crop_shape[1],
                                            :orig_crop_shape[2]]
            tm_crop = tm_crop[:orig_crop_shape[0],
                              :orig_crop_shape[1],
                              :orig_crop_shape[2]]
            og_crop = og_crop[:orig_crop_shape[0],
                              :orig_crop_shape[1],
                              :orig_crop_shape[2]]

        # ── Step 6: 将融合后的 patch 直接写回完整 CT ──
        # DiffTumor 原版逻辑: 在 96³ patch 内已完成 mask_blur 融合,
        # 直接将 patch 区域替换回完整 CT 即可, 不需要额外的 organ_edge 边界融合
        final_ct = full_ct.copy()
        final_ct[x0:x1, y0:y1, z0:z1] = blended_native

        # 生成最终肿瘤 mask
        # 注意: 肿瘤voxel (label=26) 和器官voxel (label=1) 在132-class mask中不重叠
        # 一个voxel只能有一个标签值, 所以 tm_crop & og_crop = 0 是常态
        # 直接用 tm_crop 作为肿瘤区域
        full_tumor_mask = np.zeros_like(full_ct, dtype=np.uint8)
        full_tumor_mask[x0:x1, y0:y1, z0:z1] = tm_crop.astype(np.uint8)

        meta = {
            "organ": organ_type,
            "phase": phase,
            "crop_native": [int(x) for x in [x0, x1, y0, y1, z0, z1]],
            "shape": list(full_ct.shape),
        }
        if output_mode in ("patch_96", "both"):
            meta["patch_96_hu"] = blended_hu.copy()
            meta["patch_96_mask"] = tm_t[0, 0].cpu().numpy().astype(np.uint8)

        return final_ct, full_tumor_mask, meta

    def _preprocess_crop(
        self,
        ct_path: str,
        organ_path: str,
        tumor_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预处理裁剪区域: 重采样 1mm³ + HU 裁剪归一化 + 转 Tensor。

        Returns:
            (ct_tensor, tumor_tensor, organ_tensor) 各为 (1,1,D,H,W)
        """
        import SimpleITK as sitk

        # 加载
        ct_sitk = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        organ_sitk = sitk.ReadImage(organ_path, sitk.sitkUInt8)
        tumor_sitk = sitk.ReadImage(tumor_path, sitk.sitkUInt8)

        # 重采样到 1mm³
        ct_sitk = self._resample(ct_sitk, TARGET_SPACING, sitk.sitkLinear)
        organ_sitk = self._resample(organ_sitk, TARGET_SPACING, sitk.sitkNearestNeighbor)
        tumor_sitk = self._resample(tumor_sitk, TARGET_SPACING, sitk.sitkNearestNeighbor)

        # HU 裁剪 + 归一化
        ct_arr = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        ct_arr = np.clip(ct_arr, HU_MIN, HU_MAX)
        ct_arr = (ct_arr - HU_MIN) / (HU_MAX - HU_MIN)

        # Mask 提取
        tumor_arr = sitk.GetArrayFromImage(tumor_sitk).astype(bool)
        organ_arr = sitk.GetArrayFromImage(organ_sitk).astype(bool)

        # 转 Tensor
        ct_t = torch.from_numpy(ct_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)
        tm_t = torch.from_numpy(tumor_arr).bool().unsqueeze(0).unsqueeze(0).to(self.device)
        og_t = torch.from_numpy(organ_arr).bool().unsqueeze(0).unsqueeze(0).to(self.device)

        return ct_t, tm_t, og_t

    @staticmethod
    def _resample(image, target_spacing, interpolator):
        """SimpleITK 重采样"""
        orig_spacing = np.array(image.GetSpacing())
        orig_size = np.array(image.GetSize())
        target_spacing = np.array(target_spacing)
        new_size = (orig_size * orig_spacing / target_spacing).astype(np.int32).tolist()

        resampler = __import__("SimpleITK").ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing.tolist())
        resampler.SetSize(new_size)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        return resampler.Execute(image)

    @staticmethod
    def _create_synthetic_tumor_mask(
        organ_mask: np.ndarray,
        spacing: Tuple[float, ...],
        phase: str,
    ) -> np.ndarray:
        """在器官 mask 内部创建一个合成椭球形肿瘤区域 (用于零样本器官)。

        Args:
            organ_mask: (D,H,W) bool, 器官区域
            spacing: 体素间距
            phase: early → 小肿瘤, noearly → 大肿瘤
        """
        from scipy.ndimage import gaussian_filter

        # 找到器官中心
        idx = np.argwhere(organ_mask)
        if len(idx) == 0:
            return np.zeros_like(organ_mask, dtype=bool)

        center = idx.mean(axis=0).astype(int)

        # 肿瘤半径 (物理 mm)
        if phase == "early":
            r_mm = np.random.uniform(3, 8)
        else:
            r_mm = np.random.uniform(10, 25)

        # 创建椭球
        r_vox = [r_mm / s for s in spacing]
        tumor = np.zeros_like(organ_mask, dtype=np.float32)
        for i, (c, r) in enumerate(zip(center, r_vox)):
            sl = slice(max(0, int(c - 2*r)), min(tumor.shape[i], int(c + 2*r)))
            # 简化: 在中心放一个球形
        z, y, x = np.ogrid[:tumor.shape[0], :tumor.shape[1], :tumor.shape[2]]
        dist = ((z - center[0]) / r_vox[0])**2 + \
               ((y - center[1]) / r_vox[1])**2 + \
               ((x - center[2]) / r_vox[2])**2
        tumor = (dist <= 1.0) & organ_mask

        return tumor


# ═══════════════════════════════════════════════════════════════
#  完整管线: MAISI 生成 → DiffTumor 注入 → 输出
# ═══════════════════════════════════════════════════════════════

def run_tumor_pipeline(
    task: TumorTask,
    device: str = "cpu",
    pipeline_config_path: str = None,
    tumor_paths_path: str = None,
) -> TumorPipelineResult:
    """执行完整的肿瘤生成管线。

    Phase 1: MAISI 生成基础 CT + mask
    Phase 2: DiffTumor 生成肿瘤纹理
    Phase 3: 融合嵌入
    """
    import nibabel as nib
    from .sample import LDMSampler
    from .utils import define_instance

    result = TumorPipelineResult(organ=task.organ)
    t_start = time.time()

    # ── 加载配置 ──
    adapter = TumorConfigAdapter(pipeline_config_path)
    maisi_params = adapter.task_to_maisi_params(task)
    phase = maisi_params["phase"]
    organ_info = maisi_params["organ_info"]
    result.phase = phase

    # 加载 tumor_paths.json
    if tumor_paths_path is None:
        tumor_paths_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "tumor_paths.json"
        )
    with open(tumor_paths_path, "r", encoding="utf-8") as f:
        tumor_paths = json.load(f)

    vqgan_ckpt = tumor_paths.get("vqgan_ckpt_path", "")
    diffusion_dir = tumor_paths.get("diffusion_ckpt_dir", "")
    diffumor_repo = tumor_paths.get("diffumor_repo_dir", "")

    if not vqgan_ckpt or not diffusion_dir or not diffumor_repo:
        result.status = "fail"
        result.error = (
            "tumor_paths.json 中的路径未配置。请设置:\n"
            "  - vqgan_ckpt_path: VQGAN 权重路径\n"
            "  - diffusion_ckpt_dir: 扩散模型权重目录\n"
            "  - diffumor_repo_dir: DiffTumor 源码目录"
        )
        return result

    try:
        # ── Phase 1: MAISI 基础 CT 生成 ──
        logger.info(f"Phase 1: MAISI 生成基础 CT ({task.organ}, {task.size_category})")

        # 此处需要由调用方预先完成 MAISI 生成, 并提供 CT 和 mask 的文件路径
        # 或者我们在此处调用 MAISI 的 LDMSampler
        # 为灵活性, 本函数接受已生成的 CT/mask ndarray

        # 如果外部提供了 mask_file, 使用它; 否则使用 maisi 生成的 mask
        # 具体实现取决于调用方式

        logger.info("请使用 run_tumor_pipeline_from_files() 传入已生成的 CT/mask 文件")
        result.status = "fail"
        result.error = "请使用 run_tumor_pipeline_from_files() 传入 MAISI 已生成的 CT 和 mask 文件"
        return result

    except Exception as e:
        result.status = "fail"
        result.error = str(e)
        logger.error(f"管线失败: {e}")
        import traceback
        traceback.print_exc()
        return result


def run_tumor_pipeline_from_files(
    ct_path: str,
    mask_path: str,
    task: TumorTask,
    device: str = "cpu",
    pipeline_config_path: str = None,
    tumor_paths_path: str = None,
) -> TumorPipelineResult:
    """从已有的 MAISI 生成文件执行肿瘤纹理注入管线。

    这是主要的入口函数。调用方先用 MAISI 生成基础 CT + mask,
    然后调用此函数注入 DiffTumor 肿瘤纹理。

    Args:
        ct_path: MAISI 生成的 CT NIfTI 文件路径
        mask_path: MAISI 生成的 mask NIfTI 文件路径 (132-class)
        task: 肿瘤任务配置
        device: 计算设备
        pipeline_config_path: 管线配置路径
        tumor_paths_path: DiffTumor 路径配置路径
    """
    import nibabel as nib

    result = TumorPipelineResult(organ=task.organ)
    t_start = time.time()

    # ── 加载配置 ──
    adapter = TumorConfigAdapter(pipeline_config_path)
    maisi_params = adapter.task_to_maisi_params(task)
    phase = maisi_params["phase"]
    organ_info = maisi_params["organ_info"]
    result.phase = phase

    # 加载 tumor_paths.json
    if tumor_paths_path is None:
        tumor_paths_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "tumor_paths.json"
        )
    with open(tumor_paths_path, "r", encoding="utf-8") as f:
        tumor_paths = json.load(f)

    vqgan_ckpt = tumor_paths.get("vqgan_ckpt_path", "")
    diffusion_dir = tumor_paths.get("diffusion_ckpt_dir", "")
    diffumor_repo = tumor_paths.get("diffumor_repo_dir", "")

    if not vqgan_ckpt or not diffusion_dir or not diffumor_repo:
        result.status = "fail"
        result.error = (
            "tumor_paths.json 路径未配置。请设置 vqgan_ckpt_path, "
            "diffusion_ckpt_dir, diffumor_repo_dir。"
        )
        return result

    try:
        # ── 加载 MAISI 生成的 CT 和 mask ──
        logger.info(f"Loading MAISI output: CT={ct_path}, Mask={mask_path}")
        ct_nii = nib.load(ct_path)
        full_ct = ct_nii.get_fdata().astype(np.float32)
        affine = ct_nii.affine.copy()
        spacing = np.array(ct_nii.header.get_zooms()[:3])

        mask_nii = nib.load(mask_path)
        full_mask = mask_nii.get_fdata().astype(np.int32)
        organ_vox = (full_mask == organ_info['organ_label']).sum()
        tumor_vox = (full_mask == organ_info['tumor_label']).sum() if organ_info['tumor_label'] > 0 else 0
        logger.info(f"organ vox={organ_vox}, tumor vox={tumor_vox}")

        # ── Phase 2+3: DiffTumor 纹理注入 + 融合 ──
        logger.info(f"Phase 2: DiffTumor 纹理注入 ({task.organ}, phase={phase})")

        injector = TumorTextureInjector(
            vqgan_ckpt_path=vqgan_ckpt,
            diffusion_ckpt_dir=diffusion_dir,
            diffumor_repo_dir=diffumor_repo,
            device=device,
        )

        final_ct, tumor_mask, meta = injector.inject_tumor(
            full_ct=full_ct,
            full_mask=full_mask,
            organ=task.organ,
            phase=phase,
            spacing=tuple(spacing),
            affine=affine,
            output_mode=task.output,
            eta=task.eta,
        )

        # ── 保存输出 ──
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "tumor_ct", organ_info["organ_type"]
        )
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = task.output_name or f"{task.organ}_{task.size_category}_{phase}_{timestamp}"

        out_paths = []

        # 保存完整 CT
        if task.output in ("full_ct", "both"):
            ct_out = os.path.join(output_dir, f"{base_name}.nii.gz")
            v = 2
            while os.path.exists(ct_out):
                ct_out = os.path.join(output_dir, f"{base_name}_v{v}.nii.gz")
                v += 1
            nib.save(nib.Nifti1Image(final_ct.astype(np.float32), affine), ct_out)
            out_paths.append(ct_out)

            # 保存肿瘤 mask
            mask_out = os.path.join(output_dir, f"{base_name}_tumor_mask.nii.gz")
            if not os.path.exists(mask_out):
                nib.save(nib.Nifti1Image(tumor_mask, affine), mask_out)
            out_paths.append(mask_out)

        # 保存 96³ patch
        if task.output in ("patch_96", "both") and "patch_96_hu" in meta:
            patch_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output", "tumor_patch", organ_info["organ_type"]
            )
            os.makedirs(patch_dir, exist_ok=True)
            patch_aff = np.eye(4)
            patch_path = os.path.join(patch_dir, f"{base_name}.nii.gz")
            nib.save(
                nib.Nifti1Image(meta["patch_96_hu"].astype(np.float32), patch_aff),
                patch_path
            )
            out_paths.append(patch_path)

            # 保存 patch mask
            patch_mask_path = os.path.join(patch_dir, f"{base_name}_mask.nii.gz")
            if not os.path.exists(patch_mask_path) and "patch_96_mask" in meta:
                nib.save(
                    nib.Nifti1Image(meta["patch_96_mask"], patch_aff),
                    patch_mask_path
                )
            out_paths.append(patch_mask_path)

        # 统计
        dt = time.time() - t_start
        t_hu = final_ct[tumor_mask > 0] if tumor_mask.sum() > 0 else np.array([0])
        result.status = "ok"
        result.output_paths = out_paths
        result.tumor_hu_mean = round(float(t_hu.mean()), 1)
        result.tumor_hu_std = round(float(t_hu.std()), 1)
        result.time_s = round(dt, 1)

        logger.info(
            f"  OK  {task.organ}/{task.size_category}  "
            f"vox={tumor_mask.sum():,}  HU={t_hu.mean():.0f}±{t_hu.std():.0f}  "
            f"time={dt:.0f}s"
        )

    except Exception as e:
        result.status = "fail"
        result.error = str(e)
        logger.error(f"管线失败: {e}")
        import traceback
        traceback.print_exc()

    # 释放 DiffTumor GPU 显存
    if 'injector' in dir():
        del injector
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
