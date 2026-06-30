"""
Compute 2.5D FID for medical CT images -- single GPU, Windows compatible.

Based on MAISI's compute_fid_2-5d_ct.py but removes torch.distributed/NCCL
dependency. Includes statistical diagnostics to assess FID reliability.

对比逻辑: 含肿瘤真实CT vs 含肿瘤合成CT
  FID衡量的是两组数据的分布距离，越低越接近真实分布。

Usage:
    # 默认配置 (8GB VRAM友好)
    python compute_fid_single_gpu.py \
        --real_dataset_root data/autopet_ct50 \
        --real_filelist data/autopet_ct50/filelist.txt \
        --synth_dataset_root data/fid_tumor_ct_50 \
        --synth_filelist data/fid_tumor_ct_50/filelist.txt \
        --target_shape 256x256x128 \
        --enable_resampling_spacing 1.0x1.0x1.0 \
        --enable_center_slices_ratio 0.4 \
        --num_images 48

    # 论文对标配置 (需要更多VRAM，先pre-crop再resample避免OOM)
    python compute_fid_single_gpu.py --paper_config \
        --real_dataset_root data/autopet_ct50 \
        --real_filelist data/autopet_ct50/filelist.txt \
        --synth_dataset_root data/fid_tumor_ct_50 \
        --synth_filelist data/fid_tumor_ct_50/filelist.txt \
        --num_images 48
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import fire
import monai
import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics.fid import FIDMetric
from monai.transforms import Compose

# ------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------
logger = logging.getLogger("fid_single_gpu")
if not logger.handlers:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------------------
# Utility functions (from MAISI compute_fid_2-5d_ct.py)
# ------------------------------------------------------------------------------

def drop_empty_slice(slices, empty_threshold: float):
    """Decide which 2D slices to keep by checking max intensity."""
    outputs = []
    n_drop = 0
    for s in slices:
        largest_unique = torch.max(torch.unique(s))
        if largest_unique < empty_threshold:
            outputs.append(False)
            n_drop += 1
        else:
            outputs.append(True)
    logger.info(f"Empty slice drop rate {round((n_drop / len(slices)) * 100, 1)}%")
    return outputs


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    """Subtract ImageNet per-channel means [0.406, 0.456, 0.485]."""
    mean = [0.406, 0.456, 0.485]
    x[:, 0, ...] -= mean[0]
    x[:, 1, ...] -= mean[1]
    x[:, 2, ...] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """Average out spatial dimensions to produce a 1D feature vector."""
    dim = len(x.shape)
    if dim == 2:
        return x
    if dim == 3:
        return x.mean([2], keepdim=keepdim)
    if dim == 4:
        return x.mean([2, 3], keepdim=keepdim)
    if dim == 5:
        return x.mean([2, 3, 4], keepdim=keepdim)
    return x


def radimagenet_intensity_normalisation(volume: torch.Tensor, norm2d: bool = False) -> torch.Tensor:
    """Intensity normalization for RadImageNet ResNet."""
    dim = len(volume.shape)
    if dim == 4 and norm2d:
        max2d, _ = torch.max(volume, dim=2, keepdim=True)
        max2d, _ = torch.max(max2d, dim=3, keepdim=True)
        min2d, _ = torch.min(volume, dim=2, keepdim=True)
        min2d, _ = torch.min(min2d, dim=3, keepdim=True)
        volume = (volume - min2d) / (max2d - min2d + 1e-10)
        return subtract_mean(volume)
    elif dim == 4:
        max3d = torch.max(volume)
        min3d = torch.min(volume)
        volume = (volume - min3d) / (max3d - min3d + 1e-10)
        return subtract_mean(volume)
    if dim == 5:
        maxval = torch.max(volume)
        minval = torch.min(volume)
        volume = (volume - minval) / (maxval - minval + 1e-10)
        return subtract_mean(volume)
    return volume


def get_features_2p5d(
    image: torch.Tensor,
    feature_network: torch.nn.Module,
    center_slices: bool = False,
    center_slices_ratio: float = 1.0,
    sample_every_k: int = 1,
    xy_only: bool = True,
    drop_empty: bool = False,
    empty_threshold: float = -700,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Extract 2.5D features from a 3D image by slicing along XY, YZ, ZX planes."""
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1, 1)

    # Convert from 'RGB'->(R,G,B) to (B,G,R)
    image = image[:, [2, 1, 0], ...]

    B, C, H, W, D = image.size()
    with torch.no_grad():
        # --- XY-plane slicing along D ---
        if center_slices:
            start_d = int((1.0 - center_slices_ratio) / 2.0 * D)
            end_d = int((1.0 + center_slices_ratio) / 2.0 * D)
            slices = torch.unbind(image[:, :, :, :, start_d:end_d:sample_every_k], dim=-1)
        else:
            slices = torch.unbind(image, dim=-1)

        if drop_empty:
            mapping_index = drop_empty_slice(slices, empty_threshold)
        else:
            mapping_index = [True for _ in range(len(slices))]

        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]

        feature_image_xy = feature_network.forward(images_2d)
        feature_image_xy = spatial_average(feature_image_xy, keepdim=False)
        if xy_only:
            return feature_image_xy, None, None

        # --- YZ-plane slicing along H ---
        if center_slices:
            start_h = int((1.0 - center_slices_ratio) / 2.0 * H)
            end_h = int((1.0 + center_slices_ratio) / 2.0 * H)
            slices = torch.unbind(image[:, :, start_h:end_h:sample_every_k, :, :], dim=2)
        else:
            slices = torch.unbind(image, dim=2)

        if drop_empty:
            mapping_index = drop_empty_slice(slices, empty_threshold)
        else:
            mapping_index = [True for _ in range(len(slices))]

        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]

        feature_image_yz = feature_network.forward(images_2d)
        feature_image_yz = spatial_average(feature_image_yz, keepdim=False)

        # --- ZX-plane slicing along W ---
        if center_slices:
            start_w = int((1.0 - center_slices_ratio) / 2.0 * W)
            end_w = int((1.0 + center_slices_ratio) / 2.0 * W)
            slices = torch.unbind(image[:, :, :, start_w:end_w:sample_every_k, :], dim=3)
        else:
            slices = torch.unbind(image, dim=3)

        if drop_empty:
            mapping_index = drop_empty_slice(slices, empty_threshold)
        else:
            mapping_index = [True for _ in range(len(slices))]

        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]

        feature_image_zx = feature_network.forward(images_2d)
        feature_image_zx = spatial_average(feature_image_zx, keepdim=False)

    return feature_image_xy, feature_image_yz, feature_image_zx


# ------------------------------------------------------------------------------
# Statistical diagnostics
# ------------------------------------------------------------------------------

def diagnose_features(features: torch.Tensor, label: str) -> dict:
    """Diagnose statistical reliability of feature matrix for FID.

    Checks singular value distribution, effective rank, and whether
    the covariance matrix is well-conditioned enough for reliable
    sqrtm computation in FID.
    """
    # features shape: (N, D) -- N samples, D feature dimensions
    N, D = features.shape

    # Center the features
    centered = features - features.mean(dim=0, keepdim=True)

    # Singular value decomposition
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        return {
            "label": label,
            "N": N, "D": D,
            "effective_rank": 0,
            "rank_ratio": 0.0,
            "condition_number": float('inf'),
            "top_singular_values": [],
            "reliability": "unreliable",
            "note": "SVD failed -- features are degenerate",
        }

    # Effective rank: number of singular values > threshold
    # Threshold: max(S) * max(N, D) * eps (numerical tolerance)
    eps = max(S[0].item() * max(N, D) * torch.finfo(S.dtype).eps, 1e-8)
    effective_rank = int((S > eps).sum().item())

    # Condition number
    if S[0] > eps:
        condition_number = S[0].item() / max(S[min(effective_rank - 1, len(S) - 1)].item(), eps)
    else:
        condition_number = float('inf')

    # Ratio of effective rank to feature dimension
    rank_ratio = effective_rank / D if D > 0 else 0

    # Reliability assessment based on effective rank ratio:
    #   >= 95%: reliable (sqrtm accurate enough for FID)
    #   >= 80%: acceptable (FID may have small bias, still usable)
    #   <  80%: unreliable (covariance too ill-conditioned)
    if rank_ratio >= 0.95:
        reliability = "reliable"
    elif rank_ratio >= 0.80:
        reliability = "acceptable"
    else:
        reliability = "unreliable"

    return {
        "label": label,
        "N": N, "D": D,
        "effective_rank": effective_rank,
        "rank_ratio": round(rank_ratio, 3),
        "condition_number": round(condition_number, 2),
        "top_singular_values": [round(s.item(), 4) for s in S[:5]],
        "reliability": reliability,
        "note": "",
    }


def print_diagnosis(diag: dict):
    """Print diagnostic information for feature matrix."""
    N, D = diag['N'], diag['D']
    er = diag['effective_rank']
    rank_pct = diag['rank_ratio'] * 100
    reliability = diag['reliability']

    print(f"  N={N}, D={D}")
    print(f"  effective rank: {er} / {D} ({rank_pct:.1f}%)")
    print(f"  condition number: {diag['condition_number']:.2e}")
    print(f"  top-5 singular values: {diag['top_singular_values']}")

    if reliability == "reliable":
        print(f"  [OK] reliable -- rank ratio >= 95%, sqrtm accurate for FID")
    elif reliability == "acceptable":
        print(f"  [~] acceptable -- rank ratio >= 80%, FID may have small bias")
        print(f"      -> consider increasing sample count for more precise FID")
    else:
        print(f"  [!!] unreliable -- rank ratio < 80%, covariance too ill-conditioned")
        print(f"       -> FID value may be significantly biased (too low or too high)")
        print(f"       -> need more samples or lower-dimensional features")


# ------------------------------------------------------------------------------
# Feature network loading
# ------------------------------------------------------------------------------

def load_feature_network() -> tuple[torch.nn.Module, str, int]:
    """Load RadImageNet ResNet50 and return (network, name, feature_dim).

    Only RadImageNet ResNet50 is supported -- it is the only feature network
    used in the MAISI paper (Table 2, Table 3) for FID evaluation.

    feature_dim=2048 is the output dimension after spatial_average (avgpool).
    Note: the model is moved to GPU later in compute_fid_for_network().
    """
    try:
        feature_network = torch.hub.load(
            "Warvito/radimagenet-models",
            model="radimagenet_resnet50",
            verbose=True,
            trust_repo=True,
        )
        logger.info("RadImageNet ResNet50 loaded successfully")
        return feature_network, "radimagenet_resnet50", 2048
    except Exception as e:
        logger.error(
            f"RadImageNet ResNet50 failed to load: {e}\n"
            f"This is the only feature network supported for FID evaluation.\n"
            f"It must be downloaded on first run (~100MB, requires internet).\n"
            f"Cached at ~/.cache/torch/hub/ after first download."
        )
        raise


# ------------------------------------------------------------------------------
# FID computation core
# ------------------------------------------------------------------------------

def compute_fid_for_network(
    feature_network: torch.nn.Module,
    model_name: str,
    feature_dim: int,
    real_filenames: list,
    synth_filenames: list,
    transforms: Compose,
    device: torch.device,
    enable_center_slices: bool,
    center_slices_ratio: float,
    output_root: str,
    real_features_dir: str,
    synth_features_dir: str,
    real_dataset_root: str,
    synth_dataset_root: str,
    ignore_existing: bool,
) -> dict:
    """Compute FID for a single feature network, with diagnostics."""

    feature_network.to(device)
    feature_network.eval()

    # --- Extract features: Real ---
    logger.info(f"Extracting REAL features with {model_name} ({feature_dim}d)")
    output_root_real = os.path.join(output_root, real_features_dir)
    real_features_xy, real_features_yz, real_features_zx = [], [], []

    real_ds = monai.data.Dataset(data=real_filenames, transform=transforms)
    real_loader = monai.data.DataLoader(real_ds, num_workers=0, batch_size=1, shuffle=False)

    for idx, batch_data in enumerate(real_loader, start=1):
        img = batch_data["image"].to(device)
        fn = img.meta["filename_or_obj"][0]
        logger.info(f"Real {idx}/{len(real_filenames)}: {os.path.basename(fn)}")

        out_fp = fn.replace(real_dataset_root, output_root_real).replace(".nii.gz", ".pt")
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if (not ignore_existing) and os.path.isfile(out_fp):
            feats = torch.load(out_fp, weights_only=True)
        else:
            img_t = img.as_tensor()
            logger.info(f"  image shape: {tuple(img_t.shape)}")
            feats = get_features_2p5d(
                img_t, feature_network,
                center_slices=enable_center_slices,
                center_slices_ratio=center_slices_ratio,
                xy_only=False,
            )
            logger.info(f"  feats: {feats[0].shape}")
            torch.save(feats, out_fp)

        real_features_xy.append(feats[0])
        real_features_yz.append(feats[1])
        real_features_zx.append(feats[2])

    real_features_xy = torch.vstack(real_features_xy)
    real_features_yz = torch.vstack(real_features_yz)
    real_features_zx = torch.vstack(real_features_zx)

    # --- Extract features: Synthetic ---
    logger.info(f"Extracting SYNTH features with {model_name} ({feature_dim}d)")
    output_root_synth = os.path.join(output_root, synth_features_dir)
    synth_features_xy, synth_features_yz, synth_features_zx = [], [], []

    synth_ds = monai.data.Dataset(data=synth_filenames, transform=transforms)
    synth_loader = monai.data.DataLoader(synth_ds, num_workers=0, batch_size=1, shuffle=False)

    for idx, batch_data in enumerate(synth_loader, start=1):
        img = batch_data["image"].to(device)
        fn = img.meta["filename_or_obj"][0]
        logger.info(f"Synth {idx}/{len(synth_filenames)}: {os.path.basename(fn)}")

        out_fp = fn.replace(synth_dataset_root, output_root_synth).replace(".nii.gz", ".pt")
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if (not ignore_existing) and os.path.isfile(out_fp):
            feats = torch.load(out_fp, weights_only=True)
        else:
            img_t = img.as_tensor()
            logger.info(f"  image shape: {tuple(img_t.shape)}")
            feats = get_features_2p5d(
                img_t, feature_network,
                center_slices=enable_center_slices,
                center_slices_ratio=center_slices_ratio,
                xy_only=False,
            )
            logger.info(f"  feats: {feats[0].shape}")
            torch.save(feats, out_fp)

        synth_features_xy.append(feats[0])
        synth_features_yz.append(feats[1])
        synth_features_zx.append(feats[2])

    synth_features_xy = torch.vstack(synth_features_xy)
    synth_features_yz = torch.vstack(synth_features_yz)
    synth_features_zx = torch.vstack(synth_features_zx)

    # --- Compute FID ---
    logger.info("Computing FID")
    fid_metric = FIDMetric()

    fid_xy = fid_metric(synth_features_xy, real_features_xy)
    fid_yz = fid_metric(synth_features_yz, real_features_yz)
    fid_zx = fid_metric(synth_features_zx, real_features_zx)
    fid_avg = (fid_xy + fid_yz + fid_zx) / 3.0

    # --- Diagnostics ---
    diag_real_xy = diagnose_features(real_features_xy, "real_xy")
    diag_synth_xy = diagnose_features(synth_features_xy, "synth_xy")

    return {
        "model_name": model_name,
        "feature_dim": feature_dim,
        "fid_xy": fid_xy.item(),
        "fid_yz": fid_yz.item(),
        "fid_zx": fid_zx.item(),
        "fid_avg": fid_avg.item(),
        "diag_real": diag_real_xy,
        "diag_synth": diag_synth_xy,
        "n_real": len(real_filenames),
        "n_synth": len(synth_filenames),
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(
    real_dataset_root: str = "data/autopet_ct50",
    real_filelist: str = "data/autopet_ct50/filelist.txt",
    real_features_dir: str = "autopet_real",
    synth_dataset_root: str = "data/fid_tumor_ct_50",
    synth_filelist: str = "data/fid_tumor_ct_50/filelist.txt",
    synth_features_dir: str = "tumor_ct_synth",
    enable_center_slices_ratio: float = 0.4,
    enable_padding: bool = True,
    enable_center_cropping: bool = True,
    enable_resampling_spacing: str = "1.0x1.0x1.0",
    ignore_existing: bool = False,
    num_images: int = 48,
    output_root: str = "data/fid_features",
    target_shape: str = "256x256x128",
    paper_config: bool = False,
):
    """
    Compute 2.5D FID for medical CT -- single GPU, Windows compatible, with diagnostics.

    Uses RadImageNet ResNet50 (2048d features), the same feature network as the
    MAISI paper (arXiv:2409.11169v3, Table 2 and Table 3).

    Transform order matches the MAISI paper script (compute_fid_2-5d_ct.py):
      resample -> pad -> crop -> scale_intensity
    For large CTs (e.g. autoPET 400x400x588), a pre-crop is applied before
    resampling to avoid OOM -- this is the only deviation from the paper script,
    and is documented in the output report.
    """
    # --- Resolve project root ---
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)

    def _resolve(p: str) -> str:
        """Resolve a path: absolute as-is, relative -> try CWD first, then project root."""
        if os.path.isabs(p):
            return p
        if os.path.exists(p):
            return os.path.abspath(p)
        candidate = os.path.join(_project_root, p)
        if os.path.exists(candidate):
            return candidate
        return candidate

    real_dataset_root = _resolve(real_dataset_root)
    real_filelist = _resolve(real_filelist)
    synth_dataset_root = _resolve(synth_dataset_root)
    synth_filelist = _resolve(synth_filelist)
    output_root = _resolve(output_root)

    # --- Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Project root: {_project_root}")

    # --- Parse bools ---
    if not isinstance(enable_padding, bool):
        enable_padding = str(enable_padding).lower() == "true"
    if not isinstance(enable_center_cropping, bool):
        enable_center_cropping = str(enable_center_cropping).lower() == "true"
    if not isinstance(ignore_existing, bool):
        ignore_existing = str(ignore_existing).lower() == "true"
    if not isinstance(paper_config, bool):
        paper_config = str(paper_config).lower() == "true"

    # --- Paper config override ---
    if paper_config:
        target_shape = "512x512x512"
        enable_center_slices_ratio = 0.4
        enable_resampling_spacing = "1.0x1.0x1.0"
        enable_padding = True
        enable_center_cropping = True
        logger.info("Using PAPER config: 512^3, center_slices=0.4, resampling 1mm^3")

    enable_center_slices = enable_center_slices_ratio is not None
    enable_resampling = enable_resampling_spacing is not None

    logger.info(f"Real: {real_dataset_root}")
    logger.info(f"Synth: {synth_dataset_root}")
    logger.info(f"center_slices_ratio: {enable_center_slices_ratio}")
    logger.info(f"padding: {enable_padding}, center_crop: {enable_center_cropping}")
    logger.info(f"resampling: {enable_resampling_spacing}")
    logger.info(f"target_shape: {target_shape}")

    # --- Parse shape/spacings ---
    t_shape = [int(x) for x in target_shape.split("x")]
    target_shape_tuple = tuple(t_shape)

    if enable_resampling:
        rs_spacing_tuple = tuple(float(x) for x in enable_resampling_spacing.split("x"))
    else:
        rs_spacing_tuple = (1.0, 1.0, 1.0)

    center_slices_ratio_final = enable_center_slices_ratio if enable_center_slices else 1.0

    # --- Prepare datasets ---
    with open(real_filelist) as rf:
        real_lines = [line.strip() for line in rf.readlines() if line.strip()]
    real_lines.sort()
    real_lines = real_lines[:num_images]
    real_filenames = [{"image": os.path.join(real_dataset_root, f)} for f in real_lines]

    with open(synth_filelist) as sf:
        synth_lines = [line.strip() for line in sf.readlines() if line.strip()]
    synth_lines.sort()
    synth_lines = synth_lines[:num_images]
    synth_filenames = [{"image": os.path.join(synth_dataset_root, f)} for f in synth_lines]

    logger.info(f"Real: {len(real_filenames)} images, Synth: {len(synth_filenames)} images")

    # --- Build MONAI transforms ---
    # Transform order matches MAISI paper script (compute_fid_2-5d_ct.py lines 561-570):
    #   resample -> pad -> crop -> scale_intensity
    #
    # For large CTs (autoPET: 400x400x588, spacing~2mm), resampling to 1mm^3
    # before cropping would create huge volumes (~800x800x1176) and cause OOM.
    # Solution: pre-crop to target_shape before resampling, then resample,
    # then pad/crop to exact target_shape. This only affects the pre-crop step
    # which removes peripheral slices that would be cropped away anyway.
    transform_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
    ]
    # Pre-crop to target_shape to avoid OOM during resampling of large CTs.
    # This is applied BEFORE resampling, unlike the paper script which does
    # resampling first. For CTs already smaller than target_shape, this is
    # a no-op. For larger CTs, it removes peripheral slices that would be
    # cropped away later anyway, so the final result is equivalent.
    pre_crop_applied = False
    if enable_center_cropping and enable_resampling:
        transform_list.append(
            monai.transforms.CenterSpatialCropd(keys=["image"], roi_size=target_shape_tuple)
        )
        pre_crop_applied = True

    if enable_resampling:
        transform_list.append(
            monai.transforms.Spacingd(keys=["image"], pixdim=rs_spacing_tuple, mode=["bilinear"])
        )
    if enable_padding:
        transform_list.append(
            monai.transforms.SpatialPadd(
                keys=["image"], spatial_size=target_shape_tuple, mode="constant", value=-1000
            )
        )
    if enable_center_cropping:
        transform_list.append(
            monai.transforms.CenterSpatialCropd(keys=["image"], roi_size=target_shape_tuple)
        )
    transform_list.append(
        monai.transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=-1000, b_max=1000, clip=True
        )
    )
    transforms = Compose(transform_list)

    # --- Load feature network ---
    feature_network, net_name, feature_dim = load_feature_network()

    # Feature dir includes network name to avoid collisions with old runs
    real_feat_dir = f"{real_features_dir}_{net_name}"
    synth_feat_dir = f"{synth_features_dir}_{net_name}"

    # --- Run FID ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Running FID with {net_name}")
    logger.info(f"{'='*60}")

    result = compute_fid_for_network(
        feature_network=feature_network,
        model_name=net_name,
        feature_dim=feature_dim,
        real_filenames=real_filenames,
        synth_filenames=synth_filenames,
        transforms=transforms,
        device=device,
        enable_center_slices=enable_center_slices,
        center_slices_ratio=center_slices_ratio_final,
        output_root=output_root,
        real_features_dir=real_feat_dir,
        synth_features_dir=synth_feat_dir,
        real_dataset_root=real_dataset_root,
        synth_dataset_root=synth_dataset_root,
        ignore_existing=ignore_existing,
    )

    # --- Print final report ---
    print()
    print("=" * 60)
    print("FID Realism Evaluation Report")
    print("=" * 60)
    print()
    print("Evaluation: real tumor CT vs synthetic tumor CT")
    print(f"  Real:    {real_dataset_root} ({len(real_filenames)} cases)")
    print(f"  Synth:   {synth_dataset_root} ({len(synth_filenames)} cases)")
    print(f"  Both sets contain tumors; FID measures distribution distance")
    print()
    print(f"Feature network: {result['model_name']} ({result['feature_dim']}d)")
    print(f"  (RadImageNet ResNet50, pretrained on medical images)")
    print()
    print(f"Preprocessing:")
    print(f"  target_shape:  {target_shape}")
    print(f"  resampling:    {enable_resampling_spacing}")
    print(f"  center_slices: {enable_center_slices_ratio}")
    if pre_crop_applied:
        print(f"  pre-crop:      applied before resampling (avoids OOM for large CTs)")
        print(f"                 note: paper script does resample first; pre-crop removes")
        print(f"                 only slices that would be cropped away anyway")
    if paper_config:
        print(f"  paper_config:  512^3, may require >8GB VRAM")
    print()

    print(f"--- FID Results ---")
    print(f"  FID Axial:    {result['fid_xy']:.3f}")
    print(f"  FID Sagittal: {result['fid_yz']:.3f}")
    print(f"  FID Coronal:  {result['fid_zx']:.3f}")
    print(f"  FID Average:  {result['fid_avg']:.3f}")
    print()

    print(f"--- Statistical Diagnostics ---")
    print(f"  Real features:")
    print_diagnosis(result['diag_real'])
    print(f"  Synth features:")
    print_diagnosis(result['diag_synth'])
    print()

    # --- Reliability summary ---
    diag = result['diag_real']
    diag_s = result['diag_synth']
    reliability = diag['reliability']
    reliability_s = diag_s['reliability']
    worst = min(reliability, reliability_s)

    print("--- Reliability Summary ---")
    print(f"  {result['model_name']} ({diag['D']}d, {diag['N']} slice features):")
    print(f"    real  effective rank: {diag['effective_rank']}/{diag['D']} ({diag['rank_ratio']*100:.1f}%) -> {reliability}")
    print(f"    synth effective rank: {diag_s['effective_rank']}/{diag_s['D']} ({diag_s['rank_ratio']*100:.1f}%) -> {reliability_s}")
    if worst == "reliable":
        print(f"  Overall: FID value is statistically reliable")
    elif worst == "acceptable":
        print(f"  Overall: FID value is usable but may have small bias")
        print(f"           Increasing sample count would improve precision")
    else:
        print(f"  Overall: FID value is NOT reliable -- covariance too ill-conditioned")
    print()

    # --- Data leakage note ---
    print("--- Data Leakage Note ---")
    print("  autoPET 2023 is marked as 'testing only' in the MAISI paper")
    print("  (Table S3: autoPET23 testing only, 200 volumes).")
    print("  It was NOT used for MAISI DM or ControlNet training.")
    print("  Our autoPET50 subset may overlap with the paper's autoPET200,")
    print("  but since autoPET is a pure evaluation set, this is not leakage.")
    print()

    # --- Paper reference ---
    print("--- Paper Reference (MAISI, arXiv:2409.11169v3) ---")
    print("  Table 3: FID of MAISI DM vs autoPET 2023 (200 cases, RadImageNet ResNet50)")
    print("    MAISI DM (ddpm-ct):  Axial=3.301, Sagittal=5.838, Coronal=9.109, Avg=6.083")
    print("    DDPM:    Avg=22.608")
    print("    LDM:     Avg=12.379")
    print("    HA-GAN:  Avg=13.757")
    print()
    print("  Important: the paper's FID evaluates tumor-FREE synthetic CT")
    print("  (MAISI DM without ControlNet) against tumor-containing real CT (autoPET).")
    print("  Our FID evaluates tumor-CONTAINING synthetic CT (bridge pipeline)")
    print("  against tumor-containing real CT (autoPET).")
    print("  These are different evaluation targets and NOT directly comparable.")
    print()
    print("  Other differences from paper setup:")
    print(f"    Paper: 200 cases, 512^3, resample before crop")
    print(f"    Ours:  {len(real_filenames)} cases, {target_shape}, {'pre-crop then resample' if pre_crop_applied else 'same order'}")
    print("    -> Our FID should NOT be numerically compared with paper's 6.083")
    print("=" * 60)

    return [result]


if __name__ == "__main__":
    fire.Fire(main)
