#!/usr/bin/env python3
"""
bridge_maisi_mask.py — MAISI + Mask 项目自动桥接

将 Mask 项目生成的精确肿瘤 mask 叠加到 MAISI 的 132 类标签 mask 上，
输出合并后的 mask，可直接作为 MAISI ControlNet 的输入。

管线 (Path C):
  MAISI 生成 CT + 132类mask (不含肿瘤)
    → 本脚本在器官内画精确肿瘤
    → MAISI ControlNet 按合并mask生成最终 CT

用法:
  # 单任务 — 在一个器官上画一个肿瘤
  python bridge_maisi_mask.py \
    --ct output/synthetic_ct.nii.gz \
    --label output/synthetic_label.nii.gz \
    --organ liver --size medium

  # 批量 — JSON 配置文件
  python bridge_maisi_mask.py --config batch_config.json

  # 辅助
  python bridge_maisi_mask.py --example      # 打印示例配置
  python bridge_maisi_mask.py --list-organs  # 列出支持的器官

依赖:
  - Mask 项目 (work/Mask/): create_mask() in Step5/src/mask_generator.py
  - nibabel, numpy, scipy
  - MAISI 不直接依赖 — 仅读取其输出的 NIfTI 文件
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

# ── 兼容 Windows GBK 终端 ──
_sys_stdout = sys.stdout
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )
except (AttributeError, OSError):
    pass  # 非 Windows 或已重定向


# ══════════════════════════════════════════════════════════════
#  路径解析
# ══════════════════════════════════════════════════════════════

def _get_script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _get_work_dir() -> str:
    return os.path.dirname(_get_script_dir())


def _get_mask_project_root() -> str:
    """定位 Mask 项目根目录。

    查找顺序:
      1. 环境变量 MASK_PROJECT_DIR (最高优先级)
      2. {work_dir}/Mask (同仓库 worktree: git worktree add Mask Mask)
      3. {script_dir}/../Mask (平级目录)
      4. {work_dir}/../Mask (父目录平级: git worktree add ../Mask Mask)
    """
    # 1. 环境变量覆盖
    env_dir = os.environ.get("MASK_PROJECT_DIR", "")
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)

    work_dir = _get_work_dir()
    script_dir = _get_script_dir()

    # 2-4. 依次尝试各个候选位置
    candidates = [
        os.path.join(work_dir, "Mask"),           # git worktree add Mask Mask
        os.path.join(script_dir, "..", "Mask"),   # 平级目录
        os.path.join(work_dir, "..", "Mask"),     # git worktree add ../Mask Mask
    ]
    for mask_dir in candidates:
        mask_dir = os.path.abspath(mask_dir)
        if os.path.isdir(mask_dir):
            return mask_dir

    # 找不到 → 给出清晰指引
    raise FileNotFoundError(
        f"Mask 项目未找到。\n"
        f"已检查:\n"
        f"  - {os.path.abspath(candidates[0])}\n"
        f"  - {os.path.abspath(candidates[1])}\n"
        f"  - {os.path.abspath(candidates[2])}\n"
        f"解决方法 (任选一):\n"
        f"  1. 设置环境变量: export MASK_PROJECT_DIR=/path/to/Mask\n"
        f"  2. git worktree add Mask Mask  (在仓库根目录执行)\n"
        f"  3. git worktree add ../Mask Mask  (在仓库根目录执行)"
    )


def _setup_mask_import():
    """添加 Mask 项目路径到 sys.path，使 create_mask 可导入。"""
    mask_root = _get_mask_project_root()
    step5_src = os.path.join(mask_root, "Step5", "src")
    step1_src = os.path.join(mask_root, "Step1", "src")
    for p in [step5_src, step1_src]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_path(path: str) -> str:
    """解析相对路径: 先检查绝对路径，再尝试相对于工作目录和当前目录。"""
    if os.path.isabs(path):
        return path
    # 相对于工作目录
    rel_work = os.path.join(_get_work_dir(), path)
    if os.path.exists(rel_work):
        return os.path.abspath(rel_work)
    # 相对于当前目录
    if os.path.exists(path):
        return os.path.abspath(path)
    # 不存在 — 返回相对于工作目录的路径 (用于后续友好报错)
    return os.path.abspath(rel_work)


# ══════════════════════════════════════════════════════════════
#  常量
# ══════════════════════════════════════════════════════════════

# 器官名 → 132类标签中的器官 ID
ORGAN_LABEL_MAP: Dict[str, int] = {
    "liver":     1,
    "pancreas":  4,
    "kidney":    5,       # 左肾=5, 右肾=14 → 两者都检测
    "colon":     62,
    "esophagus": 11,
    "uterus":    161,
    "lung":      20,
    "bone":      21,
}

# 右肾标签 (MAISI 区分左右肾)
KIDNEY_RIGHT_LABEL = 14

# 肺叶标签 (MAISI 区分5个肺叶，总括标签20在实际输出中不使用)
LUNG_LABELS = {28, 29, 30, 31, 32}  # 左上/左下/右上/右中/右下

# 骨标签 (脊椎 + 肋骨 + 骨盆等，总括标签21在实际输出中不使用)
BONE_LABELS = set(range(33, 58)) | set(range(63, 97))  # 脊椎33-57 + 肋骨/骨盆63-96

# 器官 → 132类标签中的肿瘤 ID
TUMOR_LABEL_MAP: Dict[str, int] = {
    "liver":     26,      # hepatic tumor
    "pancreas":  24,      # pancreatic tumor
    "kidney":    116,     # kidney cyst
    "colon":     27,      # colon cancer
    "lung":      23,      # lung tumor
    "bone":      128,     # bone lesion
    "esophagus": 0,       # MAISI 无独立食管肿瘤标签
    "uterus":    0,       # MAISI 无独立子宫肿瘤标签
}

# Body envelope — MAISI ControlNet 必需
BODY_ENVELOPE_LABEL = 200

# 体素值为 0 表示 background (air)
BACKGROUND_LABEL = 0

# 尺寸档位 → 半径范围 (mm)
SIZE_RADIUS_RANGES: Dict[str, Tuple[float, float]] = {
    "tiny":   (2.0, 5.0),
    "small":  (5.0, 10.0),
    "medium": (10.0, 20.0),
    "large":  (20.0, 35.0),
}

# MAISI 132类标签系统中的有效标签 (来自 label_dict_ctmr.json)
# 用于 validate_user_mask 识别
_MAISI_VALID_LABELS = frozenset({
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
    178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
})

# 默认 shape_config 文件 (与本脚本同目录)
_DEFAULT_SHAPE_CONFIG_PATH = os.path.join(_get_script_dir(), "DEFAULT_SHAPE_CONFIG.json")


# ══════════════════════════════════════════════════════════════
#  默认配置加载
# ══════════════════════════════════════════════════════════════

def _load_default_shape_config() -> dict:
    """加载默认的肿瘤形状配置。"""
    if os.path.exists(_DEFAULT_SHAPE_CONFIG_PATH):
        with open(_DEFAULT_SHAPE_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 移除文档字段
        return {k: v for k, v in cfg.items() if not k.startswith("_")}
    # 内置默认值
    return {
        "method": "ellipsoid",
        "axis_ratio_range": [0.8, 1.2],
        "elastic_deformation": {"enabled": True, "alpha": 4.0, "sigma": 3.0},
        "center_sampling": {"method": "distance_weighted", "power": 3.0},
        "salt_noise": {"enabled": True, "probability": 0.02},
        "gaussian_filter": {"enabled": True, "sigma_mm": 1.0},
        "scaling_clipping": {"enabled": True},
    }


# ══════════════════════════════════════════════════════════════
#  核心: 单次桥接
# ══════════════════════════════════════════════════════════════

def load_maisi_data(
    ct_path: str,
    label_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """加载 MAISI 输出的 CT 和 132 类标签。

    Returns:
        ct_data:    (D, H, W) float32, HU 值
        label_data: (D, H, W) int32, 132-class 整数标签
        affine:     (4, 4) float64, 仿射矩阵
        spacing:    (dz, dy, dx) float, 体素间距 mm/voxel
    """
    ct_img = nib.load(ct_path)
    label_img = nib.load(label_path)

    ct_data = ct_img.get_fdata().astype(np.float32)
    label_data = label_img.get_fdata().astype(np.int32)
    affine = ct_img.affine.copy()
    spacing = tuple(float(s) for s in ct_img.header.get_zooms()[:3])

    if ct_data.shape != label_data.shape:
        raise ValueError(
            f"CT 和 label 形状不一致: CT={ct_data.shape}, label={label_data.shape}"
        )

    return ct_data, label_data, affine, spacing


def extract_organ_mask(label_data: np.ndarray, organ: str) -> np.ndarray:
    """从 132 类标签中提取指定器官的二值 mask。

    kidney 特殊处理: 合并左肾 (5) 和右肾 (14)。
    lung  特殊处理: 合并5个肺叶 (28-32) — 总括标签20在实际输出中不使用。
    bone  特殊处理: 合并脊椎 (33-57) + 肋骨/骨盆 (63-96) — 总括标签21在实际输出中不使用。
    """
    organ_label = ORGAN_LABEL_MAP.get(organ)
    if organ_label is None:
        raise ValueError(
            f"不支持的器官: {organ}。支持: {list(ORGAN_LABEL_MAP.keys())}"
        )

    if organ == "kidney":
        mask = (label_data == 5) | (label_data == KIDNEY_RIGHT_LABEL)
    elif organ == "lung":
        mask = np.isin(label_data, list(LUNG_LABELS))
    elif organ == "bone":
        mask = np.isin(label_data, list(BONE_LABELS))
    else:
        mask = (label_data == organ_label)

    return np.asarray(mask, dtype=bool)


def sample_tumor_center(
    organ_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    radius_mm: float,
    rng: np.random.Generator,
    power: float = 3.0,
) -> np.ndarray:
    """在器官内部采样一个肿瘤中心点，使用距离变换加权偏向深处。

    距离变换 dist^power 加权采样让中心偏向器官深处，减少肿瘤大部分
    在器官边界外的情况。溢出器官边界的部分不裁剪，保留真实浸润特征。

    Args:
        organ_mask: 器官二值 mask
        spacing: 体素间距 (dz, dy, dx) mm/voxel
        radius_mm: 实际采样的肿瘤半径 (mm) (用于日志，不影响采样)
        rng: 随机数生成器
        power: 距离变换加权指数，越大越偏向深处 (默认 3.0)
            power=0 → 均匀随机 (旧行为)
            power=2 → 中等偏向深处
            power=3 → 强偏向深处 (推荐默认)
    """
    indices = np.argwhere(organ_mask)  # (N, 3) — (z, y, x)
    if len(indices) == 0:
        raise ValueError("器官 mask 为空，无法放置肿瘤")

    if power <= 0:
        # power=0: 均匀随机 (旧行为兼容)
        idx = rng.integers(0, len(indices))
        return indices[idx].astype(np.float64)

    # 距离变换: 每个 voxel 到器官边界的最短距离 (mm)
    dist = distance_transform_edt(organ_mask, spacing)
    distances = dist[organ_mask]  # 只取器官内 voxel 的距离值

    # dist^power 加权: 深处 voxel 权重大，边缘 voxel 权重小
    # max(dist, 1.0) 防止距离<1mm的voxel权重趋零
    weights = np.maximum(distances, 1.0) ** power
    weights = weights / weights.sum()

    idx = rng.choice(len(indices), p=weights)
    return indices[idx].astype(np.float64)


def sample_radius(size_category: str, rng: np.random.Generator) -> float:
    """根据尺寸类别随机采样肿瘤半径 (mm)。"""
    r_min, r_max = SIZE_RADIUS_RANGES.get(size_category, (5.0, 10.0))
    return float(rng.uniform(r_min, r_max))


def ensure_body_envelope(
    merged: np.ndarray,
    ct_data: np.ndarray,
    body_threshold_hu: float = -200.0,
) -> np.ndarray:
    """确保 mask 包含 body envelope (label 200)。

    如果已存在，直接返回。
    如果不存在，使用 CT 阈值识别身体区域，将未标记的身体体素设为 200。

    MAISI ControlNet 依赖 label 200 来识别身体区域边界，
    缺少它会导致生成质量严重下降。
    """
    if np.any(merged == BODY_ENVELOPE_LABEL):
        return merged  # 已存在，无需添加

    # 身体区域: CT 值高于阈值 (空气 ≈ -1000 HU, 软组织 > -200 HU)
    body_region = ct_data > body_threshold_hu

    # 将身体区域内未被任何标签覆盖的体素设为 200
    unlabeled = (merged == BACKGROUND_LABEL)
    to_fill = body_region & unlabeled

    merged = merged.copy()
    merged[to_fill] = BODY_ENVELOPE_LABEL

    filled_voxels = int(to_fill.sum())
    if filled_voxels > 0:
        print(f"  ℹ Body envelope 缺失 → 已自动生成 ({filled_voxels:,} voxels, "
              f"CT > {body_threshold_hu} HU)")

    return merged


def bridge_single(
    ct_path: str,
    label_path: str,
    organ: str,
    size_category: str,
    shape_config: Optional[dict] = None,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    seed: Optional[int] = None,
    center_zyx: Optional[Tuple[float, float, float]] = None,
    radius_mm: Optional[float] = None,
    dry_run: bool = False,
) -> dict:
    """执行单次桥接: 在 MAISI mask 上画一个肿瘤，返回合并的 132 类标签。

    Args:
        ct_path:      MAISI 合成的 CT NIfTI 路径
        label_path:   MAISI 合成的 132 类标签 NIfTI 路径
        organ:        目标器官名 (liver, pancreas, kidney, ...)
        size_category: 肿瘤尺寸档 (tiny/small/medium/large)
        shape_config: 肿瘤形状配置 (None=默认)
        output_dir:   输出目录 (None=与 label 同目录)
        output_name:  自定义输出文件名 (None=自动)
        seed:         随机种子 (None=随机)
        center_zyx:   精确肿瘤中心 (z,y,x) 体素坐标 (None=随机)
        radius_mm:    精确肿瘤半径 mm (None=按类别随机)
        dry_run:      仅计算，不保存文件

    Returns:
        dict with keys:
          status, output_path, organ, size_category,
          center_zyx, radius_mm, tumor_voxels, organ_voxels,
          tumor_label_id, organ_label_id, shape, spacing,
          has_body_envelope, time_s, reason (if failed)
    """
    t_start = time.time()

    # ── 导入 Mask 项目 ──
    _setup_mask_import()
    from mask_generator import create_mask

    # ── 准备参数 ──
    rng = np.random.default_rng(seed)
    if shape_config is None:
        shape_config = _load_default_shape_config()

    # ── Step 1: 加载 MAISI 数据 ──
    ct_data, label_data, affine, spacing = load_maisi_data(ct_path, label_path)
    shape = ct_data.shape  # (D, H, W)

    # ── Step 2: 提取器官 mask ──
    organ_mask = extract_organ_mask(label_data, organ)
    organ_voxels = int(organ_mask.sum())

    if organ_voxels < 10:
        return {
            "status": "skip",
            "reason": f"器官 {organ} 体素过少 ({organ_voxels}) — 跳过",
            "organ_voxels": organ_voxels,
            "organ": organ,
            "size_category": size_category,
            "time_s": round(time.time() - t_start, 1),
        }

    # ── Step 3: 确定肿瘤半径和位置 ──
    # 先确定半径 (影响中心采样所需的 margin)
    if radius_mm is not None:
        radius = float(radius_mm)
    else:
        radius = sample_radius(size_category, rng)

    # 获取弹性形变参数
    elastic_cfg = shape_config.get("elastic_deformation", {})
    use_elastic = elastic_cfg.get("enabled", True)

    # 获取中心采样参数
    sampling_cfg = shape_config.get("center_sampling", {})
    sampling_power = float(sampling_cfg.get("power", 3.0))

    if center_zyx is not None:
        center = np.array(center_zyx, dtype=np.float64)
        c_int = tuple(center.astype(int))
        if (c_int[0] < 0 or c_int[0] >= shape[0]
            or c_int[1] < 0 or c_int[1] >= shape[1]
            or c_int[2] < 0 or c_int[2] >= shape[2]):
            return {
                "status": "fail",
                "reason": f"指定中心 {c_int} 超出 CT 范围 {shape}",
                "organ": organ,
                "size_category": size_category,
                "organ_voxels": organ_voxels,
                "time_s": round(time.time() - t_start, 1),
            }
        if not organ_mask[c_int]:
            print(f"  ⚠ 指定中心 {c_int} 不在器官 {organ} 内部，将继续生成")
    else:
        # 距离变换加权采样让中心偏向器官深处
        center = sample_tumor_center(
            organ_mask, spacing, radius, rng, power=sampling_power,
        )

    # ── Step 4-4.5: 生成肿瘤 (带重试) ──
    # 不裁剪到器官内部 — 保留弹性形变产生的真实浸润特征（部分溢出器官边界）。
    # 重试策略: 体积太小 (<200 voxels) 时换中心重新采样。
    #   前 3 次: 用原始 shape_config (含弹性形变), 换不同中心
    #   后 2 次: 关闭弹性形变, 确保小肿瘤有足够体素
    MIN_TUMOR_VOXELS = 200
    MAX_RETRIES = 5
    tumor_mask = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            center = sample_tumor_center(
                organ_mask, spacing, radius, rng, power=sampling_power,
            )

        # 超过 3 次失败 → 关闭弹性形变
        cfg = shape_config
        if attempt >= 3 and use_elastic:
            cfg = {**shape_config,
                   "elastic_deformation": {"enabled": False}}
            if attempt == 3:
                print(f"  ℹ 重试 {attempt+1}/{MAX_RETRIES}: 临时关闭弹性形变")

        tumor_mask = create_mask(
            center_zyx=tuple(float(c) for c in center),
            radius_mm=radius,
            shape=shape,
            spacing=spacing,
            shape_config=cfg,
            rng=rng,
        )

        # 不裁剪: 保留真实浸润 (溢出器官边界是医学合理的)
        if tumor_mask.sum() >= MIN_TUMOR_VOXELS:
            break  # 成功

    tumor_voxels = int(tumor_mask.sum())

    if tumor_voxels < MIN_TUMOR_VOXELS:
        return {
            "status": "fail",
            "reason": (f"肿瘤体素过少 ({tumor_voxels}, "
                       f"期望 >= {MIN_TUMOR_VOXELS}, "
                       f"已重试 {MAX_RETRIES} 次) — "
                       f"建议: 换更大的 size_category"),
            "organ": organ,
            "size_category": size_category,
            "center_zyx": [float(c) for c in center],
            "radius_mm": radius,
            "organ_voxels": organ_voxels,
            "time_s": round(time.time() - t_start, 1),
        }

    # ── Step 5: 验证肿瘤质量 ──
    # overlap 由中心深度和 alpha 自然决定，不再强制 100%。
    # 距离变换加权采样 + alpha=4 使 overlap 通常在医学合理范围。
    tumor_in_organ = (tumor_mask.astype(bool) & organ_mask).sum()
    overlap_ratio = tumor_in_organ / max(tumor_voxels, 1)

    # 安全约束: 肿瘤显著超出器官时裁剪到器官内。
    # 触发条件 (满足任一): ① 肿瘤 > 1.2x 器官体积  ② <50% 肿瘤在器官内
    # 正常情况保留溢出作为浸润特征; 极端情况强制约束。
    tumor_to_organ_ratio = tumor_voxels / max(organ_voxels, 1)
    need_constraint = (tumor_to_organ_ratio > 1.2) or (overlap_ratio < 0.5)

    if need_constraint:
        constraint_reason = ('ratio=%.1fx' % tumor_to_organ_ratio) if tumor_to_organ_ratio > 1.2 else ('overlap=%.0f%%' % (overlap_ratio * 100))
        tumor_before = tumor_voxels
        tumor_mask = (tumor_mask.astype(bool) & organ_mask).astype(np.uint8)
        tumor_voxels = int(tumor_mask.sum())
        overlap_ratio = 1.0
        print(f"  ⚠ 安全约束触发 ({constraint_reason}) → "
              f"裁剪 {tumor_before - tumor_voxels:,} voxels 器官外肿瘤")

    # 注意: 安全约束触发后 overlap_ratio=1.0，此检查跳过。
    # 仅在约束未触发 (overlap 40-49%) 时作为早期预警。
    if 0.4 <= overlap_ratio < 0.5:
        print(f"  ⚠ 仅 {overlap_ratio:.0%} 的肿瘤在器官内 (未触发约束) — "
              f"边界浸润较多，可考虑增大 center_sampling.power 或减小 radius_mm")

    # ── Step 6: 将肿瘤合并到 132 类标签 ──
    tumor_label_id = TUMOR_LABEL_MAP.get(organ, 0)

    if tumor_label_id == 0:
        # 零样本器官 (esophagus/uterus): MAISI 无独立肿瘤标签
        # 使用器官标签本身 — ControlNet 会尝试在该区域生成肿瘤纹理
        tumor_label_id = ORGAN_LABEL_MAP[organ]
        print(f"  ⚠ {organ} 无独立肿瘤标签 (zero-shot)，使用器官标签 {tumor_label_id}")

    merged = label_data.copy()
    merged[tumor_mask > 0] = tumor_label_id

    # ── Step 7: 确保 body envelope 存在 ──
    has_body = bool(np.any(merged == BODY_ENVELOPE_LABEL))
    merged = ensure_body_envelope(merged, ct_data)
    has_body_after = bool(np.any(merged == BODY_ENVELOPE_LABEL))

    # ── Step 8: 验证标签合法性 ──
    unique_after = set(np.unique(merged).astype(int).tolist())
    unknown = sorted(v for v in unique_after if v not in _MAISI_VALID_LABELS)
    if unknown:
        print(f"  ⚠ 输出包含 {len(unknown)} 个非 MAISI 标签: {unknown[:10]}")

    # ── Step 9: 保存 ──
    if dry_run:
        output_path = None
    else:
        out_dir = _resolve_path(output_dir) if output_dir else os.path.dirname(os.path.abspath(label_path))
        os.makedirs(out_dir, exist_ok=True)

        if output_name:
            base = output_name.replace(".nii.gz", "").replace(".nii", "")
        else:
            # 自动命名: {label_basename}_{organ}_{size}_tumor[_v2].nii.gz
            label_base = os.path.splitext(os.path.splitext(
                os.path.basename(label_path))[0])[0]
            base = f"{label_base}_{organ}_{size_category}_tumor"

        output_path = os.path.join(out_dir, f"{base}.nii.gz")

        # 自动版本化: 文件已存在时加 _v2, _v3...
        if not output_name:
            v = 2
            while os.path.exists(output_path):
                output_path = os.path.join(out_dir, f"{base}_v{v}.nii.gz")
                v += 1

        nib.save(
            nib.Nifti1Image(merged.astype(np.uint8), affine),
            output_path,
        )

    elapsed = time.time() - t_start

    return {
        "status": "ok",
        "output_path": output_path,
        "organ": organ,
        "size_category": size_category,
        "center_zyx": [float(c) for c in center],
        "radius_mm": radius,
        "tumor_voxels": tumor_voxels,
        "overlap_ratio": float(overlap_ratio),
        "organ_voxels": organ_voxels,
        "tumor_label_id": tumor_label_id,
        "organ_label_id": ORGAN_LABEL_MAP[organ],
        "shape": list(shape),
        "spacing": list(spacing),
        "has_body_envelope": has_body_after,
        "had_body_envelope": has_body,
        "time_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════
#  批量模式: JSON 配置 → 多个桥接任务
# ══════════════════════════════════════════════════════════════

def run_config(config: dict) -> List[dict]:
    """执行 JSON 配置文件中的批量桥接任务。

    配置格式见 example_bridge_config.json。
    """
    tasks_raw = config.get("tasks", [])
    if not tasks_raw:
        print("错误: 配置中没有 'tasks' 数组。")
        return []

    maisi_cfg = config.get("maisi", {})
    ct_path = _resolve_path(maisi_cfg.get("ct_path", ""))
    label_path = _resolve_path(maisi_cfg.get("label_path", ""))

    if not ct_path or not label_path:
        print("错误: 请在 'maisi' 节中指定 ct_path 和 label_path。")
        return []

    for path, name in [(ct_path, "CT"), (label_path, "Label")]:
        if not os.path.exists(path):
            print(f"错误: {name} 文件不存在: {path}")
            return []

    shape_config = config.get("shape_config")  # None → 默认
    output_dir = config.get("output_dir")
    if output_dir:
        output_dir = _resolve_path(output_dir)
    global_seed = config.get("seed")

    # 展开 repeat
    expanded = []
    for task in tasks_raw:
        if not isinstance(task, dict):
            continue
        # 跳过纯注释对象 (只含 _comment 键)
        if set(task.keys()) <= {"_comment"}:
            continue
        n = max(1, task.get("repeat", 1))
        for j in range(n):
            t = {k: v for k, v in task.items() if not k.startswith("_")}
            t.pop("repeat", None)
            expanded.append(t)

    print(f"{'='*60}")
    print(f"MAISI + Mask 桥接 — {len(expanded)} 个任务")
    print(f"CT:   {ct_path}")
    print(f"Mask: {label_path}")
    print(f"{'='*60}\n")

    results = []
    ok = skip = fail = 0
    t_total = time.time()

    for i, task_dict in enumerate(expanded):
        organ = task_dict.get("organ", "liver")
        size_cat = task_dict.get("size_category", "medium")
        seed = task_dict.get("seed", global_seed)

        print(f"[{i+1}/{len(expanded)}] {organ:12s} / {size_cat:6s}  ",
              end="", flush=True)

        result = bridge_single(
            ct_path=ct_path,
            label_path=label_path,
            organ=organ,
            size_category=size_cat,
            shape_config=shape_config,
            output_dir=output_dir,
            output_name=task_dict.get("output_name"),
            seed=seed,
            center_zyx=tuple(task_dict["center_zyx"])
                       if task_dict.get("center_zyx") else None,
            radius_mm=task_dict.get("radius_mm"),
        )
        results.append(result)

        if result["status"] == "ok":
            ok += 1
            fname = os.path.basename(result["output_path"]) if result["output_path"] else "(dry)"
            print(f"✓  r={result['radius_mm']:.1f}mm  "
                  f"vox={result['tumor_voxels']:,}  "
                  f"overlap={result.get('overlap_ratio', 0):.0%}  "
                  f"→ {fname}")
        elif result["status"] == "skip":
            skip += 1
            print(f"⊘ {result.get('reason', '')}")
        else:
            fail += 1
            print(f"✗ {result.get('reason', '')}")

    total_t = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"完成: {total_t:.1f}s | 成功={ok}  跳过={skip}  失败={fail}")

    return results


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

EXAMPLE_CONFIG_TEXT = """{
  "_description": "MAISI + Mask 桥接配置示例",
  "maisi": {
    "ct_path": "output/synthetic_ct.nii.gz",
    "label_path": "output/synthetic_label.nii.gz"
  },
  "tasks": [
    {"organ": "liver",    "size_category": "medium", "repeat": 2},
    {"organ": "pancreas", "size_category": "small",  "repeat": 1},
    {"organ": "kidney",   "size_category": "large",  "repeat": 2},
    {"organ": "colon",    "size_category": "medium", "repeat": 1}
  ],
  "shape_config": null,
  "output_dir": null,
  "seed": null
}"""


def main():
    parser = argparse.ArgumentParser(
        description="MAISI + Mask 桥接 — 将精确肿瘤 mask 合并到 MAISI 132 类标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python bridge_maisi_mask.py --ct ct.nii.gz --label label.nii.gz "
            "--organ liver --size medium\n"
            "  python bridge_maisi_mask.py --config batch_config.json\n"
            "  python bridge_maisi_mask.py --example\n"
            "  python bridge_maisi_mask.py --list-organs\n"
        ),
    )

    # 模式选择
    parser.add_argument("--config", "-c", metavar="JSON",
                        help="JSON 批量配置文件路径")
    parser.add_argument("--example", action="store_true",
                        help="打印示例 JSON 配置并退出")
    parser.add_argument("--list-organs", action="store_true",
                        help="列出支持的器官映射并退出")

    # 单任务模式
    parser.add_argument("--ct", metavar="PATH",
                        help="MAISI 合成的 CT 文件 (.nii.gz)")
    parser.add_argument("--label", metavar="PATH",
                        help="MAISI 合成的 132 类标签文件 (.nii.gz)")
    parser.add_argument("--organ", choices=list(ORGAN_LABEL_MAP.keys()),
                        help="目标器官")
    parser.add_argument("--size", choices=list(SIZE_RADIUS_RANGES.keys()),
                        default="medium",
                        help="肿瘤尺寸类别 (默认: medium)")
    parser.add_argument("--output-dir", "-o", metavar="DIR",
                        help="输出目录 (默认: 与 label 同目录)")
    parser.add_argument("--output-name", metavar="NAME",
                        help="自定义输出文件名")
    parser.add_argument("--seed", type=int,
                        help="随机种子 (默认: 随机)")
    parser.add_argument("--radius-mm", type=float, metavar="MM",
                        help="精确指定肿瘤半径 mm (默认: 按类别随机)")
    parser.add_argument("--center", nargs=3, type=float,
                        metavar=("Z", "Y", "X"),
                        help="精确指定肿瘤中心体素坐标")
    parser.add_argument("--shape-config", metavar="JSON_FILE",
                        help="自定义 shape_config JSON 文件路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算不保存文件")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="输出详细 JSON 结果")

    args = parser.parse_args()

    # ── 特殊模式 ──
    if args.example:
        print(EXAMPLE_CONFIG_TEXT)
        return

    if args.list_organs:
        _print_organ_table()
        return

    # ── 加载自定义 shape_config ──
    shape_config = None
    if args.shape_config:
        with open(args.shape_config, "r", encoding="utf-8") as f:
            shape_config = json.load(f)
        shape_config = {k: v for k, v in shape_config.items()
                        if not k.startswith("_")}

    # ── 批量模式: 显式 --config 或自动检测 prompts.json ──
    config_path = args.config
    if not config_path:
        # 自动检测当前目录或脚本目录的 prompts.json
        candidates = [
            os.path.join(os.getcwd(), "prompts.json"),
            os.path.join(_get_script_dir(), "prompts.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not config.get("tasks"):
            print(f"提示: {config_path} 的 tasks 数组为空，请添加任务后重试。")
            return
        maisi_cfg = config.get("maisi", {})
        if not maisi_cfg.get("ct_path") or not maisi_cfg.get("label_path"):
            print(f"错误: {config_path} 中 maisi.ct_path 或 maisi.label_path 为空。")
            return
        if shape_config is not None:
            config["shape_config"] = shape_config
        run_config(config)
        return

    # ── 单任务模式 ──
    if not args.ct or not args.label or not args.organ:
        parser.error(
            "单任务模式需要 --ct, --label, --organ。"
            "或使用 --config JSON 批量模式。"
            "或运行 `python bridge_maisi_mask.py --example` 查看示例。"
        )

    result = bridge_single(
        ct_path=_resolve_path(args.ct),
        label_path=_resolve_path(args.label),
        organ=args.organ,
        size_category=args.size,
        shape_config=shape_config,
        output_dir=_resolve_path(args.output_dir) if args.output_dir else None,
        output_name=args.output_name,
        seed=args.seed,
        center_zyx=tuple(args.center) if args.center else None,
        radius_mm=args.radius_mm,
        dry_run=args.dry_run,
    )

    if args.verbose:
        # 移除不可序列化的字段
        printable = {k: v for k, v in result.items()
                     if k not in ("time_s",)}
        printable["time_s"] = result["time_s"]
        print(json.dumps(printable, indent=2, ensure_ascii=False))
    elif result["status"] == "ok":
        print(f"\n✓ 桥接完成!")
        print(f"  器官:       {result['organ']} "
              f"(器官标签={result['organ_label_id']}, 肿瘤标签={result['tumor_label_id']})")
        print(f"  肿瘤尺寸:   {result['size_category']} "
              f"(半径 {result['radius_mm']:.1f}mm, {result['tumor_voxels']:,} voxels)")
        print(f"  中心 (zyx): [{result['center_zyx'][0]:.1f}, "
              f"{result['center_zyx'][1]:.1f}, {result['center_zyx'][2]:.1f}]")
        print(f"  Body envelope: "
              f"{'✓ 原有' if result.get('had_body_envelope') else '✓ 已自动生成'}")
        print(f"  耗时:       {result['time_s']}s")
        if result["output_path"]:
            print(f"  输出:       {result['output_path']}")
            print(f"\n  → 下一步: 用此 mask 运行 MAISI 步骤2:")
            print(f"    python scripts/infer_image_from_mask.py "
                  f"--mask {result['output_path']}")
    else:
        print(f"\n✗ {result['status']}: {result.get('reason', '')}")


def _print_organ_table():
    """打印支持的器官映射表。"""
    print("\n支持的器官映射 (132-class label system):\n")
    print(f"  {'器官':<12} {'器官标签':<10} {'肿瘤标签':<12} 备注")
    print(f"  {'-'*50}")
    for organ, org_label in ORGAN_LABEL_MAP.items():
        tumor_label = TUMOR_LABEL_MAP.get(organ, 0)
        if tumor_label == 0:
            tumor_str = f"{'—':>6}     "
            note = "(zero-shot, 无独立肿瘤标签)"
        else:
            tumor_str = f"{tumor_label:<12}"
            note = ""
        if organ == "kidney":
            note = f"左肾=5 右肾=14 囊肿=116"
        print(f"  {organ:<12} {org_label:<10} {tumor_str} {note}")
    print()


if __name__ == "__main__":
    main()
