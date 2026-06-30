#!/usr/bin/env python3
"""
bridge_maisi_mask.py — MAISI + Mask 项目自动桥接

将 Mask 项目生成的精确肿瘤 mask 叠加到 MAISI 的 132 类标签 mask 上，
输出合并后的 mask，可直接作为 MAISI ControlNet 的输入。

管线 (Path C):
  MAISI 生成 CT + 132类mask (自带肿瘤已清除)
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
    # 原有7个（MAISI有对应肿瘤标签）
    "liver":     1,
    "pancreas":  4,
    "kidney":    5,       # 左肾=5, 右肾=14 → 两者都检测
    "colon":     62,
    "esophagus": 11,
    "lung":      20,
    "bone":      21,
    # 新增12个（MAISI无对应肿瘤标签，第四步新建）
    "stomach":      12,
    "bladder":      15,
    "prostate":     118,
    "thyroid":      126,
    "brain":        22,
    "adrenal":      8,       # 右肾上腺=8, 左肾上腺=9
    "small_bowel":  19,
    "duodenum":     13,
    "gallbladder":  10,
    "spleen":       3,
    "heart":        115,
    "uterus":       161,
}

# 右肾标签 (MAISI 区分左右肾)
KIDNEY_RIGHT_LABEL = 14

# 肺叶标签 (MAISI 区分5个肺叶，总括标签20在实际输出中不使用)
LUNG_LABELS = {28, 29, 30, 31, 32}  # 左上/左下/右上/右中/右下

# 骨标签 (脊椎 + 肋骨 + 骨盆等，总括标签21在实际输出中不使用)
BONE_LABELS = set(range(33, 58)) | set(range(63, 97))  # 脊椎33-57 + 肋骨/骨盆63-96

# 器官 → 132类标签中的肿瘤 ID
TUMOR_LABEL_MAP: Dict[str, int] = {
    # 原有7个（MAISI 132类中已有）
    "liver":     26,      # hepatic tumor
    "pancreas":  24,      # pancreatic tumor
    "kidney":    116,     # kidney cyst
    "colon":     27,      # colon cancer
    "lung":      23,      # lung tumor
    "bone":      128,     # bone lesion
    "esophagus": 133,     # 食管癌 (第四步新建)
    # 新增11个（第四步新建肿瘤标签 134-144）
    "stomach":      134,  # 胃癌
    "bladder":      135,  # 膀胱癌
    "prostate":     136,  # 前列腺癌
    "thyroid":      137,  # 甲状腺癌
    "brain":        138,  # 脑肿瘤
    "adrenal":      139,  # 肾上腺肿瘤
    "small_bowel":  140,  # 小肠癌
    "duodenum":     141,  # 十二指肠癌
    "gallbladder":  142,  # 胆囊癌
    "spleen":       143,  # 脾脏肿瘤
    "heart":        144,  # 心脏肿瘤
    "uterus":       0,    # MAISI 无独立子宫肿瘤标签
}

# MAISI 自带肿瘤标签 → 还原为对应器官标签 (A方案)
# 清除不可控的数据库自带肿瘤，保证桥接画的肿瘤是 mask 中唯一来源。
# 直接还原的标签 (有明确器官映射):
#   原有5个: 26→1, 116→5, 117→14, 24→4, 27→62
#   第四步新增12个: 133→11, 134→12, 135→15, 136→118, 137→126,
#                   138→22, 139→8, 140→19, 141→13, 142→10, 143→3, 144→115
# 特殊处理的2个标签 (碎片结构，需逐voxel还原):
#   23→最近肺叶 (28-32), 128→最近骨碎片 (33-57/63-96)
TUMOR_TO_ORGAN_RESTORE_MAP: Dict[int, int] = {
    # 原有5个 (MAISI 132类中已有)
    26:  1,    # 肝肿瘤 → 肝脏
    116: 5,    # 左肾囊肿 → 左肾
    117: 14,   # 右肾囊肿 → 右肾
    24:  4,    # 胰腺肿瘤 → 胰腺
    27:  62,   # 结肠癌 → 结肠
    # 第四步新增12个
    133: 11,   # 食管癌 → 食管
    134: 12,   # 胃癌 → 胃
    135: 15,   # 膀胱癌 → 膀胱
    136: 118,  # 前列腺癌 → 前列腺
    137: 126,  # 甲状腺癌 → 甲状腺
    138: 22,   # 脑肿瘤 → 脑
    139: 8,    # 肾上腺肿瘤 → 右肾上腺
    140: 19,   # 小肠癌 → 小肠
    141: 13,   # 十二指肠癌 → 十二指肠
    142: 10,   # 胆囊癌 → 胆囊
    143: 3,    # 脾脏肿瘤 → 脾脏
    144: 115,  # 心脏肿瘤 → 心脏
}

# 肿瘤标签中需要逐voxel还原的特殊标签 (肺/骨是碎片结构)
TUMOR_LABELS_VOXELWISE = {23, 128}  # 肺肿瘤, 骨病变

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

# ── 器官级参数覆盖 — 预留但经验证无效 ──
# TODO: 以下参数组合已通过 20-seed 系统验证，结论是"不应启用"：
#
# 1. bone alpha=8:   medium avg overlap 从 71.2%→78.6%, large 从 85.7%→97.9%
#    原因: alpha=8 增大溢出 → 安全约束触发更频繁 (medium 2→9次, large 13→19次)
#    → 裁剪后 overlap=100% → 平均值反而更高
#
# 2. bone center_power=-0.5 (边缘采样): 约束触发从 0/5→3/5
#    原因: 边缘体素跨越多块骨碎片或落在碎片外侧 → 大量溢出 → 触发约束
#
# 3. colon alpha=6:   约束触发从 0/5→4/5, overlap 全部变成 100%
#    原因: 和 bone 一样的模式——增大 alpha → 增大溢出 → 更多约束 → 更差结果
#
# 4. colon center_power=1.5: 和 alpha=6 组合效果同样差
#
# 5. kidney alpha=2:   large avg overlap 从 82%→69% (降到合理范围70-90%以下)
#
# 6. pancreas 自动降级 large→medium: medium 本身也有 50% 约束触发率，
#    安全约束已兜底，降级不是根本解决方案
#
# 根本原因: bone/colon/pancreas 的高 overlap 是解剖结构决定的:
#   - bone: 分散碎片结构 (脊椎+肋骨+骨盆)
#   - colon: 管状弯曲结构 (深处管壁足够厚包裹整个椭球)
#   - pancreas: 体积太小 (106cm³, max_depth=13.6mm)
# 安全约束 + alpha=4 + dist³ 已是最优折中，任何单参数调整都让某个指标变差。
#
ORGAN_OVERRIDES: Dict[str, dict] = {
    "bone": {
        "alpha": 8.0,
        "center_power": -0.5,
        "min_organ_depth_mm": 20.0,
    },
    "colon": {
        "alpha": 6.0,
        "center_power": 1.5,
    },
    "kidney": {
        "alpha": 2.0,
        "min_organ_depth_mm": 18.0,
    },
    "pancreas": {
        "max_size": "medium",
        "min_organ_depth_mm": 15.0,
    },
}

# 尺寸档位降级顺序 (用于自动降级) — 未启用，见上方 ORGAN_OVERRIDES 注释
_SIZE_DOWNGRADE_ORDER = ["large", "medium", "small", "tiny"]

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


def remove_existing_tumors(
    label_data: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """清除 MAISI 自带的所有肿瘤标签，还原为对应器官标签。

    A方案: 保证桥接后的 mask 中只有桥接代码画的肿瘤，
    消除不可控的 MAISI 数据库自带肿瘤。

    对于有明确器官映射的肿瘤标签 (26→1, 116→5, 117→14, 24→4, 27→62):
      直接批量还原

    对于肺肿瘤 (23) 和骨病变 (128):
      逐 voxel 找最近邻域的肺叶/骨标签还原。
      肺和骨是碎片结构 (肺=5叶, 骨=数十块脊椎/肋骨/骨盆)，
      不能把整个肿瘤区域还原到同一个标签。

    Args:
        label_data: 原始 MAISI 132类标签 (D, H, W) int32

    Returns:
        (cleaned_label_data, removal_info)
        cleaned_label_data: 清除后的标签数组 (不修改原始)
        removal_info: dict with keys:
          removed_labels: {label_id: voxel_count} — 每个标签清除了多少voxels
          total_removed_voxels: int — 总清除voxels
          restored_targets: {label_id: target_label_or_"逐voxel还原"} — 还原目标
    """
    cleaned = label_data.copy()

    removed_labels: Dict[int, int] = {}
    restored_targets: Dict[int, str] = {}
    total_removed = 0

    # ── 批量还原: 有明确器官映射的肿瘤标签 ──
    for tumor_label, organ_label in TUMOR_TO_ORGAN_RESTORE_MAP.items():
        mask = (cleaned == tumor_label)
        count = int(mask.sum())
        if count > 0:
            cleaned[mask] = organ_label
            removed_labels[tumor_label] = count
            restored_targets[tumor_label] = str(organ_label)
            total_removed += count

    # ── 逐voxel还原: 肺肿瘤 (23) ──
    lung_tumor_mask = (cleaned == 23)
    lung_count = int(lung_tumor_mask.sum())
    if lung_count > 0:
        coords = np.argwhere(lung_tumor_mask)
        for c in coords:
            # 在 2-voxel 邻域找最常见的肺叶标签
            z_lo = max(0, c[0] - 2)
            z_hi = min(cleaned.shape[0], c[0] + 3)
            y_lo = max(0, c[1] - 2)
            y_hi = min(cleaned.shape[1], c[1] + 3)
            x_lo = max(0, c[2] - 2)
            x_hi = min(cleaned.shape[2], c[2] + 3)

            neighborhood = cleaned[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
            lung_neighbors = [int(v) for v in neighborhood.flat if v in LUNG_LABELS]

            if lung_neighbors:
                # 取最常见的肺叶标签
                best_label = max(set(lung_neighbors), key=lung_neighbors.count)
                cleaned[c[0], c[1], c[2]] = best_label
            else:
                # 无邻域肺叶 → 用距离变换找最近肺叶
                lung_region = np.isin(cleaned, list(LUNG_LABELS))
                if lung_region.any():
                    dist = distance_transform_edt(~lung_region)
                    # 找肿瘤voxel位置上距离最小的肺叶voxel
                    dist_at_tumor = dist[c[0], c[1], c[2]]
                    # 在所有肺叶voxel中找离该voxel最近的那个
                    lung_coords = np.argwhere(lung_region)
                    dists_to_lung = np.sqrt(
                        ((lung_coords - c) ** 2).sum(axis=1)
                    )
                    closest_idx = dists_to_lung.argmin()
                    closest_label = cleaned[
                        lung_coords[closest_idx][0],
                        lung_coords[closest_idx][1],
                        lung_coords[closest_idx][2],
                    ]
                    cleaned[c[0], c[1], c[2]] = closest_label
                else:
                    # 极端情况: mask中无肺叶 → fallback到右下叶(32)
                    cleaned[c[0], c[1], c[2]] = 32

        removed_labels[23] = lung_count
        restored_targets[23] = "逐voxel→最近肺叶"
        total_removed += lung_count

    # ── 逐voxel还原: 骨病变 (128) ──
    bone_tumor_mask = (cleaned == 128)
    bone_count = int(bone_tumor_mask.sum())
    if bone_count > 0:
        coords = np.argwhere(bone_tumor_mask)
        for c in coords:
            z_lo = max(0, c[0] - 2)
            z_hi = min(cleaned.shape[0], c[0] + 3)
            y_lo = max(0, c[1] - 2)
            y_hi = min(cleaned.shape[1], c[1] + 3)
            x_lo = max(0, c[2] - 2)
            x_hi = min(cleaned.shape[2], c[2] + 3)

            neighborhood = cleaned[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
            bone_neighbors = [int(v) for v in neighborhood.flat if v in BONE_LABELS]

            if bone_neighbors:
                best_label = max(set(bone_neighbors), key=bone_neighbors.count)
                cleaned[c[0], c[1], c[2]] = best_label
            else:
                bone_region = np.isin(cleaned, list(BONE_LABELS))
                if bone_region.any():
                    bone_coords = np.argwhere(bone_region)
                    dists_to_bone = np.sqrt(
                        ((bone_coords - c) ** 2).sum(axis=1)
                    )
                    closest_idx = dists_to_bone.argmin()
                    closest_label = cleaned[
                        bone_coords[closest_idx][0],
                        bone_coords[closest_idx][1],
                        bone_coords[closest_idx][2],
                    ]
                    cleaned[c[0], c[1], c[2]] = closest_label
                else:
                    # 极端情况: mask中无骨 → fallback到腰椎L2(40)
                    cleaned[c[0], c[1], c[2]] = 40

        removed_labels[128] = bone_count
        restored_targets[128] = "逐voxel→最近骨碎片"
        total_removed += bone_count

    removal_info = {
        "removed_labels": removed_labels,
        "total_removed_voxels": total_removed,
        "restored_targets": restored_targets,
    }

    return cleaned, removal_info


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
          has_body_envelope, removal_info, time_s, reason (if failed)
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

    # ── Step 1.5: 清除 MAISI 自带肿瘤标签 (A方案) ──
    # MAISI 数据库自带的肿瘤标签不可控，清除后保证桥接画的肿瘤
    # 是 mask 中唯一的肿瘤来源，CT 中也只有桥接画的肿瘤区域有肿瘤纹理。
    label_data, removal_info = remove_existing_tumors(label_data)
    if removal_info["total_removed_voxels"] > 0:
        print(f"  ℹ 清除自带肿瘤: {removal_info['total_removed_voxels']:,} voxels")
        for tl, cnt in removal_info["removed_labels"].items():
            target = removal_info["restored_targets"].get(tl, "?")
            print(f"    label {tl} ({cnt:,} voxels) → {target}")

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
            "removal_info": removal_info,
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
                "removal_info": removal_info,
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
            "removal_info": removal_info,
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

    # ── Step 9: 保存 (分目录存放 4 组输出) ──
    # 每次运行输出 4 组文件:
    #   01_original_ct/   — 清除自带肿瘤后的 CT + label (中间输出①)
    #   02_tumor_mask/     — 单独的肿瘤二值 mask (中间输出②)
    #   03_tumor_region_ct/— 合并后 label 中肿瘤区域的局部裁切 (中间输出③)
    #   04_final_merged/   — 合并后的完整 132类 label (最终输出④)
    if dry_run:
        task_dir = None
        output_path = None
    else:
        # ── 任务根目录 ──
        base_out_dir = output_dir or os.path.dirname(os.path.abspath(label_path))

        if output_name:
            base = output_name.replace(".nii.gz", "").replace(".nii", "")
        else:
            label_base = os.path.splitext(os.path.splitext(
                os.path.basename(label_path))[0])[0]
            base = f"{label_base}_{organ}_{size_category}_tumor"

        # 自动版本化: 目录已存在时加 _v2, _v3...
        task_dir = os.path.join(base_out_dir, base)
        if not output_name:
            v = 2
            while os.path.exists(task_dir):
                task_dir = os.path.join(base_out_dir, f"{base}_v{v}")
                v += 1

        # ── 创建 4 个子目录 ──
        dir_01 = os.path.join(task_dir, "01_original_ct")
        dir_02 = os.path.join(task_dir, "02_tumor_mask")
        dir_03 = os.path.join(task_dir, "03_tumor_region_ct")
        dir_04 = os.path.join(task_dir, "04_final_merged")
        for d in [dir_01, dir_02, dir_03, dir_04]:
            os.makedirs(d, exist_ok=True)

        # ── 中间输出①: 清除自带肿瘤后的 CT + label ──
        nib.save(
            nib.Nifti1Image(ct_data, affine),
            os.path.join(dir_01, "ct.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(label_data.astype(np.uint8), affine),
            os.path.join(dir_01, "label_no_tumor.nii.gz"),
        )
        # 保存清除信息 JSON
        import json as _json
        with open(os.path.join(dir_01, "removal_info.json"), "w", encoding="utf-8") as f:
            _json.dump(removal_info, f, indent=2, ensure_ascii=False)

        # ── 中间输出②: 单独的肿瘤二值 mask ──
        nib.save(
            nib.Nifti1Image(tumor_mask.astype(np.uint8), affine),
            os.path.join(dir_02, "tumor_mask.nii.gz"),
        )

        # ── 中间输出③: 合并后 label 中肿瘤区域的局部裁切 ──
        # 从合并后的 label (含新肿瘤标签) 中按 tumor_mask 的 bbox 裁切，
        # 展示新肿瘤标签嵌入在周围器官标签中的空间关系。
        tumor_coords = np.argwhere(tumor_mask > 0)
        if len(tumor_coords) > 0:
            # 计算 bounding box (加 10 voxel margin，多看周围器官)
            margin = 10
            bb_min = tumor_coords.min(axis=0) - margin
            bb_max = tumor_coords.max(axis=0) + margin + 1  # +1 for slice end
            # clip to valid range
            bb_min = np.maximum(bb_min, 0)
            bb_max = np.minimum(bb_max, np.array(shape))

            # 裁切合并后的 label (含肿瘤标签26等)
            region_merged = merged[
                bb_min[0]:bb_max[0],
                bb_min[1]:bb_max[1],
                bb_min[2]:bb_max[2],
            ]
            # 裁切原始 CT (对照纹理)
            region_ct = ct_data[
                bb_min[0]:bb_max[0],
                bb_min[1]:bb_max[1],
                bb_min[2]:bb_max[2],
            ]

            # 调整 affine: 将 origin 移到 bounding box 角点
            region_affine = affine.copy()
            shift_vox = bb_min.astype(float)
            for col in range(3):
                region_affine[:3, 3] += affine[:3, col] * shift_vox[col]

            nib.save(
                nib.Nifti1Image(region_merged.astype(np.uint8), region_affine),
                os.path.join(dir_03, "merged_label_region.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(region_ct, region_affine),
                os.path.join(dir_03, "ct_region.nii.gz"),
            )

        # ── 最终输出④: 合并后的完整 132类 label ──
        output_path = os.path.join(dir_04, "merged_label.nii.gz")
        nib.save(
            nib.Nifti1Image(merged.astype(np.uint8), affine),
            output_path,
        )

        print(f"  → 任务目录: {task_dir}")

    elapsed = time.time() - t_start

    return {
        "status": "ok",
        "output_path": output_path,
        "task_dir": task_dir,
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
        "removal_info": removal_info,
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
        if result.get("task_dir"):
            print(f"  任务目录:   {result['task_dir']}")
            print(f"    01_original_ct/    — 清除自带肿瘤后的 CT + label")
            print(f"    02_tumor_mask/      — 单独的肿瘤二值 mask")
            print(f"    03_tumor_region_ct/ — 合并后 label 肿瘤局部裁切 + 原始CT对照")
            print(f"    04_final_merged/    — 合并后的完整 label")
            print(f"\n  → 下一步: 用此 mask 运行 MAISI 步骤2:")
            print(f"    python scripts/infer_image_from_mask.py "
                  f"--mask {result['task_dir']}/04_final_merged/merged_label.nii.gz")
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
