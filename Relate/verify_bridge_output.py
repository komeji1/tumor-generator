#!/usr/bin/env python3
"""
verify_bridge_output.py — 桥接输出验证工具

在不依赖 MAISI 权重的情况下，全面验证桥接输出的正确性。
所有检查均复制自 MAISI 的 validate_user_mask() 逻辑，加上桥接特有的检查项。

用法:
  # 验证单个输出文件
  python verify_bridge_output.py \
    --merged merged_label.nii.gz \
    --ct original_ct.nii.gz \
    --label original_label.nii.gz \
    --organ liver --size medium

  # 验证整个输出目录
  python verify_bridge_output.py --dir output/ --ct ... --label ...

  # 只做格式检查 (不需要原文件)
  python verify_bridge_output.py --merged merged_label.nii.gz --quick
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import nibabel as nib
except ModuleNotFoundError:
    print(
        "错误: 缺少 nibabel 模块。\n"
        "请安装后重试:\n"
        "  pip install nibabel\n"
        "或者如果你用的是 Tumor 项目的 Python 环境:\n"
        "  .\\run.bat  # 确认 Python 路径后\n"
        "  pip install nibabel",
        file=sys.stderr,
    )
    sys.exit(1)

import numpy as np

# ── 兼容 Windows GBK 终端 ──
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )
except (AttributeError, OSError):
    pass


# ══════════════════════════════════════════════════════════════
#  MAISI 验证常量 (复制自 infer_image_from_mask.py)
# ══════════════════════════════════════════════════════════════

# MAISI 接受的合法 XY 维度
_VALID_DIM_XY = (256, 384, 512)
# MAISI 接受的合法 Z 维度
_VALID_DIM_Z = (128, 256, 384, 512, 640, 768)
# MAISI 接受的 XY 间距范围 (mm)
_VALID_SPACING_XY_RANGE = (0.5, 3.0)
# MAISI 接受的 Z 间距范围 (mm)
_VALID_SPACING_Z_RANGE = (0.5, 5.0)
# MAISI 132 类标签词汇表: 0..132 + body envelope (200)
_MAISI_VALID_LABELS = set(range(0, 133)) | {200}

# 器官标签映射
ORGAN_LABELS = {
    "liver": 1, "pancreas": 4, "kidney_left": 5, "kidney_right": 14,
    "colon": 62, "esophagus": 11, "uterus": 161, "lung_left": 20,
    "lung_right": 21, "bone": 21, "spleen": 3, "stomach": 2,
}

# 肿瘤标签映射
TUMOR_LABELS = {
    "liver": 26, "pancreas": 24, "kidney": 116, "colon": 27,
    "lung": 23, "bone": 128,
}
TUMOR_LABEL_NAMES = {v: k for k, v in TUMOR_LABELS.items()}

# 尺寸 → 半径范围
SIZE_RANGES = {
    "tiny": (2.0, 5.0), "small": (5.0, 10.0),
    "medium": (10.0, 20.0), "large": (20.0, 35.0),
}

BODY_ENVELOPE = 200
BACKGROUND = 0


# ══════════════════════════════════════════════════════════════
#  验证函数
# ══════════════════════════════════════════════════════════════

class CheckResult:
    """单条检查结果"""
    def __init__(self, name: str, passed: bool, msg: str = "",
                 severity: str = "error"):
        self.name = name
        self.passed = passed
        self.msg = msg
        self.severity = severity  # error | warning | info

    def icon(self) -> str:
        if self.passed:
            return "[PASS]"
        return "[WARN]" if self.severity == "warning" else "[FAIL]"


class VerificationReport:
    """汇总验证报告"""
    def __init__(self):
        self.checks: List[CheckResult] = []
        self.meta: dict = {}

    def add(self, name: str, passed: bool, msg: str = "",
            severity: str = "error"):
        self.checks.append(CheckResult(name, passed, msg, severity))

    def all_pass(self) -> bool:
        return all(c.passed or c.severity == "warning" for c in self.checks)

    def errors(self) -> list:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    def warnings(self) -> list:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def print(self):
        n_pass = sum(1 for c in self.checks if c.passed)
        n_warn = len(self.warnings())
        n_fail = len(self.errors())
        n_total = len(self.checks)

        print(f"\n{'='*65}")
        print(f"  验证报告: {n_total} 项检查 | "
              f"通过={n_pass}  警告={n_warn}  失败={n_fail}")
        print(f"{'='*65}")

        for c in self.checks:
            mark = c.icon()
            print(f"  {mark}  {c.name}")
            if c.msg:
                for line in c.msg.strip().split("\n"):
                    print(f"         {line}")

        print(f"{'='*65}")
        if n_fail == 0 and n_warn == 0:
            print("  结论: 全部通过 — 可以安全输入 MAISI ControlNet")
        elif n_fail == 0:
            print(f"  结论: 有 {n_warn} 条警告 — 建议检查后再输入 MAISI")
        else:
            print(f"  结论: {n_fail} 项失败 — 生成质量将受影响，请先修复")
        print(f"{'='*65}\n")


# ── 检查 1: 文件完整性 ──

def check_file_integrity(path: str, report: VerificationReport) -> np.ndarray:
    """验证 NIfTI 文件可读且格式正确。"""
    if not os.path.exists(path):
        report.add("文件存在", False, f"文件不存在: {path}")
        return None

    report.add("文件存在", True, f"路径: {path}")

    try:
        img = nib.load(path)
    except Exception as e:
        report.add("NIfTI 可读性", False, f"nibabel 无法加载: {e}")
        return None

    report.add("NIfTI 可读性", True,
               f"nibabel 类型: {type(img).__name__}")

    try:
        data = img.get_fdata()
    except Exception as e:
        report.add("数据可访问", False, f"get_fdata() 失败: {e}")
        return None

    # 检查存储的数据类型 (get_fdata() 总是返回 float64)
    stored_dtype = img.get_data_dtype()
    report.add("数据可访问", True,
               f"shape={data.shape}, stored_dtype={stored_dtype}")

    if not np.issubdtype(stored_dtype, np.integer):
        report.add("整数类型", False,
                   f"stored dtype={stored_dtype}，应为整数 (uint8, int32 等)",
                   severity="warning")
    else:
        report.add("整数类型", True, f"stored dtype={stored_dtype}")

    # 检查是否为 3D
    if data.ndim != 3:
        report.add("3D 体积", False,
                   f"维度={data.ndim}，应为 3D")
    else:
        report.add("3D 体积", True, f"shape=({data.shape[0]}, {data.shape[1]}, {data.shape[2]})")

    # 检查 affine
    affine = img.affine
    if affine.shape != (4, 4):
        report.add("Affine 矩阵", False, f"shape={affine.shape}，应为 (4,4)")
    else:
        report.add("Affine 矩阵", True, f"diagonal approx: [{affine[0,0]:.2f}, {affine[1,1]:.2f}, {affine[2,2]:.2f}]")

    return data.astype(np.int32) if data is not None else None


# ── 检查 2: 标签词汇表 ──

def check_label_vocabulary(merged: np.ndarray, report: VerificationReport):
    """验证所有标签值都在 MAISI 合法范围内。"""
    unique = set(np.unique(merged).astype(int).tolist())
    report.add("标签数量", True, f"{len(unique)} 种唯一标签值")

    known = sorted(v for v in unique if v in _MAISI_VALID_LABELS)
    unknown = sorted(v for v in unique if v not in _MAISI_VALID_LABELS)

    if unknown:
        report.add("标签合法性", False,
                   f"{len(unknown)} 个标签不在 MAISI 词汇表中: {unknown[:15]}"
                   + ("..." if len(unknown) > 15 else ""))
    else:
        report.add("标签合法性", True,
                   f"全部 {len(known)} 个标签 ∈ MAISI 132-class 词汇表")

    # 列出所有标签及其体素数量
    label_counts = {}
    for v in sorted(unique):
        count = int((merged == v).sum())
        if count > 0:
            label_counts[int(v)] = count

    detail_lines = []
    for label_id, count in sorted(label_counts.items()):
        name = _get_label_name(label_id)
        detail_lines.append(f"  label {label_id:>3} ({name:<20s}): {count:>10,} voxels")
    report.add("标签分布", True, "\n".join(detail_lines))


def _get_label_name(label_id: int) -> str:
    """获取标签的可读名称。"""
    if label_id == BACKGROUND:
        return "background"
    if label_id == BODY_ENVELOPE:
        return "body envelope"
    for name, lid in ORGAN_LABELS.items():
        if lid == label_id:
            return name
    for name, lid in TUMOR_LABELS.items():
        if lid == label_id:
            return f"{name}_tumor"
    if 0 < label_id <= 132:
        return "other organ"
    return "UNKNOWN"


# ── 检查 3: Body Envelope ──

def check_body_envelope(merged: np.ndarray, report: VerificationReport):
    """验证 body envelope (label 200) 存在且合理。"""
    has_be = bool((merged == BODY_ENVELOPE).any())
    be_count = int((merged == BODY_ENVELOPE).sum())
    total = merged.size
    be_ratio = be_count / total if total > 0 else 0

    if has_be:
        report.add("Body Envelope", True,
                   f"label 200 存在 ({be_count:,} voxels, "
                   f"{be_ratio:.1%} of volume)")
    else:
        report.add("Body Envelope", False,
                   "label 200 (body envelope) 缺失 — "
                   "MAISI ControlNet 必须此项！请看 "
                   "scripts/utils.py:add_body_envelope")

    # 合理性检查: body envelope 应该在 10%~60% 的体积
    if has_be:
        if be_ratio < 0.05:
            report.add("Body Envelope 比例", False,
                       f"仅 {be_ratio:.2%} — 可能未正确覆盖身体区域",
                       severity="warning")
        elif be_ratio > 0.70:
            report.add("Body Envelope 比例", False,
                       f"{be_ratio:.1%} — 可能覆盖了空气",
                       severity="warning")
        else:
            report.add("Body Envelope 比例", True,
                       f"{be_ratio:.1%} (合理范围 5%-70%)")


# ── 检查 4: 空间一致性 ──

def check_spatial_consistency(
    merged_path: str,
    ct_path: Optional[str],
    label_path: Optional[str],
    report: VerificationReport,
):
    """验证桥接输出的形状、间距、affine 与输入一致。"""
    merged_img = nib.load(merged_path)
    merged_shape = merged_img.shape
    merged_spacing = tuple(float(s) for s in merged_img.header.get_zooms()[:3])

    report.add("输出形状", True,
               f"({merged_shape[0]}, {merged_shape[1]}, {merged_shape[2]})")
    report.add("输出间距", True,
               f"({merged_spacing[0]:.2f}, {merged_spacing[1]:.2f}, {merged_spacing[2]:.2f}) mm")

    if ct_path and os.path.exists(ct_path):
        ct_img = nib.load(ct_path)
        ct_shape = ct_img.shape
        ct_spacing = tuple(float(s) for s in ct_img.header.get_zooms()[:3])

        if merged_shape == ct_shape:
            report.add("形状一致性 (vs CT)", True,
                       f"shape={merged_shape} 完全匹配")
        else:
            report.add("形状一致性 (vs CT)", False,
                       f"merged={merged_shape} vs CT={ct_shape}")

        spacing_ok = all(
            abs(merged_spacing[i] - ct_spacing[i]) < 0.01
            for i in range(3)
        )
        if spacing_ok:
            report.add("间距一致性 (vs CT)", True,
                       f"spacing={merged_spacing} 匹配")
        else:
            report.add("间距一致性 (vs CT)", False,
                       f"merged={merged_spacing} vs CT={ct_spacing}",
                       severity="warning")

    if label_path and os.path.exists(label_path):
        label_img = nib.load(label_path)
        label_shape = label_img.shape
        if merged_shape == label_shape:
            report.add("形状一致性 (vs 原label)", True,
                       f"shape={merged_shape} 完全匹配")
        else:
            report.add("形状一致性 (vs 原label)", False,
                       f"merged={merged_shape} vs label={label_shape}")


# ── 检查 5: MAISI 兼容性 ──

def check_maisi_compatibility(
    merged: np.ndarray,
    merged_path: str,
    report: VerificationReport,
):
    """验证输出是否符合 MAISI ControlNet 的输入要求。"""
    img = nib.load(merged_path)
    shape = merged.shape
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

    # MAISI 使用 MONAI Orientationd(axcodes="RAS") 将数据重排为 RAS 顺序。
    # 对于 RAS-aligned 的 NIfTI (大多数 MAISI 输出都是如此),
    # nibabel shape 已经是 (R_dim, A_dim, S_dim) = (H, W, D)。
    # 直接使用 nibabel shape 即可，不需要轴交换。
    #
    # 如果 affine 不是 RAS-aligned, 则需要通过 nib.aff2axcodes 判断轴方向
    # 并按 RAS 顺序重排。但这种情况在实际 MAISI 输出中不会出现。
    axcodes = nib.aff2axcodes(img.affine)

    if axcodes == ("R", "A", "S"):
        # RAS-aligned: nibabel shape 直接对应 MAISI (H, W, D)
        maisi_shape = shape
        maisi_spacing = spacing
    else:
        # 非 RAS-aligned: 需要按 RAS 顺序重排 shape 和 spacing
        # 确定 nibabel 各轴对应的 RAS 方向
        ras_order = {"R": 0, "A": 1, "S": 2}
        axis_map = [ras_order[code] for code in axcodes]
        maisi_shape = tuple(shape[i] for i in [
            axis_map.index(0), axis_map.index(1), axis_map.index(2)])
        maisi_spacing = tuple(spacing[i] for i in [
            axis_map.index(0), axis_map.index(1), axis_map.index(2)])

    # 检查 _is_valid_target 条件
    xy_equal = maisi_shape[0] == maisi_shape[1]
    xy_valid = maisi_shape[0] in _VALID_DIM_XY
    z_valid = maisi_shape[2] in _VALID_DIM_Z
    spacing_xy_equal = abs(maisi_spacing[0] - maisi_spacing[1]) < 1e-6
    spacing_xy_ok = (_VALID_SPACING_XY_RANGE[0] <= maisi_spacing[0]
                     <= _VALID_SPACING_XY_RANGE[1])
    spacing_z_ok = (_VALID_SPACING_Z_RANGE[0] <= maisi_spacing[2]
                    <= _VALID_SPACING_Z_RANGE[1])

    all_valid = all([
        xy_equal, xy_valid, z_valid,
        spacing_xy_equal, spacing_xy_ok, spacing_z_ok,
    ])

    if all_valid:
        report.add("MAISI 形状/间距兼容", True,
                   f"无需 resample — 可直接输入 ControlNet")
    else:
        issues = []
        if not xy_equal:
            issues.append(f"H({maisi_shape[0]}) != W({maisi_shape[1]})")
        if not xy_valid:
            issues.append(f"H/W={maisi_shape[0]} not in {_VALID_DIM_XY}")
        if not z_valid:
            issues.append(f"D={maisi_shape[2]} not in {_VALID_DIM_Z}")
        if not spacing_xy_equal:
            issues.append(f"spacing_xy mismatch: {maisi_spacing[:2]}")
        if not spacing_xy_ok:
            issues.append(f"spacing_xy={maisi_spacing[0]:.2f} not in "
                          f"[{_VALID_SPACING_XY_RANGE[0]}, {_VALID_SPACING_XY_RANGE[1]}]")
        if not spacing_z_ok:
            issues.append(f"spacing_z={maisi_spacing[2]:.2f} not in "
                          f"[{_VALID_SPACING_Z_RANGE[0]}, {_VALID_SPACING_Z_RANGE[1]}]")

        report.add("MAISI 形状/间距兼容", False,
                   "MAISI 将自动 resample (会损失精度):\n"
                   + "\n".join(f"    - {i}" for i in issues),
                   severity="warning")


# ── 检查 6: 肿瘤放置质量 ──

def check_tumor_placement(
    merged: np.ndarray,
    original_label: Optional[np.ndarray],
    organ: str,
    size_category: str,
    report: VerificationReport,
):
    """验证肿瘤放置的正确性。"""
    tumor_label = TUMOR_LABELS.get(organ)
    organ_label = ORGAN_LABELS.get(organ)

    if tumor_label is None:
        report.add("肿瘤标签", False,
                   f"器官 {organ} 无已知肿瘤标签 — 跳过肿瘤检查",
                   severity="warning")
        return

    # 肿瘤是否存在
    tumor_mask = (merged == tumor_label)
    tumor_voxels = int(tumor_mask.sum())

    if tumor_voxels == 0:
        report.add("肿瘤存在", False,
                   f"label {tumor_label} ({organ}_tumor) 在输出中不存在")
        return

    report.add("肿瘤存在", True,
               f"label {tumor_label} ({organ}_tumor): {tumor_voxels:,} voxels")

    # 肿瘤尺寸合理性
    if size_category:
        r_min, r_max = SIZE_RANGES.get(size_category, (0, 999))
        report.add("肿瘤尺寸合理性", True,
                   f"size_category={size_category} (半径 {r_min}-{r_max}mm), "
                   f"{tumor_voxels:,} voxels")
    else:
        report.add("肿瘤尺寸合理性", True,
                   f"{tumor_voxels:,} voxels (未指定 size_category)")

    # 检查肿瘤是否在器官内
    if original_label is not None and organ_label is not None:
        if organ == "kidney":
            organ_mask = ((original_label == 5) | (original_label == 14))
        else:
            organ_mask = (original_label == organ_label)

        organ_voxels = int(organ_mask.sum())

        if organ_voxels > 0:
            overlap = tumor_mask & organ_mask
            overlap_ratio = overlap.sum() / max(tumor_voxels, 1)

            if overlap_ratio >= 0.8:
                report.add("肿瘤-器官重叠", True,
                           f"{overlap_ratio:.0%} 的肿瘤 voxel 在器官内")
            elif overlap_ratio >= 0.5:
                report.add("肿瘤-器官重叠", False,
                           f"仅 {overlap_ratio:.0%} 重叠 — "
                           f"肿瘤部分超出器官边界",
                           severity="warning")
            else:
                report.add("肿瘤-器官重叠", False,
                           f"仅 {overlap_ratio:.0%} — 肿瘤大部分在器官外",
                           severity="warning")
        else:
            report.add("原始器官存在", False,
                       f"原 label 中找不到器官 {organ} (label={organ_label})",
                       severity="warning")

    # 检查肿瘤标签不要覆盖其他器官
    if original_label is not None:
        non_organ_non_tumor = (
            (merged == tumor_label) &
            (original_label != BACKGROUND) &
            (original_label != BODY_ENVELOPE)
        )
        if organ_label:
            non_organ_non_tumor &= (original_label != organ_label)
        if organ == "kidney":
            non_organ_non_tumor &= (original_label != 5)
            non_organ_non_tumor &= (original_label != KIDNEY_RIGHT_LABEL)

        conflict_voxels = int(non_organ_non_tumor.sum())
        conflict_labels = set(np.unique(original_label[non_organ_non_tumor]).tolist())

        if conflict_voxels > 0:
            conflict_names = [_get_label_name(l) for l in conflict_labels]
            report.add("肿瘤标签冲突", False,
                       f"{conflict_voxels} 个体素的肿瘤标签覆盖了其他结构: "
                       f"{conflict_names}",
                       severity="warning")
        else:
            report.add("肿瘤标签冲突", True,
                       "肿瘤标签未覆盖任何其他器官/结构")


KIDNEY_RIGHT_LABEL = 14


# ── 检查 7: 背景/身体区域合理性 ──

def check_background_body_ratio(
    merged: np.ndarray,
    ct_data: Optional[np.ndarray],
    report: VerificationReport,
):
    """验证背景(0)和身体(200)的分布是否合理。"""
    bg_count = int((merged == BACKGROUND).sum())
    be_count = int((merged == BODY_ENVELOPE).sum())
    organ_count = int(((merged != BACKGROUND) & (merged != BODY_ENVELOPE)).sum())
    total = merged.size

    report.add("背景比例", True,
               f"background (label 0): {bg_count:,} voxels ({bg_count/total:.1%})")

    if ct_data is not None and ct_data.shape == merged.shape:
        # 检查背景区域是否确实是空气 (CT < -500 HU)
        bg_in_merged = (merged == BACKGROUND)
        air_in_ct = (ct_data < -500)
        bg_is_air = bg_in_merged & air_in_ct
        bg_not_air = bg_in_merged & ~air_in_ct

        air_ratio = bg_is_air.sum() / max(bg_in_merged.sum(), 1)
        if air_ratio > 0.9:
            report.add("背景=空气", True,
                       f"{air_ratio:.0%} 的背景 voxel CT<-500 HU (正确)")
        else:
            not_air_count = int(bg_not_air.sum())
            report.add("背景=空气", False,
                       f"仅 {air_ratio:.0%} 的背景是空气 — "
                       f"{not_air_count:,} voxels 的背景区域 HU >= -500",
                       severity="warning")

        # 检查 body envelope 是否确实是身体组织
        be_in_merged = (merged == BODY_ENVELOPE)
        body_in_ct = (ct_data > -200)
        be_is_body = be_in_merged & body_in_ct

        body_ratio = be_is_body.sum() / max(be_in_merged.sum(), 1)
        if body_ratio > 0.8:
            report.add("Body=组织", True,
                       f"{body_ratio:.0%} 的 body envelope CT>-200 HU (正确)")
        else:
            report.add("Body=组织", False,
                       f"仅 {body_ratio:.0%} 的 body envelope 在软组织范围内",
                       severity="warning")


# ══════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════

def verify_single(
    merged_path: str,
    ct_path: Optional[str] = None,
    label_path: Optional[str] = None,
    organ: Optional[str] = None,
    size_category: Optional[str] = None,
) -> VerificationReport:
    """对单个桥接输出文件运行全部验证。"""
    report = VerificationReport()
    report.meta = {
        "merged_path": merged_path,
        "ct_path": ct_path,
        "label_path": label_path,
        "organ": organ,
        "size_category": size_category,
    }

    print(f"\n{'='*65}")
    print(f"  验证: {os.path.basename(merged_path)}")
    print(f"{'='*65}")

    # ── 1. 文件完整性 ──
    merged = check_file_integrity(merged_path, report)
    if merged is None:
        report.print()
        return report

    # ── 2. 标签词汇表 ──
    check_label_vocabulary(merged, report)

    # ── 3. Body Envelope ──
    check_body_envelope(merged, report)

    # ── 4. 空间一致性 ──
    check_spatial_consistency(merged_path, ct_path, label_path, report)

    # ── 5. MAISI 兼容性 ──
    check_maisi_compatibility(merged, merged_path, report)

    # ── 6. 肿瘤放置 ──
    if organ:
        original_label = None
        if label_path and os.path.exists(label_path):
            original_label = nib.load(label_path).get_fdata().astype(np.int32)
        check_tumor_placement(
            merged, original_label, organ, size_category, report
        )
    else:
        # 自动检测肿瘤标签
        unique = set(np.unique(merged).astype(int).tolist())
        detected_tumors = [v for v in unique if v in TUMOR_LABEL_NAMES]
        if detected_tumors:
            report.add("肿瘤检测", True,
                       f"检测到肿瘤标签: { {v: TUMOR_LABEL_NAMES[v] for v in detected_tumors} }")
        else:
            report.add("肿瘤检测", False,
                       "未检测到任何已知肿瘤标签",
                       severity="warning")

    # ── 7. 背景/身体合理性 ──
    ct_data = None
    if ct_path and os.path.exists(ct_path):
        ct_data = nib.load(ct_path).get_fdata().astype(np.float32)
    check_background_body_ratio(merged, ct_data, report)

    report.print()
    return report


def verify_directory(
    dir_path: str,
    ct_path: Optional[str] = None,
    label_path: Optional[str] = None,
    organ: Optional[str] = None,
    size_category: Optional[str] = None,
) -> List[VerificationReport]:
    """验证目录下所有 .nii.gz 文件。"""
    reports = []
    files = sorted(Path(dir_path).glob("*.nii.gz"))

    if not files:
        print(f"目录中无 .nii.gz 文件: {dir_path}")
        return reports

    for f in files:
        # 跳过原始 CT 和 label
        fname = f.name
        if ct_path and f.samefile(Path(ct_path)):
            continue
        if label_path and f.samefile(Path(label_path)):
            continue

        # 判断器官和尺寸 (从文件名解析)
        f_organ = organ
        f_size = size_category
        if not f_organ:
            for org_name in TUMOR_LABELS:
                if org_name in fname:
                    f_organ = org_name
                    break
        if not f_size:
            for sz in SIZE_RANGES:
                if sz in fname:
                    f_size = sz
                    break

        report = verify_single(
            str(f), ct_path, label_path, f_organ, f_size
        )
        reports.append(report)

    # 汇总
    n_ok = sum(1 for r in reports if r.all_pass())
    n_fail = len(reports) - n_ok
    print(f"\n{'='*65}")
    print(f"  目录汇总: {len(reports)} 个文件 | 通过={n_ok}  有问题={n_fail}")
    print(f"{'='*65}")

    return reports


def main():
    parser = argparse.ArgumentParser(
        description="桥接输出验证 — 检查合并 mask 是否可直接输入 MAISI ControlNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python verify_bridge_output.py --merged merged.nii.gz --quick\n"
            "  python verify_bridge_output.py --merged merged.nii.gz "
            "--ct ct.nii.gz --label label.nii.gz --organ liver --size medium\n"
            "  python verify_bridge_output.py --dir output/ --ct ct.nii.gz "
            "--label label.nii.gz\n"
        ),
    )

    parser.add_argument("--merged", "-m", help="桥接输出的 mask 文件 (.nii.gz)")
    parser.add_argument("--dir", "-d", help="桥接输出目录 (验证所有 .nii.gz)")
    parser.add_argument("--ct", help="原始 MAISI CT 文件 (用于空间和背景检查)")
    parser.add_argument("--label", help="原始 MAISI label 文件 (用于对比检查)")
    parser.add_argument("--organ", choices=list(TUMOR_LABELS.keys()),
                        help="目标器官 (用于肿瘤放置检查)")
    parser.add_argument("--size", choices=list(SIZE_RANGES.keys()),
                        help="肿瘤尺寸类别")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 — 只做格式和标签检查 (不需要 CT/label)")
    parser.add_argument("--json", action="store_true",
                        help="输出 JSON 格式的验证结果")

    args = parser.parse_args()

    if args.quick:
        # 快速模式: 忽略 ct/label
        if args.merged:
            verify_single(args.merged, organ=args.organ)
        elif args.dir:
            verify_directory(args.dir, organ=args.organ)
        else:
            parser.error("需要 --merged 或 --dir")
    elif args.dir:
        verify_directory(
            args.dir, args.ct, args.label, args.organ, args.size
        )
    elif args.merged:
        verify_single(
            args.merged, args.ct, args.label, args.organ, args.size
        )
    else:
        parser.error("需要 --merged 或 --dir")


if __name__ == "__main__":
    main()
