#!/usr/bin/env python3
"""
label_colormap_export.py — 导出 MAISI 132类 label 的颜色映射文件

为 label NIfTI 生成配套的颜色映射文件，让 3D Slicer / ITK-SNAP
打开 label 时自动显示彩色（不用手动配置）。

输出格式:
  - .ctbl  — 3D Slicer 颜色表 (最通用)
  - label_colors.txt — ITK-SNAP 格式

用法:
  python label_colormap_export.py --label path/to/merged_label.nii.gz

  # 批量 — 自动扫描任务目录
  python label_colormap_export.py --scan-dir output/
"""

from __future__ import annotations

import argparse
import os
import sys

import nibabel as nib
import numpy as np

# 导入颜色定义
_script_dir = os.path.dirname(os.path.abspath(__file__))
relate_dir = _script_dir if os.path.basename(_script_dir) == "Relate" else os.path.join(os.path.dirname(_script_dir), "Relate")
if relate_dir not in sys.path:
    sys.path.insert(0, relate_dir)

from label_colorize import LABEL_COLORS, LABEL_NAMES, _default_color


# ══════════════════════════════════════════════════════════════
#  3D Slicer .ctbl 格式
# ══════════════════════════════════════════════════════════════

def export_slicer_ctbl(label_path: str, output_dir: str | None = None):
    """生成 3D Slicer .ctbl 颜色表文件。"""

    # 读取 label 中实际出现的标签
    label_data = nib.load(label_path).get_fdata().astype(np.int32)
    present_labels = sorted(set(np.unique(label_data).astype(int).tolist()))

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(label_path))
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.splitext(os.path.basename(label_path))[0])[0]

    # ── 3D Slicer .ctbl ──
    # 格式: 每行 "编号 名称 R G B A"
    ctbl_path = os.path.join(output_dir, f"{base}_colormap.ctbl")
    with open(ctbl_path, "w", encoding="utf-8") as f:
        for label_id in range(0, 201):
            color = LABEL_COLORS.get(label_id, _default_color(label_id))
            name = LABEL_NAMES.get(label_id, f"label_{label_id}")
            # RGBA, 0-255范围
            r, g, b = color
            a = 255 if label_id != 0 else 0  # 背景透明
            f.write(f"{label_id} {name} {r} {g} {b} {a}\n")

        # 预留肿瘤标签 133-140
        for label_id in range(133, 141):
            color = LABEL_COLORS.get(label_id, _default_color(label_id))
            name = LABEL_NAMES.get(label_id, f"预留肿瘤{label_id-132}")
            r, g, b = color
            f.write(f"{label_id} {name} {r} {g} {b} 255\n")

    print(f"  → 3D Slicer 颜色表: {ctbl_path}")

    # ── ITK-SNAP label_colors.txt ──
    # 格式: 每行 "编号 R G B A 名称"
    itk_path = os.path.join(output_dir, f"{base}_itksnap_labels.txt")
    with open(itk_path, "w", encoding="utf-8") as f:
        # ITK-SNAP 头部
        f.write("# ITK-SNAP Label Description File\n")
        f.write("# Rows: label_id, R, G, B, A, label_name\n")
        for label_id in present_labels:
            color = LABEL_COLORS.get(label_id, _default_color(label_id))
            name = LABEL_NAMES.get(label_id, f"label_{label_id}")
            r, g, b = color
            a = 255 if label_id != 0 else 0
            f.write(f"{label_id} {r} {g} {b} {a} {name}\n")

    print(f"  → ITK-SNAP 颜色表: {itk_path}")

    # ── 打印使用说明 ──
    print()
    print("  ── 如何在查看器中显示彩色 ──")
    print()
    print("  3D Slicer:")
    print("    1. 加载 merged_label.nii.gz")
    print("    2. 在 Volume Rendering 或 Segmentation 模块")
    print("    3. 选择 Color → Load Color Table → 选择 .ctbl 文件")
    print()
    print("  ITK-SNAP:")
    print("    1. 打开 ITK-SNAP, 加载 CT + label")
    print("    2. Segmentation → Load Label Description → 选择 .txt 文件")
    print()
    print("  通用方法 (适用于任何查看器):")
    print("    打开 merged_label_colored.nii.gz (RGB NIfTI)")
    print("    这个文件每个 voxel 是 RGB 颜色值，不需要额外配置")

    return {
        "label_path": label_path,
        "ctbl_path": ctbl_path,
        "itk_path": itk_path,
        "present_labels": present_labels,
    }


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="导出 MAISI label 颜色映射文件 (3D Slicer .ctbl + ITK-SNAP .txt)"
    )

    parser.add_argument("--label", metavar="PATH",
                        help="label NIfTI 文件路径")
    parser.add_argument("--scan-dir", metavar="DIR",
                        help="扫描目录下的任务目录，批量导出")
    parser.add_argument("--output-dir", "-o", metavar="DIR",
                        help="输出目录")

    args = parser.parse_args()

    if args.scan_dir:
        import glob as _glob
        pattern = os.path.join(args.scan_dir, "**", "04_final_merged", "merged_label.nii.gz")
        files = sorted(_glob.glob(pattern, recursive=True))
        for f in files:
            print(f"处理: {os.path.basename(f)}")
            export_slicer_ctbl(f)
            print()
        return

    if not args.label:
        parser.error("需要 --label 或 --scan-dir")

    result = export_slicer_ctbl(args.label, args.output_dir)
    print(f"\n✓ 颜色映射导出完成!")


if __name__ == "__main__":
    main()
