#!/usr/bin/env python3
"""
export_tumor_removed_ct.py — 输出清除自带肿瘤后的 CT + label

读取 MAISI 步骤1生成的 CT 和 132类 label，执行 Step 1.5 清除自带肿瘤，
保存清除后的 CT（原样不变）和 label（肿瘤标签还原为器官标签）。

用法:
  # 基本用法 — 指定 CT 和 label 文件
  python export_tumor_removed_ct.py \
    --ct output/sample_image.nii.gz \
    --label output/sample_label.nii.gz

  # 指定输出目录
  python export_tumor_removed_ct.py \
    --ct output/sample_image.nii.gz \
    --label output/sample_label.nii.gz \
    --output-dir output/removed

  # 批量 — 自动扫描 output/ 目录下的 _image + _label_full 配对
  python export_tumor_removed_ct.py \
    --scan-dir output/

  # 仅打印清除信息，不保存文件
  python export_tumor_removed_ct.py \
    --ct output/sample_image.nii.gz \
    --label output/sample_label.nii.gz \
    --dry-run

依赖:
  - Relate/bridge_maisi_mask.py 中的 remove_existing_tumors, load_maisi_data
  - nibabel, numpy
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import glob

import nibabel as nib
import numpy as np

# ── 兼容 Windows GBK 终端 ──
_sys_stdout = sys.stdout
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )
except (AttributeError, OSError):
    pass

# ── 导入桥接模块 ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
_work_dir = os.path.dirname(_script_dir)
_mask_project_root_candidates = [
    os.path.join(_work_dir, "Mask"),
    os.path.join(_script_dir, "..", "Mask"),
    os.path.join(_work_dir, "..", "Mask"),
]

# 添加 Mask 项目路径（remove_existing_tumors 不需要 Mask，但保持兼容）
for p in _mask_project_root_candidates:
    p = os.path.abspath(p)
    if os.path.isdir(p):
        step5_src = os.path.join(p, "Step5", "src")
        step1_src = os.path.join(p, "Step1", "src")
        for sp in [step5_src, step1_src]:
            if sp not in sys.path:
                sys.path.insert(0, sp)
        break

# 添加 Relate 目录
relate_dir = _script_dir if os.path.basename(_script_dir) == "Relate" else os.path.join(_work_dir, "Relate")
if relate_dir not in sys.path:
    sys.path.insert(0, relate_dir)

from bridge_maisi_mask import remove_existing_tumors, load_maisi_data


# ══════════════════════════════════════════════════════════════
#  单文件处理
# ══════════════════════════════════════════════════════════════

def process_single(
    ct_path: str,
    label_path: str,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> dict:
    """处理单个 CT + label 对，清除自带肿瘤后保存。"""

    # ── 加载 ──
    ct_data, label_data, affine, spacing = load_maisi_data(ct_path, label_path)

    # ── 清除自带肿瘤 ──
    cleaned_label, removal_info = remove_existing_tumors(label_data)

    # ── 打印清除信息 ──
    total = removal_info["total_removed_voxels"]
    if total > 0:
        print(f"  ✓ 清除自带肿瘤: {total:,} voxels")
        for tl, cnt in removal_info["removed_labels"].items():
            target = removal_info["restored_targets"].get(tl, "?")
            print(f"    label {tl} → {target}  ({cnt:,} voxels)")
    else:
        print(f"  ℹ 此 label 中无自带肿瘤标签")

    # ── 保存 ──
    if dry_run:
        output_ct_path = None
        output_label_path = None
    else:
        # 输出目录: 默认与 label 同目录下的 removed/ 子目录
        if output_dir is None:
            label_dir = os.path.dirname(os.path.abspath(label_path))
            output_dir = os.path.join(label_dir, "removed_tumor")

        os.makedirs(output_dir, exist_ok=True)

        # 文件名: 基于原始 label 文件名，加 _no_tumor 后缀
        label_base = os.path.splitext(os.path.splitext(
            os.path.basename(label_path))[0])[0]

        # CT 文件名: 基于原始 CT 文件名，加 _no_tumor 后缀
        ct_base = os.path.splitext(os.path.splitext(
            os.path.basename(ct_path))[0])[0]

        output_label_path = os.path.join(output_dir, f"{label_base}_no_tumor.nii.gz")
        output_ct_path = os.path.join(output_dir, f"{ct_base}_no_tumor.nii.gz")

        # 保存 CT (原样不变)
        nib.save(
            nib.Nifti1Image(ct_data, affine),
            output_ct_path,
        )
        print(f"  → CT 已保存: {output_ct_path}")

        # 保存 label (清除后)
        nib.save(
            nib.Nifti1Image(cleaned_label.astype(np.uint8), affine),
            output_label_path,
        )
        print(f"  → Label 已保存: {output_label_path}")

        # 保存清除信息 JSON
        info_path = os.path.join(output_dir, f"{label_base}_removal_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(removal_info, f, indent=2, ensure_ascii=False)
        print(f"  → 清除信息已保存: {info_path}")

    return {
        "ct_path": ct_path,
        "label_path": label_path,
        "output_ct_path": output_ct_path,
        "output_label_path": output_label_path,
        "removal_info": removal_info,
    }


# ══════════════════════════════════════════════════════════════
#  批量扫描
# ══════════════════════════════════════════════════════════════

def scan_and_process(scan_dir: str, output_dir: str | None = None, dry_run: bool = False):
    """扫描目录下的 _image + _label_full 配对，批量处理。"""

    # 找所有 label_full 文件
    label_files = sorted(glob.glob(os.path.join(scan_dir, "*_label_full*.nii.gz")))

    if not label_files:
        print(f"在 {scan_dir} 中未找到 *_label_full*.nii.gz 文件")
        return

    print(f"找到 {len(label_files)} 个 label 文件")
    print(f"{'='*60}\n")

    results = []
    for label_path in label_files:
        # 推算对应的 CT 文件名
        label_base = os.path.splitext(os.path.splitext(
            os.path.basename(label_path))[0])[0]

        # label_base 格式: sample_20260619_184801_368695_label_full
        # CT 文件名格式:   sample_20260619_184801_368695_image
        ct_base = label_base.replace("_label_full", "_image")
        ct_path = os.path.join(scan_dir, f"{ct_base}.nii.gz")

        if not os.path.exists(ct_path):
            print(f"⚠ 找不到对应 CT: {ct_path}，跳过 {label_path}")
            continue

        print(f"[{len(results)+1}/{len(label_files)}] {os.path.basename(label_path)}")
        result = process_single(ct_path, label_path, output_dir, dry_run)
        results.append(result)
        print()

    # 汇总
    total_removed = sum(r["removal_info"]["total_removed_voxels"] for r in results)
    with_tumor = sum(1 for r in results if r["removal_info"]["total_removed_voxels"] > 0)
    print(f"{'='*60}")
    print(f"完成: {len(results)} 个文件")
    print(f"含自带肿瘤: {with_tumor}/{len(results)}")
    print(f"总清除 voxels: {total_removed:,}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="输出清除自带肿瘤后的 CT + label NIfTI 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python export_tumor_removed_ct.py --ct ct.nii.gz --label label.nii.gz\n"
            "  python export_tumor_removed_ct.py --scan-dir output/\n"
            "  python export_tumor_removed_ct.py --ct ct.nii.gz --label label.nii.gz --dry-run\n"
        ),
    )

    # 单文件模式
    parser.add_argument("--ct", metavar="PATH",
                        help="MAISI 合成的 CT 文件 (.nii.gz)")
    parser.add_argument("--label", metavar="PATH",
                        help="MAISI 合成的 132类标签文件 (.nii.gz)")

    # 批量模式
    parser.add_argument("--scan-dir", metavar="DIR",
                        help="扫描目录下的 _image + _label_full 配对，批量处理")

    # 输出选项
    parser.add_argument("--output-dir", "-o", metavar="DIR",
                        help="输出目录 (默认: label 同目录下的 removed_tumor/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印清除信息，不保存文件")

    args = parser.parse_args()

    # ── 批量模式 ──
    if args.scan_dir:
        scan_and_process(args.scan_dir, args.output_dir, args.dry_run)
        return

    # ── 单文件模式 ──
    if not args.ct or not args.label:
        parser.error(
            "单文件模式需要 --ct 和 --label。"
            "或使用 --scan-dir 批量模式。"
        )

    result = process_single(args.ct, args.label, args.output_dir, args.dry_run)

    if not args.dry_run and result["output_ct_path"]:
        print(f"\n✓ 清除完成!")
        print(f"  原始 CT:    {result['ct_path']}")
        print(f"  原始 Label: {result['label_path']}")
        print(f"  清除后 CT:  {result['output_ct_path']}")
        print(f"  清除后 Label: {result['output_label_path']}")
        print(f"  清除 voxels: {result['removal_info']['total_removed_voxels']:,}")


if __name__ == "__main__":
    main()
