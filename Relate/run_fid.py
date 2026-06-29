#!/usr/bin/env python3
"""
run_fid.py — FID 真实性评估一键运行脚本

用法:
    python run_fid.py              # 默认配置 (8GB VRAM友好, 使用缓存)
    python run_fid.py --force      # 强制重新提取特征 (transform变更后使用)
    python run_fid.py --paper      # 论文对标配置 (512^3, 需要更多VRAM)
    python run_fid.py --force --paper  # 论文配置 + 强制重提
"""

import subprocess
import sys
import os


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FID Realism Evaluation")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extract features (ignore cache)")
    parser.add_argument("--paper", action="store_true",
                        help="Use paper config: 512^3, 1mm^3, center_slices=0.4")
    parser.add_argument("--num_images", type=int, default=48,
                        help="Number of images to evaluate (default: 48)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    fid_script = os.path.join(script_dir, "compute_fid_single_gpu.py")
    real_root = os.path.join(project_root, "data", "autopet_ct50")
    real_list = os.path.join(real_root, "filelist.txt")
    synth_root = os.path.join(project_root, "data", "fid_tumor_ct_50")
    synth_list = os.path.join(synth_root, "filelist.txt")
    output_root = os.path.join(project_root, "data", "fid_features")

    # Check data exists
    for path, name in [
        (real_root, "autoPET real CT"),
        (real_list, "autoPET filelist"),
        (synth_root, "synthetic tumor CT"),
        (synth_list, "synthetic filelist"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)

    # Build command
    cmd = [
        sys.executable, fid_script,
        "--real_dataset_root", real_root,
        "--real_filelist", real_list,
        "--real_features_dir", "autopet_real",
        "--synth_dataset_root", synth_root,
        "--synth_filelist", synth_list,
        "--synth_features_dir", "tumor_ct_synth",
        "--num_images", str(args.num_images),
        "--output_root", output_root,
    ]

    if args.paper:
        cmd.append("--paper_config")
    else:
        cmd += [
            "--target_shape", "256x256x128",
            "--enable_resampling_spacing", "1.0x1.0x1.0",
            "--enable_center_slices_ratio", "0.4",
            "--enable_padding", "True",
            "--enable_center_cropping", "True",
        ]

    cmd.append("--ignore_existing")
    cmd.append("True" if args.force else "False")

    # Print info
    print("=" * 60)
    print("FID Realism Evaluation")
    print("=" * 60)
    print(f"  Config:    {'paper (512^3)' if args.paper else 'default (256x256x128)'}")
    print(f"  Force:     {args.force}")
    print(f"  N images:  {args.num_images}")
    print(f"  Real:      {real_root}")
    print(f"  Synth:     {synth_root}")
    print(f"  Output:    {output_root}")
    print("=" * 60)
    print()

    # Run
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
