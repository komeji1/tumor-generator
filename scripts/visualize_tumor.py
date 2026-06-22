"""
肿瘤生成结果可视化工具

用法:
  python -m scripts.visualize_tumor                          # 可视化最新生成结果
  python -m scripts.visualize_tumor --ct path.nii.gz         # 指定 CT 文件
  python -m scripts.visualize_tumor --ct ct.nii.gz --mask mask.nii.gz
  python -m scripts.visualize_tumor --diff                   # 生成差值图 (需要原始 CT)
"""

import argparse
import os
import sys
from pathlib import Path


def find_latest_output(project_root: str, organ: str = None):
    """查找最新的生成结果"""
    tumor_dir = os.path.join(project_root, "output", "tumor_ct")
    if not os.path.isdir(tumor_dir):
        return None, None

    # 搜索所有器官子目录
    candidates = []
    for subdir in Path(tumor_dir).iterdir():
        if not subdir.is_dir():
            continue
        if organ and organ not in subdir.name:
            continue
        for f in subdir.glob("*.nii.gz"):
            if "tumor_mask" not in f.name:
                mask_file = f.parent / f.name.replace(".nii.gz", "_tumor_mask.nii.gz")
                if mask_file.exists():
                    candidates.append((str(f), str(mask_file)))

    if not candidates:
        return None, None

    # 按修改时间排序，返回最新的
    candidates.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    return candidates[0]


def visualize(ct_path: str, mask_path: str = None, orig_ct_path: str = None,
              output_path: str = None, show_diff: bool = False):
    """生成可视化图片"""
    import nibabel as nib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    ct_data = nib.load(ct_path).get_fdata().astype(np.float32)

    if mask_path:
        mask_data = nib.load(mask_path).get_fdata().astype(np.int32)
        tumor_mask = (mask_data > 0)
    else:
        tumor_mask = np.zeros_like(ct_data, dtype=bool)

    # 找肿瘤中心
    if tumor_mask.any():
        tumor_idx = np.argwhere(tumor_mask)
        center = tumor_idx.mean(axis=0).astype(int)
    else:
        center = np.array(ct_data.shape) // 2

    orig_data = None
    if orig_ct_path:
        orig_data = nib.load(orig_ct_path).get_fdata().astype(np.float32)

    if show_diff and orig_data is not None:
        diff = ct_data - orig_data
        ncols = 3
    else:
        ncols = 2

    fig, axes = plt.subplots(2, ncols, figsize=(8 * ncols, 14))

    views = [
        ('Axial', center[2], lambda d, c: d[:, :, c]),
        ('Coronal', center[1], lambda d, c: d[:, c, :]),
        ('Sagittal', center[0], lambda d, c: d[c, :, :]),
    ]

    for col, (view_name, coord, slicer) in enumerate(views[:ncols]):
        ct_slice = slicer(ct_data, coord)
        tm_slice = slicer(tumor_mask, coord) if mask_path else None

        # Row 0: Full slice
        axes[0, col].imshow(ct_slice, cmap='gray', vmin=-200, vmax=400)
        if tm_slice is not None and tm_slice.any():
            overlay = np.zeros((*ct_slice.shape, 4))
            boundary = binary_dilation(tm_slice, iterations=1) & ~tm_slice
            overlay[tm_slice] = [1, 0, 0, 0.3]
            overlay[boundary] = [1, 1, 0, 0.8]
            axes[0, col].imshow(overlay)
        axes[0, col].set_title(f'{view_name} @ {coord}', fontsize=12)
        axes[0, col].axis('off')

        # Row 1: Zoomed to tumor
        cx, cy = center[1], center[0]
        r = 40
        zoom = ct_slice[max(0,cx-r):cx+r, max(0,cy-r):cy+r]
        tm_zoom = tm_slice[max(0,cx-r):cx+r, max(0,cy-r):cy+r] if tm_slice is not None else None

        if show_diff and orig_data is not None and col < 1:
            diff_slice = slicer(diff, coord)
            zoom_diff = diff_slice[max(0,cx-r):cx+r, max(0,cy-r):cy+r]
            im = axes[1, col].imshow(zoom_diff, cmap='RdBu_r', vmin=-150, vmax=150)
            if tm_zoom is not None:
                tm_ov = np.zeros((*zoom_diff.shape, 4))
                tm_ov[tm_zoom] = [0, 1, 0, 0.15]
                axes[1, col].imshow(tm_ov)
            axes[1, col].set_title(f'{view_name} Diff (blue=lower, red=higher)', fontsize=12)
            plt.colorbar(im, ax=axes[1, col], shrink=0.7, label='HU change')
        else:
            axes[1, col].imshow(zoom, cmap='gray', vmin=-200, vmax=400)
            if tm_zoom is not None and tm_zoom.any():
                ov = np.zeros((*zoom.shape, 4))
                bd = binary_dilation(tm_zoom, iterations=1) & ~tm_zoom
                ov[tm_zoom] = [1, 0, 0, 0.25]
                ov[bd] = [1, 1, 0, 0.8]
                axes[1, col].imshow(ov)
            axes[1, col].set_title(f'{view_name} Zoomed', fontsize=12)
        axes[1, col].axis('off')

    # Stats text
    if tumor_mask.any():
        tumor_hu = ct_data[tumor_mask]
        stats = f'Tumor: {tumor_mask.sum():,} voxels, HU={tumor_hu.mean():.0f}+-{tumor_hu.std():.0f}'
    else:
        stats = 'No tumor mask loaded'
    fig.suptitle(f'Tumor Visualization\n{stats}', fontsize=14, fontweight='bold')

    if output_path is None:
        output_path = os.path.join(os.path.dirname(ct_path), 'visualization.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='肿瘤生成结果可视化')
    parser.add_argument('--ct', help='CT NIfTI 文件路径')
    parser.add_argument('--mask', help='Tumor mask NIfTI 文件路径')
    parser.add_argument('--orig-ct', help='原始 CT (用于差值图)')
    parser.add_argument('--output', '-o', help='输出图片路径')
    parser.add_argument('--diff', action='store_true', help='生成差值图')
    parser.add_argument('--organ', help='指定器官 (用于自动查找最新输出)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.ct:
        ct_path = args.ct
        mask_path = args.mask
    else:
        ct_path, mask_path = find_latest_output(project_root, args.organ)
        if ct_path is None:
            print('未找到生成结果。请用 --ct 指定文件路径。')
            return
        print(f'Using latest output: {ct_path}')

    visualize(ct_path, mask_path, args.orig_ct, args.output, args.diff)


if __name__ == '__main__':
    main()
