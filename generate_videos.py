"""
为每个器官生成3个扫描视频: mask / CT / 叠加
方向修正: 头=上, 前=上(axial)/左(sagittal)
用法: python generate_videos.py
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from PIL import Image

MASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(MASK_DIR, 'output', 'real_ct')
CT_DIR = os.path.join(MASK_DIR, 'data', 'ct')

ORGANS = ['liver_lesion', 'pancreatic_lesion', 'kidney_lesion', 'colon_lesion',
         'esophagus_tumor', 'endometrioma_tumor']

FPS = 4
SLICES_PER_AXIS = 24
EDGE_MARGIN = 0.10
TARGET_SIZE = (640, 640)


def pick_best_mask(organ_dir):
    files = [f for f in os.listdir(organ_dir) if f.endswith('.nii.gz')]
    if not files: return None, None
    best_file, best_vol = None, 0
    for f in files:
        data = (nib.load(os.path.join(organ_dir, f)).get_fdata() > 0).astype(np.uint8)
        vol = int(data.sum())
        if 100 < vol < 500000 and vol > best_vol:
            best_vol = vol; best_file = f
    if best_file is None: return None, None
    fp = os.path.join(organ_dir, best_file)
    data = (nib.load(fp).get_fdata() > 0).astype(np.uint8)
    return fp, {'filename': best_file, 'volume': best_vol, 'shape': data.shape, 'data': data}


def find_ct(mask_path):
    """Find matching CT by shape since renumbered masks don't carry CT ID."""
    mask_shape = nib.load(mask_path).shape
    for ct_id in sorted(os.listdir(CT_DIR)):
        ct_p = os.path.join(CT_DIR, ct_id, 'ct.nii.gz')
        if os.path.exists(ct_p) and nib.load(ct_p).shape == mask_shape:
            return ct_p, ct_id
    return None, None


def load_ct(ct_path):
    img = nib.load(ct_path)
    data = img.get_fdata()
    data = np.clip(data, -175, 250)
    data = (data + 175) / 425
    aff = img.affine
    dz = float(np.linalg.norm(aff[:3, 2]))
    dy = float(np.linalg.norm(aff[:3, 1]))
    dx = float(np.linalg.norm(aff[:3, 0]))
    return data, (dz, dy, dx)


def bb_from_mask(mask):
    z_idx = np.any(mask, axis=(1,2))
    y_idx = np.any(mask, axis=(0,2))
    x_idx = np.any(mask, axis=(0,1))
    if not z_idx.any(): return None
    zr = np.where(z_idx)[0]; yr = np.where(y_idx)[0]; xr = np.where(x_idx)[0]
    zm = max(1,int((zr[-1]-zr[0])*EDGE_MARGIN))
    ym = max(1,int((yr[-1]-yr[0])*EDGE_MARGIN))
    xm = max(1,int((xr[-1]-xr[0])*EDGE_MARGIN))
    D,H,W = mask.shape
    return (max(0,zr[0]-zm), min(D-1,zr[-1]+zm),
            max(0,yr[0]-ym), min(H-1,yr[-1]+ym),
            max(0,xr[0]-xm), min(W-1,xr[-1]+xm))


def bb_from_body(ct_data):
    """从CT找出人体范围(软组织以上HU值的区域)"""
    body = ct_data > 0.05  # HU > -150 roughly
    z_idx = np.any(body, axis=(1,2))
    y_idx = np.any(body, axis=(0,2))
    x_idx = np.any(body, axis=(0,1))
    if not z_idx.any(): return None
    zr = np.where(z_idx)[0]; yr = np.where(y_idx)[0]; xr = np.where(x_idx)[0]
    D,H,W = ct_data.shape
    # Small margin for body context
    margin = 5
    return (max(0,zr[0]-margin), min(D-1,zr[-1]+margin),
            max(0,yr[0]-margin), min(H-1,yr[-1]+margin),
            max(0,xr[0]-margin), min(W-1,xr[-1]+margin))


def render_frame(img_2d, title, xlabel, ylabel, aspect):
    """img_2d: (rows, cols) — rows=vertical on screen"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_2d, cmap='gray', vmin=0, vmax=1, aspect=aspect)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(fig)
    return frame


def render_overlay(ct_2d, mask_2d, title, xlabel, ylabel, aspect):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(ct_2d, cmap='gray', vmin=0, vmax=1, aspect=aspect)
    ov = np.ma.masked_where(mask_2d == 0, mask_2d)
    ax.imshow(ov, cmap='Reds', alpha=0.5, vmin=0, vmax=1, aspect=aspect)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(fig)
    return frame


def make_video(mask_data, ct_data, spacing, output_path, title, video_type):
    dz, dy, dx = spacing

    # Choose bounding box:
    #   mask video  = tight tumor crop
    #   CT & overlay = same zoomed region around tumor (for direct comparison)
    if video_type == 'mask':
        bb = bb_from_mask(mask_data)
    elif video_type in ('ct', 'overlay'):
        mb = bb_from_mask(mask_data)
        if mb is None:
            bb = bb_from_body(ct_data) or (0, mask_data.shape[0]-1, 0, mask_data.shape[1]-1, 0, mask_data.shape[2]-1)
        else:
            mz0, mz1, my0, my1, mx0, mx1 = mb

            # Compute tumor center (voxel coords)
            tumor_center_z = (mz0 + mz1) / 2.0
            tumor_center_y = (my0 + my1) / 2.0
            tumor_center_x = (mx0 + mx1) / 2.0

            # Actual tumor volume in mm³
            tumor_vox_count = int(mask_data[mz0:mz1+1, my0:my1+1, mx0:mx1+1].sum())
            voxel_vol_mm3 = dz * dy * dx
            tumor_vol_mm3 = tumor_vox_count * voxel_vol_mm3

            # Adaptive display margin in mm (smaller tumor → tighter zoom)
            import math
            margin_mm = 15.0 + 55.0 * (1.0 - math.exp(-tumor_vol_mm3 / 3000.0))
            margin_mm = max(20.0, min(80.0, margin_mm))

            # Convert margin to voxels in each axis
            margin_z = int(margin_mm / dz)
            margin_y = int(margin_mm / dy)
            margin_x = int(margin_mm / dx)

            D, H, W = mask_data.shape
            bb = (max(0, int(tumor_center_z) - margin_z),
                  min(D-1, int(tumor_center_z) + margin_z),
                  max(0, int(tumor_center_y) - margin_y),
                  min(H-1, int(tumor_center_y) + margin_y),
                  max(0, int(tumor_center_x) - margin_x),
                  min(W-1, int(tumor_center_x) + margin_x))

    if bb is None: print("  WARNING: empty"); return False
    z_lo, z_hi, y_lo, y_hi, x_lo, x_hi = bb

    frames = []

    # === Axial: XY plane, sweep Z ===
    # Anterior=TOP (flip Y), Patient Right=LEFT (flip X)
    for z in np.linspace(z_lo, z_hi, SLICES_PER_AXIS).astype(int):
        if video_type == 'mask':
            img = mask_data[z, y_lo:y_hi+1, x_lo:x_hi+1]
        elif video_type == 'ct':
            img = ct_data[z, y_lo:y_hi+1, x_lo:x_hi+1]
        else:
            img = ct_data[z, y_lo:y_hi+1, x_lo:x_hi+1]
        # Flip Y so Anterior=TOP, flip X so Right=LEFT
        img_disp = np.flipud(np.fliplr(img))
        if video_type == 'overlay':
            m = np.flipud(np.fliplr(mask_data[z, y_lo:y_hi+1, x_lo:x_hi+1]))
            frames.append(render_overlay(img_disp, m,
                         f'{title}  Axial Z={z}', 'R ← → L', 'A ← → P', dy/dx))
        else:
            frames.append(render_frame(img_disp,
                         f'{title}  Axial Z={z}', 'R ← → L', 'A ← → P', dy/dx))

    # === Coronal: XZ plane, sweep Y ===
    # Head=TOP (Z ascending), Patient Right=LEFT (flip X)
    # data[:, y, x] → (Z, X), need Z vertical and X horizontal
    for y in np.linspace(y_lo, y_hi, SLICES_PER_AXIS).astype(int):
        if video_type == 'mask':
            img = mask_data[z_lo:z_hi+1, y, x_lo:x_hi+1]
        elif video_type == 'ct':
            img = ct_data[z_lo:z_hi+1, y, x_lo:x_hi+1]
        else:
            img = ct_data[z_lo:z_hi+1, y, x_lo:x_hi+1]
        # img is (Z, X). Need Z as rows (vertical), X as cols (horizontal)
        # Rows=Z: Z=z_lo(feet) at row0, Z=z_hi(head) at row-1
        # Flip rows so Head=TOP: np.flipud
        # Flip cols so Right=LEFT: np.fliplr
        img_disp = np.flipud(np.fliplr(img))
        if video_type == 'overlay':
            m = np.flipud(np.fliplr(mask_data[z_lo:z_hi+1, y, x_lo:x_hi+1]))
            frames.append(render_overlay(img_disp, m,
                         f'{title}  Coronal Y={y}', 'R ← → L', 'S ← Z → I', dx/dz))
        else:
            frames.append(render_frame(img_disp,
                         f'{title}  Coronal Y={y}', 'R ← → L', 'S ← Z → I', dx/dz))

    # === Sagittal: YZ plane, sweep X ===
    # Head=TOP (Z ascending), Anterior=LEFT
    # data[:, :, x] → (Z, Y). Need Z vertical, Y horizontal
    for x in np.linspace(x_lo, x_hi, SLICES_PER_AXIS).astype(int):
        if video_type == 'mask':
            img = mask_data[z_lo:z_hi+1, y_lo:y_hi+1, x]
        elif video_type == 'ct':
            img = ct_data[z_lo:z_hi+1, y_lo:y_hi+1, x]
        else:
            img = ct_data[z_lo:z_hi+1, y_lo:y_hi+1, x]
        # img is (Z, Y). Need Z as rows (vertical=Head at top), Y as cols (horizontal)
        # Row0=Z=z_lo(feet), Row-1=Z=z_hi(head). Flip rows → Head=TOP
        # Col0=Y=y_lo(Posterior), Col-1=Y=y_hi(Anterior). Anterior should be LEFT
        #   → need to NOT flip cols (Posterior=RIGHT, Anterior=LEFT)
        img_disp = np.flipud(img)  # only flip rows for Head=TOP, don't flip cols
        if video_type == 'overlay':
            m = np.flipud(mask_data[z_lo:z_hi+1, y_lo:y_hi+1, x])
            frames.append(render_overlay(img_disp, m,
                         f'{title}  Sagittal X={x}', 'A ← Y → P', 'S ← Z → I', dy/dz))
        else:
            frames.append(render_frame(img_disp,
                         f'{title}  Sagittal X={x}', 'A ← Y → P', 'S ← Z → I', dy/dz))

    # Resize all frames to same size
    resized = []
    for f in frames:
        img = Image.fromarray(f).resize(TARGET_SIZE, Image.LANCZOS)
        resized.append(np.array(img))

    writer = imageio.get_writer(output_path, fps=FPS, codec='libx264', quality=8)
    for f in resized: writer.append_data(f)
    writer.close()
    return True


def main():
    print("=" * 60)
    print("Tumor Mask Video Generator")
    print("=" * 60)
    total = 0

    for organ in ORGANS:
        organ_dir = os.path.join(OUTPUT_BASE, organ)
        if not os.path.isdir(organ_dir): continue

        best_path, info = pick_best_mask(organ_dir)
        if best_path is None: continue
        mask_data = info['data']
        base = info['filename'].replace('.nii.gz', '')
        print(f"\n{organ}: {info['filename']} ({info['volume']:,} voxels)")

        ct_path, ct_id = find_ct(info['filename'])
        if ct_path is None: continue
        ct_data, spacing = load_ct(ct_path)
        print(f"  CT: {ct_id} ({ct_data.shape}) spacing={spacing}")

        vdir = os.path.join(organ_dir, 'video')
        os.makedirs(vdir, exist_ok=True)

        for i, (suffix, vtype, use_ct) in enumerate([
            ('_mask.mp4', 'mask', False),
            ('_ct.mp4', 'ct', True),
            ('_overlay.mp4', 'overlay', True)
        ]):
            print(f"  [{i+1}/3] {vtype}...")
            p = os.path.join(vdir, base + suffix)
            ct_for_video = ct_data if use_ct else np.zeros_like(mask_data)
            if make_video(mask_data, ct_for_video, spacing, p,
                         f'{organ} {vtype.title()}', vtype):
                sz = os.path.getsize(p) // 1024
                print(f"    -> {os.path.basename(p)} ({sz} KB)")
                total += 1

    print(f"\n{'='*60}")
    print(f"Done: {total} videos")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
