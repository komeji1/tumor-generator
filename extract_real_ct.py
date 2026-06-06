"""
提取真实 CT 数据并与合成器官 mask 组合

用法: python extract_real_ct.py [n_scans]
默认提取前 30 个 CT 扫描到 data/ct/
"""

import os, sys, tarfile, shutil, json
import numpy as np
import nibabel as nib

MASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MASK_DIR, 'data')
CT_DIR = os.path.join(DATA_DIR, 'ct')
LABEL_DIR = os.path.join(DATA_DIR, 'organ_labels')

TAR_PATH = os.path.join(DATA_DIR, 'tmp', 'AbdomenAtlas2.0Mini_ct_00000001_00000500.tar.gz')

def main():
    n_scans = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    if not os.path.exists(TAR_PATH):
        print(f"CT archive not found: {TAR_PATH}")
        print("Please wait for download to complete.")
        sys.exit(1)

    # ── 列出 tar 中前 N 个目录名 ───────────────────
    print(f"Reading archive: {TAR_PATH}")
    with tarfile.open(TAR_PATH, 'r:gz') as tf:
        members = tf.getmembers()

    # 获取唯一的 CT 目录
    ct_dirs = sorted(set(
        m.name.split('/')[0] for m in members
        if '/' in m.name and not m.name.startswith('.')
    ))
    print(f"Total CT scans in archive: {len(ct_dirs)}")
    selected = ct_dirs[:n_scans]
    print(f"Extracting {len(selected)} scans...")

    # ── 只提取前 N 个 ──────────────────────────────
    os.makedirs(CT_DIR, exist_ok=True)
    extracted = 0
    with tarfile.open(TAR_PATH, 'r:gz') as tf:
        for d in selected:
            dst = os.path.join(CT_DIR, d)
            if os.path.exists(dst):
                extracted += 1
                continue

            # 提取该目录下所有文件
            dir_members = [m for m in members if m.name.startswith(d + '/')]
            for m in dir_members:
                tf.extract(m, CT_DIR)

            # tar 会创建 CT_DIR/d/...，需要移动到 CT_DIR/d/
            src = os.path.join(CT_DIR, d)
            extracted += 1

            if extracted % 5 == 0:
                print(f"  [{extracted}/{len(selected)}] {d}")

    print(f"Extracted: {extracted} CT scans → {CT_DIR}")

    # ── 为每个 CT 生成对应的器官 mask ──────────────
    print(f"\nGenerating organ masks for real CTs...")
    sys.path.insert(0, os.path.join(MASK_DIR, 'Step1', 'src'))
    from utils import compute_ellipsoid_dist

    # 器官定义 — 根据真实 CT 调整位置
    # 先用第一个 CT 推断体积范围
    first_ct_dir = os.path.join(CT_DIR, selected[0])
    first_ct_path = None
    for f in os.listdir(first_ct_dir):
        if f.endswith('.nii.gz'):
            first_ct_path = os.path.join(first_ct_dir, f)
            break

    if first_ct_path:
        img = nib.load(first_ct_path)
        ct_shape = img.get_fdata().shape
        print(f"  CT shape: {ct_shape}")
    else:
        ct_shape = (128, 256, 256)
        print(f"  CT shape (default): {ct_shape}")

    D, H, W = ct_shape

    # 动态调整器官位置到 CT 中合理位置
    ORGAN_DEFS = {
        'liver':       {'center': (int(D*0.50), int(H*0.40), int(W*0.45)),
                        'radii': (int(D*0.18), int(H*0.20), int(W*0.18)),
                        'hu_range': (30, 80)},
        'pancreas':    {'center': (int(D*0.48), int(H*0.55), int(W*0.40)),
                        'radii': (int(D*0.08), int(H*0.12), int(W*0.06)),
                        'hu_range': (20, 60)},
        'kidney_left': {'center': (int(D*0.42), int(H*0.32), int(W*0.38)),
                        'radii': (int(D*0.10), int(H*0.08), int(W*0.07)),
                        'hu_range': (30, 90)},
        'colon':       {'center': (int(D*0.38), int(H*0.42), int(W*0.42)),
                        'radii': (int(D*0.12), int(H*0.14), int(W*0.10)),
                        'hu_range': (-20, 40)},
        'esophagus':   {'center': (int(D*0.55), int(H*0.65), int(W*0.50)),
                        'radii': (int(D*0.16), int(H*0.06), int(W*0.05)),
                        'hu_range': (10, 50)},
        'uterus':      {'center': (int(D*0.30), int(H*0.25), int(W*0.40)),
                        'radii': (int(D*0.08), int(H*0.10), int(W*0.09)),
                        'hu_range': (20, 70)},
    }

    rng = np.random.default_rng(42)

    for sample_dir in selected:
        sid = sample_dir
        label_sample_dir = os.path.join(LABEL_DIR, sid, 'segmentations')
        os.makedirs(label_sample_dir, exist_ok=True)

        for organ_name, cfg in ORGAN_DEFS.items():
            cz, cy, cx = cfg['center']
            rz, ry, rx = cfg['radii']

            # 加微弱扰动
            rz2 = rz * rng.uniform(0.95, 1.05)
            ry2 = ry * rng.uniform(0.95, 1.05)
            rx2 = rx * rng.uniform(0.95, 1.05)

            dist = compute_ellipsoid_dist(ct_shape, (cz, cy, cx), (rz2, ry2, rx2))
            mask = (dist <= 1.0).astype(np.uint8)

            mask_path = os.path.join(label_sample_dir, f'{organ_name}.nii.gz')
            nib.save(nib.Nifti1Image(mask, img.affine if first_ct_path else np.eye(4)), mask_path)

    print(f"  Organ masks generated for {len(selected)} scans")

    # ── 生成 manifest ──────────────────────────────
    sys.path.insert(0, os.path.join(MASK_DIR, 'Step2', 'src'))
    from data_loader import build_manifest, save_manifest_csv

    organ_cfg = [
        {'name': 'liver_lesion', 'organ_label_file': 'liver.nii.gz'},
        {'name': 'pancreatic_lesion', 'organ_label_file': 'pancreas.nii.gz'},
        {'name': 'kidney_lesion', 'organ_label_file': 'kidney_left.nii.gz'},
        {'name': 'colon_lesion', 'organ_label_file': 'colon.nii.gz'},
        {'name': 'esophagus_tumor', 'organ_label_file': 'esophagus.nii.gz'},
        {'name': 'endometrioma_tumor', 'organ_label_file': 'uterus.nii.gz'},
    ]

    manifest = build_manifest(CT_DIR, LABEL_DIR, organ_cfg)
    manifest_path = os.path.join(DATA_DIR, 'manifest.csv')
    save_manifest_csv(manifest, manifest_path)

    existing = sum(1 for m in manifest if m['exists'])
    print(f"\n{'='*60}")
    print(f"Done")
    print(f"{'='*60}")
    print(f"  Real CTs:     {CT_DIR} ({extracted} scans)")
    print(f"  Organ masks:  {LABEL_DIR} (synthetic, organ-matched)")
    print(f"  Manifest:     {manifest_path} ({existing}/{len(manifest)} available)")
    print(f"\nRun pipeline:")
    print(f"  python Step6/src/main.py --seed 42")


if __name__ == "__main__":
    main()
