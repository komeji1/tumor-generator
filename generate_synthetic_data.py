"""
生成合成数据 — 供 Tumor Mask Generator 测试和验证使用

产出: data/ 目录下的 CT 扫描和器官分割 mask
    6 器官 × 每器官足够样本，总计可生成 120+ tumor masks

用法: python generate_synthetic_data.py
"""

import os, sys, json
import numpy as np
import nibabel as nib

# 项目根目录
MASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MASK_DIR, 'data')
CT_DIR = os.path.join(DATA_DIR, 'ct')
LABEL_DIR = os.path.join(DATA_DIR, 'organ_labels')

os.makedirs(CT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ============================================================
# 器官定义 — 每种器官的形状、位置、大小参数
# ============================================================
# (z_range, y_range, x_range) 在 80x128x128 体积中的位置
ORGAN_DEFS = {
    'liver': {
        'center': (64, 56, 56),
        'radii': (40, 52, 48),       # 肝: 很大
        'hu_range': (30, 80),
    },
    'pancreas': {
        'center': (66, 90, 64),
        'radii': (22, 36, 22),       # 胰腺: 放大版
        'hu_range': (20, 60),
    },
    'kidney_left': {
        'center': (58, 48, 60),
        'radii': (26, 22, 22),       # 左肾
        'hu_range': (30, 90),
    },
    'colon': {
        'center': (50, 56, 64),
        'radii': (28, 32, 24),       # 结肠
        'hu_range': (-20, 40),
    },
    'esophagus': {
        'center': (70, 110, 80),
        'radii': (36, 16, 16),       # 食管: 放大版
        'hu_range': (10, 50),
    },
    'uterus': {
        'center': (40, 40, 64),
        'radii': (22, 26, 26),       # 子宫: 放大版
        'hu_range': (20, 70),
    },
}

# ============================================================
# 生成 CT 体积 — 模拟腹部 CT HU 值分布
# ============================================================
def make_ct(shape=(80, 128, 128), spacing=(2.0, 1.0, 1.0), local_rng=None):
    """
    创建合成 CT 体积。
    每个器官的位置和大小有 ±15% 随机变异，模拟个体差异。
    """
    if local_rng is None:
        local_rng = np.random.default_rng()

    ct = np.full(shape, -100, dtype=np.int16)  # 背景 ≈ 脂肪

    for name, cfg in ORGAN_DEFS.items():
        # 每个样本的器官位置 ±10% 随机偏移
        cz0, cy0, cx0 = cfg['center']
        rz0, ry0, rx0 = cfg['radii']
        hu_lo, hu_hi = cfg['hu_range']

        # 位置随机偏移 (最多 ±15% 的半径范围)
        cz = cz0 + local_rng.uniform(-rz0 * 0.15, rz0 * 0.15)
        cy = cy0 + local_rng.uniform(-ry0 * 0.15, ry0 * 0.15)
        cx = cx0 + local_rng.uniform(-rx0 * 0.15, rx0 * 0.15)

        # 大小随机变异 (0.85 ~ 1.15)
        rz = rz0 * local_rng.uniform(0.85, 1.15)
        ry = ry0 * local_rng.uniform(0.85, 1.15)
        rx = rx0 * local_rng.uniform(0.85, 1.15)

        # HU 范围随机偏移 (每样本 ±10 HU)
        hu_shift = local_rng.integers(-10, 10)
        hu_l = hu_lo + hu_shift
        hu_h = hu_hi + hu_shift

        Z, Y, X = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt(
            ((Z - cz) / rz) ** 2 +
            ((Y - cy) / ry) ** 2 +
            ((X - cx) / rx) ** 2
        )
        organ_mask = dist <= 1.0
        hu_vals = local_rng.integers(hu_l, hu_h, size=int(organ_mask.sum()), dtype=np.int16)
        ct[organ_mask] = hu_vals

    # 全局噪声 (每样本独立)
    noise = local_rng.integers(-8, 8, size=shape, dtype=np.int16)
    ct = ct + noise
    ct = np.clip(ct, -175, 250)

    # 构建 affine
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x
    affine[1, 1] = spacing[1]  # y
    affine[2, 2] = spacing[0]  # z

    return ct.astype(np.int16), affine


# ============================================================
# 生成器官分割 mask
# ============================================================
def make_organ_mask(organ_name, shape=(80, 128, 128), local_rng=None):
    """
    为指定器官创建二值分割 mask。
    每个样本的器官位置和大小有 ±15% 随机变异。
    """
    if local_rng is None:
        local_rng = np.random.default_rng()

    cfg = ORGAN_DEFS[organ_name]
    cz0, cy0, cx0 = cfg['center']
    rz0, ry0, rx0 = cfg['radii']

    # 位置随机偏移 (最多 ±15% 的半径范围)
    cz = cz0 + local_rng.uniform(-rz0 * 0.15, rz0 * 0.15)
    cy = cy0 + local_rng.uniform(-ry0 * 0.15, ry0 * 0.15)
    cx = cx0 + local_rng.uniform(-rx0 * 0.15, rx0 * 0.15)

    # 大小随机变异 (0.85 ~ 1.15)
    rz = rz0 * local_rng.uniform(0.85, 1.15)
    ry = ry0 * local_rng.uniform(0.85, 1.15)
    rx = rx0 * local_rng.uniform(0.85, 1.15)

    Z, Y, X = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        ((Z - cz) / rz) ** 2 +
        ((Y - cy) / ry) ** 2 +
        ((X - cx) / rx) ** 2
    )
    mask = (dist <= 1.0).astype(np.uint8)

    return mask


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("生成合成数据")
    print("=" * 60)

    n_samples = 30  # 每个器官 30 个样本
    shape = (128, 160, 160)  # 大体积，确保器官经得起 margin 腐蚀
    spacing = (1.0, 1.0, 1.0)  # 各向同性

    organ_list = list(ORGAN_DEFS.keys())
    print(f"\n器官: {organ_list}")
    print(f"每器官样本数: {n_samples}")
    print(f"CT 体积: {shape}")

    for sample_id in range(n_samples):
        sid = f"SYNTH_{sample_id:03d}"
        rng_i = np.random.default_rng(42 + sample_id)

        # ── 创建 CT (每样本独立随机) ──────────────
        ct, affine = make_ct(shape, spacing, local_rng=rng_i)

        ct_sample_dir = os.path.join(CT_DIR, sid)
        os.makedirs(ct_sample_dir, exist_ok=True)
        ct_path = os.path.join(ct_sample_dir, 'ct.nii.gz')
        nib.save(nib.Nifti1Image(ct, affine), ct_path)

        # ── 创建器官分割 ─────────────────────────
        label_sample_dir = os.path.join(LABEL_DIR, sid, 'segmentations')
        os.makedirs(label_sample_dir, exist_ok=True)

        for organ_name in organ_list:
            mask = make_organ_mask(organ_name, shape, local_rng=rng_i)
            mask_path = os.path.join(label_sample_dir, f'{organ_name}.nii.gz')
            nib.save(nib.Nifti1Image(mask, affine), mask_path)

        if sample_id % 10 == 0:
            print(f"  [{sample_id:3d}/{n_samples}] {sid}: CT + {len(organ_list)} organ masks")

    # ── 生成 manifest.csv ────────────────────────
    sys.path.insert(0, os.path.join(MASK_DIR, 'Step1', 'src'))
    sys.path.insert(0, os.path.join(MASK_DIR, 'Step2', 'src'))
    from data_loader import build_manifest, save_manifest_csv

    # 构建轻量 config
    organ_cfg = [
        {'name': 'liver_lesion', 'organ_label_file': 'liver.nii.gz', 'count': 20},
        {'name': 'pancreatic_lesion', 'organ_label_file': 'pancreas.nii.gz', 'count': 20},
        {'name': 'kidney_lesion', 'organ_label_file': 'kidney_left.nii.gz', 'count': 20},
        {'name': 'colon_lesion', 'organ_label_file': 'colon.nii.gz', 'count': 20},
        {'name': 'esophagus_tumor', 'organ_label_file': 'esophagus.nii.gz', 'count': 20},
        {'name': 'endometrioma_tumor', 'organ_label_file': 'uterus.nii.gz', 'count': 20},
    ]

    manifest = build_manifest(CT_DIR, LABEL_DIR, organ_cfg)
    manifest_path = os.path.join(DATA_DIR, 'manifest.csv')
    save_manifest_csv(manifest, manifest_path)

    existing = sum(1 for m in manifest if m['exists'])
    print(f"\n{'='*60}")
    print(f"数据生成完成")
    print(f"{'='*60}")
    print(f"  CT:        {CT_DIR} ({n_samples} 个扫描)")
    print(f"  Organ Labels: {LABEL_DIR}")
    print(f"  Manifest:  {manifest_path} ({existing}/{len(manifest)} 可用)")
    print(f"\n现在可以运行: python Step6/src/main.py")

    # ── 更新 config ──────────────────────────────
    config_src = os.path.join(MASK_DIR, 'Step0', 'config', 'generation_config.json')
    with open(config_src, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 更新数据路径
    config['data']['ct_dir'] = 'data/ct/'
    config['data']['organ_label_dir'] = 'data/organ_labels/'
    config['data']['manifest_path'] = 'data/manifest.csv'

    # 更新器官标签文件名
    label_name_map = {
        'liver_lesion': 'liver.nii.gz',
        'pancreatic_lesion': 'pancreas.nii.gz',
        'kidney_lesion': 'kidney_left.nii.gz',
        'colon_lesion': 'colon.nii.gz',
        'esophagus_tumor': 'esophagus.nii.gz',
        'endometrioma_tumor': 'uterus.nii.gz',
    }
    for o in config['organs']:
        if o['name'] in label_name_map:
            o['organ_label_file'] = label_name_map[o['name']]

    # 调小 margin 以适应合成数据
    config['placement']['margin']['feather_mm'] = 2
    config['placement']['margin']['safety_mm'] = 3
    config['shape']['gaussian_filter']['sigma_mm'] = 0.5

    with open(config_src, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n配置已更新: {config_src}")
    print(f"  - data.ct_dir → data/ct/")
    print(f"  - data.organ_label_dir → data/organ_labels/")
    print(f"  - organ_label_file 映射已更新")


if __name__ == "__main__":
    main()
