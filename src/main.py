"""
Step 2.1: 批量生成入口 (方案A — 96³ patch 输出)

用法:
  python main.py --organ liver_lesion --mask-index 0 --phase early

输出:
  Tumor/output/synthetic_ct/{organ}/{organ}_s{idx}__{ct_id}.nii.gz  (96³, 1mm³)
  Tumor/output/tumor_labels/{organ}/{organ}_s{idx}__{ct_id}.nii.gz  (96³, 1mm³)
  Tumor/output/metadata/{organ}_s{idx}__{ct_id}.json
"""

import sys, os, argparse, time, json, glob, warnings
import torch, numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(__file__))
from ct_preprocessor import CTPreprocessor
from condition_builder import ConditionBuilder
from diffusion_engine import DiffusionEngine
from texture_blender import TextureBlender, patch_to_hu

warnings.filterwarnings("ignore")

MASK_ROOT  = r"C:\Users\33067\.claude\work\Mask\output\real_ct"
CT_ROOT    = r"C:\Users\33067\.claude\work\Mask\data\ct"
LABEL_ROOT = r"C:\Users\33067\.claude\work\Mask\data\organ_labels"
CKPT_ROOT  = r"C:\Users\33067\.claude\work\Tumor\checkpoints"
OUT_ROOT   = r"C:\Users\33067\.claude\work\Tumor\output"

# 器官名标准化
ORGAN_SHORT_MAP = {
    "liver_lesion": "liver", "pancreatic_lesion": "pancreas",
    "kidney_lesion": "kidney", "colon_lesion": "colon",
    "esophagus_tumor": "esophagus", "endometrioma_tumor": "uterus",
}
# 零样本器官 (无 noearly 权重, 始终用 early)
ZERO_SHOT_ORGANS = {"colon", "esophagus", "uterus"}


def organ_short(organ_type: str) -> str:
    return ORGAN_SHORT_MAP.get(organ_type, organ_type)


# 器官→标签文件映射
ORGAN_LABEL_MAP = {
    "liver_lesion":        "liver.nii.gz",
    "pancreatic_lesion":   "pancreas.nii.gz",
    "kidney_lesion":       "kidney_left.nii.gz",
    "colon_lesion":        "colon.nii.gz",
    "esophagus_tumor":     "esophagus.nii.gz",
    "endometrioma_tumor":  "uterus.nii.gz",
}


def generate_one(organ_type: str, mask_path: str, device: str = "cpu"):
    """单次生成: mask → 96³ 合成CT patch"""

    # 解析 mask 文件名: {organ}_t{idx}__{ct_id}.nii.gz
    basename = os.path.basename(mask_path).replace(".nii.gz", "")
    ct_id = basename.split("__")[1]
    out_id = re.sub(r'_t(\d)', r'_s\1', basename)  # _t00→_s00, won't corrupt "tumor"

    ct_path    = os.path.join(CT_ROOT, ct_id, "ct.nii.gz")
    label_file = ORGAN_LABEL_MAP.get(organ_type)
    organ_path = os.path.join(LABEL_ROOT, ct_id, "segmentations", label_file)

    if not os.path.exists(organ_path):
        # fallback: kidney_left → kidney_right
        if "kidney_left" in label_file:
            organ_path = os.path.join(LABEL_ROOT, ct_id, "segmentations", "kidney_right.nii.gz")
        if not os.path.exists(organ_path):
            raise FileNotFoundError(f"Organ mask not found: {organ_path}")

    # 加载原始CT获取 spacing → 确定物理裁剪范围
    orig_nii = nib.load(ct_path)
    spacing = np.array(orig_nii.header.get_zooms()[:3])
    orig_ct = orig_nii.get_fdata().astype(np.float32)
    affine = orig_nii.affine.copy()
    tumor_full = (nib.load(mask_path).get_fdata() > 0)
    if tumor_full.sum() < 10:
        raise ValueError(f"Mask too small ({int(tumor_full.sum())} voxels), skip")

    # 找到肿瘤质心, 在原生空间裁剪 ≥48mm 物理范围 → 保证 1mm³ 下 ≥96³
    tumor_idx = np.argwhere(tumor_full)
    ctr = tumor_idx.mean(axis=0).astype(int)
    half_phys = 48.0  # mm
    half = [int(np.ceil(half_phys / s)) for s in spacing]
    hw, hh, hd = half[0], half[1], half[2]
    x0, x1 = max(0, ctr[0]-hw), min(orig_ct.shape[0], ctr[0]+hw)
    y0, y1 = max(0, ctr[1]-hh), min(orig_ct.shape[1], ctr[1]+hh)
    z0, z1 = max(0, ctr[2]-hd), min(orig_ct.shape[2], ctr[2]+hd)

    ct_crop  = orig_ct[x0:x1, y0:y1, z0:z1].copy()
    tm_crop  = tumor_full[x0:x1, y0:y1, z0:z1].copy()
    og_crop  = (nib.load(organ_path).get_fdata() > 0)[x0:x1, y0:y1, z0:z1].copy()

    # 物理尺寸不足96mm时用零填充补足 (边界case: 小器官靠近CT边缘)
    need_phys = [96.0, 96.0, 96.0]
    for i, (s, need) in enumerate(zip(spacing, need_phys)):
        current = ct_crop.shape[i] * s
        if current < need:
            pad_voxels = int(np.ceil((need - current) / s))
            pad_width = [(0, 0), (0, 0), (0, 0)]
            pad_width[i] = (0, pad_voxels)
            ct_crop = np.pad(ct_crop, pad_width, mode='constant', constant_values=ct_crop.min())
            tm_crop = np.pad(tm_crop, pad_width, mode='constant', constant_values=0)
            og_crop = np.pad(og_crop, pad_width, mode='constant', constant_values=0)

    # 写临时文件 → CTPreprocessor 重采样到 1mm³
    import tempfile
    tmp = tempfile.gettempdir()
    for name, arr in [("ct", ct_crop), ("org", og_crop.astype(np.uint8)), ("tm", tm_crop.astype(np.uint8))]:
        nib.save(nib.Nifti1Image(arr, affine), os.path.join(tmp, f"_e2e_{name}.nii.gz"))

    pre = CTPreprocessor(device)
    r = pre.process(
        os.path.join(tmp, "_e2e_ct.nii.gz"),
        os.path.join(tmp, "_e2e_org.nii.gz"),
        os.path.join(tmp, "_e2e_tm.nii.gz"),
        organ_short(organ_type),
    )

    # 中心裁剪到 96³
    ct_t  = r.ct_tensor
    tm_t  = r.tumor_mask_tensor
    d, h, w = ct_t.shape[2:]
    ct_t = ct_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]
    tm_t = tm_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]

    # 确定尺寸类别 + 选择权重
    from utils import compute_tumor_radius_voxel
    radius_mm = compute_tumor_radius_voxel(tm_crop, spacing)
    oshort = organ_short(organ_type)
    # 零样本器官强制 early (无 noearly 权重)
    phase = "early" if (radius_mm <= 10 or oshort in ZERO_SHOT_ORGANS) else "noearly"

    # 条件构造
    vqgan_ckpt = os.path.join(CKPT_ROOT, "AutoencoderModel", "AutoencoderModel.ckpt")
    builder = ConditionBuilder(vqgan_ckpt, device)
    cond = builder.build(ct_t, tm_t)

    # 扩散生成
    engine = DiffusionEngine(vqgan_ckpt, os.path.join(CKPT_ROOT, "DiffusionModel"),
                             oshort, phase, device)
    synthetic = engine.generate(cond)

    # Alpha 混合
    blender = TextureBlender(device)
    blended = blender.blend(ct_t, synthetic, tm_t, oshort)

    # 保存 96³ patch
    os.makedirs(os.path.join(OUT_ROOT, "synthetic_ct", organ_type), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "tumor_labels", organ_type), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "metadata"), exist_ok=True)

    ct_out = os.path.join(OUT_ROOT, "synthetic_ct", organ_type, f"{out_id}.nii.gz")
    mask_out = os.path.join(OUT_ROOT, "tumor_labels", organ_type, f"{out_id}.nii.gz")

    # 用 1mm³ 各向同性 affine 保存
    iso_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(patch_to_hu(blended), iso_affine), ct_out)
    nib.save(nib.Nifti1Image(tm_t.squeeze().cpu().numpy().astype(np.uint8), iso_affine), mask_out)

    meta = {
        "organ": organ_type, "ct_id": ct_id, "mask_source": mask_path,
        "crop_native": [int(x) for x in [x0, x1, y0, y1, z0, z1]],
        "radius_mm": round(float(radius_mm), 1), "phase": phase, "weight_used": engine.WEIGHT_MAP[oshort][0],
        "hu_organ": r.hu_stats,
    }
    meta_path = os.path.join(OUT_ROOT, "metadata", f"{out_id}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup temp
    for name in ["ct", "org", "tm"]:
        os.remove(os.path.join(tmp, f"_e2e_{name}.nii.gz"))

    return ct_out, mask_out, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ", default="liver_lesion")
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--phase", default=None)  # auto-detect
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(MASK_ROOT, args.organ, "*.nii.gz")))
    if not masks:
        print(f"No masks found for {args.organ}")
        sys.exit(1)

    mask_path = masks[args.mask_index % len(masks)]
    print(f"Generating: {mask_path}")
    t0 = time.time()
    ct_out, mask_out, meta = generate_one(args.organ, mask_path, args.device)
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"  CT:    {ct_out}")
    print(f"  Mask:  {mask_out}")
    print(f"  Meta:  {json.dumps(meta, indent=2)[:300]}")
