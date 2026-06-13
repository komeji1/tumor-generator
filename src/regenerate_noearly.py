"""
Quick script: regenerate pancreas, kidney, liver full-CT using noearly weights.
Paper: mid-late tumors (r>10mm) use T=200 DDIM S=50 → larger, more visible lesions.
"""
import sys, os, time, numpy as np, nibabel as nib, SimpleITK as sitk, torch
sys.path.insert(0, os.path.dirname(__file__))
from ct_preprocessor import CTPreprocessor
from condition_builder import ConditionBuilder
from diffusion_engine import DiffusionEngine
from texture_blender import TextureBlender, patch_to_hu

CKPT_ROOT = r"C:\Users\33067\.claude\work\Tumor\checkpoints"
OUT_ROOT  = r"C:\Users\33067\.claude\work\Tumor\output"
MASK_ROOT  = r"C:\Users\33067\.claude\work\Mask"

# Organ → (bdmap_id, ct_dir, organ_mask_rel, tumor_mask_rel, tumor_file)
# kidney uses combined left+right organ mask
TASKS = [
    ("pancreas", "pancreatic_lesion", "BDMAP_00000019", "pancreas.nii.gz",
     "pancreatic_lesion_t00__BDMAP_00000019.nii.gz"),
    ("kidney",   "kidney_lesion",     "BDMAP_00000019", None,  # combined L+R
     "kidney_lesion_t00__BDMAP_00000019.nii.gz"),
    ("liver",    "liver_lesion",      "BDMAP_00000012", "liver.nii.gz",
     "liver_lesion_t00__BDMAP_00000012.nii.gz"),
]


def generate_one_noearly(organ_short, organ_type, bdmap_id, organ_file, tumor_file, device="cpu"):
    """Run full-CT embedding with noearly weights."""
    ct_path = os.path.join(MASK_ROOT, "data", "ct", bdmap_id, "ct.nii.gz")
    tm_path = os.path.join(MASK_ROOT, "output", "real_ct", organ_type, tumor_file)

    # Organ mask: either specific file or combined kidney_left+kidney_right
    if organ_file:
        og_path = os.path.join(MASK_ROOT, "data", "organ_labels", bdmap_id,
                               "segmentations", organ_file)
    else:
        # Combine kidney_left + kidney_right
        kl = os.path.join(MASK_ROOT, "data", "organ_labels", bdmap_id,
                         "segmentations", "kidney_left.nii.gz")
        kr = os.path.join(MASK_ROOT, "data", "organ_labels", bdmap_id,
                         "segmentations", "kidney_right.nii.gz")
        kl_data = nib.load(kl).get_fdata() > 0
        kr_data = nib.load(kr).get_fdata() > 0
        combined = (kl_data | kr_data).astype(np.uint8)
        # Save temp combined organ mask
        og_path = os.path.join(r"D:\Users\33067\claude-data\downloads\_tmp",
                               f"kidney_combined_{bdmap_id}.nii.gz")
        ref_nii = nib.load(kl)
        nib.save(nib.Nifti1Image(combined, ref_nii.affine, ref_nii.header), og_path)

    print(f"  CT:       {ct_path}")
    print(f"  Organ:    {og_path}")
    print(f"  Tumor:    {tm_path}")

    t0 = time.time()

    # -- Load originals --
    orig_nii = nib.load(ct_path)
    full_ct = orig_nii.get_fdata().astype(np.float32)
    spacing = np.array(orig_nii.header.get_zooms()[:3])
    affine  = orig_nii.affine.copy()

    tm_data = nib.load(tm_path).get_fdata() > 0
    og_data = nib.load(og_path).get_fdata() > 0
    print(f"  Tumor voxels: {tm_data.sum():,}")

    # -- Crop tumor region in native space --
    t_idx = np.argwhere(tm_data)
    ctr = t_idx.mean(axis=0).astype(int)
    half_phys = 48.0
    half = [int(np.ceil(half_phys / s)) for s in spacing]
    x0, x1 = max(0, ctr[0]-half[0]), min(full_ct.shape[0], ctr[0]+half[0])
    y0, y1 = max(0, ctr[1]-half[1]), min(full_ct.shape[1], ctr[1]+half[1])
    z0, z1 = max(0, ctr[2]-half[2]), min(full_ct.shape[2], ctr[2]+half[2])

    ct_crop = full_ct[x0:x1, y0:y1, z0:z1].copy()
    tm_crop = tm_data[x0:x1, y0:y1, z0:z1].copy()
    og_crop = og_data[x0:x1, y0:y1, z0:z1].copy()

    # -- Temp files → preprocessor → 1mm^3 → 96^3 --
    tmp = r"D:\Users\33067\claude-data\downloads\_tmp"
    os.makedirs(tmp, exist_ok=True)
    real_aff = np.diag(list(spacing) + [1.0])
    for name, arr in [("ct", ct_crop), ("org", og_crop.astype(np.int16)),
                       ("tm", tm_crop.astype(np.int16))]:
        nib.save(nib.Nifti1Image(arr.astype(np.float32), real_aff),
                 os.path.join(tmp, f"noearly_{name}.nii.gz"))

    pre = CTPreprocessor(device)
    r = pre.process(os.path.join(tmp, "noearly_ct.nii.gz"),
                    os.path.join(tmp, "noearly_org.nii.gz"),
                    os.path.join(tmp, "noearly_tm.nii.gz"), organ_short)
    ct_t = r.ct_tensor; tm_t = r.tumor_mask_tensor
    d, h, w = ct_t.shape[2:]
    ct_t = ct_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]
    tm_t = tm_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]

    # -- Diffusion: NOEARLY weights --
    vqgan_ckpt = os.path.join(CKPT_ROOT, "AutoencoderModel", "AutoencoderModel.ckpt")
    diff_dir   = os.path.join(CKPT_ROOT, "DiffusionModel")
    builder = ConditionBuilder(vqgan_ckpt, device)
    cond = builder.build(ct_t, tm_t)
    engine = DiffusionEngine(vqgan_ckpt, diff_dir, organ_short, "noearly", device)
    synthetic = engine.generate(cond)

    # -- Blend --
    blender = TextureBlender(device)
    blended = blender.blend(ct_t, synthetic, tm_t, organ_short, random_sigma=False)
    blended_hu = patch_to_hu(blended)

    # -- Resample back to native crop space --
    blended_sitk = sitk.GetImageFromArray(blended_hu.transpose(2, 1, 0))
    blended_sitk.SetSpacing((1.0, 1.0, 1.0))
    native_crop_sitk = sitk.GetImageFromArray(ct_crop.transpose(2, 1, 0))
    native_crop_sitk.SetSpacing([float(s) for s in spacing])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(native_crop_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    blended_native = resampler.Execute(blended_sitk)
    blended_native_arr = sitk.GetArrayFromImage(blended_native).transpose(2, 1, 0)

    # -- Soft organ-boundary blending --
    from scipy.ndimage import gaussian_filter
    final_ct = full_ct.copy()
    tumor_edge = gaussian_filter(tm_crop.astype(np.float32), sigma=8.0)
    organ_edge = gaussian_filter(og_crop.astype(np.float32), sigma=10.0)
    edge = tumor_edge * organ_edge
    edge = np.clip(edge, 0, 1)
    native_region = final_ct[x0:x1, y0:y1, z0:z1]
    final_ct[x0:x1, y0:y1, z0:z1] = (1 - edge) * native_region + edge * blended_native_arr

    full_mask = np.zeros_like(full_ct, dtype=np.uint8)
    full_mask[x0:x1, y0:y1, z0:z1] = tm_crop & og_crop

    # -- Save --
    out_dir = os.path.join(OUT_ROOT, "full_ct", organ_type)
    os.makedirs(out_dir, exist_ok=True)
    base = f"{organ_type}_n00__{bdmap_id}"
    nib.save(nib.Nifti1Image(final_ct.astype(np.float32), affine),
             os.path.join(out_dir, f"{base}.nii.gz"))
    nib.save(nib.Nifti1Image(full_mask.astype(np.uint8), affine),
             os.path.join(out_dir, f"{base}_mask.nii.gz"))

    # -- Verify --
    orig_data = orig_nii.get_fdata()
    t_hu = final_ct[full_mask > 0]
    o_hu = orig_data[full_mask > 0]
    nt = ~(full_mask > 0)
    nt_diff = np.abs(final_ct[nt] - orig_data[nt])

    dt = time.time() - t0
    diff_pct = np.mean(np.abs(t_hu - o_hu) > 20) * 100
    print(f"  Done in {dt:.0f}s")
    print(f"  Mask: {int(full_mask.sum()):,}/{int(tm_data.sum()):,}  "
          f"HU: synthetic={t_hu.mean():.0f}+/-{t_hu.std():.0f}  "
          f"orig={o_hu.mean():.0f}+/-{o_hu.std():.0f}")
    print(f"  |diff|={np.abs(t_hu-o_hu).mean():.0f}HU  >20HU={diff_pct:.0f}%  "
          f"non-tumor max={nt_diff.max():.1f}HU")
    return out_dir, base, diff_pct


if __name__ == "__main__":
    device = "cpu"
    print("=" * 60)
    print("Regenerating with NOEARLY weights (T=200 DDIM S=50)")
    print("=" * 60)

    for organ_short, organ_type, bdmap_id, organ_file, tumor_file in TASKS:
        print(f"\n--- {organ_type} ({organ_short}) noearly ---")
        try:
            generate_one_noearly(organ_short, organ_type, bdmap_id,
                                organ_file, tumor_file, device)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done! Compare in Tumor/output/full_ct/:")
    print("  pancreas: pancreatic_lesion_e00 (early) vs pancreatic_lesion_n00 (noearly)")
    print("  kidney:   kidney_lesion_e00 (early)     vs kidney_lesion_n00 (noearly)")
    print("  liver:    liver_lesion_e00 (early)      vs liver_lesion_n00 (noearly)")
