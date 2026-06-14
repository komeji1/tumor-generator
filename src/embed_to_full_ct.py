"""
Full-CT tumor embedding (DiffTumor paper method)

Process:
  1. Load full CT + tumor mask (native space)
  2. Crop tumor region, resample to 1mm3, generate 96^3 synthetic
  3. Blend ONLY tumor area (paper formula: mask=0 stays original)
  4. Resample blended patch back to native space
  5. Embed into full CT at exact crop location
  6. Output: full-size synthetic CT (=original except tumor area)
"""

import os, sys, time, numpy as np, nibabel as nib, SimpleITK as sitk, torch

# 项目根目录 (src/ 的父目录)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VQGAN_CKPT, DIFFUSION_DIR, FULL_CT_DIR, TEMP_DIR

sys.path.insert(0, os.path.dirname(__file__))
from ct_preprocessor import CTPreprocessor
from condition_builder import ConditionBuilder
from diffusion_engine import DiffusionEngine
from texture_blender import TextureBlender, patch_to_hu


def embed_tumor_full_ct(
    ct_path: str,
    organ_mask_path: str,
    tumor_mask_path: str,
    organ_type: str,
    device: str = "cpu",
    phase: str = "early",
    output_mode: str = "both",
    eta: float = 0.0,
) -> tuple:
    """
    Generate synthetic tumor and embed into full CT.

    Args:
        phase: "early" (T=4 DDPM) or "noearly" (T=200 DDIM S=50)
        output_mode: "full_ct" | "patch_96" | "both"
        eta: DDIM stochasticity, 0=deterministic, 1=max diversity (noearly only)

    Returns: (full_ct_native, tumor_mask_native, affine, meta)
    """
    oshort = {"liver_lesion":"liver","pancreatic_lesion":"pancreas",
              "kidney_lesion":"kidney","colon_lesion":"colon",
              "esophagus_tumor":"esophagus","endometrioma_tumor":"uterus"}[organ_type]

    # -- Step 1: Load originals --
    orig_nii = nib.load(ct_path)
    full_ct = orig_nii.get_fdata().astype(np.float32)
    spacing = np.array(orig_nii.header.get_zooms()[:3])
    affine  = orig_nii.affine.copy()

    tm_data = nib.load(tumor_mask_path).get_fdata() > 0
    og_data = nib.load(organ_mask_path).get_fdata() > 0

    # -- Step 2: Crop tumor region in native space --
    t_idx = np.argwhere(tm_data)
    ctr = t_idx.mean(axis=0).astype(int)
    half_phys = 48.0
    half = [int(np.ceil(half_phys / s)) for s in spacing]
    x0, x1 = max(0, ctr[0]-half[0]), min(full_ct.shape[0], ctr[0]+half[0])
    y0, y1 = max(0, ctr[1]-half[1]), min(full_ct.shape[1], ctr[1]+half[1])
    z0, z1 = max(0, ctr[2]-half[2]), min(full_ct.shape[2], ctr[2]+half[2])

    ct_crop  = full_ct[x0:x1, y0:y1, z0:z1].copy()
    tm_crop  = tm_data[x0:x1, y0:y1, z0:z1].copy()
    og_crop  = og_data[x0:x1, y0:y1, z0:z1].copy()

    # Save original crop shape for trimming after padding+resample round-trip
    orig_crop_shape = ct_crop.shape

    # Pad if physical extent < 96mm (edge cases: small organs near CT boundary)
    need_phys = [96.0, 96.0, 96.0]
    for i, (s, need) in enumerate(zip(spacing, need_phys)):
        current = ct_crop.shape[i] * s
        if current < need:
            pad_voxels = int(np.ceil((need - current) / s))
            pad_width = [(0, 0), (0, 0), (0, 0)]
            pad_width[i] = (0, pad_voxels)
            ct_crop = np.pad(ct_crop, pad_width, mode='constant',
                             constant_values=ct_crop.min())
            tm_crop = np.pad(tm_crop, pad_width, mode='constant', constant_values=0)
            og_crop = np.pad(og_crop, pad_width, mode='constant', constant_values=0)

    # -- Step 3: Temp files -> preprocessor -> 1mm^3 96^3 --
    tmp = TEMP_DIR
    os.makedirs(tmp, exist_ok=True)
    real_aff = np.diag(list(spacing) + [1.0])
    for name, arr in [("ct", ct_crop), ("org", og_crop.astype(np.int16)),
                       ("tm", tm_crop.astype(np.int16))]:
        nib.save(nib.Nifti1Image(arr.astype(np.float32), real_aff),
                 os.path.join(tmp, f"embed_{name}.nii.gz"))

    pre = CTPreprocessor(device)
    r = pre.process(os.path.join(tmp, "embed_ct.nii.gz"),
                    os.path.join(tmp, "embed_org.nii.gz"),
                    os.path.join(tmp, "embed_tm.nii.gz"), oshort)
    ct_t = r.ct_tensor; tm_t = r.tumor_mask_tensor
    d, h, w = ct_t.shape[2:]
    ct_t = ct_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]
    tm_t = tm_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]

    # -- Step 4: Diffusion generation --
    builder = ConditionBuilder(VQGAN_CKPT, device)
    cond = builder.build(ct_t, tm_t)
    engine = DiffusionEngine(VQGAN_CKPT, DIFFUSION_DIR, oshort, phase, device)
    synthetic = engine.generate(cond, eta=eta)

    # -- Step 5: Blend (paper formula, fixed sigma for demo) --
    blender = TextureBlender(device)
    blended = blender.blend(ct_t, synthetic, tm_t, oshort, random_sigma=False)
    blended_hu = patch_to_hu(blended)

    # -- Step 6: Resample blended patch back to native crop space --
    blended_sitk = sitk.GetImageFromArray(blended_hu.transpose(2, 1, 0))
    blended_sitk.SetSpacing((1.0, 1.0, 1.0))
    native_crop_sitk = sitk.GetImageFromArray(ct_crop.transpose(2, 1, 0))
    native_crop_sitk.SetSpacing([float(s) for s in spacing])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(native_crop_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # preserve tumor texture contrast
    blended_native = resampler.Execute(blended_sitk)
    blended_native_arr = sitk.GetArrayFromImage(blended_native).transpose(2, 1, 0)

    # Trim padding back to original crop shape
    if blended_native_arr.shape != orig_crop_shape:
        blended_native_arr = blended_native_arr[:orig_crop_shape[0],
                                                :orig_crop_shape[1],
                                                :orig_crop_shape[2]]
        tm_crop = tm_crop[:orig_crop_shape[0], :orig_crop_shape[1], :orig_crop_shape[2]]
        og_crop = og_crop[:orig_crop_shape[0], :orig_crop_shape[1], :orig_crop_shape[2]]

    # -- Step 7: Soft organ-boundary blending --
    final_ct = full_ct.copy()
    from scipy.ndimage import gaussian_filter
    tumor_edge = gaussian_filter(tm_crop.astype(np.float32), sigma=8.0)
    organ_edge = gaussian_filter(og_crop.astype(np.float32), sigma=10.0)
    edge = tumor_edge * organ_edge
    edge = np.clip(edge, 0, 1)
    native_region = final_ct[x0:x1, y0:y1, z0:z1]
    final_ct[x0:x1, y0:y1, z0:z1] = (1 - edge) * native_region + edge * blended_native_arr

    # Final mask
    full_mask = np.zeros_like(full_ct, dtype=np.uint8)
    full_mask[x0:x1, y0:y1, z0:z1] = tm_crop & og_crop

    meta = {
        "organ": organ_type, "crop_native": [int(x) for x in [x0, x1, y0, y1, z0, z1]],
        "shape": list(full_ct.shape), "weight": phase,
    }
    if output_mode in ("patch_96", "both"):
        meta["patch_96_hu"] = blended_hu.copy()  # (96,96,96) HU, 1mm^3 isotropic
        meta["patch_96_mask"] = tm_t[0, 0].cpu().numpy().astype(np.uint8)  # (96,96,96) binary
    return final_ct, full_mask, affine, meta


def save_full_ct(final_ct, full_mask, affine, organ_type, base_name, out_root=None):
    """Save full-CT output. If file exists, append _v2, _v3... instead of overwriting."""
    d = os.path.join(out_root or FULL_CT_DIR, organ_type)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{base_name}.nii.gz")
    # Auto-version: don't overwrite
    v = 2
    while os.path.exists(path):
        path = os.path.join(d, f"{base_name}_v{v}.nii.gz")
        v += 1
    nib.save(nib.Nifti1Image(final_ct.astype(np.float32), affine), path)
    # Check base mask path (not versioned), skip if exists
    base_mask = os.path.join(d, f"{base_name}_mask.nii.gz")
    if not os.path.exists(base_mask):
        nib.save(nib.Nifti1Image(full_mask.astype(np.uint8), affine),
                 path.replace(".nii.gz", "_mask.nii.gz"))
    return path


if __name__ == "__main__":
    ct  = r"C:\Users\33067\.claude\work\Mask\data\ct\BDMAP_00000012\ct.nii.gz"
    og  = r"C:\Users\33067\.claude\work\Mask\data\organ_labels\BDMAP_00000012\segmentations\liver.nii.gz"
    tm  = r"C:\Users\33067\.claude\work\Mask\output\real_ct\liver_lesion\liver_lesion_t00__BDMAP_00000012.nii.gz"

    t0 = time.time()
    final_ct, full_mask, affine, meta = embed_tumor_full_ct(ct, og, tm, "liver_lesion", "cpu")
    dt = time.time() - t0

    save_full_ct(final_ct, full_mask, affine, "liver_lesion", "liver_lesion_e00__BDMAP_00000012")

    orig_data = nib.load(ct).get_fdata()
    t_hu = final_ct[full_mask > 0]
    o_hu = orig_data[full_mask > 0]
    nt = ~(full_mask > 0)
    nt_diff = np.abs(final_ct[nt] - orig_data[nt])

    print(f"Full CT embedded: shape={final_ct.shape}, time={dt:.0f}s")
    print(f"Tumor voxels:    {int(full_mask.sum()):,}")
    print(f"Tumor HU:        {t_hu.mean():.0f}+/-{t_hu.std():.0f} (orig={o_hu.mean():.0f})")
    print(f"Non-tumor diff:  max={nt_diff.max():.1f}HU mean={nt_diff.mean():.4f}HU")
    print(f"Match orig CT:   shape={final_ct.shape==orig_data.shape}, non-tumor same={nt_diff.max()<1}")
