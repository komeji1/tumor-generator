"""
Batch full-CT generation for all organs (except colon — waiting for training).
Uses the fixed embed_to_full_ct.py with padding support.
"""
import sys, os, time, glob, numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(__file__))
from embed_to_full_ct import embed_tumor_full_ct, save_full_ct

MASK_ROOT  = r"C:\Users\33067\.claude\work\Mask"
OUT_ROOT   = r"C:\Users\33067\.claude\work\Tumor\output"

# Organ mapping: mask_dir_name → (organ_short, organ_seg_file)
ORGAN_MAP = {
    "liver_lesion":        ("liver",    "liver.nii.gz"),
    "pancreatic_lesion":   ("pancreas", "pancreas.nii.gz"),
    "kidney_lesion":       ("kidney",   None),  # combined L+R
    "esophagus_tumor":     ("esophagus","esophagus.nii.gz"),
    "endometrioma_tumor":  ("uterus",   "uterus.nii.gz"),
    # "colon_lesion":      ("colon",    "colon.nii.gz"),  # wait for training
}

TMP_DIR = r"D:\Users\33067\claude-data\downloads\_tmp"
os.makedirs(TMP_DIR, exist_ok=True)


def get_organ_mask_path(bdmap_id, seg_file):
    """Get organ mask path, handling kidney combined case."""
    if seg_file:
        return os.path.join(MASK_ROOT, "data", "organ_labels", bdmap_id,
                           "segmentations", seg_file)
    else:
        # Kidney: combine left+right
        kl = os.path.join(MASK_ROOT, "data", "organ_labels", bdmap_id,
                         "segmentations", "kidney_left.nii.gz")
        kr = os.path.join(MASK_ROOT, "data", "organ_labels", bdmap_id,
                         "segmentations", "kidney_right.nii.gz")
        out = os.path.join(TMP_DIR, f"kidney_combined_{bdmap_id}.nii.gz")
        if os.path.exists(out):
            return out
        if not os.path.exists(kl) or not os.path.exists(kr):
            # Try single kidney
            for kf in [kl, kr]:
                if os.path.exists(kf):
                    return kf
            return None
        kl_data = nib.load(kl).get_fdata() > 0
        kr_data = nib.load(kr).get_fdata() > 0
        ref = nib.load(kl)
        nib.save(nib.Nifti1Image((kl_data | kr_data).astype(np.uint8),
                                 ref.affine, ref.header), out)
        return out


def batch_generate(device="cpu"):
    total_ok = 0
    total_skip = 0
    total_fail = 0
    t_start = time.time()

    for mask_dir, (oshort, seg_file) in sorted(ORGAN_MAP.items()):
        mask_glob = os.path.join(MASK_ROOT, "output", "real_ct", mask_dir, "*.nii.gz")
        mask_files = sorted(glob.glob(mask_glob))
        if not mask_files:
            print(f"\n{mask_dir}: NO MASKS FOUND")
            continue

        print(f"\n{'='*50}")
        print(f"{mask_dir} ({oshort}): {len(mask_files)} masks")
        print(f"{'='*50}")

        ok = skip = fail = 0

        for i, mask_path in enumerate(mask_files):
            base = os.path.basename(mask_path).replace(".nii.gz", "")
            # Parse BDMAP ID from filename: *_BDMAP_XXXXXXXX.*
            import re
            m = re.search(r'BDMAP_(\d{8})', base)
            if not m:
                print(f"  [{i+1}/{len(mask_files)}] SKIP {base}: no BDMAP ID")
                skip += 1
                continue
            bdmap_id = m.group(0)

            ct_path = os.path.join(MASK_ROOT, "data", "ct", bdmap_id, "ct.nii.gz")
            og_path = get_organ_mask_path(bdmap_id, seg_file)

            if not os.path.exists(ct_path):
                print(f"  [{i+1}/{len(mask_files)}] SKIP {base}: no CT")
                skip += 1
                continue
            if not og_path or not os.path.exists(og_path):
                print(f"  [{i+1}/{len(mask_files)}] SKIP {base}: no organ mask")
                skip += 1
                continue

            # Quick pre-check: mask has enough voxels
            try:
                tm_data = nib.load(mask_path).get_fdata() > 0
                if tm_data.sum() < 10:
                    print(f"  [{i+1}/{len(mask_files)}] SKIP {base}: <10 voxels")
                    skip += 1
                    continue
            except Exception as e:
                print(f"  [{i+1}/{len(mask_files)}] FAIL {base}: load error: {e}")
                fail += 1
                continue

            t0 = time.time()
            try:
                fc, fm, aff, meta = embed_tumor_full_ct(
                    ct_path, og_path, mask_path, mask_dir, device)
                out_base = re.sub(r'_t(\d)', r'_s\1', base)  # _t00→_s00, won't corrupt "tumor"
                save_full_ct(fc, fm, aff, mask_dir, out_base)
                dt = time.time() - t0
                ok += 1

                if (ok % 10 == 0) or (ok <= 3):
                    t_hu = fc[fm > 0]
                    print(f"  [{i+1}/{len(mask_files)}] OK  {out_base}  "
                          f"vox={int(fm.sum()):,}  HU={t_hu.mean():.0f}  "
                          f"time={dt:.0f}s")
            except Exception as e:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(mask_files)}] FAIL {base}: {e} ({dt:.0f}s)")
                fail += 1

        print(f"  --- {mask_dir}: ok={ok} skip={skip} fail={fail} ---")
        total_ok += ok
        total_skip += skip
        total_fail += fail

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {total_time/60:.0f}min")
    print(f"  OK: {total_ok}  Skip: {total_skip}  Fail: {total_fail}")
    print(f"  Output: {OUT_ROOT}/full_ct/{{organ}}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    batch_generate(args.device)
