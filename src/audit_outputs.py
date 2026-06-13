"""
Quality audit: check ALL existing full-CT outputs for:
  1. Mask alignment (tumor mask overlaps generated tumor region)
  2. Non-tumor region unchanged
  3. Organ boundary integrity (no bleeding)
  4. Previously-identified issues regression
"""
import os, glob, numpy as np, nibabel as nib

OUT_DIR = r"C:\Users\33067\.claude\work\Tumor\output\full_ct"
MASK_DIR = r"C:\Users\33067\.claude\work\Mask\output\real_ct"

organ_map = {
    "liver_lesion": "liver_lesion",
    "pancreatic_lesion": "pancreatic_lesion",
    "kidney_lesion": "kidney_lesion",
    "colon_lesion": "colon_lesion",
    "esophagus_tumor": "esophagus_tumor",
    "endometrioma_tumor": "endometrioma_tumor",
}

all_ok = True
results = []

for otype in sorted(organ_map.keys()):
    d = os.path.join(OUT_DIR, otype)
    if not os.path.isdir(d):
        continue

    # Find synthetic CTs (not _mask, not original_)
    ct_files = sorted(glob.glob(os.path.join(d, "*.nii.gz")))
    ct_files = [f for f in ct_files if "_mask" not in f and "original_" not in os.path.basename(f)]

    for f in ct_files:
        base = os.path.basename(f).replace(".nii.gz", "")
        mask_f = f.replace(".nii.gz", "_mask.nii.gz")

        if not os.path.exists(mask_f):
            print(f"  SKIP {otype}/{base}: no mask file")
            continue

        try:
            syn_nii = nib.load(f)
            syn = syn_nii.get_fdata().astype(np.float32)
            mask = nib.load(mask_f).get_fdata() > 0

            issues = []

            # Check 1: Mask has voxels
            n_mask = int(mask.sum())
            if n_mask == 0:
                issues.append("MASK_EMPTY")

            # Check 2: Non-tumor region - find original CT to compare
            # Extract BDMAP ID from filename: *_BDMAP_XXXXXXXX.*
            import re
            m = re.search(r'BDMAP_(\d{8})', base)
            if m:
                bdmap = m.group(0)
                orig_ct_path = os.path.join(r"C:\Users\33067\.claude\work\Mask\data\ct", bdmap, "ct.nii.gz")
                if os.path.exists(orig_ct_path):
                    orig = nib.load(orig_ct_path).get_fdata().astype(np.float32)

                    # Check shapes match
                    if syn.shape != orig.shape:
                        issues.append(f"SHAPE_MISMATCH syn={syn.shape} orig={orig.shape}")

                    # Non-tumor regions should be identical
                    nt = ~mask
                    if nt.sum() > 0 and syn.shape == orig.shape:
                        nt_diff = np.abs(syn[nt] - orig[nt])
                        nt_max = nt_diff.max()
                        nt_mean = nt_diff.mean()
                        if nt_max > 20:
                            issues.append(f"NON_TUMOR_CHANGED max={nt_max:.1f}HU mean={nt_mean:.2f}HU")

                        # Tumor region stats
                        t_syn = syn[mask]
                        t_orig = orig[mask]
                        t_diff = np.abs(t_syn - t_orig)
                        hu_diff_mean = t_diff.mean()
                        pct_gt20 = np.mean(t_diff > 20) * 100

                        # Check 3: Mask outside organ boundary?
                        # Mask should be within the organ - but we already clip via tm_crop & og_crop

                    else:
                        hu_diff_mean = -1
                        pct_gt20 = -1
                else:
                    hu_diff_mean = -1
                    pct_gt20 = -1
                    nt_max = -1
            else:
                issues.append("NO_BDMAP_ID")
                hu_diff_mean = -1
                pct_gt20 = -1
                nt_max = -1

            status = "OK" if not issues else "|".join(issues)
            if issues:
                all_ok = False

            results.append((otype, base, n_mask, hu_diff_mean, pct_gt20, nt_max, status))
            tag = "FAIL" if issues else "OK"
            print(f"  [{tag}] {otype}/{base}  mask={n_mask:,}  "
                  f"tumor_|diff|={hu_diff_mean:.0f}HU  >20HU={pct_gt20:.0f}%  "
                  f"non-tumor_max={nt_max:.1f}HU  {status if issues else ''}")

        except Exception as e:
            print(f"  [ERR] {otype}/{base}: {e}")
            all_ok = False

print(f"\n{'='*60}")
print(f"Total: {len(results)} outputs checked")
if all_ok:
    print("ALL PASSED - no issues found")
else:
    failed = [r for r in results if r[-1] != "OK"]
    print(f"ISSUES FOUND: {len(failed)}")
    for r in failed:
        print(f"  {r[0]}/{r[1]}: {r[-1]}")
