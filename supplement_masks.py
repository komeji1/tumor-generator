"""
Supplement tumor masks to reach 50 successful per organ.
Skips existing masks and continues numbering from last index.
"""
import os, sys, json, time, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Step6', 'src'))
from main import generate_one, load_config
from Step2.src.data_loader import build_manifest

MASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(MASK_DIR, 'output', 'real_ct')

TARGET = 50

def count_existing_masks(organ_type):
    organ_dir = os.path.join(OUTPUT_BASE, organ_type)
    if not os.path.isdir(organ_dir):
        return 0
    files = glob.glob(os.path.join(organ_dir, '*.nii.gz'))
    return len(files)

def get_last_index(organ_type):
    organ_dir = os.path.join(OUTPUT_BASE, organ_type)
    if not os.path.isdir(organ_dir):
        return -1
    files = glob.glob(os.path.join(organ_dir, '*.nii.gz'))
    if not files:
        return -1
    # Extract t## from filenames
    max_idx = -1
    for f in files:
        name = os.path.basename(f)
        if '_t' in name:
            try:
                idx = int(name.split('_t')[-1].split('.')[0])
                max_idx = max(max_idx, idx)
            except:
                pass
    return max_idx

def main():
    config = load_config(os.path.join(MASK_DIR, 'Step0', 'config', 'generation_config.json'))

    # Build manifest
    ct_dir = os.path.join(MASK_DIR, config['data']['ct_dir'])
    label_dir = os.path.join(MASK_DIR, config['data']['organ_label_dir'])
    manifest = build_manifest(ct_dir, label_dir, config['organs'])
    available = [m for m in manifest if m['exists']]

    # Group by organ
    from collections import defaultdict
    by_organ = defaultdict(list)
    for m in available:
        by_organ[m['organ_type']].append(m)

    print("=" * 60)
    print("Supplement Masks to 50 per organ")
    print("=" * 60)

    total_success = 0
    total_attempts = 0
    start_time = time.time()

    for organ_cfg in config['organs']:
        organ_type = organ_cfg['name']
        organ_samples = by_organ.get(organ_type, [])

        if not organ_samples:
            print(f"\n{organ_type}: NO SAMPLES - SKIPPED")
            continue

        existing = count_existing_masks(organ_type)
        needed = TARGET - existing
        last_idx = get_last_index(organ_type)

        print(f"\n{organ_type}: {existing}/{TARGET} existing, need {needed} more")
        if needed <= 0:
            print("  Already complete, skipping")
            continue

        success_count = 0
        attempt = 0
        max_attempts = needed * 80  # More attempts for hard organs

        # --- Pre-filter: find CTs that can fit a tiny tumor ---
        import random as _random
        import nibabel as nib

        viable_samples = []
        for s in organ_samples:
            if not os.path.exists(s['organ_mask_path']):
                continue
            try:
                # Quick check: can this organ fit even a tiny tumor?
                om = nib.load(s['organ_mask_path'])
                spacing_vals = tuple(abs(x) for x in [om.affine[2,2], om.affine[1,1], om.affine[0,0]])
                mean_sp = float(np.mean(spacing_vals))
                # Tiny tumor: r=2mm → r_vox = 2/mean_sp
                r_vox_min = 2.0 / mean_sp if mean_sp > 0 else 2.0
                margin_min = r_vox_min + 0.5/mean_sp + 1.0/mean_sp  # feather=0.5, safety=1
                import scipy.ndimage as ndi
                organ_data = om.get_fdata().astype(np.uint8)
                eroded = ndi.distance_transform_edt(organ_data.astype(bool)) > margin_min
                if eroded.sum() >= 10:  # At least 10 valid voxels
                    viable_samples.append(s)
            except Exception:
                pass

        if viable_samples:
            print(f"  Viable CTs: {len(viable_samples)}/{len(organ_samples)}")
        else:
            # Fall back to all samples
            print(f"  No viable CTs with pre-filter! Using all {len(organ_samples)}")
            viable_samples = list(organ_samples)

        _random.shuffle(viable_samples)

        while success_count < needed and attempt < max_attempts:
            sample = viable_samples[attempt % len(viable_samples)]
            next_idx = last_idx + success_count + 1
            sample_id = f"{sample['sample_id']}_t{next_idx:02d}"

            seed = 42 + total_attempts + attempt
            rng = np.random.default_rng(seed)

            meta = generate_one(
                ct_path=sample['ct_path'],
                organ_mask_path=sample['organ_mask_path'],
                organ_type=organ_type,
                organ_label=sample['organ_label'],
                sample_id=sample_id,
                config=config,
                rng=rng,
            )

            total_attempts += 1
            attempt += 1

            if meta['success']:
                # Verify non-zero volume
                actual_vol = meta.get('final_volume_voxels', meta.get('mask_volume_voxels', 0))
                if actual_vol < 5:  # Reject too-small masks
                    if attempt % 20 == 0:
                        print(f'  ... attempt {attempt}, {success_count} successful (rejected tiny vol={actual_vol})')
                    continue
                success_count += 1
                total_success += 1
                print(f"  [{existing + success_count:2d}/{TARGET}] {meta.get('size_category','?'):6s} r={meta.get('radius_mm',0):.0f}mm OK")
            else:
                if attempt % 20 == 0:
                    print(f"  ... attempt {attempt}, {success_count} successful so far")

        if success_count < needed:
            print(f"  WARNING: Only {success_count}/{needed} successful after {attempt} attempts")
        else:
            print(f"  Complete: {existing + success_count}/{TARGET}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Summary: {total_success} new masks in {elapsed:.0f}s ({elapsed/max(1,total_attempts):.1f}s/attempt)")
    print(f"Total attempts: {total_attempts}, success rate: {total_success/max(1,total_attempts):.0%}")
    print(f"{'='*60}")

    # Final count
    print("\nFinal counts:")
    for organ_cfg in config['organs']:
        organ_type = organ_cfg['name']
        count = count_existing_masks(organ_type)
        print(f"  {organ_type}: {count}/{TARGET}")

if __name__ == '__main__':
    main()