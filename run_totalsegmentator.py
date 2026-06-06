"""
Batch run TotalSegmentator on remaining CTs.
Only extracts: liver, pancreas, kidney_left, colon, esophagus (and kidney_right for future use)
Usage: python run_totalsegmentator.py
"""
import os, sys, subprocess, time

MASK_DIR = os.path.dirname(os.path.abspath(__file__))
CT_DIR = os.path.join(MASK_DIR, 'data', 'ct')
LABEL_DIR = os.path.join(MASK_DIR, 'data', 'organ_labels')

ROI_NAMES = ['liver', 'pancreas', 'kidney_left', 'kidney_right', 'colon', 'esophagus']

def find_remaining_cts():
    """Find CTs that need TotalSegmentator processing."""
    ct_ids = sorted([d for d in os.listdir(CT_DIR)
                     if os.path.isdir(os.path.join(CT_DIR, d)) and d.startswith('BDMAP')])

    remaining = []
    for ct_id in ct_ids:
        seg_dir = os.path.join(LABEL_DIR, ct_id, 'segmentations')
        if not os.path.isdir(seg_dir):
            remaining.append((ct_id, 'missing'))
        else:
            # Check if all required organs exist
            missing_organs = []
            for roi in ROI_NAMES:
                if not os.path.exists(os.path.join(seg_dir, f'{roi}.nii.gz')):
                    missing_organs.append(roi)
            if missing_organs:
                remaining.append((ct_id, f'partial: {missing_organs}'))
    return remaining

def process_ct(ct_id):
    """Run TotalSegmentator on one CT."""
    ct_path = os.path.join(CT_DIR, ct_id, 'ct.nii.gz')
    out_dir = os.path.join(LABEL_DIR, ct_id, 'segmentations')

    if not os.path.exists(ct_path):
        print(f"  ERROR: CT not found: {ct_path}")
        return False

    os.makedirs(out_dir, exist_ok=True)

    # TotalSegmentator with fast mode, only save needed ROI subsets
    cmd = [
        sys.executable, '-m', 'totalsegmentator',
        '-i', ct_path,
        '-o', out_dir,
        '--fast',
        '--task', 'total',
        '-rs'] + ROI_NAMES + [
        '--device', 'cpu',
    ]

    print(f"  Running TotalSegmentator --fast (CPU)...")
    print(f"  Output: {out_dir}")
    t0 = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        if result.returncode == 0:
            # Verify outputs
            ok = True
            for roi in ROI_NAMES:
                fpath = os.path.join(out_dir, f'{roi}.nii.gz')
                if os.path.exists(fpath):
                    sz = os.path.getsize(fpath) // 1024
                    print(f"    {roi}.nii.gz ({sz} KB)")
                else:
                    print(f"    MISSING: {roi}.nii.gz")
                    ok = False
            print(f"  Done in {elapsed:.0f}s")
            return ok
        else:
            print(f"  FAILED (exit {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (10 min)")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    remaining = find_remaining_cts()

    if not remaining:
        print("All CTs have complete segmentations!")
        return

    print(f"{'='*60}")
    print(f"TotalSegmentator Batch Process")
    print(f"{'='*60}")
    print(f"Remaining: {len(remaining)} CTs")
    for ct_id, status in remaining:
        print(f"  {ct_id}: {status}")
    print()

    completed, failed = 0, 0
    for i, (ct_id, status) in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {ct_id} ({status})")
        if process_ct(ct_id):
            completed += 1
        else:
            failed += 1
        print()

    print(f"{'='*60}")
    print(f"Complete: {completed} succeeded, {failed} failed")

    # Final count
    final = find_remaining_cts()
    if final:
        print(f"Still incomplete: {len(final)}")
        for ct_id, status in final:
            print(f"  {ct_id}: {status}")
    else:
        print("All CTs fully segmented!")

    print(f"{'='*60}")

if __name__ == '__main__':
    main()
