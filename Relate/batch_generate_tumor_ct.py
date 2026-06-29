"""
Batch pipeline: generate 50 tumor-containing CTs for FID evaluation.

7 organs × 3 size categories, each CT gets exactly 1 tumor type,
50 different base CTs → 50 independent synthetic tumor CTs → reliable FID.

Usage:
    python batch_generate_tumor_ct.py step1     # paint tumor masks (~1 min)
    python batch_generate_tumor_ct.py step2     # MAISI generate CTs (~2.5 hrs)
    python batch_generate_tumor_ct.py collect   # collect results for FID
    python batch_generate_tumor_ct.py fid       # compute FID
    python batch_generate_tumor_ct.py all       # step1 + collect (no step2)
"""

from __future__ import annotations

import os
import sys
import json
import glob
import random
import subprocess
import shutil
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "tumor_ct_batch"
FID_SCRIPT = ROOT / "Relate" / "compute_fid_single_gpu.py"

# ══════════════════════════════════════════════════════════════
# Tumor assignment: 7 organs × 3 sizes, diverse coverage
# ══════════════════════════════════════════════════════════════
#
# Each CT gets exactly 1 tumor → 50 unique base CTs used.
# esophagus has no "large" (organ too small, tumor overflows badly).
# bone has more "large" (bone fragments are dispersed, large tumors
# are realistic for metastatic bone lesions).

ASSIGNMENTS = [
    # organ,       count,  size
    ("liver",       3,     "small"),
    ("liver",       4,     "medium"),
    ("liver",       3,     "large"),
    ("pancreas",    3,     "small"),
    ("pancreas",    3,     "medium"),
    ("pancreas",    2,     "large"),
    ("kidney",      3,     "small"),
    ("kidney",      3,     "medium"),
    ("kidney",      2,     "large"),
    ("colon",       2,     "small"),
    ("colon",       3,     "medium"),
    ("colon",       3,     "large"),
    ("lung",        2,     "small"),
    ("lung",        3,     "medium"),
    ("lung",        3,     "large"),
    ("bone",        2,     "small"),
    ("bone",        3,     "medium"),
    ("bone",        5,     "large"),
    ("esophagus",   2,     "small"),
    ("esophagus",   2,     "medium"),
]  # total = 50


def get_step1_pairs():
    """Get all step1 CT+label pairs with _label_full.nii.gz."""
    pairs = []
    for label_file in sorted((ROOT / "output").glob("sample_*_label_full.nii.gz")):
        base = label_file.name.replace("_label_full.nii.gz", "")
        ct_file = ROOT / "output" / f"{base}_image.nii.gz"
        if ct_file.exists():
            pairs.append({"base": base, "ct": str(ct_file), "label": str(label_file)})
    return pairs


def step1_paint_masks():
    """Step 1: Paint tumor masks on 50 step1 CTs via direct import."""
    sys.path.insert(0, str(ROOT / "Relate"))
    from bridge_maisi_mask import bridge_single, _resolve_path

    pairs = get_step1_pairs()
    print(f"Found {len(pairs)} step1 CT+label pairs")

    # Build tasks: each assignment uses a different base CT
    tasks = []
    idx = 0
    for organ, count, size in ASSIGNMENTS:
        for _ in range(count):
            if idx < len(pairs):
                tasks.append({"organ": organ, "size": size, "pair": pairs[idx]})
                idx += 1

    if len(tasks) < 50:
        print(f"Warning: only {len(tasks)} tasks (need 53 pairs for 50 tasks)")

    tasks = tasks[:50]
    print(f"Planned {len(tasks)} tumor painting tasks")
    print("\nAssignment summary:")
    summary = {}
    for t in tasks:
        key = f"{t['organ']}_{t['size']}"
        summary[key] = summary.get(key, 0) + 1
    for key, cnt in sorted(summary.items()):
        print(f"  {key}: {cnt}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    used_seeds = set()
    for i, task in enumerate(tasks):
        pair = task["pair"]
        organ = task["organ"]
        size = task["size"]

        # Generate unique seed
        seed = random.randint(0, 999999)
        while seed in used_seeds:
            seed = random.randint(0, 999999)
        used_seeds.add(seed)

        print(f"\n[{i+1}/{len(tasks)}] {organ} {size} seed={seed} → {pair['base']}")

        try:
            result = bridge_single(
                ct_path=_resolve_path(pair["ct"]),
                label_path=_resolve_path(pair["label"]),
                organ=organ,
                size_category=size,
                seed=seed,
                output_dir=str(OUTPUT_DIR),
            )

            if result["status"] == "ok":
                merged_path = result.get("output_path", "")
                # The merged label is inside a task directory:
                #   output/tumor_ct_batch/{base}_{organ}_{size}_tumor/04_final_merged/merged_label.nii.gz
                results.append({
                    "status": "ok",
                    "organ": organ,
                    "size": size,
                    "seed": seed,
                    "tumor_label_path": merged_path,
                    "original_ct": pair["ct"],
                    "original_label": pair["label"],
                    "base": pair["base"],
                    "radius_mm": result.get("radius_mm", 0),
                    "overlap": result.get("overlap_ratio", 0),
                    "tumor_voxels": result.get("tumor_voxels", 0),
                    "task_dir": result.get("task_dir", ""),
                })
                print(f"  OK: r={result.get('radius_mm',0):.1f}mm "
                      f"overlap={result.get('overlap_ratio',0):.0%} "
                      f"vox={result.get('tumor_voxels',0):,}")
            elif result["status"] == "skip":
                results.append({
                    "status": "skip",
                    "organ": organ,
                    "size": size,
                    "seed": seed,
                    "reason": result.get("reason", ""),
                    "base": pair["base"],
                })
                print(f"  SKIP: {result.get('reason', '')}")
            else:
                results.append({
                    "status": "fail",
                    "organ": organ,
                    "size": size,
                    "seed": seed,
                    "reason": result.get("reason", ""),
                    "base": pair["base"],
                })
                print(f"  FAIL: {result.get('reason', '')}")
        except Exception as e:
            results.append({
                "status": "fail",
                "organ": organ,
                "size": size,
                "seed": seed,
                "reason": str(e),
                "base": pair["base"],
            })
            print(f"  FAIL: {str(e)}")

    ok = [r for r in results if r["status"] == "ok"]
    skip = [r for r in results if r["status"] == "skip"]
    fail = [r for r in results if r["status"] == "fail"]
    print(f"\n{'='*60}")
    print(f"Step 1 complete: {len(ok)} OK, {len(skip)} SKIP, {len(fail)} FAIL "
          f"out of {len(tasks)}")

    # Print organ/size distribution of successful tasks
    if ok:
        print("\nSuccessful task distribution:")
        success_summary = {}
        for r in ok:
            key = f"{r['organ']}_{r['size']}"
            success_summary[key] = success_summary.get(key, 0) + 1
        for key, cnt in sorted(success_summary.items()):
            print(f"  {key}: {cnt}")

    with open(OUTPUT_DIR / "batch_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def step2_generate_cts():
    """Step 2: Run MAISI infer_image_from_mask.py for each tumor label.

    Uses single-file CLI (no DDP required).
    This step takes ~3-5 min per image, total ~2.5 hours for 50 images.
    """
    results_path = OUTPUT_DIR / "batch_results.json"
    if not results_path.exists():
        print("No batch_results.json. Run step 1 first.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    ok_results = [r for r in results if r["status"] == "ok"]
    # Only generate for tasks that haven't been generated yet
    pending = [r for r in ok_results if r.get("ct_generated") != "ok"]
    print(f"Found {len(ok_results)} tumor masks, {len(pending)} pending generation")

    if not pending:
        print("All CTs already generated!")
        return

    for i, r in enumerate(pending):
        mask_path = r["tumor_label_path"]
        seed = random.randint(0, 999999)

        cmd = [
            sys.executable, "-m", "scripts.infer_image_from_mask",
            "--mask", str(_resolve_path(mask_path)),
            "--environment-file", str(ROOT / "configs" / "environment_rflow-ct.json"),
            "--inference-file", str(ROOT / "configs" / "config_infer_8g_256x256x128.json"),
            "--config-file", str(ROOT / "configs" / "config_network_rflow.json"),
            "--random-seed", str(seed),
        ]

        print(f"\n[{i+1}/{len(pending)}] Generating CT for {r['organ']} {r['size']}")
        print(f"  Mask: {mask_path}")
        print(f"  Seed: {seed}")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace",
            )
            if proc.returncode == 0:
                r["ct_generated"] = "ok"
                # Find the generated image file
                # MAISI puts output in output/ directory with timestamp
                print(f"  OK")
            else:
                r["ct_generated"] = "fail"
                r["fail_reason"] = proc.stderr[:500]
                print(f"  FAIL: {proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            r["ct_generated"] = "fail"
            r["fail_reason"] = "timeout (>600s)"
            print(f"  TIMEOUT")
        except Exception as e:
            r["ct_generated"] = "fail"
            r["fail_reason"] = str(e)
            print(f"  FAIL: {str(e)}")

        # Save progress after each task
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    ok_gen = [r for r in ok_results if r.get("ct_generated") == "ok"]
    fail_gen = [r for r in ok_results if r.get("ct_generated") == "fail"]
    print(f"\nStep 2: {len(ok_gen)} OK / {len(fail_gen)} FAIL "
          f"out of {len(ok_results)} total")


def collect_generated_cts():
    """Collect all generated tumor CTs into a flat directory for FID."""
    ct_files = set()

    # Find all tumor CTs in output tree
    for pattern_root in [ROOT / "output", OUTPUT_DIR]:
        for f in pattern_root.glob("**/*tumor*image*.nii.gz"):
            # Skip old step2 outputs (those 8 from the single base CT)
            # Only include ones from batch_results.json
            ct_files.add(f)

    # Also check batch_results.json for generated CTs
    results_path = OUTPUT_DIR / "batch_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        ok_results = [r for r in results if r["status"] == "ok" and r.get("ct_generated") == "ok"]
        print(f"From batch_results.json: {len(ok_results)} successfully generated CTs")
        for r in ok_results:
            mask_dir = Path(r["task_dir"])
            # Look for _image.nii.gz in the task output
            for img in mask_dir.glob("*image*.nii.gz"):
                ct_files.add(img)

    # Find all step1 (no tumor) CTs for reference
    step1_files = []
    for f in sorted((ROOT / "output").glob("sample_*_image.nii.gz")):
        if "tumor" not in f.name:
            step1_files.append(f)

    ct_files = sorted(ct_files)
    step1_files = sorted(step1_files)
    print(f"Found {len(ct_files)} tumor CTs and {len(step1_files)} step1 CTs")

    if len(ct_files) >= 10:
        fid_dir = ROOT / "data" / "fid_tumor_ct_50"

        fid_dir.mkdir(parents=True, exist_ok=True)

        # Copy tumor CTs to flat directory
        n_copied = 0
        for f in ct_files[:50]:
            dst = fid_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            n_copied += 1

        # Create filelist
        with open(fid_dir / "filelist.txt", "w") as fl:
            for f in sorted(fid_dir.glob("*.nii.gz")):
                fl.write(f.name + "\n")

        print(f"\nCollected {n_copied} tumor CTs → {fid_dir}")
        print(f"  filelist.txt: {fid_dir / 'filelist.txt'}")
        return n_copied
    else:
        print(f"Not enough tumor CTs ({len(ct_files)}). Run step2 first.")
        return 0


def compute_fid(num_images=50):
    """Compute FID: autoPET (real with tumor) vs synthetic tumor CTs."""
    fid_dir = ROOT / "data" / "fid_tumor_ct_50"
    if not fid_dir.exists() or not (fid_dir / "filelist.txt").exists():
        n = collect_generated_cts()
        if n == 0:
            print("No tumor CTs available for FID.")
            return

    cmd = [
        sys.executable, str(FID_SCRIPT),
        "--real_dataset_root", str(ROOT / "data" / "autopet_ct50"),
        "--real_filelist", str(ROOT / "data" / "autopet_ct50" / "filelist.txt"),
        "--real_features_dir", "autopet_real",
        "--synth_dataset_root", str(fid_dir),
        "--synth_filelist", str(fid_dir / "filelist.txt"),
        "--synth_features_dir", "tumor_ct_50",
        "--model_name", "radimagenet_resnet50",
        "--target_shape", "256x256x128",
        "--enable_resampling_spacing", "1.0x1.0x1.0",
        "--enable_center_slices_ratio", "0.4",
        "--enable_padding", "True",
        "--enable_center_cropping", "True",
        "--ignore_existing", "False",
        "--num_images", str(num_images),
        "--output_root", str(ROOT / "data" / "fid_features"),
    ]

    print("FID command:")
    print("  " + " ".join(cmd[:6]) + " ...")
    subprocess.run(cmd)


def main(step="all"):
    """Run the batch pipeline."""
    print(f"{'='*60}")
    print(f"Batch Tumor CT Generation — {step}")
    print(f"{'='*60}\n")

    if step == "step1":
        step1_paint_masks()
    elif step == "step2":
        step2_generate_cts()
    elif step == "collect":
        collect_generated_cts()
    elif step == "fid":
        compute_fid()
    elif step == "all":
        results = step1_paint_masks()
        ok = [r for r in results if r["status"] == "ok"]
        if ok:
            print(f"\nStep 1 done: {len(ok)} tumor masks painted.")
            print("Run step 2 (MAISI generation, ~2.5 hrs):")
            print("  python batch_generate_tumor_ct.py step2")
        else:
            print("No tumor masks generated. Check errors above.")


def _resolve_path(p):
    """Resolve path relative to ROOT."""
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    return str(p.resolve())


if __name__ == "__main__":
    import fire
    fire.Fire(main)
