"""Download LIDC-IDRI Chest CT from HuggingFace for MAISI training.

Uses raw requests to download from HuggingFace CDN, bypassing the
huggingface_hub library's httpx SSL proxy issues entirely.

Usage:
    python download_lidc_hf.py --num_scans 450
    python download_lidc_hf.py --num_scans 5   # quick test

Requirements:
    pip install requests
"""

import glob
import json
import os
import sys
import time
# shutil is used by hf_hub_download only, not needed in requests mode

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

HF_DATASET = "hourouu/LIDC_IDRI_PROCESSED"
DEFAULT_OUTPUT = "./temp_work_dir/demo_train_datasets/LIDC"

# HuggingFace CDN URL pattern for dataset files
# Format: https://huggingface.co/datasets/{repo_id}/resolve/main/{filepath}
HF_CDN_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# Patient IDs from LIDC-IDRI (0001-1010, some gaps)
KNOWN_IDS = list(range(1, 1011))


def download_file(url, local_path, proxy, max_retries=5):
    """Download a file using requests (works reliably with HTTPS proxies)."""
    import requests
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Suppress InsecureRequestWarning when verify=False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    if proxy:
        session.proxies = {"https": proxy, "http": proxy}
    retry = Retry(total=max_retries, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for attempt in range(max_retries):
        try:
            # verify=False bypasses SSL certificate issues with proxies like Clash
            # This is safe for downloading public dataset files
            r = session.get(url, timeout=300, stream=True, allow_redirects=True, verify=False)
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)

            # Verify file size
            if total > 0 and downloaded != total:
                print(f"    Size mismatch: expected {total}, got {downloaded}")
                os.remove(local_path)
                continue

            return True

        except (requests.exceptions.RequestException, TimeoutError, OSError, ConnectionError) as e:
            print(f"    Attempt {attempt+1}/{max_retries} failed: {type(e).__name__}: {str(e)[:100]}")
            if os.path.exists(local_path):
                os.remove(local_path)
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Waiting {wait}s before retry ...")
                time.sleep(wait)

    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download LIDC-IDRI CT from HuggingFace")
    parser.add_argument("--num_scans", type=int, default=450, help="Number of CT scans to download")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Training data ratio")
    parser.add_argument("--proxy", type=str, default="", help="Proxy URL (empty=direct, e.g. http://127.0.0.1:7890)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LIDC-IDRI HuggingFace Downloader for MAISI Training")
    print("=" * 60)
    print(f"  Dataset      : {HF_DATASET}")
    print(f"  Target scans : {args.num_scans}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Proxy        : {args.proxy}")
    print(f"  Method       : direct requests (no huggingface_hub httpx)")
    print()

    # Check existing files
    existing_nii = sorted(glob.glob(os.path.join(output_dir, "*.nii.gz")))
    existing_names = {os.path.basename(f) for f in existing_nii}
    print(f"  Found {len(existing_nii)} existing NIfTI files.")

    # Step 1: Generate file list locally
    print("\n[Step 1] Generating file list ...")
    selected_ids = KNOWN_IDS[:args.num_scans]
    to_download = []
    for pid in selected_ids:
        basename = f"LIDC-IDRI-{pid:04d}_img.nii.gz"
        if basename not in existing_names:
            local_path = os.path.join(output_dir, basename)
            hf_path = f"images/{basename}"
            url = HF_CDN_BASE.format(repo=HF_DATASET, path=hf_path)
            to_download.append((pid, basename, local_path, url))

    print(f"  Need to download: {len(to_download)} files")
    print(f"  Already have: {len(existing_nii)} files")

    if not to_download:
        print("  All files already downloaded!")
    else:
        # Step 2: Download files
        print(f"\n[Step 2] Downloading {len(to_download)} CT scans ...")
        success = len(existing_nii)
        fail = 0

        for idx, (pid, basename, local_path, url) in enumerate(to_download):
            print(f"  [{idx+1}/{len(to_download)}] {basename} ...")

            ok = download_file(url, local_path, args.proxy)

            if ok and os.path.exists(local_path):
                fsize = os.path.getsize(local_path) / 1024 / 1024
                success += 1
                print(f"  [{idx+1}/{len(to_download)}] OK: {basename} ({fsize:.1f} MB) -- total OK: {success}, failed: {fail}")
            else:
                fail += 1
                print(f"  [{idx+1}/{len(to_download)}] FAILED: {basename} -- total OK: {success}, failed: {fail}")

            # Small pause every 100 files to avoid rate limiting
            if (idx + 1) % 100 == 0:
                print(f"  Pausing 10s after {idx+1} files ...")
                time.sleep(10)

    # Refresh file list
    all_nii = sorted(glob.glob(os.path.join(output_dir, "*.nii.gz")))
    print(f"\n  Total NIfTI files: {len(all_nii)}")

    # Step 3: Split train/val
    print("\n[Step 3] Organizing dataset ...")
    n_train = int(args.train_ratio * len(all_nii))
    train_files = all_nii[:n_train]
    val_files = all_nii[n_train:]

    print(f"  Training   : {len(train_files)}")
    print(f"  Validation : {len(val_files)}")

    # Step 4: Save data list JSON
    train_dicts = [{"image": f, "class": "ct"} for f in train_files]
    val_dicts = [{"image": f, "class": "ct"} for f in val_files]

    datalist_path = os.path.join(output_dir, "lidc_datalist.json")
    with open(datalist_path, "w") as f:
        json.dump({"training": train_dicts, "validation": val_dicts}, f, indent=2)

    total_size = sum(os.path.getsize(f) for f in all_nii)
    print(f"  Total NIfTI size : {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Data list saved  : {datalist_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Total files  : {len(all_nii)}")
    print(f"  Failed       : {fail}")
    print(f"  Training     : {len(train_files)}")
    print(f"  Validation   : {len(val_files)}")
    print(f"  Output       : {output_dir}")
    print("=" * 60)

    print("\n[USAGE] To use in train_vae_tutorial.py:")
    print(f"   Data list: {datalist_path}")
    print(f"   NIfTI dir: {output_dir}")


if __name__ == "__main__":
    main()