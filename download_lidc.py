"""Download LIDC-IDRI Chest CT dataset subset for MAISI training.

Strategy: Use TCIA NBIA API to get a bulk download manifest, then download
series ZIPs in parallel. Falls back to manual patient-by-patient download
if the manifest API is slow.

Usage:
    python download_lidc.py --num_scans 5
    python download_lidc.py --num_scans 450

Requirements:
    pip install pydicom requests nibabel
"""

import glob
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TCIA_API = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"
NBIA_API = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"
DEFAULT_OUTPUT = "./temp_work_dir/demo_train_datasets/LIDC"
# Force stdout unbuffered so we see progress immediately
sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def make_session():
    """Create a requests session with retry logic."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504]
        )
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def api_get(url, timeout=180, retries=5):
    """GET with retries and exponential backoff."""
    s = make_session()
    for i in range(retries):
        try:
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"    API attempt {i+1}/{retries} failed: {e}")
            if i < retries - 1:
                time.sleep(10 * (i + 1))
    raise RuntimeError(f"API call failed after {retries} attempts: {url}")


# ---------------------------------------------------------------------------
# Step 1: Get all LIDC CT series UIDs
# ---------------------------------------------------------------------------
def get_lidc_ct_series_list(num_needed):
    """Return list of (PatientID, SeriesInstanceUID) for LIDC CT scans.

    Tries two approaches:
      1) Bulk getSeries?collection=LIDC-IDRI&Modality=CT  (one request)
      2) Fallback: patient-by-patient (slow but more reliable)
    """
    print("[Step 1] Fetching LIDC CT series list from TCIA ...")

    # --- Approach A: single bulk query ---
    try:
        url = f"{TCIA_API}/getSeries?collection=LIDC-IDRI&Modality=CT"
        print(f"  Trying bulk query: {url[:80]}...")
        r = api_get(url, timeout=300)
        all_series = r.json()

        ct_series = []
        seen = set()
        for s in all_series:
            uid = s.get("SeriesInstanceUID", "")
            count = int(s.get("ImageCount", 0))
            pid = s.get("PatientID", "")
            # Filter: enough slices, not duplicate
            if uid and uid not in seen and count >= 80:
                ct_series.append((pid, uid, count))
                seen.add(uid)

        ct_series.sort(key=lambda x: x[2], reverse=True)  # most slices first
        result = [(p, u) for p, u, _ in ct_series[:num_needed]]
        print(f"  Bulk query returned {len(ct_series)} CT series, selecting {len(result)}.")
        if len(result) >= num_needed:
            return result
        print(f"  Only found {len(result)} series, will try patient-by-patient for more.")
    except Exception as e:
        print(f"  Bulk query failed: {e}")
        print("  Falling back to patient-by-patient search ...")

    # --- Approach B: patient-by-patient ---
    print("  Getting patient list ...")
    r = api_get(f"{TCIA_API}/getPatient?collection=LIDC-IDRI", timeout=180)
    patients = r.json()
    patient_ids = [p["PatientID"] for p in patients]
    print(f"  Found {len(patient_ids)} patients. Scanning for CT series ...")

    ct_series = []
    seen = set()
    for i, pid in enumerate(patient_ids):
        if len(ct_series) >= num_needed:
            break
        try:
            # Get studies for this patient
            r = api_get(f"{TCIA_API}/getPatientStudy?PatientID={pid}", timeout=120)
            studies = r.json()
            for study in studies:
                study_uid = study.get("StudyInstanceUID", "")
                if not study_uid:
                    continue
                r2 = api_get(
                    f"{TCIA_API}/getSeries?PatientID={pid}&StudyInstanceUID={study_uid}",
                    timeout=120,
                )
                for s in r2.json():
                    uid = s.get("SeriesInstanceUID", "")
                    count = int(s.get("ImageCount", 0))
                    mod = s.get("Modality", "")
                    if mod == "CT" and uid and uid not in seen and count >= 80:
                        ct_series.append((pid, uid))
                        seen.add(uid)
                        print(f"    [{len(ct_series)}/{num_needed}] {pid} series={uid[:16]}... ({count} slices)")
                        if len(ct_series) >= num_needed:
                            break
                if len(ct_series) >= num_needed:
                    break
        except Exception as e:
            print(f"    Skipping patient {pid}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  ... scanned {i+1}/{len(patient_ids)} patients, found {len(ct_series)} CT series")

    print(f"  Total CT series found: {len(ct_series)}")
    return ct_series[:num_needed]


# ---------------------------------------------------------------------------
# Step 2: Download a single series as ZIP, convert to NIfTI
# ---------------------------------------------------------------------------
def download_and_convert(series_info, output_dir, temp_dir):
    """Download one CT series as ZIP, convert to NIfTI, clean up.

    Args:
        series_info: (PatientID, SeriesInstanceUID)
        output_dir: where to save NIfTI
        temp_dir: temp dir for ZIP / DICOM

    Returns:
        (nifti_path, True) on success, or (None, False) on failure
    """
    pid, series_uid = series_info
    short_uid = series_uid[:8]
    nifti_path = os.path.join(output_dir, f"LIDC-IDRI-{pid}_{short_uid}.nii.gz")

    # Skip if already exists
    if os.path.exists(nifti_path):
        return nifti_path, True

    zip_path = os.path.join(temp_dir, f"{series_uid}.zip")
    extract_dir = os.path.join(temp_dir, f"extract_{short_uid}")

    try:
        # Download
        url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/wado/getImage?SeriesInstanceUID={series_uid}"
        s = make_session()
        for attempt in range(3):
            try:
                print(f"  Downloading {pid}/{short_uid}... (attempt {attempt+1})")
                r = s.get(url, timeout=600, stream=True)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        done += len(chunk)
                if total > 0:
                    print(f"  Downloaded {pid}/{short_uid}: {done/1024/1024:.1f} MB")
                break
            except Exception as e:
                print(f"  Download attempt {attempt+1} failed: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if attempt < 2:
                    time.sleep(15)
                else:
                    return None, False

        # Extract
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Find DICOM directory
        dcm_dirs = []
        for root, _, files in os.walk(extract_dir):
            dcm_count = sum(1 for f in files if f.endswith(".dcm") or (os.path.isfile(os.path.join(root, f)) and not f.endswith(".xml") and not f.endswith(".json")))
            if dcm_count > 10:
                dcm_dirs.append(root)

        # Convert
        converted = False
        for dcm_dir in dcm_dirs:
            if dicom_to_nifti(dcm_dir, nifti_path):
                converted = True
                break

        if not converted:
            print(f"  FAILED to convert {pid}/{short_uid}")
            return None, False

        print(f"  OK: {os.path.basename(nifti_path)}")
        return nifti_path, True

    except Exception as e:
        print(f"  ERROR {pid}/{short_uid}: {e}")
        return None, False

    finally:
        # Always clean up temp files
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)


def dicom_to_nifti(dcm_dir, output_path):
    """Convert a DICOM series directory to NIfTI."""
    dcm_files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    if not dcm_files:
        all_files = sorted(glob.glob(os.path.join(dcm_dir, "*")))
        dcm_files = [f for f in all_files if os.path.isfile(f) and not f.endswith((".xml", ".json", ".txt"))]

    if not dcm_files:
        return False

    try:
        slices = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, "pixel_array") and hasattr(ds, "ImagePositionPatient"):
                    slices.append(ds)
            except Exception:
                continue

        if len(slices) < 20:
            return False

        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
        slope = float(slices[0].get("RescaleSlope", 1))
        intercept = float(slices[0].get("RescaleIntercept", 0))
        volume = volume * slope + intercept

        pos = np.array([float(x) for x in slices[0].ImagePositionPatient])
        last_pos = np.array([float(x) for x in slices[-1].ImagePositionPatient])
        n_slices = len(slices)
        ps = [float(x) for x in slices[0].PixelSpacing]
        slice_thick = float(slices[0].get("SliceThickness", abs(last_pos[2] - pos[2]) / max(n_slices - 1, 1)))

        orient = [float(x) for x in slices[0].ImageOrientationPatient]
        row_cos = np.array(orient[:3])
        col_cos = np.array(orient[3:6])
        slice_cos = np.cross(row_cos, col_cos)

        affine = np.eye(4)
        affine[:3, 0] = row_cos * ps[1]
        affine[:3, 1] = col_cos * ps[0]
        affine[:3, 2] = slice_cos * slice_thick
        affine[:3, 3] = pos

        nib.save(nib.Nifti1Image(volume, affine), output_path)
        return True

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download LIDC-IDRI CT subset for MAISI")
    parser.add_argument("--num_scans", type=int, default=5, help="Number of CT scans to download")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Training data ratio")
    parser.add_argument("--workers", type=int, default=2, help="Parallel download workers (default 2)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp_download")
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print("LIDC-IDRI Chest CT Downloader for MAISI Training")
    print("=" * 60)
    print(f"  Target scans : {args.num_scans}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Workers      : {args.workers}")
    print()

    # Check existing
    existing = sorted(glob.glob(os.path.join(output_dir, "*.nii.gz")))
    if existing:
        print(f"  Found {len(existing)} existing NIfTI files (will skip).")

    # Step 1: Get series list
    ct_series = get_lidc_ct_series_list(args.num_scans)
    if not ct_series:
        print("ERROR: No CT series found. Check network connectivity.")
        sys.exit(1)

    # Step 2: Download & convert
    print(f"\n[Step 2] Downloading and converting {len(ct_series)} CT scans ...")
    nifti_files = list(existing)  # start with existing
    success = len(existing)
    fail = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_and_convert, info, output_dir, temp_dir): info
            for info in ct_series
        }
        for future in as_completed(futures):
            info = futures[future]
            try:
                path, ok = future.result()
                if ok and path:
                    success += 1
                    nifti_files.append(path)
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"  FATAL error for {info}: {e}")

            print(f"  Progress: {success + fail}/{len(ct_series)} (OK={success}, fail={fail})")

    # Clean up temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Deduplicate
    nifti_files = sorted(set(f for f in nifti_files if os.path.exists(f)))

    # Step 3: Split train/val
    print(f"\n[Step 3] Organizing dataset ...")
    n_train = int(args.train_ratio * len(nifti_files))
    train_files = nifti_files[:n_train]
    val_files = nifti_files[n_train:]

    print(f"  Training   : {len(train_files)}")
    print(f"  Validation : {len(val_files)}")

    # Step 4: Save data list JSON
    train_dicts = [{"image": f, "class": "ct"} for f in train_files]
    val_dicts = [{"image": f, "class": "ct"} for f in val_files]
    datalist_path = os.path.join(output_dir, "lidc_datalist.json")
    with open(datalist_path, "w") as f:
        json.dump({"training": train_dicts, "validation": val_dicts}, f, indent=2)

    print(f"\n  Data list saved: {datalist_path}")

    # Disk usage
    total_size = sum(os.path.getsize(f) for f in nifti_files if os.path.exists(f))
    print(f"  Total NIfTI size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Total NIfTI  : {len(nifti_files)}")
    print(f"  Training     : {len(train_files)}")
    print(f"  Validation   : {len(val_files)}")
    print(f"  Failed       : {fail}")
    print(f"  Output       : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
