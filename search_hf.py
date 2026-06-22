import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
from huggingface_hub import HfApi, hf_hub_url

api = HfApi()

print("Checking file sizes in hourouu/LIDC_IDRI_PROCESSED ...")
files = api.list_repo_files("hourouu/LIDC_IDRI_PROCESSED", repo_type="dataset")
nii_files = [f for f in files if f.endswith('.nii.gz')]

# Get sizes for a sample of files
print(f"Total NIfTI files: {len(nii_files)}")

# Try to get file info via repo_info
info = api.repo_info("hourouu/LIDC_IDRI_PROCESSED", repo_type="dataset")
print(f"Repo size_on_disk: {getattr(info, 'size_on_disk', 'N/A')}")

# Check a few individual files for size
sample_files = nii_files[:3]
for f in sample_files:
    try:
        url = hf_hub_url("hourouu/LIDC_IDRI_PROCESSED", f, repo_type="dataset")
        import requests
        session = requests.Session()
        session.proxies = {"https": "http://127.0.0.1:7890", "http": "http://127.0.0.1:7890"}
        r = session.head(url, timeout=30, allow_redirects=True)
        size = int(r.headers.get("content-length", 0))
        print(f"  {f}: {size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"  {f}: Error ({e})")

# Estimate total
if len(nii_files) > 0:
    print(f"\nEstimating total size (assuming ~60-80MB per file):")
    est_low = len(nii_files) * 60 / 1024  # GB
    est_high = len(nii_files) * 80 / 1024  # GB
    print(f"  Low estimate: {est_low:.1f} GB ({len(nii_files)} files x 60MB)")
    print(f"  High estimate: {est_high:.1f} GB ({len(nii_files)} files x 80MB)")

print(f"\nBut we only need 450 files, estimated:")
print(f"  ~26-35 GB for 450 files")