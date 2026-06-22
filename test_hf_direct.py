"""Quick test: can we reach HuggingFace CDN directly?"""
import requests
import time

url = "https://huggingface.co/datasets/hourouu/LIDC_IDRI_PROCESSED/resolve/main/images/LIDC-IDRI-0071_img.nii.gz"

print("Testing HuggingFace CDN (direct, no proxy)...")
try:
    start = time.time()
    r = requests.get(url, timeout=60, stream=True, allow_redirects=True)
    print(f"Status: {r.status_code}")
    print(f"Headers: {dict(r.headers)}")
    total = int(r.headers.get("content-length", 0))
    print(f"Content-Length: {total / 1024 / 1024:.1f} MB")
    elapsed = time.time() - start
    print(f"Time to get headers: {elapsed:.1f}s")

    # Try downloading 1MB to test actual transfer speed
    downloaded = 0
    for chunk in r.iter_content(chunk_size=65536):
        downloaded += len(chunk)
        if downloaded >= 1_000_000:
            break
    elapsed2 = time.time() - start
    print(f"Downloaded {downloaded / 1024 / 1024:.2f} MB in {elapsed2:.1f}s")
    if elapsed2 > 0:
        speed = downloaded / elapsed2 / 1024 / 1024
        print(f"Speed: {speed:.2f} MB/s")
    r.close()
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\nDone.")