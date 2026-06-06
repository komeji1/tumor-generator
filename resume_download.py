"""续传CT下载 — Python实现，自动重试，断点续传"""
import os, time, urllib.request

URL = "https://hf-mirror.com/datasets/MrGiovanni/AbdomenAtlas2.0Mini/resolve/main/AbdomenAtlas2.0Mini_ct_00000001_00000500.tar.gz"
OUTPUT = "C:/Users/33067/.claude/work/Mask/data/tmp/AbdomenAtlas2.0Mini_ct_00000001_00000500.tar.gz"
TOTAL = 30587802640

existing = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
print(f"Resuming from byte {existing} ({existing/TOTAL*100:.1f}%)")

max_retries = 20
for attempt in range(max_retries):
    try:
        req = urllib.request.Request(URL)
        if existing > 0:
            req.add_header('Range', f'bytes={existing}-')

        resp = urllib.request.urlopen(req, timeout=30)

        with open(OUTPUT, 'ab') as f:
            chunk_size = 8 * 1024 * 1024  # 8 MB
            downloaded = existing
            t0 = time.time()
            last_report = t0

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_report > 5:
                    pct = downloaded / TOTAL * 100
                    mb = downloaded / 1e6
                    speed = (downloaded - existing) / (now - t0) / 1e6 if now > t0 else 0
                    remaining = (TOTAL - downloaded) / (speed * 1e6) if speed > 0 else 0
                    print(f"\r  {mb:.0f}/{TOTAL/1e6:.0f} MB ({pct:.1f}%)  {speed:.1f} MB/s  ETA: {remaining:.0f}s", end='', flush=True)
                    last_report = now

        print(f"\nDone! {downloaded/TOTAL*100:.1f}%")
        break

    except Exception as e:
        print(f"\nAttempt {attempt+1}/{max_retries}: {e}")
        existing = os.path.getsize(OUTPUT)
        time.sleep(min(30, (attempt + 1) * 5))
else:
    print("All retries exhausted.")
