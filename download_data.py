"""
数据下载脚本 — 获取 AbdomenAtlas2.0Mini 子集
目标: 足够 6器官×20样本 = 120 masks 的 CT + 标签数据

来源:
    CT:   HuggingFace datasets/MrGiovanni/AbdomenAtlas2.0Mini
    标签: JHU http://www.cs.jhu.edu/~zongwei/dataset/

用法: python download_data.py
"""

import os, sys, tarfile, urllib.request, time, shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TARGET_DIR = os.path.join(DATA_DIR, 'AbdomenAtlas2.0')
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'tmp'), exist_ok=True)

def download(url, dest, desc=""):
    """带进度条的下载"""
    print(f"\n[{desc}] Downloading...")
    print(f"  URL: {url[:100]}...")

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end='', flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"\n  Done: {size_mb:.1f} MB")

# ============================================================
# Step 1: 下载标签 (约几百 MB, 单个文件)
# ============================================================
LABEL_URL = "http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlas2.0Mini_label.tar.gz"
label_tar = os.path.join(DATA_DIR, 'tmp', 'label.tar.gz')

if not os.path.exists(label_tar):
    download(LABEL_URL, label_tar, "Step 1: 标签")
else:
    print(f"\n[Step 1] 标签已存在: {label_tar} ({os.path.getsize(label_tar)/1e6:.0f} MB)")

# 解压标签
label_extract_dir = os.path.join(DATA_DIR, 'tmp', 'label_extracted')
if not os.path.exists(label_extract_dir):
    print("  解压标签...")
    os.makedirs(label_extract_dir, exist_ok=True)
    with tarfile.open(label_tar, 'r:gz') as tf:
        tf.extractall(label_extract_dir)

    # 检查结构
    for root, dirs, files in os.walk(label_extract_dir):
        depth = root.replace(label_extract_dir, '').count(os.sep)
        if depth <= 3:
            print(f"  {root}")
            if depth <= 2 and len(files) > 0:
                for f in files[:5]:
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    ... and {len(files)-5} more files")
else:
    print("  标签已解压")

# 复制 label 到目标目录
label_src = os.path.join(label_extract_dir, 'label')
if os.path.isdir(label_src):
    label_dst = os.path.join(TARGET_DIR, 'label')
    if not os.path.exists(label_dst):
        print(f"  复制 label/ → {label_dst}")
        shutil.copytree(label_src, label_dst)
    else:
        print(f"  label/ 已存在: {label_dst}")

# ============================================================
# Step 2: 下载 CT (每 500 个 CT 一个 tar.gz, 约 2-5 GB/个)
# 我们只需要 ~30-50 个 CT 即可满足 6×20 的需求
# 下载第一个分卷 (含前 500 个 CT) 足够
# ============================================================
CT_BASE = "https://huggingface.co/datasets/MrGiovanni/AbdomenAtlas2.0Mini/resolve/main"
CT_FILE = "AbdomenAtlas2.0Mini_ct_00000001_00000500.tar.gz"
CT_URL = f"{CT_BASE}/{CT_FILE}?download=true"
ct_tar = os.path.join(DATA_DIR, 'tmp', CT_FILE)

if not os.path.exists(ct_tar):
    print(f"\n[Step 2] CT 数据 (第一个分卷, 约 500 个扫描)")
    print(f"  注意: 此文件较大 (~2-5 GB), 可能需要几分钟")
    try:
        download(CT_URL, ct_tar, "Step 2: CT")
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"  请手动下载: {CT_URL}")
        sys.exit(1)
else:
    size_gb = os.path.getsize(ct_tar) / 1e9
    print(f"\n[Step 2] CT 已存在: {ct_tar} ({size_gb:.1f} GB)")

# 解压 CT
ct_extract_dir = os.path.join(DATA_DIR, 'tmp', 'ct_extracted')
if not os.path.exists(ct_extract_dir):
    print("  解压 CT (这可能需要几分钟)...")
    os.makedirs(ct_extract_dir, exist_ok=True)
    with tarfile.open(ct_tar, 'r:gz') as tf:
        tf.extractall(ct_extract_dir)

    # 统计
    ct_dirs = [d for d in os.listdir(ct_extract_dir) if os.path.isdir(os.path.join(ct_extract_dir, d))]
    print(f"  解压完成: {len(ct_dirs)} 个 CT 扫描")
else:
    ct_dirs = [d for d in os.listdir(ct_extract_dir) if os.path.isdir(os.path.join(ct_extract_dir, d))]
    print(f"  CT 已解压: {len(ct_dirs)} 个扫描")

# 复制 CT 到目标目录
ct_src = ct_extract_dir
ct_dst = os.path.join(TARGET_DIR, 'ct')
if not os.path.exists(ct_dst):
    print(f"  复制 ct/ → {ct_dst}")
    os.makedirs(ct_dst, exist_ok=True)
    copied = 0
    for d in sorted(os.listdir(ct_src)):
        src_path = os.path.join(ct_src, d)
        if os.path.isdir(src_path):
            dst_path = os.path.join(ct_dst, d)
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
                copied += 1
    print(f"  复制了 {copied} 个 CT 扫描")
else:
    ct_count = len([d for d in os.listdir(ct_dst) if os.path.isdir(os.path.join(ct_dst, d))])
    print(f"  ct/ 已存在: {ct_count} 个扫描")

# ============================================================
# Step 3: 汇总
# ============================================================
print("\n" + "=" * 60)
print("下载完成")
print("=" * 60)

ct_count = 0
if os.path.isdir(ct_dst):
    ct_count = len([d for d in os.listdir(ct_dst) if os.path.isdir(os.path.join(ct_dst, d))])

label_count = 0
if os.path.isdir(label_dst):
    label_count = len([d for d in os.listdir(label_dst) if os.path.isdir(os.path.join(label_dst, d))])

print(f"  CT 扫描: {ct_count} 个")
print(f"  标签目录: {label_count} 个")
print(f"  数据目录: {TARGET_DIR}")

# 检查数据结构
if label_count > 0:
    sample_dir = os.path.join(label_dst, sorted(os.listdir(label_dst))[0])
    print(f"\n  示例标签结构 ({os.path.basename(sample_dir)}):")
    for root, dirs, files in os.walk(sample_dir):
        for f in sorted(files)[:10]:
            fp = os.path.join(root, f)
            size_kb = os.path.getsize(fp) / 1024
            rel = os.path.relpath(fp, sample_dir)
            print(f"    {rel} ({size_kb:.0f} KB)")
        break

if ct_count > 0:
    sample_ct = os.path.join(ct_dst, sorted(os.listdir(ct_dst))[0])
    print(f"\n  示例 CT 结构 ({os.path.basename(sample_ct)}):")
    for f in sorted(os.listdir(sample_ct))[:5]:
        print(f"    {f}")
