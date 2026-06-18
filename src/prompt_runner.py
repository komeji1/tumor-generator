"""
肿瘤生成提示词执行器 (Prompt Runner)

双层接口:
  1. JSON 配置文件 → 批量精确控制 (推荐)
  2. CLI 命令行 → 快速单次生成

用法:
  python prompt_runner.py config.json           # JSON 配置
  python prompt_runner.py --quick               # CLI 快速模式

设计依据: AI肿瘤生成提示词设计分析.md §八 + DiffTumor 实际条件编码
"""

import sys, os, time, json, glob, re, argparse, random, textwrap, warnings
import numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CT_DIR, LABEL_DIR, MASK_DIR, FULL_CT_DIR, PATCH_96_DIR, TEMP_DIR

sys.path.insert(0, os.path.dirname(__file__))
from embed_to_full_ct import embed_tumor_full_ct, save_full_ct

# ─── 器官/权重路由 ───
ORGAN_SHORT = {
    "liver": "liver", "pancreas": "pancreas", "kidney": "kidney",
    "colon": "colon", "esophagus": "esophagus", "uterus": "uterus",
}
ORGAN_TYPE_MAP = {  # organ → mask_dir_name
    "liver": "liver_lesion", "pancreas": "pancreatic_lesion",
    "kidney": "kidney_lesion", "colon": "colon_lesion",
    "esophagus": "esophagus_tumor", "uterus": "endometrioma_tumor",
}
ORGAN_SEG_FILE = {
    "liver": "liver.nii.gz", "pancreas": "pancreas.nii.gz",
    "kidney": None,  # combined L+R
    "colon": "colon.nii.gz", "esophagus": "esophagus.nii.gz",
    "uterus": "uterus.nii.gz",
}
ORGAN_WEIGHT = {
    "liver": ("liver_early.pt", "liver_noearly.pt"),
    "pancreas": ("pancreas_early.pt", "pancreas_noearly.pt"),
    "kidney": ("kidney_early.pt", "kidney_noearly.pt"),
    "colon": ("colon_early.pt", "colon_early.pt"),     # 训练中
    "esophagus": ("liver_early.pt", "liver_early.pt"),  # 零样本
    "uterus": ("liver_early.pt", "liver_early.pt"),    # 零样本
}
SIZE_CATEGORIES = {
    "tiny":   {"r_mm": [1, 5],   "max_vox": 524,   "phase": "early"},
    "small":  {"r_mm": [5, 10],  "max_vox": 4189,  "phase": "early"},
    "medium": {"r_mm": [10, 20], "max_vox": 33510, "phase": "noearly"},
    "large":  {"r_mm": [20, 50], "max_vox": 999999,"phase": "noearly"},
}
# max_vox: approximate max voxel count at 1mm³ for sphere radius = r_max
OUTPUT_FORMATS = ["full_ct", "patch_96", "both"]



# ═══════════════════════════════════════════════════════════════
# 核心: 单任务执行
# ═══════════════════════════════════════════════════════════════

def resolve_mask(organ: str, bdmap_id: str = None, size_category: str = None,
                 mask_index: int = 0, mask_file: str = None) -> str:
    """根据条件解析/选择肿瘤 mask 文件路径。size_category 过滤只选对应尺寸的 mask。"""
    mask_dir = os.path.join(MASK_DIR, ORGAN_TYPE_MAP[organ])

    if mask_file:
        path = os.path.join(mask_dir, mask_file)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Mask not found: {path}")

    all_masks = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
    if not all_masks:
        raise FileNotFoundError(f"No masks found in {mask_dir}")
    candidates = all_masks

    # 按 size_category 过滤 — 找不到则自动降级
    if size_category and size_category in SIZE_CATEGORIES:
        keys = list(SIZE_CATEGORIES.keys())
        cat_idx = keys.index(size_category)
        index_file = os.path.join(MASK_DIR, "mask_size_index.json")
        org_key = ORGAN_TYPE_MAP[organ]
        idx_data = None
        if os.path.exists(index_file):
            try:
                idx_data = json.load(open(index_file, "r", encoding="utf-8"))
            except Exception:
                pass
        # Try requested size, then downgrade
        for attempt in range(cat_idx + 1):
            cat = keys[cat_idx - attempt]
            max_vox = SIZE_CATEGORIES[cat]["max_vox"]
            min_vox = SIZE_CATEGORIES[keys[cat_idx - attempt - 1]]["max_vox"] if cat_idx - attempt > 0 else 0
            sized = []
            if idx_data:
                sized = [m for m in candidates if any(
                    min_vox < info.get("voxels", 0) <= max_vox
                    for name, info in idx_data.get(org_key, {}).items()
                    if m.endswith(name))]
            else:
                for m in candidates:
                    try:
                        n = int((nib.load(m).get_fdata() > 0).sum())
                        if min_vox < n <= max_vox:
                            sized.append(m)
                    except Exception:
                        pass
            if sized:
                if cat != size_category:
                    print(f"  NOTE: no {size_category} masks for {organ}, downgraded to {cat}")
                candidates = sized
                break

    # 按 BDMAP ID 筛选
    if bdmap_id:
        candidates = [m for m in candidates if bdmap_id in os.path.basename(m)]
        if not candidates:
            avail = sorted(set(
                re.search(r'BDMAP_(\d{8})', os.path.basename(m)).group(0)
                for m in all_masks
                if re.search(r'BDMAP_(\d{8})', os.path.basename(m))
            ))
            raise FileNotFoundError(
                f"No masks for {bdmap_id} in {organ}.\n"
                f"  Available BDMAP IDs for {organ}: {', '.join(avail[:10])}{'...' if len(avail)>10 else ''}\n"
                f"  Tip: set \"host_ct\": null for random selection"
            )

    if mask_index >= len(candidates):
        mask_index = mask_index % len(candidates)
    return candidates[mask_index]


def resolve_ct(organ: str, bdmap_id: str, mask_path: str) -> tuple:
    """解析 CT 和器官 mask 路径"""
    # 从 mask 文件名提取 BDMAP ID
    m = re.search(r'BDMAP_(\d{8})', os.path.basename(mask_path))
    if m:
        bdmap_id = m.group(0)

    ct_path = os.path.join(CT_DIR, bdmap_id, "ct.nii.gz")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT not found: {ct_path}")

    seg_file = ORGAN_SEG_FILE[organ]
    if seg_file:
        og_path = os.path.join(LABEL_DIR, bdmap_id, "segmentations", seg_file)
    else:
        # Kidney: combine left+right
        og_path = _get_kidney_combined(bdmap_id)

    if not os.path.exists(og_path):
        raise FileNotFoundError(f"Organ mask not found: {og_path}")

    return ct_path, og_path


def _get_kidney_combined(bdmap_id: str) -> str:
    """组合左肾+右肾 mask"""
    out = os.path.join(TEMP_DIR, f"kidney_combined_{bdmap_id}.nii.gz")
    if os.path.exists(out):
        return out
    kl = os.path.join(LABEL_DIR, bdmap_id, "segmentations", "kidney_left.nii.gz")
    kr = os.path.join(LABEL_DIR, bdmap_id, "segmentations", "kidney_right.nii.gz")
    kl_data = nib.load(kl).get_fdata() > 0
    kr_data = nib.load(kr).get_fdata() > 0
    ref = nib.load(kl)
    nib.save(nib.Nifti1Image((kl_data | kr_data).astype(np.uint8),
                             ref.affine, ref.header), out)
    return out


def run_one_task(task: dict, device: str = "cpu") -> dict:
    """执行单个生成任务, 返回结果字典"""
    organ = task["organ"]
    oshort = ORGAN_SHORT[organ]
    organ_type = ORGAN_TYPE_MAP[organ]
    size_cat = task.get("size_category", "small")
    radius_mm = task.get("radius_mm", None)
    phase = task.get("phase") or SIZE_CATEGORIES.get(size_cat, {}).get("phase", "early")

    # 如果此器官没有 noearly 权重，强制降级为 early
    early_wt, noearly_wt = ORGAN_WEIGHT[organ]
    if phase == "noearly" and early_wt == noearly_wt:
        print(f"  NOTE: {organ} has no noearly weights, downgrading phase to early")
        phase = "early"

    output_fmt = task.get("output", "both")
    bdmap_id = task.get("host_ct")

    t_start = time.time()

    # 1. 解析 mask (host_ct=null 时随机 mask_index)
    mask_idx = task.get("mask_index", 0)
    if bdmap_id is None and "mask_index" not in task:
        mask_idx = random.randint(0, 99)  # random, not always largest
    mask_path = resolve_mask(
        organ, bdmap_id=bdmap_id,
        size_category=size_cat,
        mask_index=mask_idx,
        mask_file=task.get("mask_file"),
    )

    # 2. 解析 CT + 器官 mask
    ct_path, og_path = resolve_ct(organ, bdmap_id, mask_path)

    # 3. 预检查: mask 体积
    tm_data = nib.load(mask_path).get_fdata() > 0
    n_vox = int(tm_data.sum())
    if n_vox < 10:
        return {"status": "skip", "reason": f"mask too small ({n_vox} vox)",
                "mask_path": mask_path}

    # 4. 生成 full-CT (核心)
    result = {"status": "ok", "organ": organ, "size_category": size_cat,
              "phase": phase, "mask_path": mask_path, "mask_voxels": n_vox}

    try:
        eta = task.get("eta", 0.0)
        # Optional metadata (docs only; mask+CT already encode these implicitly)
        meta_info = {k: task[k] for k in ("radius_mm", "position", "hu_stats") if k in task}
        fc, fm, aff, meta = embed_tumor_full_ct(
            ct_path, og_path, mask_path, organ_type, device, phase, output_fmt, eta=eta)
        meta["user_spec"] = meta_info

        base = os.path.basename(mask_path).replace(".nii.gz", "")
        base = re.sub(r'_t(\d)', r'_s\1', base)  # _t00→_s00, won't corrupt "tumor"
        if task.get("output_name"):
            base = task["output_name"]

        # 5. 保存
        out_paths = []
        if output_fmt in ("full_ct", "both"):
            p = save_full_ct(fc, fm, aff, organ_type, base)
            out_paths.append(p)
        if output_fmt in ("patch_96", "both") and "patch_96_hu" in meta:
            patch_dir = os.path.join(PATCH_96_DIR, organ_type)
            os.makedirs(patch_dir, exist_ok=True)
            patch_aff = np.diag([1.0, 1.0, 1.0, 1.0])
            patch_path = os.path.join(patch_dir, f"{base}.nii.gz")
            v = 2
            while os.path.exists(patch_path):
                patch_path = os.path.join(patch_dir, f"{base}_v{v}.nii.gz")
                v += 1
            nib.save(nib.Nifti1Image(meta["patch_96_hu"].astype(np.float32), patch_aff), patch_path)
            base_mask = os.path.join(patch_dir, f"{base}_mask.nii.gz")
            if not os.path.exists(base_mask):
                patch_mask = meta.get("patch_96_mask", np.zeros((96, 96, 96), dtype=np.uint8))
                nib.save(nib.Nifti1Image(patch_mask, patch_aff), base_mask)
            out_paths.append(patch_path)

        dt = time.time() - t_start
        t_hu = fc[fm > 0] if fm.sum() > 0 else np.array([0])
        result.update({
            "output_paths": out_paths, "time_s": round(dt, 1),
            "tumor_hu_mean": round(float(t_hu.mean()), 1),
            "tumor_hu_std": round(float(t_hu.std()), 1),
            "weight_used": ORGAN_WEIGHT[organ][0 if phase == "early" else 1],
        })
        what = output_fmt.replace("_", "-")
        print(f"  OK  {organ}/{size_cat}  vox={n_vox:,}  HU={t_hu.mean():.0f}  "
              f"time={dt:.0f}s  → {what}")

    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)
        print(f"  FAIL  {organ}/{size_cat}: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
# JSON 配置接口
# ═══════════════════════════════════════════════════════════════

EXAMPLE_CONFIG = {
    "_description": "肿瘤生成提示词配置 — 依据 AI肿瘤生成提示词设计分析.md §八-B",
    "_note": "DiffTumor 实际使用 cond=concat([z_healthy, mask_downsampled]), 四要素隐式编码于mask+CT中",

    "tasks": [
        {
            "organ": "liver",
            "size_category": "medium",
            "host_ct": "BDMAP_00000012",
            "mask_index": 0,
            "phase": "noearly",
            "output": "full_ct",
            "output_name": "demo_liver_medium_noearly",
        },
        {
            "organ": "esophagus",
            "size_category": "tiny",
            "host_ct": None,
            "mask_index": 0,
            "phase": "early",
            "output": "full_ct",
        },
        {
            "organ": "kidney",
            "size_category": "small",
            "host_ct": "BDMAP_00000019",
            "mask_index": 2,
            "phase": "early",
            "output": "full_ct",
        },
    ],

    "global": {
        "device": "cpu",
        "output_root": None,
    }
}


def run_config(config: dict) -> list:
    """执行 JSON 配置文件中的所有任务"""
    tasks = config.get("tasks", [])
    if not tasks:
        print("ERROR: no tasks in config")
        return []

    global_cfg = config.get("global", {})
    device = global_cfg.get("device", "cpu")

    print(f"{'='*60}")
    print(f"Prompt Runner: {len(tasks)} task(s)")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    results = []
    ok = skip = fail = 0
    t_start = time.time()

    # 展开 repeat 字段: {"organ":"liver", "repeat":5} → 5 个任务
    expanded = []
    for task in tasks:
        n = max(1, task.get("repeat", 1))
        user_set_idx = "mask_index" in task
        for j in range(n):
            t = dict(task)
            if user_set_idx:
                t["mask_index"] = task["mask_index"] + j
            t.pop("repeat", None)
            expanded.append(t)

    for i, task in enumerate(expanded):
        # 合并 global 设置
        task.setdefault("device", device)
        p = task.get("phase") or SIZE_CATEGORIES.get(
            task.get("size_category", "small"), {}).get("phase", "early")
        print(f"[{i+1}/{len(expanded)}] {task.get('organ')}/{task.get('size_category','?')}"
              f"  phase={p}  host={task.get('host_ct','any')}")

        r = run_one_task(task, device)
        results.append(r)

        if r["status"] == "ok":
            ok += 1
        elif r["status"] == "skip":
            skip += 1
        else:
            fail += 1

    total_t = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Complete: {total_t/60:.1f}min | OK={ok} Skip={skip} Fail={fail}")
    print(f"Output: {FULL_CT_DIR}/{{organ}}/")
    return results


# ═══════════════════════════════════════════════════════════════
# CLI 快速模式
# ═══════════════════════════════════════════════════════════════

def cli_quick(args):
    """CLI 快速模式: 命令行参数 → JSON 配置 → 执行"""
    task = {
        "organ": args.organ,
        "size_category": args.size,
        "phase": args.phase or None,
        "host_ct": args.host,
        "mask_index": args.index,
        "output": args.output,
    }
    if args.name:
        task["output_name"] = args.name

    config = {"tasks": [task], "global": {"device": args.device}}
    return run_config(config)


def print_example():
    """打印示例配置"""
    print(json.dumps(EXAMPLE_CONFIG, indent=2, ensure_ascii=False))


def list_available(args):
    """列出可用的 mask 资源"""
    print(f"{'Organ':<20} {'Masks':>6}  {'Host CTs':>10}  {'Weight (early)':<30}  {'Weight (noearly)':<30}")
    print("-" * 110)
    for organ in ["liver", "pancreas", "kidney", "colon", "esophagus", "uterus"]:
        mask_dir = os.path.join(MASK_DIR, ORGAN_TYPE_MAP[organ])
        masks = glob.glob(os.path.join(mask_dir, "*.nii.gz")) if os.path.isdir(mask_dir) else []
        n_masks = len(masks)

        # Count unique host CTs
        bdmaps = set()
        for m in masks:
            m2 = re.search(r'BDMAP_(\d{8})', os.path.basename(m))
            if m2:
                bdmaps.add(m2.group(0))
        n_cts = len(bdmaps)

        we, wn = ORGAN_WEIGHT[organ]
        print(f"{organ:<20} {n_masks:>6}  {n_cts:>10}  {we:<30}  {wn:<30}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="肿瘤生成提示词执行器 — JSON配置 或 CLI快速模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例:
          python prompt_runner.py config.json          # JSON 配置批量执行
          python prompt_runner.py --example            # 输出示例 JSON 配置
          python prompt_runner.py --list               # 列出可用资源
          python prompt_runner.py --quick --organ liver --size medium  # CLI 快速模式
        """))
    parser.add_argument("config", nargs="?", help="JSON 配置文件路径")
    parser.add_argument("--quick", action="store_true", help="CLI 快速模式")
    parser.add_argument("--example", action="store_true", help="输出示例 JSON 配置")
    parser.add_argument("--list", action="store_true", help="列出可用的 mask 和 CT 资源")
    parser.add_argument("--organ", choices=list(ORGAN_SHORT.keys()), help="器官类型")
    parser.add_argument("--size", choices=list(SIZE_CATEGORIES.keys()), default="small",
                       help="肿瘤尺寸类别 (默认: small)")
    parser.add_argument("--phase", choices=["early", "noearly"],
                       help="权重阶段 (默认: 按尺寸自动选择)")
    parser.add_argument("--host", help="宿主 CT 的 BDMAP ID (默认: 随机)")
    parser.add_argument("--index", type=int, default=0, help="Mask 序号 (默认: 0)")
    parser.add_argument("--output", choices=OUTPUT_FORMATS, default="both",
                       help="输出格式 (默认: both)")
    parser.add_argument("--name", help="自定义输出文件名")
    parser.add_argument("--device", default="cpu", help="计算设备 (默认: cpu)")

    args = parser.parse_args()

    if args.example:
        print_example()
    elif args.list:
        list_available(args)
    elif args.quick:
        if not args.organ:
            parser.error("--quick 模式需要 --organ 参数")
        cli_quick(args)
    elif args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {args.config}")
            print(f"  {e}")
            return

        # Auto-detect config type: "organs" (no "tasks") → mask, "tasks" → tumor
        if "organs" in config and "tasks" not in config:
            from run_mask_gen import main as run_mask
            run_mask(args.config)
        else:
            run_config(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nFATAL: {e}")
        traceback.print_exc()

