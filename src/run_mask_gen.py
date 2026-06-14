"""
Mask generation bridge: reads mask_config.json → calls Mask project directly.
Usage: python src/run_mask_gen.py [mask_config.json]
"""
import sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MASK_PROJECT_DIR

MASK_MAIN = os.path.join(MASK_PROJECT_DIR, "Step6", "src", "main.py")
MASK_CONFIG_TEMPLATE = os.path.join(MASK_PROJECT_DIR, "Step0", "config", "generation_config.json")
THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(THIS_DIR, "mask_config.json")

ORGAN_TYPE_MAP = {
    "liver": "liver_lesion", "pancreas": "pancreatic_lesion",
    "kidney": "kidney_lesion", "colon": "colon_lesion",
    "esophagus": "esophagus_tumor", "uterus": "endometrioma_tumor",
}
SIZE_WEIGHTS = {"tiny": 0, "small": 4, "medium": 2, "large": 1}


def translate_config(mask_cfg):
    """Turn mask_config.json entries into Mask project config format."""
    template = json.load(open(MASK_CONFIG_TEMPLATE, "r", encoding="utf-8"))

    cfg = {
        "project": template["project"],
        "data": template["data"],
        "size_categories": template["size_categories"],
        "shape": template["shape"],
        "placement": template["placement"],
        "preprocessing": template["preprocessing"],
        "output": template["output"],
        "logging": template["logging"],
        "organs": [],
    }

    global_naming = mask_cfg.get("naming")

    for entry in mask_cfg.get("organs", []):
        organ = entry["organ"]
        count = entry["count"]
        sizes = [s.strip() for s in entry.get("size_mix", "small").split(",")]

        cat_weights = {}
        for cat_name in template["size_categories"]["categories"]:
            cat_weights[cat_name] = SIZE_WEIGHTS.get(cat_name, 4) if cat_name in sizes else 0

        naming = entry.get("naming") or global_naming
        # Accept int/float, convert to string; skip empty
        if naming is not None and not isinstance(naming, str):
            naming = str(naming)
        naming_pattern = None
        if isinstance(naming, str) and naming.strip() and naming != "None":
            naming_pattern = naming if "{" in naming else f"{naming}_{{organ_type}}_{{sample_id}}"

        organ_entry = {
            "name": ORGAN_TYPE_MAP[organ],
            "organ_label_file": template["organs"][0]["organ_label_file"],
            "organ_name": organ,
            "count": count,
            "size_categories": cat_weights,
            "naming_pattern": naming_pattern,
        }
        cfg["organs"].append(organ_entry)

    return cfg


SIZE_INDEX_FILE = os.path.join(MASK_PROJECT_DIR, "output", "real_ct", "mask_size_index.json")

SIZE_THRESHOLDS = {"tiny": 524, "small": 4189, "medium": 33510, "large": 999999}


def _write_size_index():
    """Scan Mask output dir and write mask_size_index.json for fast size filtering."""
    import glob as _glob
    index = {}
    mask_root = os.path.join(MASK_PROJECT_DIR, "output", "real_ct")
    for organ_dir in sorted(os.listdir(mask_root)):
        organ_path = os.path.join(mask_root, organ_dir)
        if not os.path.isdir(organ_path):
            continue
        index[organ_dir] = {}
        for mf in sorted(_glob.glob(os.path.join(organ_path, "*.nii.gz"))):
            if "_mask" in mf:
                continue
            try:
                n = int((nib.load(mf).get_fdata() > 0).sum())
                # Classify size
                cat = "tiny"
                for name in ["small", "medium", "large"]:
                    if n > SIZE_THRESHOLDS[name]:
                        cat = name
                index[organ_dir][os.path.basename(mf)] = {"voxels": n, "size_category": cat}
            except Exception:
                pass
    json.dump(index, open(SIZE_INDEX_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    total = sum(len(v) for v in index.values())
    print(f"  Size index: {total} masks → {SIZE_INDEX_FILE}")


def main(config_path=None):
    cfg_path = config_path or CONFIG_FILE
    if not os.path.exists(cfg_path):
        print(f"ERROR: {cfg_path} not found")
        return

    mask_cfg = json.load(open(cfg_path, "r", encoding="utf-8"))

    # Import Mask project modules
    for step in ['Step0', 'Step1', 'Step2', 'Step3', 'Step4', 'Step5']:
        step_src = os.path.join(MASK_PROJECT_DIR, step, 'src')
        if step_src not in sys.path:
            sys.path.insert(0, step_src)

    import importlib.util
    spec = importlib.util.spec_from_file_location("mask_main", MASK_MAIN)
    mask_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_main)
    generate_batch = mask_main.generate_batch

    gen_cfg = translate_config(mask_cfg)

    # Point paths to actual data
    gen_cfg["data"]["ct_dir"] = os.path.join(MASK_PROJECT_DIR, "data", "ct")
    gen_cfg["data"]["organ_label_dir"] = os.path.join(MASK_PROJECT_DIR, "data", "organ_labels")
    gen_cfg["project"]["output_dir"] = os.path.join(MASK_PROJECT_DIR, "output", "real_ct")
    gen_cfg["logging"]["log_file"] = os.path.join(MASK_PROJECT_DIR, "output", "real_ct", "generation_log.json")
    gen_cfg["logging"]["stats_file"] = os.path.join(MASK_PROJECT_DIR, "output", "real_ct", "statistics.json")
    # Naming: first organ's naming_pattern overrides output default
    for o in gen_cfg["organs"]:
        if o.get("naming_pattern"):
            gen_cfg["output"]["naming_pattern"] = o["naming_pattern"]
            break

    print(f"Generating masks...")
    for o in gen_cfg["organs"]:
        sizes = [k for k, v in o["size_categories"].items() if v > 0]
        print(f"  {o['name']}: {o['count']} masks, sizes={sizes}")

    results = generate_batch(gen_cfg, rng_seed=random.randint(0, 2**31))
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"Done: {ok} OK, {fail} failed → {MASK_PROJECT_DIR}/output/real_ct/{{organ}}/")

    # Write size index for fast lookup by prompt_runner
    _write_size_index()
