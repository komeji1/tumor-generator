"""
Tumor 项目统一配置

换机器只需:
  1. 复制整个 Tumor/ 目录
  2. 修改 Tumor/paths.json 中的外部路径
  3. 安装依赖: pip install torch nibabel SimpleITK numpy scipy

所有内部路径 (checkpoints/, output/, trained_weights/) 自动相对项目根目录。
"""
import os, json

# ─── 项目根目录 (config.py 所在目录) ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── 加载外部路径配置 ───
_paths_file = os.path.join(PROJECT_ROOT, "paths.json")
if os.path.exists(_paths_file):
    with open(_paths_file, "r", encoding="utf-8") as f:
        _external = json.load(f)
else:
    _external = {}

# ─── 内部路径 (相对于项目根, 换机器无需修改) ───
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "output")
TRAINED_DIR     = os.path.join(PROJECT_ROOT, "trained_weights")

# ─── 外部路径 (机器相关, 在 paths.json 中配置) ───
MASK_PROJECT_DIR   = _external.get("mask_project_dir",
    r"C:\Users\33067\.claude\work\Mask")
DIFFTUMOR_REPO_DIR = _external.get("diffumor_repo_dir",
    r"D:\Users\33067\claude-data\DiffTumor\STEP3.SegmentationModel")
DATA_DOWNLOAD_DIR  = _external.get("data_download_dir",
    r"D:\Users\33067\claude-data\downloads")
TEMP_DIR           = _external.get("temp_dir",
    os.path.join(DATA_DOWNLOAD_DIR, "_tmp"))

# ─── 派生路径 ───
# CT + 器官标签
CT_DIR    = os.path.join(MASK_PROJECT_DIR, "data", "ct")
LABEL_DIR = os.path.join(MASK_PROJECT_DIR, "data", "organ_labels")
MASK_DIR  = os.path.join(MASK_PROJECT_DIR, "output", "real_ct")

# 权重
VQGAN_CKPT    = os.path.join(CHECKPOINTS_DIR, "AutoencoderModel", "AutoencoderModel.ckpt")
DIFFUSION_DIR = os.path.join(CHECKPOINTS_DIR, "DiffusionModel")

# 训练数据
COLON_CT_DIR    = os.path.join(DATA_DOWNLOAD_DIR, "colon_train_data", "CT")
COLON_LABEL_DIR = os.path.join(DATA_DOWNLOAD_DIR, "colon_train_data", "Labels", "colon_tumor_early")
COLON_IDX_FILE  = os.path.join(DATA_DOWNLOAD_DIR, "colon_train_data", "cross_eval",
                               "colon_tumor_data_early_fold", "real_tumor_train_0.txt")

# 输出
FULL_CT_DIR = os.path.join(OUTPUT_DIR, "full_ct")
PATCH_96_DIR = os.path.join(OUTPUT_DIR, "synthetic_ct")

# 确保必要目录存在
os.makedirs(FULL_CT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)


def verify():
    """检查关键文件是否存在, 打印状态"""
    checks = {
        "VQGAN": VQGAN_CKPT,
        "liver_early": os.path.join(DIFFUSION_DIR, "liver_early.pt"),
        "Mask project": MASK_PROJECT_DIR,
        "DiffTumor repo": DIFFTUMOR_REPO_DIR,
        "Data downloads": DATA_DOWNLOAD_DIR,
    }
    for name, path in checks.items():
        ok = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{ok}] {name}: {path}")
