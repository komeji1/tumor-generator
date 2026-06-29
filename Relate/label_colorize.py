#!/usr/bin/env python3
"""
label_colorize.py — 将 MAISI 132类 label NIfTI 上色为 RGB 可视化图像

为每个 label (0-200) 分配一个 RGB 颜色，生成彩色 NIfTI 和 PNG 截图。
肿瘤标签使用醒目的红色系，预留第四步新增肿瘤的颜色空间。

用法:
  # 单文件
  python label_colorize.py --label path/to/merged_label.nii.gz

  # 指定输出目录
  python label_colorize.py --label path/to/merged_label.nii.gz --output-dir output/colored

  # 批量 — 自动扫描任务目录中的 04_final_merged/
  python label_colorize.py --scan-dir output/

  # 只生成PNG截图 (不生成NIfTI)
  python label_colorize.py --label path/to/merged_label.nii.gz --png-only

输出:
  - {basename}_colored.nii.gz  — RGB NIfTI (每voxel是3通道uint8颜色)
  - {basename}_colored.png     — 中间切片PNG截图
  - {basename}_legend.png      — 颜色对照表
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image

# ── 兼容 Windows GBK 终端 ──
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass


# ══════════════════════════════════════════════════════════════
#  颜色映射: MAISI 132类标签 → RGB
# ══════════════════════════════════════════════════════════════
# 设计原则:
#   1. 同类器官用相近色系 (如5个肺叶都是蓝色系的5种深浅)
#   2. 肿瘤标签用醒目红色系，与正常器官区分
#   3. 预留第四步新增肿瘤的颜色 (133-140 红色系不同深浅)
#   4. 背景=纯黑，body envelope=深灰

LABEL_COLORS: dict[int, tuple[int, int, int]] = {
    # ── 背景 / 特殊 ──
    0:   (0, 0, 0),         # 背景 (空气) — 纯黑
    200: (40, 40, 40),      # body envelope — 深灰

    # ── 主要器官 — 高饱和度 ──
    1:   (180, 40, 40),     # 肝脏 — 红褐色 (正常肝)
    3:   (160, 60, 160),    # 脾脏 — 紫色
    4:   (200, 160, 40),    # 胰腺 — 金黄
    5:   (40, 120, 200),    # 右肾 — 蓝 (原始label_dict里5是右肾)
    14:  (40, 80, 200),     # 左肾 — 深蓝 (14是左肾)
    10:  (80, 200, 80),     # 胆囊 — 绿色
    11:  (160, 120, 80),    # 食管 — 土黄
    12:  (120, 160, 200),   # 胃 — 浅蓝灰
    13:  (140, 180, 60),    # 十二指肠 — 黄绿
    15:  (200, 120, 160),   # 膀胱 — 浅粉
    17:  (60, 120, 60),     # 门静脉/脾静脉 — 暗绿
    19:  (180, 160, 80),    # 小肠 — 浅土黄
    62:  (160, 100, 40),    # 结肠 — 橙褐

    # ── 肿瘤标签 — 醒目红色系 ──
    26:  (255, 0, 0),       # 肝肿瘤 — 纯红 ⬅ 桥接画的新肿瘤
    23:  (255, 40, 40),     # 肺肿瘤 — 亮红
    24:  (255, 80, 0),      # 胰腺肿瘤 — 红橙
    27:  (255, 120, 0),     # 结肠癌 — 橙红
    116: (255, 0, 80),      # 左肾囊肿 — 红粉
    117: (255, 0, 120),     # 右肾囊肿 — 红紫
    128: (200, 0, 0),       # 骨病变 — 深红

    # ── 第四步新增肿瘤标签 133-144 — 红色系不同深浅 ──
    133: (255, 60, 60),     # 食管癌 — 浅红
    134: (220, 0, 60),      # 胃癌 — 深红偏粉
    135: (255, 100, 40),    # 膀胱癌 — 红橙
    136: (220, 40, 0),      # 前列腺癌 — 深红橙
    137: (255, 0, 160),     # 甲状腺癌 — 红紫
    138: (180, 0, 0),       # 脑肿瘤 — 暗红
    139: (255, 140, 0),     # 肾上腺肿瘤 — 金红
    140: (200, 60, 60),     # 小肠癌 — 灰红
    141: (180, 80, 0),      # 十二指肠癌 — 深橙
    142: (200, 0, 80),      # 胆囊癌 — 红粉
    143: (160, 0, 80),      # 脾脏肿瘤 — 暗红粉
    144: (140, 0, 0),       # 心脏肿瘤 — 极暗红

    # ── 血管 — 暗红系 ──
    6:   (140, 40, 40),     # 主动脉 — 暗红
    7:   (120, 40, 60),     # 下腔静脉 — 暗紫红
    58:  (140, 60, 40),     # 左髂动脉
    59:  (140, 60, 60),     # 右髂动脉
    60:  (100, 40, 80),     # 左髂静脉
    61:  (100, 60, 80),     # 右髂静脉
    108: (120, 40, 40),     # 左心耳
    109: (140, 50, 50),     # 头臂干
    110: (100, 40, 60),     # 左头臂静脉
    111: (100, 60, 60),     # 右头臂静脉
    112: (140, 60, 40),     # 左颈总动脉
    113: (140, 60, 60),     # 右颈总动脉
    119: (100, 40, 60),     # 肺静脉
    123: (140, 50, 40),     # 左锁骨下动脉
    124: (140, 50, 60),     # 右锁骨下动脉
    125: (100, 40, 40),     # 上腔静脉
    126: (160, 80, 80),     # 甲状腺 — 粉灰

    # ── 肾上腺 — 绿系 ──
    8:   (60, 140, 60),     # 右肾上腺
    9:   (60, 160, 60),     # 左肾上腺

    # ── 5个肺叶 — 蓝色系5种深浅 ──
    28:  (100, 160, 220),   # 左上肺叶 — 浅蓝
    29:  (60, 140, 200),    # 左下肺叶 — 蓝
    30:  (80, 180, 240),    # 右上肺叶 — 亮蓝
    31:  (40, 120, 180),    # 右中肺叶 — 深蓝
    32:  (20, 100, 160),    # 右下肺叶 — 暗蓝

    # ── 脊椎 (33-57) — 骨白色系，从腰到颈渐变 ──
    33:  (200, 200, 180),   # L5
    34:  (200, 200, 175),   # L4
    35:  (200, 200, 170),   # L3
    36:  (200, 200, 165),   # L2
    37:  (200, 200, 160),   # L1
    38:  (195, 195, 155),   # T12
    39:  (195, 195, 150),   # T11
    40:  (195, 195, 145),   # T10
    41:  (190, 190, 140),   # T9
    42:  (190, 190, 135),   # T8
    43:  (190, 190, 130),   # T7
    44:  (185, 185, 125),   # T6
    45:  (185, 185, 120),   # T5
    46:  (185, 185, 115),   # T4
    47:  (180, 180, 110),   # T3
    48:  (180, 180, 105),   # T2
    49:  (180, 180, 100),   # T1
    50:  (175, 175, 95),    # C7
    51:  (175, 175, 90),    # C6
    52:  (175, 175, 85),    # C5
    53:  (170, 170, 80),    # C4
    54:  (170, 170, 75),    # C3
    55:  (170, 170, 70),    # C2
    56:  (165, 165, 65),    # C1
    57:  (140, 180, 180),   # 气管 — 浅蓝灰

    # ── 肋骨 (63-86) — 12对肋骨，左偏黄、右偏粉 ──
    63:  (200, 180, 120),   # 左肋1
    64:  (195, 175, 115),   # 左肋2
    65:  (190, 170, 110),   # 左肋3
    66:  (185, 165, 105),   # 左肋4
    67:  (180, 160, 100),   # 左肋5
    68:  (175, 155, 95),    # 左肋6
    69:  (170, 150, 90),    # 左肋7
    70:  (165, 145, 85),    # 左肋8
    71:  (160, 140, 80),    # 左肋9
    72:  (155, 135, 75),    # 左肋10
    73:  (150, 130, 70),    # 左肋11
    74:  (145, 125, 65),    # 左肋12
    75:  (200, 160, 120),   # 右肋1
    76:  (195, 155, 115),   # 右肋2
    77:  (190, 150, 110),   # 右肋3
    78:  (185, 145, 105),   # 右肋4
    79:  (180, 140, 100),   # 右肋5
    80:  (175, 135, 95),    # 右肋6
    81:  (170, 130, 90),    # 右肋7
    82:  (165, 125, 85),    # 右肋8
    83:  (160, 120, 80),    # 右肋9
    84:  (155, 115, 75),    # 右肋10
    85:  (150, 110, 70),    # 右肋11
    86:  (145, 105, 65),    # 右肋12

    # ── 骨盆/肩胛/股骨 (87-107) — 粉白系 ──
    87:  (200, 180, 180),   # 左肱骨
    88:  (200, 160, 180),   # 右肱骨
    89:  (190, 170, 170),   # 左肩胛骨
    90:  (190, 150, 170),   # 右肩胛骨
    91:  (180, 180, 180),   # 左锁骨
    92:  (180, 160, 180),   # 右锁骨
    93:  (210, 190, 190),   # 左股骨
    94:  (210, 170, 190),   # 右股骨
    95:  (220, 200, 200),   # 左髋骨
    96:  (220, 180, 200),   # 右髋骨
    97:  (190, 190, 170),   # 骶骨
    98:  (140, 100, 60),    # 左臀大肌
    99:  (140, 100, 80),    # 右臀大肌
    100: (160, 120, 80),    # 左臀中肌
    101: (160, 120, 100),   # 右臀中肌
    102: (180, 140, 100),   # 左臀小肌
    103: (180, 140, 120),   # 右臀小肌
    104: (140, 120, 100),   # 左竖脊肌
    105: (140, 120, 120),   # 右竖脊肌
    106: (160, 140, 120),   # 左髂腰肌
    107: (160, 140, 140),   # 右髂腰肌

    # ── 胸部器官 ──
    114: (180, 140, 140),   # 肋软骨
    115: (180, 60, 60),     # 心脏 — 红
    118: (200, 120, 140),   # 前列腺
    120: (220, 210, 200),   # 头骨
    121: (160, 160, 200),   # 脊髓 — 浅蓝
    122: (200, 190, 170),   # 胸骨
    127: (200, 200, 160),   # S1脊椎
    132: (160, 200, 200),   # 气道

    # ── dummy / 不常见 ──
    2:   (60, 60, 60),      # dummy1
    16:  (60, 60, 80),      # dummy2
    18:  (80, 60, 60),      # dummy3
    20:  (60, 80, 60),      # dummy4
    21:  (80, 80, 60),      # dummy5
    22:  (180, 180, 220),   # 脑 — 浅紫蓝
    25:  (120, 80, 40),     # 肝血管 — 暗褐
    129: (80, 80, 80),      # dummy6
    130: (80, 80, 100),     # dummy7
    131: (100, 80, 80),     # dummy8
}

# label 名称 (用于 legend)
LABEL_NAMES: dict[int, str] = {
    0: "背景(空气)", 1: "肝脏", 3: "脾脏", 4: "胰腺", 5: "右肾",
    6: "主动脉", 7: "下腔静脉", 8: "右肾上腺", 9: "左肾上腺",
    10: "胆囊", 11: "食管", 12: "胃", 13: "十二指肠", 14: "左肾",
    15: "膀胱", 17: "门静脉/脾静脉", 19: "小肠", 22: "脑",
    25: "肝血管", 26: "肝肿瘤⬅桥接", 27: "结肠癌",
    23: "肺肿瘤", 24: "胰腺肿瘤", 28: "左上肺叶", 29: "左下肺叶",
    30: "右上肺叶", 31: "右中肺叶", 32: "右下肺叶",
    57: "气管", 62: "结肠", 114: "肋软骨", 115: "心脏",
    116: "左肾囊肿(已清除)", 117: "右肾囊肿(已清除)", 128: "骨病变",
    118: "前列腺", 120: "头骨", 121: "脊髓", 122: "胸骨",
    126: "甲状腺", 132: "气道", 200: "Body envelope",
    133: "食管癌", 134: "胃癌", 135: "膀胱癌",
    136: "前列腺癌", 137: "甲状腺癌", 138: "脑肿瘤",
    139: "肾上腺肿瘤", 140: "小肠癌",
    141: "十二指肠癌", 142: "胆囊癌", 143: "脾脏肿瘤", 144: "心脏肿瘤",
}

# 为脊椎/肋骨填充名称
for i in range(33, 57):
    if i == 57:
        LABEL_NAMES[i] = "气管"
        continue
    names = {33:"L5",34:"L4",35:"L3",36:"L2",37:"L1",38:"T12",39:"T11",
             40:"T10",41:"T9",42:"T8",43:"T7",44:"T6",45:"T5",46:"T4",
             47:"T3",48:"T2",49:"T1",50:"C7",51:"C6",52:"C5",53:"C4",
             54:"C3",55:"C2",56:"C1"}
    LABEL_NAMES[i] = f"脊椎{names.get(i, str(i))}"

for i in range(63, 87):
    side = "左" if i <= 74 else "右"
    num = i - 62 if i <= 74 else i - 74
    LABEL_NAMES[i] = f"{side}肋{num}"

for i in range(87, 108):
    names = {87:"左肱骨",88:"右肱骨",89:"左肩胛骨",90:"右肩胛骨",
             91:"左锁骨",92:"右锁骨",93:"左股骨",94:"右股骨",
             95:"左髋骨",96:"右髋骨",97:"骶骨",98:"左臀大肌",99:"右臀大肌",
             100:"左臀中肌",101:"右臀中肌",102:"左臀小肌",103:"右臀小肌",
             104:"左竖脊肌",105:"右竖脊肌",106:"左髂腰肌",107:"右髂腰肌"}
    LABEL_NAMES[i] = names.get(i, f"label {i}")


def _default_color(label_id: int) -> tuple[int, int, int]:
    """未在映射表中的标签 → 自动生成灰度色 (基于label ID)。"""
    gray = min(60 + (label_id * 3) % 180, 200)
    return (gray, gray, gray)


def colorize_label(label_data: np.ndarray) -> np.ndarray:
    """将整数标签数组转换为 RGB 彩色数组。

    Args:
        label_data: (D, H, W) int32 标签数组

    Returns:
        (D, H, W, 3) uint8 RGB 彩色数组
    """
    colored = np.zeros((*label_data.shape, 3), dtype=np.uint8)

    unique_labels = np.unique(label_data).astype(int)

    for label_id in unique_labels:
        color = LABEL_COLORS.get(label_id, _default_color(label_id))
        mask = (label_data == label_id)
        colored[mask, 0] = color[0]
        colored[mask, 1] = color[1]
        colored[mask, 2] = color[2]

    return colored


def save_png_slice(
    colored_data: np.ndarray,
    label_data: np.ndarray,
    output_path: str,
    axis: str = "axial",
    slice_index: int | None = None,
):
    """保存彩色切片为 PNG。

    自动选择肿瘤中心所在切片，或中间切片。
    """
    if slice_index is None:
        # 找肿瘤中心切片 (label 26)
        tumor_coords = np.argwhere(label_data == 26)
        if len(tumor_coords) > 0:
            if axis == "axial":
                slice_index = int(tumor_coords[:, 0].mean())
            elif axis == "coronal":
                slice_index = int(tumor_coords[:, 1].mean())
            else:
                slice_index = int(tumor_coords[:, 2].mean())
        else:
            # 无肿瘤 → 中间切片
            if axis == "axial":
                slice_index = colored_data.shape[0] // 2
            elif axis == "coronal":
                slice_index = colored_data.shape[1] // 2
            else:
                slice_index = colored_data.shape[2] // 2

    if axis == "axial":
        slice_2d = colored_data[slice_index, :, :, :]
    elif axis == "coronal":
        slice_2d = colored_data[:, slice_index, :, :]
    else:  # sagittal
        slice_2d = colored_data[:, :, slice_index, :]

    img = Image.fromarray(slice_2d, 'RGB')
    img.save(output_path)

    return slice_index


def save_legend(output_path: str, present_labels: list[int]):
    """生成颜色对照表 PNG。"""

    # 只展示当前文件中实际存在的标签
    present_labels = sorted(set(present_labels))

    row_height = 22
    margin = 40
    width = 400
    height = len(present_labels) * row_height + 40

    img = Image.new('RGB', (width, height), (30, 30, 30))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    y = 10
    for label_id in present_labels:
        color = LABEL_COLORS.get(label_id, _default_color(label_id))
        name = LABEL_NAMES.get(label_id, f"label {label_id}")

        # 颜色方块
        draw.rectangle([10, y, 30, y + row_height - 4], fill=color, outline=(200, 200, 200))

        # 标签编号 + 名称
        text = f"{label_id}: {name}"
        draw.text((35, y + 2), text, fill=(220, 220, 220), font=font)

        y += row_height

    img.save(output_path)


# ══════════════════════════════════════════════════════════════
#  单文件处理
# ══════════════════════════════════════════════════════════════

def process_single(
    label_path: str,
    output_dir: str | None = None,
    png_only: bool = False,
) -> dict:
    """处理单个 label NIfTI 文件，生成彩色可视化。"""

    label_img = nib.load(label_path)
    label_data = label_img.get_fdata().astype(np.int32)
    affine = label_img.affine

    # 上色
    colored_data = colorize_label(label_data)

    # 输出目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(label_path))

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.splitext(os.path.basename(label_path))[0])[0]

    # 保存 RGB NIfTI
    nii_path = None
    if not png_only:
        nii_path = os.path.join(output_dir, f"{base}_colored.nii.gz")
        # nibabel 保存 RGB: shape (x, y, z, 3), dtype uint8
        # 需要把 (D, H, W, 3) 转为 nibabel 的 (W, H, D, 3) 格式
        colored_transposed = colored_data.transpose(2, 1, 0, 3)
        colored_nii = nib.Nifti1Image(colored_transposed, affine)
        colored_nii.header.set_intent('vector')
        nib.save(colored_nii, nii_path)
        print(f"  → NIfTI: {nii_path}")

    # 保存 PNG 截图 (axial 切面)
    png_path = os.path.join(output_dir, f"{base}_colored_axial.png")
    slice_idx = save_png_slice(colored_data, label_data, png_path, axis="axial")
    print(f"  → Axial PNG (slice {slice_idx}): {png_path}")

    # 保存冠状切面 PNG
    coronal_path = os.path.join(output_dir, f"{base}_colored_coronal.png")
    save_png_slice(colored_data, label_data, coronal_path, axis="coronal")
    print(f"  → Coronal PNG: {coronal_path}")

    # 保存颜色对照表
    present_labels = np.unique(label_data).astype(int).tolist()
    legend_path = os.path.join(output_dir, f"{base}_legend.png")
    save_legend(legend_path, present_labels)
    print(f"  → Legend: {legend_path}")

    # 统计信息
    total_voxels = label_data.size
    nonzero = (label_data != 0).sum()
    tumor_labels_in_data = [l for l in present_labels if l in {23, 24, 26, 27, 116, 117, 128, 133, 134, 135, 136, 137, 138, 139, 140}]
    print(f"  总 voxels: {total_voxels:,}, 非零: {nonzero:,}")
    print(f"  出现的标签: {len(present_labels)} 个")
    if tumor_labels_in_data:
        for tl in tumor_labels_in_data:
            cnt = (label_data == tl).sum()
            print(f"  ⬅ 肿瘤标签 {tl} ({LABEL_NAMES.get(tl, '?')}): {cnt:,} voxels")

    return {
        "label_path": label_path,
        "nii_path": nii_path,
        "png_path": png_path,
        "legend_path": legend_path,
        "present_labels": present_labels,
    }


# ══════════════════════════════════════════════════════════════
#  批量扫描
# ══════════════════════════════════════════════════════════════

def scan_and_process(scan_dir: str, png_only: bool = False):
    """扫描任务目录中的 label 文件，批量上色。"""

    import glob as _glob

    # 查找 04_final_merged/merged_label.nii.gz
    pattern = os.path.join(scan_dir, "**", "04_final_merged", "merged_label.nii.gz")
    label_files = sorted(_glob.glob(pattern, recursive=True))

    if not label_files:
        # 也查找直接目录下的 label 文件
        pattern2 = os.path.join(scan_dir, "*label*.nii.gz")
        label_files = sorted(_glob.glob(pattern2))

    if not label_files:
        print(f"在 {scan_dir} 中未找到 label NIfTI 文件")
        return

    print(f"找到 {len(label_files)} 个 label 文件")
    print(f"{'='*60}\n")

    for label_path in label_files:
        print(f"处理: {os.path.basename(label_path)}")
        process_single(label_path, png_only=png_only)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="将 MAISI 132类 label NIfTI 上色为 RGB 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--label", metavar="PATH",
                        help="label NIfTI 文件路径")
    parser.add_argument("--scan-dir", metavar="DIR",
                        help="扫描目录下的任务目录，批量上色")
    parser.add_argument("--output-dir", "-o", metavar="DIR",
                        help="输出目录 (默认: 与 label 同目录)")
    parser.add_argument("--png-only", action="store_true",
                        help="只生成 PNG 截图，不生成 NIfTI")

    args = parser.parse_args()

    if args.scan_dir:
        scan_and_process(args.scan_dir, args.png_only)
        return

    if not args.label:
        parser.error("需要 --label 或 --scan-dir")

    result = process_single(args.label, args.output_dir, args.png_only)
    print(f"\n✓ 上色完成!")


if __name__ == "__main__":
    main()
