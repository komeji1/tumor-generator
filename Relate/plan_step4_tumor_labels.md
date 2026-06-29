# 第四步实现计划：新建肿瘤标签建立映射

## 目标

为 MAISI 132类中没有对应肿瘤标签的12个器官，新建肿瘤标签（133-144），
并更新 bridge_maisi_mask.py 和 label_colorize.py 中所有相关的常量映射。

## 新增肿瘤标签映射

| 优先级 | 器官 | 器官label | 新增肿瘤label | 肿瘤名称 | 还原目标(清除时) |
|--------|------|----------|-------------|---------|----------------|
| 🔴高 | 食管 | 11 | 133 | 食管癌 | 11 (食管) |
| 🔴高 | 胃 | 12 | 134 | 胃癌 | 12 (胃) |
| 🔴高 | 膀胱 | 15 | 135 | 膀胱癌 | 15 (膀胱) |
| 🔴高 | 前列腺 | 118 | 136 | 前列腺癌 | 118 (前列腺) |
| 🟡高 | 甲状腺 | 126 | 137 | 甲状腺癌 | 126 (甲状腺) |
| 🟡中 | 脑 | 22 | 138 | 脑肿瘤 | 22 (脑) |
| 🟡中 | 肾上腺 | 8 | 139 | 肾上腺肿瘤 | 8 (右肾上腺) |
| 🟢低 | 小肠 | 19 | 140 | 小肠癌 | 19 (小肠) |
| 🟢低 | 十二指肠 | 13 | 141 | 十二指肠癌 | 13 (十二指肠) |
| 🟢低 | 胆囊 | 10 | 142 | 胆囊癌 | 10 (胆囊) |
| 🟢低 | 脾脏 | 3 | 143 | 脾脏肿瘤 | 3 (脾脏) |
| 🟢低 | 心脏 | 115 | 144 | 心脏肿瘤 | 115 (心脏) |

## 修改文件1：Relate/bridge_maisi_mask.py

### 改动1：ORGAN_LABEL_MAP 新增12个器官

```python
ORGAN_LABEL_MAP: Dict[str, int] = {
    # 原有7个
    "liver":     1,
    "pancreas":  4,
    "kidney":    5,
    "colon":     62,
    "esophagus": 11,
    "uterus":    161,
    "lung":      20,
    "bone":      21,
    # 新增12个
    "stomach":      12,
    "bladder":      15,
    "prostate":     118,
    "thyroid":      126,
    "brain":        22,
    "adrenal":      8,
    "small_bowel":  19,
    "duodenum":     13,
    "gallbladder":  10,
    "spleen":       3,
    "heart":        115,
}
```

### 改动2：TUMOR_LABEL_MAP 更新

```python
TUMOR_LABEL_MAP: Dict[str, int] = {
    # 原有7个（esophagus/uterus从0改为新标签）
    "liver":     26,
    "pancreas":  24,
    "kidney":    116,
    "colon":     27,
    "lung":      23,
    "bone":      128,
    "esophagus": 133,    # 0 → 133
    # 新增11个（原uterus=0暂保留，MAISI无对应器官）
    "stomach":      134,
    "bladder":      135,
    "prostate":     136,
    "thyroid":      137,
    "brain":        138,
    "adrenal":      139,
    "small_bowel":  140,
    "duodenum":     141,
    "gallbladder":  142,
    "spleen":       143,
    "heart":        144,
    "uterus":       0,     # 仍无标签
}
```

### 改动3：TUMOR_TO_ORGAN_RESTORE_MAP 新增12个

```python
TUMOR_TO_ORGAN_RESTORE_MAP: Dict[int, int] = {
    # 原有5个
    26:  1,  116: 5,  117: 14,  24: 4,  27: 62,
    # 新增12个
    133: 11,   # 食管癌 → 食管
    134: 12,   # 胃癌 → 胃
    135: 15,   # 膀胱癌 → 膀胱
    136: 118,  # 前列腺癌 → 前列腺
    137: 126,  # 甲状腺癌 → 甲状腺
    138: 22,   # 脑肿瘤 → 脑
    139: 8,    # 肾上腺肿瘤 → 右肾上腺
    140: 19,   # 小肠癌 → 小肠
    141: 13,   # 十二指肠癌 → 十二指肠
    142: 10,   # 胆囊癌 → 胆囊
    143: 3,    # 脾脏肿瘤 → 脾脏
    144: 115,  # 心脏肿瘤 → 心脏
}
```

注意：所有12个新增肿瘤标签都是**直接还原**（有明确器官映射），不需要像肺/骨那样逐voxel还原。
- 肾上腺标签 139 还原到 8（右肾上腺），左肾上腺(9)的肿瘤也还原到 8
- 这和 kidney 类似（左右肾合并检测）

### 改动4：_MAISI_VALID_LABELS 新增 133-144

### 改动5：SIZE_RADIUS_RANGES 无需改动（通用）

## 修改文件2：Relate/label_colorize.py

### 改动：LABEL_COLORS 新增 134-144 的颜色

LABEL_NAMES 新增 134-144 的英文名称

## 不修改的内容

- `remove_existing_tumors()` — 不需要改。新增的 133-144 标签不会出现在 MAISI 原始输出中
  （MAISI 只输出 132类），所以清除步骤不需要处理这些新标签。
  但 TUMOR_TO_ORGAN_RESTORE_MAP 已包含 133-144，万一将来出现也能正确还原。
- `bridge_single()` 核心逻辑 — 不需要改。Step 6 中 `tumor_label_id = TUMOR_LABEL_MAP.get(organ, 0)`
  会自动获取新标签，esophagus 从 0 变成 133，不再走 zero-shot 分支。
