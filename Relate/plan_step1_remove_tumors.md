# 第一步实现计划：清除 MAISI 自带肿瘤标签（A方案）

## 目标

在 `bridge_maisi_mask.py` 中新增 `remove_existing_tumors()` 函数，
在 `bridge_single()` Step 1 和 Step 2 之间插入清除步骤，
保证桥接后的 mask 中只有桥接代码画的肿瘤，消除不可控的 MAISI 数据库自带肿瘤。

## 修改文件

仅修改一个文件：`Relate/bridge_maisi_mask.py`

## 具体改动

### 改动 1：新增常量 `TUMOR_TO_ORGAN_RESTORE_MAP`

位置：在现有 `TUMOR_LABEL_MAP` 常量块之后（约第171行）

内容：
```python
# MAISI 自带肿瘤标签 → 还原为对应器官标签 (A方案)
# 清除不可控的数据库自带肿瘤，保证桥接画的肿瘤是唯一来源。
# 直接还原的6个标签 (有明确器官映射):
#   26→1 (肝肿瘤→肝脏), 116→5 (左肾囊肿→左肾), 117→14 (右肾囊肿→右肾),
#   24→4 (胰腺肿瘤→胰腺), 27→62 (结肠癌→结肠)
# 特殊处理的2个标签 (碎片结构，需逐voxel还原):
#   23→最近肺叶 (28-32), 128→最近骨碎片 (33-57/63-96)
TUMOR_TO_ORGAN_RESTORE_MAP: Dict[int, int] = {
    26:  1,    # 肝肿瘤 → 肝脏
    116: 5,    # 左肾囊肿 → 左肾
    117: 14,   # 右肾囊肿 → 右肾
    24:  4,    # 胰腺肿瘤 → 胰腺
    27:  62,   # 结肠癌 → 结肠
}
# 肿瘤标签中需要逐voxel还原的特殊标签 (肺/骨是碎片结构)
TUMOR_LABELS_VOXELWISE = {23, 128}  # 肺肿瘤, 骨病变
```

### 改动 2：新增函数 `remove_existing_tumors()`

位置：在 `extract_organ_mask()` 函数之后（约第338行之后）

逻辑：
1. `cleaned = label_data.copy()` 不修改原始数组
2. 遍历 `TUMOR_TO_ORGAN_RESTORE_MAP`，批量还原 6 个有明确映射的肿瘤标签
3. 对 label 23 (肺肿瘤) 逐 voxel 还原：取 2-voxel 邻域内最常见的 LUNG_LABELS 成员；无邻域时用 distance_transform_edt 找最近肺叶
4. 对 label 128 (骨病变) 逐 voxel 还原：取 2-voxel 邻域内最常见的 BONE_LABELS 成员；无邻域时用 distance_transform_edt 找最近骨碎片
5. 返回 `(cleaned, removal_info)` 元组

`removal_info` 结构：
```python
{
    "removed_labels": {26: 13442, 116: 8593, 117: 9893},  # label→voxel count
    "total_removed_voxels": 31928,
    "restored_targets": {26: 1, 116: 5, 117: 14},         # label→还原目标
}
```

### 改动 3：修改 `bridge_single()` 函数

在 Step 1 加载之后、Step 2 提取之前，插入清除步骤：

```python
# ── Step 1: 加载 MAISI 数据 ──
ct_data, label_data, affine, spacing = load_maisi_data(ct_path, label_path)
shape = ct_data.shape

# ── Step 1.5: 清除 MAISI 自带肿瘤标签 (A方案) ──
label_data, removal_info = remove_existing_tumors(label_data)
if removal_info["total_removed_voxels"] > 0:
    print(f"  ℹ 清除自带肿瘤: {removal_info['total_removed_voxels']:,} voxels")
    for tl, cnt in removal_info["removed_labels"].items():
        target = removal_info["restored_targets"].get(tl, "逐voxel还原")
        print(f"    label {tl} ({cnt:,} voxels) → {target}")
```

同时在返回字典中添加 `removal_info` 字段：
```python
return {
    ...  # 现有字段
    "removal_info": removal_info,  # 新增
}
```

所有返回路径（ok/skip/fail）都需要包含 `removal_info`。skip 和 fail 路径在 Step 1.5 之后，所以 `removal_info` 已可用，只需加入返回字典。

### 改动 4：更新文档字符串

1. 文件顶部注释：将 "MAISI 生成 CT + 132类mask (不含肿瘤)" 改为 "MAISI 生成 CT + 132类mask (自带肿瘤已清除)"
2. `bridge_single()` docstring：添加 Step 1.5 描述

## 不改动的内容

- `TUMOR_LABEL_MAP` — 不变，桥接画肿瘤时仍用这些标签
- `ORGAN_LABEL_MAP` — 不变
- Step 3-9 逻辑 — 不变，`label_data` 已在 Step 1.5 被替换为清除版本，后续代码自动使用
- 批量模式 `run_config()` — 不变，它调用 `bridge_single()`
- CLI 参数 — 不新增参数（默认清除，无需开关）

## 验证方法（实现后）

用实测数据运行一次，验证：
1. 清除前 label 26 有 13,442 voxels → 清除后 label 26 为 0，label 1 增加 13,442 voxels
2. 清除前 label 116 有 8,593 voxels → 清除后 label 116 为 0，label 5 增加 8,593 voxels
3. 清除前 label 117 有 9,893 voxels → 清除后 label 117 为 0，label 14 增加 9,893 voxels
4. 桥接画 liver large 肿瘤后，merged 中只有桥接画的 label 26，不再有 MAISI 自带的小肿瘤区域被覆盖的情况
