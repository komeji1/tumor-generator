# validator.py — 详细解读

> **所属步骤**: Step 3 — 校验层  
> **文件路径**: `Step3/src/validator.py`  
> **依赖**: `utils.py` (Step 1) — `compute_ellipsoid_dist`  
> **被依赖**: `position_selector.py` (Step 4), `mask_generator.py` (Step 5), `main.py` (Step 6)

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [函数索引](#二函数索引)
3. [逐函数详解](#三逐函数详解)
4. [校验流程](#四校验流程)
5. [错误处理策略](#五错误处理策略)
6. [自检说明](#六自检说明)

---

## 一、文件功能概述

`validator.py` 是项目的**质量守门员**。所有位置选择和 mask 生成的结果都要经过此模块的校验，确保满足：

- 肿瘤完全在器官内部
- 肿瘤尺寸在论文定义的分类范围内
- 多肿瘤场景下互不重叠
- mask 非空且器官体积足够

### 设计原则

| 原则 | 说明 |
|------|------|
| **不抛异常** | 校验失败返回 `(False, detail)`，由调用方决定重试或放弃 |
| **统一接口** | 所有 `check_*()` 返回 `(bool, detail)` 元组 |
| **一站式** | `validate_sample()` 执行全部检查并返回完整报告 |
| **低开销** | `check_in_organ` 使用 bbox 裁剪，避免全体积距离场计算 |

---

## 二、函数索引

```
validator.py
│
├── 单项校验 ──────────────────────────────────────
│   ├── check_organ_volume(organ_mask, min_voxels) → (bool, int)
│   ├── check_size_range(radius_mm, size_category_config) → (bool, tuple)
│   ├── check_in_organ(center, radii, organ_mask) → (bool, str)
│   ├── check_mask_nonzero(mask_3d) → (bool, int)
│   └── check_not_overlapping(new_mask, existing_masks) → (bool, float)
│
└── 一站式校验 ────────────────────────────────────
    └── validate_sample(center, radii, radius_mm, mask, organ, config, ...) → dict
```

---

## 三、逐函数详解

### 3.1 `check_organ_volume(organ_mask, min_voxels=100) → (bool, int)`

**检查**: 器官体积是否 ≥ `min_voxels`。

**用途**: 过滤掉体积过小的器官，防止连最小肿瘤都放不下。例如某个 CT 扫描中食管区域只有 30 个体素 → 应跳过。

**返回值**: `(True, 108000)` 或 `(False, 0)`。

---

### 3.2 `check_size_range(radius_mm, size_category_config) → (bool, (r_min, r_max))`

**检查**: 肿瘤半径是否在论文定义的分类范围内。

**论文依据**: Scaling Tumor (ICCV 2025) §3.2 (P5):
- `tiny`: r ≤ 5 mm
- `small`: 5 < r ≤ 10 mm
- `medium`: 10 < r ≤ 20 mm
- `large`: r > 20 mm

**区间语义**: tiny 是闭区间 `[1, 5]`，其余三类是左开右闭 `(r_min, r_max]`。

**示例**:
```
check_size_range(7.5, {"r_min_mm": 5, "r_max_mm": 10})  → (True,  (5, 10))
check_size_range(5.0, {"r_min_mm": 5, "r_max_mm": 10})  → (False, (5, 10))  # 左开
check_size_range(3.0, {"r_min_mm": 1, "r_max_mm": 5})   → (True,  (1, 5))   # tiny闭区间
```

---

### 3.3 `check_in_organ(center_zyx, radius_voxel, organ_mask) → (bool, str)`

**检查**: 椭球体肿瘤是否完全在器官 mask 内部。

**算法**（分 4 步）:
```
① 检查中心点
   - 是否在 volume 范围内
   - organ_mask[center] == 1 ?

② 计算椭球 bounding box
   bbox = center ± max(radii)，裁剪到 volume 边界
   → 避免对整个 (512,512,400) CT 计算距离场

③ 在 bbox 内计算椭球距离场
   dist = compute_ellipsoid_dist(bbox_shape, bbox_center, radii)

④ 验证覆盖
   肿瘤体素 = (dist <= 1.0)
   covered = tumor_voxels & (organ_bbox > 0)
   ratio = covered / tumor_count

   ratio == 1.0 → 通过
   ratio < 1.0  → 失败（超出 organ 边界）
```

**bbox 优化效果**: 对于 40³ 的椭球在 (512,512,400) 的 CT 中，仅计算 82³ 的距离场，节省 ~99.6% 内存。

**失败消息示例**:
```
"Tumor exceeds organ boundary: 367/515 voxels (71.3%) outside.
 Center=(5,11,11), radii=(5.0,5.0,5.0)"
```

---

### 3.4 `check_mask_nonzero(mask_3d) → (bool, int)`

**检查**: 生成的 mask 是否包含至少一个非零体素。

**用途**: 捕获 mask_generator 在极端情况下的失败（如弹性形变把整个 mask 推出边界）。

---

### 3.5 `check_not_overlapping(new_mask, existing_masks, threshold=0.0) → (bool, float)`

**检查**: 新 mask 不与已有 mask 重叠。

**默认策略**: `threshold=0.0` — 不允许任何重叠。

**重叠比例计算**: `overlap_ratio = |new ∩ existing| / |new|`

---

### 3.6 `validate_sample(...) → dict`

**一站式校验入口**。依次执行全部 5 项检查，返回结构化报告。

**执行顺序**（按开销从小到大）:

```
① check_organ_volume     O(1)    — 只需 sum
② check_size_range        O(1)    — 标量比较
③ check_mask_nonzero      O(1)    — 只需 sum
④ check_in_organ          O(|bbox|) — 距离场计算
⑤ check_not_overlapping   O(|mask|) — 逐体素 AND
```

**返回结构**:
```python
{
    'passed': True,                    # 全部通过
    'checks': {
        'organ_volume':    (True, "volume=108000 voxels"),
        'size_range':      (True, "radius=10.0mm, range=(5,20]mm"),
        'mask_nonzero':    (True, "4169 nonzero voxels"),
        'in_organ':        (True, "Tumor fully inside organ (4169 voxels)"),
        'not_overlapping': (True, "max_overlap=0.0000"),
    },
    'errors': [],                      # 失败项摘要
    'warnings': [],                    # 警告（如肿瘤>器官50%体积）
}
```

**额外警告**: 如果肿瘤体积 > 器官体积的 50%，会添加一条 warning（但不会标记为失败）。这在解剖上可能不合理（肿瘤不会占器官大半），但不作为硬性约束。

---

## 四、校验流程

### 在 position_selector.py 中的使用

```python
# Step 4: 选择位置时的校验循环
for attempt in range(max_retries):
    center = sample_position(valid_region)
    
    ok, msg = check_in_organ(center, radii, organ_mask)
    if ok:
        return center  # 成功
    # 失败 → 重试

# 全部重试耗尽 → 该样本生成失败
```

### 在 mask_generator.py 中的使用

```python
# Step 5: 生成 mask 后的校验
mask = create_mask(center, radius_mm, ct_volume, config)
result = validate_sample(center, radii, radius_mm, mask, organ_mask, size_cfg)
if not result['passed']:
    # 日志记录，可能重试（调整参数后重新生成）
```

### 在 main.py 中的使用

```python
# Step 6: 每个 sample 生成完成后
result = validate_sample(...)
metadata['validation'] = result
if not result['passed']:
    metadata['success'] = False
    metadata['errors'] = result['errors']
```

---

## 五、错误处理策略

| 场景 | 校验结果 | 调用方响应 |
|------|---------|-----------|
| 器官体积不足 | `check_organ_volume → False` | 跳过该样本，记录日志 |
| 半径超出范围 | `check_size_range → False` | 不应发生（采样逻辑保证），记录 warning |
| 肿瘤不完全在器官内 | `check_in_organ → False` | Step 4 重试选位置，Step 5 调整参数后重试 |
| mask 为空 | `check_mask_nonzero → False` | 调整生成参数后重试 |
| 与已有 mask 重叠 | `check_not_overlapping → False` | Step 4 重新选位置 |
| 多项失败 | `validate_sample → passed=False` | 记录全部失败原因，跳过或重试 |

**为什么校验不抛异常？** 因为在批量生成 120 个 mask 的场景中，单个样本失败不应中断整个 batch。调用方（`main.py`）收集所有失败信息，在最终报告中汇总。

---

## 六、自检说明

| 测试 | 内容 | 验证方式 |
|------|------|----------|
| [1] check_organ_volume | 正常器官 + 空器官 | assert 布尔值 |
| [2] check_size_range | 范围内/范围外半径 | assert 区间判断 |
| [3] check_in_organ | 中心在内/边界/在外 | assert + 消息验证 |
| [4] check_mask_nonzero | 正常mask + 空mask | assert 布尔值 |
| [5] check_not_overlapping | 无重叠 + 有重叠 | assert 重叠比例 |
| [6] validate_sample | 全通过 + 3项失败 | assert passed + 错误计数 |

**运行方式**: `python Step3/src/validator.py`
