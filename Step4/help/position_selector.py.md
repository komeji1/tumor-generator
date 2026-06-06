# position_selector.py — 详细解读

> **所属步骤**: Step 4 — 位置选择层  
> **文件路径**: `Step4/src/position_selector.py`  
> **依赖**: `utils.py` (Step 1), `validator.py` (Step 3)  
> **被依赖**: `main.py` (Step 6)

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [函数索引](#二函数索引)
3. [逐函数详解](#三逐函数详解)
4. [策略对比分析](#四策略对比分析)
5. [重试循环机制](#五重试循环机制)
6. [自检说明](#六自检说明)

---

## 一、文件功能概述

`position_selector.py` 负责在器官内部为肿瘤选定一个合法的中心位置。核心流程：

```
输入: organ_mask + radius_voxel + strategy + config
  │
  ├── ① 计算 margin
  │     margin = max_radius + feather(3mm) + safety(5mm)
  │
  ├── ② 腐蚀得到有效区域
  │     valid = erode(organ_mask, margin)
  │
  ├── ③ 重试循环 (最多 max_retries 次)
  │     ├── 按策略采样 center_zyx
  │     ├── check_in_organ(center, radii, organ_mask)
  │     ├── 通过 → 返回
  │     └── 失败 → 重试
  │
  └── 输出: (z, y, x) 体素坐标
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **策略可切换** | PlacementStrategy 枚举 + config 驱动 |
| **内建重试** | 采样+验证循环，屏蔽随机失败 |
| **显式 rng** | 所有函数接受可选 Generator 参数 |

---

## 二、函数索引

```
position_selector.py
│
├── PlacementStrategy    枚举: UNIFORM / DISTANCE_WEIGHTED
│
├── compute_margin_voxel(radius_voxel, feather_mm, safety_mm, spacing_z) → float
├── sample_uniform(valid_mask, rng) → (z, y, x)
├── sample_distance_weighted(valid_mask, organ_mask, alpha, rng) → (z, y, x)
├── select_location(organ_mask, radius, spacing, strategy, ...) → (z, y, x)
└── select_location_from_config(organ_mask, radius, spacing, config, rng) → (z, y, x)
```

---

## 三、逐函数详解

### 3.1 `compute_margin_voxel(radius_voxel, feather_mm, safety_mm, spacing_z) → float`

**公式**: `margin = radius_voxel + feather_mm/spacing_z + safety_mm/spacing_z`

**为什么 feather 和 safety 除以 spacing_z？** 只有 z 轴间距可能与其他轴不同（CT 的 z 间距通常 1-5mm，而 xy 通常 0.5-1mm）。取 z 轴间距作为上界，确保腐蚀量足够。

### 3.2 `sample_uniform(valid_mask, rng) → (z, y, x)`

均匀随机采样 — 默认策略，最大化位置多样性。

**实现**: `random_sample_valid(valid_mask, n=1, rng)` → 所有 valid 体素等概率。

### 3.3 `sample_distance_weighted(valid_mask, organ_mask, alpha, rng) → (z, y, x)`

距离加权采样 — 偏向器官中心。

**算法**:
```
① distance_transform_edt(organ_mask) → distance_field
   距离场: 每个体素到最近非器官体素的欧氏距离

② 限制到 valid 区域
   indices = argwhere(valid_mask > 0)
   distances = distance_field[indices]

③ 权重计算
   weights = distances^alpha
   alpha > 0 → 距离表面越远权重越大
   alpha = 0 → 退化为 uniform

④ 按权重采样
   idx = choice(len(indices), p=weights/weights.sum())
```

**alpha 效果**:
| alpha | 效果 |
|-------|------|
| 0 | = uniform |
| 1 | 线性偏向中心 |
| 2 | 强烈偏向中心（自检中 avg_z=19.9，center=20.0） |

### 3.4 `select_location(...) → (z, y, x)`

主入口，内置重试循环。详细分析见[第五节](#五重试循环机制)。

### 3.5 `select_location_from_config(...) → (z, y, x)`

便捷入口，将 JSON 配置字典映射为函数参数。`main.py` 的标准调用方式。

---

## 四、策略对比分析

| 维度 | UNIFORM | DISTANCE_WEIGHTED |
|------|---------|-------------------|
| 位置多样性 | 最大 | 偏向中心，多样性降低 |
| 临床合理性 | 可能选在器官边缘 | 更接近临床肿瘤分布（多位于器官实质内） |
| 论文依据 | 论文未指定 → 推导 | 论文未指定 → 推导 |
| 计算开销 | O(valid_voxels) | O(organ_voxels) + 距离变换 |
| 风险 | 边缘位置可能被下游扩散模型拒绝 | 中心区域过度集中 |

**推荐**: 默认使用 UNIFORM（最大多样性），如果下游反馈边缘肿瘤生成质量差，再切换到 DISTANCE_WEIGHTED。

---

## 五、重试循环机制

```
select_location() 重试流程:
─────────────────────────────

输入: organ_mask, radius_voxel=8, spacing=(2,1,1), strategy=UNIFORM

Step 1: margin = 8 + 3/2 + 5/2 = 12 voxels
Step 2: valid = erode(organ, 12) → 7,776 voxels (7.2% of organ)
Step 3: 循环 (max_retries=50):

  attempt 1: sample → (19, 44, 50)
    check_in_organ((19,44,50), (8,8,8), organ) → True ✓
    → return (19, 44, 50)

  (worst case: 50次全部失败 → RuntimeError)
```

### 为什么会重试失败？

1. 采样点正好落在器官的突起/狭长区域边缘
2. 椭球验证（check_in_organ）发现部分体素超出器官
3. 随机采样恰好连续命中了这些"危险"区域

### 失败概率估算

```
P(单次失败) ≈ 1 - valid_volume/organ_volume

margin=12, organ=100mm×80mm×80mm:
  P(single_fail) ≈ 1 - 0.07 = 0.93  # 在这个例子中，valid只占7%
  
但 check_in_organ 使用等半径球体验证，而有效区域已经用 margin 预腐蚀...
实际上只要 center 在 valid_region 中，球体就一定在 organ 内。
因为 valid_region = erode(organ, margin)，margin 已经包含了半径。

所以 P(单次失败) ≈ 0%（理论上）
  除非 check_in_organ 的球体半径 > margin（参数不一致）
```

---

## 六、自检说明

| 测试 | 内容 |
|------|------|
| [1] compute_margin | radius=8, spacing_z=2 → margin=12 voxels |
| [2] valid_region | organ=108K → valid=7.8K voxels |
| [3] sample_uniform | 3次采样均在 valid 内 |
| [4] sample_distance_weighted | 20次采样 biased → avg_z≈center |
| [5] select_location UNIFORM | 成功返回合法位置 |
| [6] select_location DISTANCE_WEIGHTED | 成功返回合法位置 |
| [7] select_location_from_config | config驱动正常工作 |
| [8] 有效区域为空 | 正确抛出 ValueError |
| [9] 重试耗尽 | 小器官 + 大半径 → ValueError |

**运行方式**: `python Step4/src/position_selector.py`
