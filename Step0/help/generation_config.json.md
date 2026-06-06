# generation_config.json — 详细解读

> **所属步骤**: Step 0 — 项目初始化  
> **文件路径**: `Step0/config/generation_config.json`  
> **运行时路径**: 项目运行时会复制/链接到 `config/generation_config.json` 供 `main.py` 读取  
> **文件类型**: JSON 配置文件  
> **对应源码**: 无（被动数据，由 `main.py` 中的 `load_config()` 读取）

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [整体结构](#二整体结构)
3. [逐段详解](#三逐段详解)
   - [3.1 project — 项目元信息](#31-project--项目元信息)
   - [3.2 data — 数据路径配置](#32-data--数据路径配置)
   - [3.3 organs — 6类肿瘤配置](#33-organs--6类肿瘤配置)
   - [3.4 size_categories — 尺寸分类与分布权重](#34-size_categories--尺寸分类与分布权重)
   - [3.5 shape — Mask形状生成管线配置](#35-shape--mask形状生成管线配置)
   - [3.6 placement — 位置选择配置](#36-placement--位置选择配置)
   - [3.7 preprocessing — CT预处理参数](#37-preprocessing--ct预处理参数)
   - [3.8 output — 输出格式配置](#38-output--输出格式配置)
   - [3.9 logging — 日志与统计配置](#39-logging--日志与统计配置)
4. [代码如何使用此配置](#四代码如何使用此配置)
5. [关键算法分析](#五关键算法分析)
6. [设计决策与论文依据](#六设计决策与论文依据)
7. [扩展指南](#七扩展指南)

---

## 一、文件功能概述

`generation_config.json` 是 Tumor Mask Generator 项目的**唯一参数入口**。所有模块（数据加载、位置选择、Mask生成、输出保存）的可调参数均集中在此文件中定义，而非散落在各源码文件的硬编码常量中。

### 设计原则

```
┌─────────────────────────────────────────────────────────────────┐
│  原则           │  说明                                          │
│  ─────────────────────────────────────────────────────────────  │
│  单一入口       │  所有可调参数集中在一处，避免散落              │
│  自描述         │  每个参数段包含 _ref（论文出处）和 description  │
│  可扩展         │  新增器官/策略/管线步骤只需修改JSON，不改代码  │
│  类型安全        │  下游代码读取后做 schema 校验                  │
│  无代码依赖      │  纯数据文件，任何语言均可解析                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、整体结构

```
generation_config.json
│
├── project            ← 项目元信息 (名称/版本/输出目录)
├── data               ← 数据路径 (CT目录/器官标签目录/manifest路径)
├── organs[]           ← 6类肿瘤定义 (名称/class_id/对应器官文件/数量)
├── size_categories    ← 尺寸分类 (4档半径 + 4:2:1权重)
│   └── categories{}   ← tiny / small / medium / large
├── shape              ← Mask形状生成管线 (5个步骤，每步可独立开关)
│   ├── elastic_deformation
│   ├── salt_noise
│   ├── gaussian_filter
│   └── scaling_clipping
├── placement          ← 位置选择策略 (策略名/margin/重试次数)
├── preprocessing      ← CT HU值裁剪范围
├── output             ← 输出格式 (nifti/dtype/命名规则)
└── logging            ← 日志路径
```

---

## 三、逐段详解

### 3.1 project — 项目元信息

```json
"project": {
    "name": "Tumor Mask Generator",
    "version": "1.0.0",
    "output_dir": "output/real_ct/",
    "description": "..."
}
```

| 字段 | 类型 | 含义 | 使用者 |
|------|------|------|--------|
| `name` | string | 项目名称，用于日志/报告标题 | main.py |
| `version` | string | 语义化版本号，用于输出文件标注 | main.py |
| `output_dir` | string | 输出根目录，所有mask的相对路径起点 | main.py |
| `description` | string | 项目一句话描述，仅文档用途 | — |

---

### 3.2 data — 数据路径配置

```json
"data": {
    "ct_dir": "data/ct/",
    "organ_label_dir": "data/organ_labels/",
    "manifest_path": "data/manifest.csv"
}
```

| 字段 | 含义 | 使用者 |
|------|------|--------|
| `ct_dir` | 正常CT扫描 (.nii.gz) 存放目录 | `data_loader.load_ct()` |
| `organ_label_dir` | 器官分割mask存放目录，按 `BDMAP_XXXXXXXX/segmentations/` 组织 | `data_loader.load_organ_mask()` |
| `manifest_path` | 样本索引CSV文件路径，映射 CT → 器官 → 肿瘤类型 | `data_loader.build_manifest()` |

#### 数据目录预期结构

```
data/
├── ct/
│   ├── BDMAP_00000001/
│   │   └── ct.nii.gz
│   ├── BDMAP_00000002/
│   │   └── ct.nii.gz
│   └── ...
├── organ_labels/
│   ├── BDMAP_00000001/
│   │   └── segmentations/
│   │       ├── liver.nii.gz
│   │       ├── pancreas.nii.gz
│   │       ├── kidney_left.nii.gz
│   │       ├── colon.nii.gz
│   │       ├── esophagus.nii.gz
│   │       └── uterus.nii.gz
│   └── ...
└── manifest.csv
```

---

### 3.3 organs — 6类肿瘤配置

```json
"organs": [
    {
        "name": "liver_lesion",
        "class_id": 27,
        "organ_label_file": "liver.nii.gz",
        "organ_name": "liver",
        "count": 50,
        "description": "肝脏肿瘤"
    },
    // ... 共6项
]
```

#### 字段详解

| 字段 | 类型 | 含义 | 约束 |
|------|------|------|------|
| `name` | string | 肿瘤类型名称，与 AbdomenAtlas2.0 命名一致 | 必须匹配 class_map 中的名称 |
| `class_id` | int | AbdomenAtlas2.0 中的类别编号 (27-32) | 唯一，不可重复 |
| `organ_label_file` | string | 对应器官的.nii.gz文件名 | 必须存在于 organ_label_dir 的子目录中 |
| `organ_name` | string | 器官简称，用于日志/显示 | — |
| `count` | int | 该类肿瘤需生成的mask数量 | 默认 50，总和 = 300 |
| `description` | string | 中文描述 | — |

#### 与 AbdomenAtlas2.0 class_map 的对应关系

```
class_map_abdomenatlas_2_0:
    27: 'liver_lesion'        ← organs[0]
    28: 'pancreatic_lesion'   ← organs[1]
    29: 'kidney_lesion'       ← organs[2]
    30: 'colon_lesion'        ← organs[3]
    31: 'endometrioma_tumor'  ← organs[5]（注意顺序交换）
    32: 'esophagus_tumor'     ← organs[4]
```

> **注意**: class_id 31 (endometrioma_tumor) 和 32 (esophagus_tumor) 在数组中交换了顺序（不影响功能），因为按器官常见程度排列。

#### 为什么 kidney_lesion 使用 `kidney_left.nii.gz`？

肾脏是成对器官（左/右），AbdomenAtlas2.0 分别标注 `kidney_left`(class 3) 和 `kidney_right`(class 4)。配置中优先使用左肾，后续可扩展为同时使用双肾。

---

### 3.4 size_categories — 尺寸分类与分布权重

```json
"size_categories": {
    "_ref": "Scaling Tumor (ICCV 2025) §3.2 (P5) + §4.1 (P7)",
    "_note": "tiny 不在 4:2:1 比例中显式列出...",
    "categories": {
        "tiny":   { "r_min_mm": 1,  "r_max_mm": 5,  "weight": 0, ... },
        "small":  { "r_min_mm": 5,  "r_max_mm": 10, "weight": 4, ... },
        "medium": { "r_min_mm": 10, "r_max_mm": 20, "weight": 2, ... },
        "large":  { "r_min_mm": 20, "r_max_mm": 50, "weight": 1, ... }
    }
}
```

#### 字段详解

| 字段 | 类型 | 含义 |
|------|------|------|
| `r_min_mm` | float | 该类别的最小半径 (mm)，开区间 |
| `r_max_mm` | float | 该类别的最大半径 (mm)，闭区间 |
| `weight` | int | 采样权重，4:2:1 对应 small:medium:large |
| `description` | string | 中文描述 + 论文出处 |

#### 尺寸分类的数学定义

```
类别      半径范围 (mm)        区间表示
─────────────────────────────────────────
tiny       [1, 5]           r ∈ [r_min, r_max]
small      (5, 10]          r ∈ (r_min, r_max]
medium     (10, 20]         r ∈ (r_min, r_max]
large      (20, 50]         r ∈ (r_min, r_max]
```

#### 权重分布算法

```
采样概率计算 (在 main.py 的 sample_size_category 中实现):

① 收集各分类的 weight 值:
   weights = [cat['weight'] for cat in categories.values()]
   → [0, 4, 2, 1]

② 归一化为概率分布:
   probs = weights / sum(weights)
   → [0.0, 0.571, 0.286, 0.143]

③ 忽略 weight=0 的分类 (tiny):
   只从 [small, medium, large] 中按 4:2:1 采样

④ tiny 的生成策略:
   weight=0 不参与概率采样。但当器官过小无法容纳更大尺寸时，
   算法自动回退到 tiny (r=1-5mm)。对tiny肿瘤自动禁用弹性变形
   以避免边界溢出。
```

| 分类 | weight | 概率 | 50张中期望 |
|------|--------|------|-----------|
| tiny | 0 | — | 按需回退 |
| small | 4 | ~57% | ~29 张 |
| medium | 2 | ~29% | ~14 张 |
| large | 1 | ~14% | ~7 张 |

---

### 3.5 shape — Mask形状生成管线配置

```json
"shape": {
    "_ref": "DiffTumor (CVPR 2024) §3.3 (P5) + §F.1 (P22)",
    "method": "ellipsoid",
    "axis_ratio_range": [0.8, 1.2],
    "volume_conservation": true,

    "elastic_deformation": { "enabled": true, "alpha": 15, "sigma": 3 },
    "salt_noise":          { "enabled": true, "probability": 0.02 },
    "gaussian_filter":     { "enabled": true, "sigma_mm": 1.0 },
    "scaling_clipping":    { "enabled": true, "value_range": [0, 1] }
}
```

#### 管线流程

```
输入: center_zyx, radius_mm, CTVolume
  │
  ├── Step 1: create_ellipsoid()          ← method="ellipsoid"
  │   使用 axis_ratio_range 生成各向异性椭球
  │
  ├── Step 2: apply_elastic_deformation() ← enabled=true
  │   参数: alpha (变形程度), sigma (平滑度)
  │
  ├── Step 3: apply_salt_noise()          ← enabled=true
  │   参数: probability (翻转概率)
  │
  ├── Step 4: apply_gaussian_smoothing()  ← enabled=true
  │   参数: sigma_mm (高斯核宽度)
  │
  └── Step 5: scaling_clipping()          ← enabled=true
      输出: value_range [0, 1]
```

#### 各步骤参数详解

##### Step 1: 椭球生成

| 参数 | 值 | 含义 |
|------|-----|------|
| `method` | `"ellipsoid"` | 形状方法：椭球体。论文唯一指定的形状 |
| `axis_ratio_range` | `[0.8, 1.2]` | 三轴比例随机范围，各轴独立采样 |
| `volume_conservation` | `true` | 是否保持体积守恒（缩放因子 = (r1×r2×r3)^(1/3)） |

##### Step 2: 弹性形变

| 参数 | 值 | 含义 |
|------|-----|------|
| `enabled` | `true` | 总开关 |
| `alpha` | `15` | 变形程度，值越大位移越大。实现中作为位移场的缩放因子 |
| `sigma` | `3` | 高斯滤波sigma（体素单位），控制位移场的空间平滑度。值越大变形越"宏观" |

> `alpha=15, sigma=3` 的含义：先生成与mask同尺寸的随机位移场（标准正态分布），用 sigma=3 体素的高斯核平滑，再乘以 alpha=15。结果是低频、平滑的变形，模拟肿瘤边界的自然不规则性。

##### Step 3: Salt-Noise

| 参数 | 值 | 含义 |
|------|-----|------|
| `enabled` | `true` | 总开关 |
| `probability` | `0.02` | 每个mask内部体素有2%概率被翻转 (1→0)，模拟内部纹理不规则 |

##### Step 4: 高斯滤波

| 参数 | 值 | 含义 |
|------|-----|------|
| `enabled` | `true` | 总开关 |
| `sigma_mm` | `1.0` | 高斯核sigma（mm单位），会被转为体素单位 `sigma_voxel = sigma_mm / spacing` |

##### Step 5: 缩放与裁剪

| 参数 | 值 | 含义 |
|------|-----|------|
| `enabled` | `true` | 总开关 |
| `value_range` | `[0, 1]` | 最终输出值域 |

---

### 3.6 placement — 位置选择配置

```json
"placement": {
    "strategy": "uniform",
    "available_strategies": ["uniform", "distance_weighted", "subregion"],
    "margin": {
        "feather_mm": 3,
        "safety_mm": 5
    },
    "max_retries": 50,
    "distance_weighted": {
        "alpha": 1.0
    }
}
```

#### 字段详解

| 字段 | 值 | 含义 |
|------|-----|------|
| `strategy` | `"uniform"` | 当前使用的位置选择策略 |
| `available_strategies` | `["uniform", "distance_weighted", "subregion"]` | 所有已实现的策略列表 |
| `margin.feather_mm` | `0.5` | 羽化边距 (mm)，已极小化以适应小器官 |
| `margin.safety_mm` | `1` | 安全边距 (mm)，平衡边界安全与小器官兼容 |
| `max_retries` | `50` | 位置选择最大重试次数 |
| `distance_weighted.alpha` | `1.0` | distance_weighted 策略参数：prob ∝ distance^alpha |

#### margin 计算

```
有效区域 = erode(organ_mask, margin_voxel)

其中:
  margin_voxel = max_tumor_radius_voxel + feather_voxel + safety_voxel
  max_tumor_radius_voxel = max(r_max_mm) / spacing = 50 / spacing
  feather_voxel = 3 / spacing
  safety_voxel = 5 / spacing

示例: spacing = 1.0 mm/voxel, 最大large肿瘤半径 = 50mm
  → margin_voxel = 50 + 3 + 5 = 58 voxels
```

#### 各策略说明

| 策略 | 算法 | 适用场景 |
|------|------|----------|
| `uniform` | 所有valid体素等概率采样 | 默认，最大化位置多样性 |
| `distance_weighted` | 概率 ∝ distance(体素, 器官表面)^alpha | 肿瘤倾向位于器官中心时 |
| `subregion` | 先选解剖亚区(Zones)，再区内均匀 | 有明确解剖位置偏好时 |

---

### 3.7 preprocessing — CT预处理参数

```json
"preprocessing": {
    "_ref": "DiffTumor (CVPR 2024) §E.2 (P21)",
    "hu_min": -175,
    "hu_max": 250
}
```

| 字段 | 值 | 含义 | 出处 |
|------|-----|------|------|
| `hu_min` | `-175` | HU值下界，低于此值的体素被截断 | DiffTumor §E.2: "truncated to the range [−175, 250]" |
| `hu_max` | `250` | HU值上界，高于此值的体素被截断 | 同上 |

#### HU值含义

```
CT HU (Hounsfield Unit) 值范围:
  -1000  空气
  -175   论文下界（脂肪 ≈ -100 到 -50）
    0    水
   50    软组织
  250    论文上界（增强后血管/器官 ≈ 100-300）
 1000+   骨骼/金属

裁剪到 [-175, 250]:
  - 去除空气 (< -175 对肿瘤检测无意义)
  - 去除骨骼 (> 250 对腹部软组织肿瘤无意义)
  - 保留软组织对比度的有效范围
```

---

### 3.8 output — 输出格式配置

```json
"output": {
    "format": "nifti",
    "dtype": "uint8",
    "value_range": [0, 1],
    "compress": true,
    "naming_pattern": "{organ_type}_{sample_id}.nii.gz"
}
```

| 字段 | 值 | 含义 |
|------|-----|------|
| `format` | `"nifti"` | 输出格式：NIfTI |
| `dtype` | `"uint8"` | 数据类型：8位无符号整数 |
| `value_range` | `[0, 1]` | 值域：仅含 0 和 1 |
| `compress` | `true` | 是否 gzip 压缩 (.nii.gz) |
| `naming_pattern` | `"{organ_type}_{sample_id:03d}.nii.gz"` | 文件命名规则 |

#### 命名示例

```
output/real_ct/
├── liver_lesion/
│   ├── liver_lesion_t00__BDMAP_00000012.nii.gz
│   ├── liver_lesion_t01__BDMAP_00000012.nii.gz
│   └── ...
├── pancreatic_lesion/
│   ├── pancreatic_lesion_t00__BDMAP_00000019.nii.gz
│   └── ...
└── ...
```

> 命名格式: `{organ_type}_t{序号}__{CT编号}.nii.gz`，序号00-49按体积降序排列。

---

### 3.9 logging — 日志与统计配置

```json
"logging": {
    "log_file": "output/generation_log.json",
    "stats_file": "output/statistics.json",
    "verbose": true
}
```

| 字段 | 含义 |
|------|------|
| `log_file` | 逐sample的详细生成记录（位置/尺寸/类型/成功状态） |
| `stats_file` | 汇总统计（实际分布 vs 目标4:2:1分布） |
| `verbose` | 是否在控制台输出详细进度 |

---

## 四、代码如何使用此配置

### 加载方式

```python
# main.py 中的 load_config()
import json

def load_config(config_path="config/generation_config.json"):
    """
    加载JSON配置文件。
    运行时路径: config/generation_config.json (从 Step0/config/ 复制或链接而来)
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 后续可加 schema 校验
    return config
```

### 各模块读取的配置路径

```
模块                   读取的配置字段
────────────────────────────────────────────────────
data_loader.py         config['data']                  数据路径
                       config['preprocessing']          HU裁剪范围
                       
position_selector.py   config['placement']             策略/margin/重试次数
                       config['size_categories']
                       
mask_generator.py      config['shape']                 管线各步骤参数
                       config['preprocessing']         (参考，不直接使用)
                       
main.py                config['organs']                批量循环
                       config['size_categories']       采样权重
                       config['output']                命名规则
                       config['logging']               日志路径
                       config['project']               输出目录
```

---

## 五、关键算法分析

### 5.1 尺寸采样算法

```
输入: size_categories (来自config)
输出: size_category_name (如 "small"), radius_mm

算法:
  ① 从 categories 中提取 weight > 0 的分类名和权重
     items = [(k, v) for k, v in categories if v['weight'] > 0]
     → [('small', 4), ('medium', 2), ('large', 1)]

  ② 归一化权重为概率分布
     total = sum(w for _, w in items) = 7
     probs = [4/7, 2/7, 1/7]

  ③ 按概率采样分类
     cat_name = np.random.choice(names, p=probs)

  ④ 在分类的半径范围内 uniform 采样
     radius_mm = np.random.uniform(cat['r_min_mm'], cat['r_max_mm'])

  ⑤ 返回 (cat_name, radius_mm)
```

### 5.2 椭球轴比例采样（带体积守恒）

```
输入: axis_ratio_range [0.8, 1.2], volume_conservation=true
输出: (ratio_z, ratio_y, ratio_x)

算法:
  ① 各轴独立采样
     ratio_z = uniform(0.8, 1.2)
     ratio_y = uniform(0.8, 1.2)
     ratio_x = uniform(0.8, 1.2)

  ② 如果 volume_conservation:
     vol_factor = (ratio_z * ratio_y * ratio_x) ^ (1/3)
     ratio_z /= vol_factor
     ratio_y /= vol_factor
     ratio_x /= vol_factor

  ③ 最终: ratio_z * ratio_y * ratio_x = 1.0 (体积与等半径球体一致)
```

---

## 六、设计决策与论文依据

| 决策 | 配置体现 | 论文出处 |
|------|---------|----------|
| 椭球体为初始形状 | `shape.method = "ellipsoid"` | DiffTumor §3.3 (P5) |
| 弹性形变必须执行 | `shape.elastic_deformation.enabled = true` | DiffTumor §F.1 (P22) |
| salt-noise + Gaussian + scaling + clipping | `shape.salt_noise/gaussian_filter/scaling_clipping` | DiffTumor §F.1 (P22) |
| 4:2:1尺寸比例 | `size_categories.categories.*.weight = [0,4,2,1]` | Scaling §4.1 (P7) |
| 四档半径分类 | `size_categories.categories.*.r_min/r_max_mm` | Scaling §3.2 (P5) |
| HU范围 [-175, 250] | `preprocessing.hu_min/hu_max` | DiffTumor §E.2 (P21) |
| 6类器官命名 | `organs[].name` 使用 AbdomenAtlas2.0 命名 | AbdomenAtlas2.0 class_map |

---

## 七、扩展指南

### 新增肿瘤类型

在 `organs` 数组中添加一项即可：

```json
{
    "name": "spleen_lesion",
    "class_id": 33,
    "organ_label_file": "spleen.nii.gz",
    "organ_name": "spleen",
    "count": 50,
    "description": "脾脏肿瘤"
}
```

### 新增尺寸分类

在 `size_categories.categories` 中添加，设置合适的 `r_min/r_max_mm` 和 `weight`。

### 调整管线步骤

将对应步骤的 `enabled` 设为 `false` 即可关闭。例如跳过弹性形变：

```json
"elastic_deformation": { "enabled": false, ... }
```

### 切换位置策略

修改 `placement.strategy` 即可：

```json
"strategy": "distance_weighted"
```

> **注意**: 切换策略前确认对应策略的参数（如 `distance_weighted.alpha`）已配置正确。
