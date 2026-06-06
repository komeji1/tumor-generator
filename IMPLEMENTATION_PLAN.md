# Tumor Mask Generator — 逐模块实现计划

> **项目**: 肿瘤位置Mask自动生成器  
> **创建日期**: 2026-06-04  
> **原则**: 按依赖关系排序，先构建基础设施，再构建依赖模块，最后组装主流程。每个模块独立可测。

---

## 文件夹组织约定

```
Mask/
├── Step0/                        ← 每个步骤独立文件夹
│   ├── config/                   ←   该步骤的配置文件
│   ├── src/                      ←   该步骤的源码文件
│   ├── tests/                    ←   该步骤的测试文件
│   └── help/                     ←   该步骤的解读文档（与生成文件同名）
│
├── Step1/                        ← 下一步骤
│   ├── src/
│   │   └── utils.py
│   └── help/
│       └── utils.py.md
│
├── Step2/ ... Step7/
│
├── data/                         ← 共享数据目录（所有步骤共用）
│   ├── ct/
│   └── organ_labels/
├── output/                       ← 共享输出目录（运行时生成）
├── PROJECT_OVERVIEW.md
└── IMPLEMENTATION_PLAN.md         ← 本文件
```

> **注意**: `data/` 和 `output/` 在顶层共享，不属于任何单一步骤。源码在各 `StepN/src/` 中按模块独立存放，运行时会汇总到顶层 `src/`。

---

## 目录

1. [模块依赖关系图](#一模块依赖关系图)
2. [实现顺序总览](#二实现顺序总览)
3. [逐模块详细说明](#三逐模块详细说明)

---

## 一、模块依赖关系图

```
                         ┌─────────────────┐
                         │  generation_    │
                         │  config.json    │  ← Step 0: 配置文件（无代码依赖）
                         └───────┬─────────┘
                                 │ 被 main.py 读取
                                 │
                         ┌───────┴─────────┐
                         │    utils.py     │  ← Step 1: 零依赖，全模块的基础
                         └───────┬─────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
     │  data_loader   │ │  validator   │ │ (其他模块)      │
     │     .py        │ │     .py      │ │                │
     │  Step 2        │ │  Step 3      │ │                │
     └───────┬────────┘ └──────┬───────┘ └────────────────┘
             │                 │
             │    ┌────────────┘
             │    │
             ▼    ▼
     ┌────────────────┐
     │   position     │
     │  _selector.py  │  ← Step 4: 依赖 utils + data_loader + validator
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │     mask       │
     │  _generator.py │  ← Step 5: 依赖 utils + data_loader
     └───────┬────────┘
             │
             │    (所有模块就绪)
             │
             ▼
     ┌────────────────┐
     │    main.py     │  ← Step 6: 串联全部模块，批量生成
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │  tests/        │  ← Step 7: 单元测试 & 集成测试
     └────────────────┘
```

> **为什么按这个顺序？**  
> - `utils.py` 在最底层，提供坐标变换、形态学操作、HU裁剪等纯函数，所有其他模块都需要它  
> - `data_loader.py` 和 `validator.py` 只依赖 `utils.py`，两者之间互不依赖，可以并行开发  
> - `position_selector.py` 需要加载数据 + 验证位置合法性，所以依赖 `data_loader` + `validator`  
> - `mask_generator.py` 需要CT的空间参考（shape/affine），所以依赖 `data_loader`  
> - `main.py` 是胶水层，在所有模块就绪后最后组装

---

## 二、实现顺序总览

```
Step 0  ─→  generation_config.json    配置定义        (JSON文件，无代码)
Step 1  ─→  utils.py                  工具函数层       (零依赖)
Step 2  ─→  data_loader.py            数据加载层       (依赖: utils)
Step 3  ─→  validator.py              校验层          (依赖: utils)
Step 4  ─→  position_selector.py      位置选择层       (依赖: utils, data_loader, validator)
Step 5  ─→  mask_generator.py         Mask生成层      (依赖: utils, data_loader)
Step 6  ─→  main.py                   主入口+批量      (依赖: 全部模块)
Step 7  ─→  tests/                    测试验证         (依赖: 全部模块)
```

---

## 三、逐模块详细说明

---

### Step 0: `Step0/config/generation_config.json`

**依赖**: 无  
**被依赖**: `main.py`

#### 职责

定义所有生成参数，是项目的唯一参数入口。

#### 配置内容

```json
{
  "_comment": "Tumor Mask Generator — 生成参数配置",

  "project": {
    "name": "Tumor Mask Generator",
    "version": "1.0.0",
    "output_dir": "output/"
  },

  "data": {
    "ct_dir": "data/ct/",
    "organ_label_dir": "data/organ_labels/",
    "manifest_path": "data/manifest.csv"
  },

  "organs": [
    {
      "name": "liver_lesion",
      "class_id": 27,
      "organ_label_file": "liver.nii.gz",
      "count": 20
    },
    {
      "name": "pancreatic_lesion",
      "class_id": 28,
      "organ_label_file": "pancreas.nii.gz",
      "count": 20
    },
    {
      "name": "kidney_lesion",
      "class_id": 29,
      "organ_label_file": "kidney_left.nii.gz",
      "count": 20
    },
    {
      "name": "colon_lesion",
      "class_id": 30,
      "organ_label_file": "colon.nii.gz",
      "count": 20
    },
    {
      "name": "esophagus_tumor",
      "class_id": 32,
      "organ_label_file": "esophagus.nii.gz",
      "count": 20
    },
    {
      "name": "endometrioma_tumor",
      "class_id": 31,
      "organ_label_file": "uterus.nii.gz",
      "count": 20
    }
  ],

  "size_categories": {
    "tiny":   { "r_min_mm": 1,  "r_max_mm": 5,  "weight": 0.0 },
    "small":  { "r_min_mm": 5,  "r_max_mm": 10, "weight": 4 },
    "medium": { "r_min_mm": 10, "r_max_mm": 20, "weight": 2 },
    "large":  { "r_min_mm": 20, "r_max_mm": 50, "weight": 1 }
  },

  "shape": {
    "method": "ellipsoid",
    "axis_ratio_range": [0.8, 1.2],
    "volume_conservation": true,
    "elastic_deformation": {
      "enabled": true,
      "alpha": 15,
      "sigma": 3
    },
    "salt_noise": {
      "enabled": true,
      "probability": 0.02
    },
    "gaussian_filter": {
      "enabled": true,
      "sigma_mm": 1.0
    }
  },

  "placement": {
    "strategy": "uniform",
    "margin": {
      "feather_mm": 3,
      "safety_mm": 5
    },
    "max_retries": 50
  },

  "preprocessing": {
    "hu_min": -175,
    "hu_max": 250
  },

  "output": {
    "format": "nifti",
    "dtype": "uint8",
    "value_range": [0, 1],
    "compress": true,
    "naming_pattern": "{organ_type}_{sample_id:03d}.nii.gz"
  },

  "logging": {
    "log_file": "output/generation_log.json",
    "stats_file": "output/statistics.json",
    "verbose": true
  }
}
```

#### 关键参数说明

| 参数路径 | 说明 | 论文依据 |
|----------|------|----------|
| `size_categories.*.weight` | 4:2:1比例权重 | Scaling §4.1 (P7) |
| `size_categories.*.r_min/max_mm` | 四档半径范围 | Scaling §3.2 (P5) |
| `shape.method` | 椭球体 | DiffTumor §3.3 (P5) |
| `shape.elastic_deformation` | 弹性形变 | DiffTumor §F.1 (P22) |
| `preprocessing.hu_min/max` | HU裁剪范围 | DiffTumor §E.2 (P21) |

---

### Step 1: `Step1/src/utils.py`

**依赖**: 无  
**被依赖**: 全部模块

#### 职责

提供所有模块共用的纯函数工具。**不涉及任何文件IO或业务逻辑。**

#### 函数列表

```python
# ========== 坐标变换 ==========

def voxel_to_mm(voxel_coord, affine):
    """体素坐标 → 物理坐标 (mm)"""

def mm_to_voxel(mm_coord, affine):
    """物理坐标 (mm) → 体素坐标"""

def get_spacing(affine):
    """从affine矩阵提取体素间距 (mm/voxel)"""


# ========== HU值处理 ==========

def clip_hu(ct_array, hu_min=-175, hu_max=250):
    """HU值裁剪到指定范围"""

def normalize_hu(ct_array, hu_min=-175, hu_max=250):
    """HU值线性归一化到[0, 1]"""


# ========== 形态学操作 ==========

def erode_mask(mask_3d, radius_voxel):
    """3D二值mask腐蚀 (使用球结构元素)"""

def dilate_mask(mask_3d, radius_voxel):
    """3D二值mask膨胀"""


# ========== 几何计算 ==========

def compute_ellipsoid_dist(shape, center, radii):
    """计算椭球距离场: dist = √((z/rz)² + (y/ry)² + (x/rx)²)"""

def volume_from_radius(radius_mm, spacing):
    """从等效半径估算椭球体积 (体素数)"""

def compute_valid_region(organ_mask, margin_voxel):
    """腐蚀器官mask得到可采样有效区域"""


# ========== 随机采样 ==========

def random_sample_valid(valid_mask, n=1):
    """在有效区域中随机采样体素坐标"""

def random_axis_ratios(ratio_range=(0.8, 1.2)):
    """生成随机轴比例 (保持体积守恒)"""


# ========== 弹性形变 ==========

def generate_elastic_deformation_field(shape, alpha, sigma):
    """生成弹性形变位移场 (基于高斯滤波的随机位移)"""

def apply_deformation(mask_3d, displacement_field):
    """对mask施加位移场"""
```

#### 接口约定

- 所有函数不持有状态，不读取文件
- 输入/输出均为 numpy 数组或标量
- 坐标约定: `(z, y, x)` 顺序
- 需要依赖的库: `numpy`, `scipy.ndimage`

#### 为什么先做 utils？

utils.py 是纯函数集合，无外部依赖，可以**立即编写并通过单元测试验证**。后续所有模块的代码质量都建立在这些工具函数的正确性之上。

---

### Step 2: `Step2/src/data_loader.py`

**依赖**: `utils.py`  
**被依赖**: `position_selector.py`, `mask_generator.py`, `main.py`

#### 职责

加载CT影像和器官分割mask，统一接口返回标准化的数据结构。

#### 数据结构

```python
@dataclass
class CTVolume:
    """CT扫描数据结构"""
    array: np.ndarray       # (D, H, W) uint16, HU值
    affine: np.ndarray      # 4×4 affine矩阵
    spacing: tuple          # (dz, dy, dx) mm/voxel
    shape: tuple            # (D, H, W)
    path: str               # 文件路径

@dataclass
class OrganMask:
    """器官分割数据结构"""
    array: np.ndarray       # (D, H, W) uint8, {0, 1}
    affine: np.ndarray      # 4×4 (应与CT一致)
    organ_type: str         # 器官类型名称
    path: str               # 文件路径
```

#### 函数列表

```python
def load_ct(ct_path):
    """加载CT .nii.gz → CTVolume"""

def load_organ_mask(mask_path, organ_type):
    """加载器官分割mask → OrganMask"""

def load_sample(ct_path, organ_mask_path, organ_type):
    """一次加载CT + 器官mask，返回 (CTVolume, OrganMask)"""

def get_organ_bbox(organ_mask):
    """获取器官mask的bounding box (用于裁剪加速)"""

def validate_compatibility(ct, organ_mask):
    """验证CT与mask的shape/affine一致性，不一致则报错"""

def build_manifest(ct_dir, organ_label_dir, organ_config):
    """扫描数据目录，构建sample索引列表 → list[dict]"""
```

#### 需要处理的异常情况

| 情况 | 处理方式 |
|------|----------|
| CT文件不存在 | 抛出 `FileNotFoundError` 含路径 |
| 器官mask为空（全零） | 抛出 `ValueError: organ mask is empty` |
| CT与mask的shape不一致 | 抛出 `ValueError: shape mismatch` |
| CT与mask的affine不一致 | 警告但继续（可能仅平移差异） |
| CT与mask的spacing差异大 | 警告，建议检查数据 |

#### 接口约定

- 所有`load_*()`返回上述dataclass，方便IDE类型提示
- `validate_compatibility()` 在加载后自动调用
- `build_manifest()` 扫描目录并返回sample列表，供main.py批量迭代

---

### Step 3: `Step3/src/validator.py`

**依赖**: `utils.py`  
**被依赖**: `position_selector.py`, `mask_generator.py`, `main.py`

#### 职责

校验位置选择和mask生成的结果是否满足约束条件。

#### 函数列表

```python
def check_in_organ(center, mask_3d, organ_mask, radius_voxel):
    """
    检查以center为中心、radius为半径的球体是否完全在organ_mask内
    返回: (is_valid: bool, message: str)
    """

def check_not_overlapping(mask_3d, existing_masks):
    """
    检查新mask不与已有mask重叠（如需生成多肿瘤时）
    返回: (is_valid: bool, overlap_ratio: float)
    """

def check_size_range(radius_mm, size_category_config):
    """
    检查半径是否在指定size_category的范围内
    返回: (is_valid: bool, expected_range: tuple)
    """

def check_organ_volume(organ_mask, min_voxels=100):
    """
    检查器官mask的体积是否足够放置肿瘤
    返回: (is_valid: bool, volume_voxels: int)
    """

def check_mask_nonzero(mask_3d):
    """
    检查生成的mask非空
    返回: (is_valid: bool, nonzero_count: int)
    """

def validate_sample(center, radius_voxel, radius_mm, mask_3d,
                    organ_mask, size_config, existing_masks=None):
    """
    一站式校验: 依次执行上述所有检查
    返回: {
        'passed': bool,
        'checks': {name: (bool, str)},
        'errors': [str]
    }
    """
```

#### 校验规则

```
校验流程 (validate_sample):
─────────────────────────────────────────
① check_organ_volume      → 器官足够大？
② check_in_organ          → 肿瘤完全在器官内？
③ check_size_range        → 半径在分类范围内？
④ check_mask_nonzero      → mask非空？
⑤ check_not_overlapping   → 不与已有mask重叠？(可选)
```

#### 接口约定

- 所有校验函数返回`(bool, str/details)`元组，方便调用方获取失败原因
- `validate_sample()` 是一站式入口，会执行所有必要的检查
- 校验失败不抛异常，返回失败信息 → 由调用方决定重试或放弃

---

### Step 4: `Step4/src/position_selector.py`

**依赖**: `utils.py`, `data_loader.py`, `validator.py`  
**被依赖**: `main.py`

#### 职责

在器官的有效区域内，按策略为肿瘤选择一个合法的中心位置。

#### 策略枚举

```python
class PlacementStrategy(Enum):
    UNIFORM = "uniform"            # 所有valid体素等概率
    DISTANCE_WEIGHTED = "distance" # 按到边界距离加权
    SUBREGION = "subregion"        # 先选解剖亚区，再区内均匀
```

#### 函数列表

```python
def compute_valid_region(organ_mask, margin_voxel):
    """
    计算可采样有效区域 = erode(organ_mask, margin)
    margin = max_tumor_radius + feather + safety
    """

def sample_uniform(valid_mask, rng=None):
    """
    均匀随机策略: 所有valid体素等概率选取
    返回: (z, y, x) 体素坐标
    """

def sample_distance_weighted(valid_mask, organ_mask, alpha=1.0, rng=None):
    """
    距离加权策略: prob ∝ distance(体素, 器官表面)^alpha
    alpha > 0: 偏向中心
    alpha = 0: 退化为uniform
    """

def select_location(organ_mask, radius_voxel, strategy, config):
    """
    主入口: 在器官mask的有效区域内选择一个位置
    
    Args:
        organ_mask: OrganMask对象
        radius_voxel: 肿瘤半径(体素)
        strategy: PlacementStrategy
        config: 完整配置dict
    
    Returns:
        center_zyx: (z, y, x) 体素坐标
        is_valid: bool
    
    内部流程:
        ① 计算margin = radius + feather + safety
        ② 腐蚀得到valid_region
        ③ 按策略采样
        ④ 调用validator.check_in_organ() 验证
        ⑤ 失败则重试（最多max_retries次）
    """
```

#### 重试逻辑

```
select_location 重试流程:
─────────────────────────
① 计算有效区域
② 如果有效区域为空 → 抛出 ValueError (器官太小)
③ 循环 (最多 max_retries 次):
    a. 按策略采样 center_zyx
    b. check_in_organ(center, ...) 
    c. 如果通过 → 返回 center_zyx
    d. 如果失败 → 继续循环
④ 全部失败 → 抛出 RuntimeError (可调整margin或radius后重试)
```

---

### Step 5: `Step5/src/mask_generator.py`

**依赖**: `utils.py`, `data_loader.py`  
**被依赖**: `main.py`

#### 职责

以选定位置为中心，按论文管线生成二值肿瘤mask。

#### 生成管线

```
create_mask(center_zyx, radius_mm, ct_volume, config)
│
├── Step 5a: 创建基础椭球体
│   ├── mm → voxel 坐标转换
│   ├── 生成随机轴比例 (体积守恒)
│   └── compute_ellipsoid_dist(shape, center, radii) ≤ 1
│
├── Step 5b: 弹性形变 (if enabled)
│   ├── generate_elastic_deformation_field(shape, alpha, sigma)
│   └── apply_deformation(mask, field)
│
├── Step 5c: Salt-Noise (if enabled)
│   └── mask内部随机翻转少量体素
│
├── Step 5d: Gaussian Filter (if enabled)
│   └── 对mask边界做高斯平滑
│
└── Step 5e: Scaling & Clipping
    └── 缩放到{0, 1}，clip到有效范围
```

#### 函数列表

```python
def create_ellipsoid(shape, center_zyx, radii_voxel):
    """
    创建基础椭球体mask (无变形)
    论文依据: DiffTumor §3.3 (P5): "using ellipsoids"
    
    Args:
        shape: (D, H, W) CT体积形状
        center_zyx: (z, y, x) 椭球中心
        radii_voxel: (rz, ry, rx) 三轴半径(体素)
    
    Returns:
        mask: (D, H, W) uint8, {0, 1}
    """

def compute_radii_from_mm(radius_mm, spacing, axis_ratio_range=(0.8, 1.2)):
    """
    将等效半径(mm)转为三轴体素半径
    
    Args:
        radius_mm: 肿瘤等效半径
        spacing: (dz, dy, dx) mm/voxel
        axis_ratio_range: 各轴随机比例范围
    
    Returns:
        radii_voxel: (rz, ry, rx) 三轴体素半径
    """

def apply_elastic_deformation(mask, alpha=15, sigma=3):
    """
    对mask施加弹性形变，使边界不规则
    论文依据: DiffTumor §F.1 (P22): "elastic deformation"
    
    方法: 生成低频随机位移场，用scipy.ndimage.map_coordinates重采样
    """

def apply_salt_noise(mask, probability=0.02):
    """
    在mask内部随机翻转少量体素
    论文依据: DiffTumor §F.1 (P22): "salt-noise generation"
    """

def apply_gaussian_smoothing(mask, sigma_mm=1.0, spacing=None):
    """
    对mask边界做高斯平滑
    论文依据: DiffTumor §F.1 (P22): "Gaussian filtering"
    """

def create_mask(center_zyx, radius_mm, ct_volume, config):
    """
    主入口: 执行完整Mask生成管线
    
    Args:
        center_zyx: 肿瘤中心体素坐标
        radius_mm: 肿瘤等效半径(mm)
        ct_volume: CTVolume对象 (提供shape, spacing)
        config: 完整配置dict
    
    Returns:
        mask_3d: (D, H, W) uint8, {0, 1}
    
    管线流程 (严格按论文顺序):
        ① create_ellipsoid()
        ② apply_elastic_deformation()  (if enabled)
        ③ apply_salt_noise()           (if enabled)
        ④ apply_gaussian_smoothing()   (if enabled)
        ⑤ clip & cast to uint8
    """

def mask_to_nifti(mask_3d, affine, output_path):
    """
    将3D mask数组保存为.nii.gz文件
    
    Args:
        mask_3d: (D, H, W) uint8
        affine: 4×4矩阵 (使用CT的affine)
        output_path: 输出文件路径
    """
```

#### 管线开关控制

所有后处理步骤（弹性形变/噪声/滤波）均可通过config中的`enabled`字段独立开关：

```json
{
  "shape": {
    "elastic_deformation": { "enabled": true, ... },
    "salt_noise":          { "enabled": true, ... },
    "gaussian_filter":     { "enabled": true, ... }
  }
}
```

默认全部开启（与论文描述一致）。

---

### Step 6: `Step6/src/main.py`

**依赖**: 所有模块 (`utils`, `data_loader`, `position_selector`, `mask_generator`, `validator`)  
**被依赖**: 无（顶层入口）

#### 职责

串联所有模块，实现批量生成循环，输出300个mask文件（最终每器官50个，共6器官）。

#### 函数列表

```python
def load_config(config_path):
    """加载并校验JSON配置 → dict"""

def sample_size_category(size_config, rng=None):
    """
    按4:2:1权重采样size_category
    论文依据: Scaling §4.1 (P7)
    """

def sample_radius(size_category, size_config, rng=None):
    """
    在指定size_category的半径范围内uniform采样
    """

def generate_one(ct_volume, organ_mask, organ_type, config, rng=None):
    """
    生成单张mask的完整流程:
    
    ① sample_size_category() → "small"/"medium"/"large"/"tiny"
    ② sample_radius()        → radius_mm
    ③ select_location()      → center_zyx
    ④ create_mask()          → mask_3d
    ⑤ mask_to_nifti()        → 保存.nii.gz
    ⑥ 记录metadata
    
    Returns:
        metadata: dict {
            'organ_type', 'sample_id', 'size_category',
            'radius_mm', 'center_zyx', 'output_path',
            'success': bool, 'error': str|None
        }
    """

def generate_batch(config, rng_seed=42):
    """
    批量生成主循环:
    
    ① 加载config
    ② 创建output目录结构
    ③ 构建sample列表 (build_manifest)
    ④ for each organ_type × 20 samples:
         generate_one()
    ⑤ 保存 generation_log.json
    ⑥ 计算并保存 statistics.json
    ⑦ 打印汇总报告
    
    Returns:
        results: list[metadata_dict]
    """

def compute_statistics(results, config):
    """
    汇总统计:
    - 每种器官的实际尺寸分布 vs 目标分布
    - 成功率统计
    - 每个size_category的实际count
    """

def main():
    """CLI入口点: 解析参数 → generate_batch()"""
```

#### 批量生成伪代码

```python
def generate_batch(config, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    results = []
    
    for organ_cfg in config['organs']:       # 6种
        organ_type = organ_cfg['name']
        count = organ_cfg['count']           # 20
        
        for sample_id in range(count):
            # 1. 加载数据
            ct_volume, organ_mask = load_sample(...)
            
            # 2. 采样尺寸
            size_cat = sample_size_category(config['size_categories'], rng)
            radius_mm = sample_radius(size_cat, config['size_categories'], rng)
            
            # 3. 选位置 + 生成mask
            try:
                center = select_location(organ_mask, radius_voxel, ...)
                mask = create_mask(center, radius_mm, ct_volume, config)
                
                # 4. 保存
                output_path = f"output/{organ_type}/{organ_type}_{sample_id:03d}.nii.gz"
                mask_to_nifti(mask, ct_volume.affine, output_path)
                
                results.append({
                    'organ_type': organ_type,
                    'sample_id': sample_id,
                    'size_category': size_cat,
                    'radius_mm': radius_mm,
                    'center_zyx': center,
                    'output_path': output_path,
                    'success': True
                })
                
            except Exception as e:
                results.append({..., 'success': False, 'error': str(e)})
                # 继续生成下一张
                continue
    
    # 5. 保存日志和统计
    save_log(results, config['logging']['log_file'])
    stats = compute_statistics(results, config)
    save_stats(stats, config['logging']['stats_file'])
    
    return results
```

#### 异常处理策略

```
单个sample失败 → 记录错误 → 继续下一个 → 不中断整个batch
所有sample失败 → 在最终报告中标注
CT/器官文件不存在 → 预处理阶段提前报错，不进入生成循环
```

---

### Step 7: `Step7/tests/` 测试模块

**依赖**: 全部模块  
**被依赖**: 无

#### 测试文件

```
tests/
├── test_utils.py              # Step 1 测试
├── test_data_loader.py        # Step 2 测试
├── test_validator.py          # Step 3 测试
├── test_position_selector.py  # Step 4 测试
├── test_mask_generator.py     # Step 5 测试
└── test_integration.py        # 集成测试 (端到端)
```

#### 每个测试文件内容

| 测试文件 | 关键测试用例 |
|----------|------------|
| `test_utils.py` | `voxel_to_mm` 往返一致性、`erode_mask` 边界收缩量正确、`clip_hu` 裁剪范围正确、`compute_ellipsoid_dist` 距离计算正确、`random_axis_ratios` 体积守恒 |
| `test_data_loader.py` | 加载已存在的CT、加载不存在的文件、shape/affine一致性检测、空mask检测 |
| `test_validator.py` | 球体完全在器官内/部分超出/完全超出、mask非零检测、尺寸范围边界值、小器官拒绝 |
| `test_position_selector.py` | uniform采样在valid区域内、重试耗尽时抛异常、margin计算正确性 |
| `test_mask_generator.py` | 椭球mask形状正确、弹性形变不改变总体范围、噪声添加体素数在预期范围、Gaussian平滑后mask仍为{0,1}（经clipping后）、nifti保存后可正确重载 |
| `test_integration.py` | 端到端: 假数据 → generate_one → 检查输出文件存在、值域{0,1}、affine一致 |

---

## 附录A: 各模块输入/输出速查

```
模块                 输入                              输出
──────────────────────────────────────────────────────────────────────
utils.py             numpy数组/标量                      numpy数组/标量
data_loader.py       文件路径                            CTVolume / OrganMask
validator.py         center, mask, organ_mask, config    (bool, details)
position_selector.py organ_mask, radius, config          (z, y, x) 坐标
mask_generator.py    center, radius, CTVolume, config    (D,H,W) uint8 mask
main.py              config文件路径                      120个.nii.gz + log
```

## 附录B: 实现检查清单

```
Step 0: config
  □ generation_config.json 创建并验证JSON格式正确

Step 1: utils.py
  □ 坐标变换: voxel_to_mm / mm_to_voxel / get_spacing
  □ HU处理: clip_hu / normalize_hu
  □ 形态学: erode_mask / dilate_mask
  □ 几何: compute_ellipsoid_dist / volume_from_radius
  □ 变形: generate_elastic_deformation_field / apply_deformation
  □ 每个函数有docstring和类型注解

Step 2: data_loader.py
  □ load_ct / load_organ_mask / load_sample
  □ CTVolume / OrganMask dataclass
  □ validate_compatibility
  □ build_manifest

Step 3: validator.py
  □ check_in_organ / check_not_overlapping / check_size_range
  □ check_organ_volume / check_mask_nonzero
  □ validate_sample (一站式)

Step 4: position_selector.py
  □ compute_valid_region
  □ sample_uniform
  □ select_location (含重试逻辑)

Step 5: mask_generator.py
  □ create_ellipsoid
  □ apply_elastic_deformation
  □ apply_salt_noise
  □ apply_gaussian_smoothing
  □ create_mask (管线主入口)
  □ mask_to_nifti

Step 6: main.py
  □ load_config
  □ sample_size_category / sample_radius
  □ generate_one
  □ generate_batch
  □ compute_statistics
  □ main()

Step 7: tests
  □ test_utils.py
  □ test_data_loader.py
  □ test_validator.py
  □ test_position_selector.py
  □ test_mask_generator.py
  □ test_integration.py
```

---

> **开始实现**: 从 Step 0 `generation_config.json` 开始，按顺序逐模块生成代码。
