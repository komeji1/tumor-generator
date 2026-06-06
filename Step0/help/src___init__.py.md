# Step0/src/__init__.py — 详细解读

> **所属步骤**: Step 0 — 项目初始化  
> **文件路径**: `Step0/src/__init__.py`  
> **运行时路径**: 项目运行时会复制/链接到 `src/__init__.py` 作为 Python 包入口  
> **文件类型**: Python 包初始化文件

---

## 一、文件功能概述

`src/__init__.py` 是 Python 包的标识文件，将 `src/` 目录标记为一个 Python 包（package），使得：

1. 其他模块可以通过 `from src import utils` 等方式导入
2. 测试文件可以通过 `from src.xxx import yyy` 访问源码模块
3. IDE 可以正确识别 `src/` 下的模块并提供自动补全

### 当前内容

```python
# Tumor Mask Generator
# 肿瘤位置Mask自动生成器
#
# 项目结构:
#   utils.py              - 工具函数 (坐标变换/形态学操作/HU处理)
#   data_loader.py         - 数据加载 (CT + 器官mask)
#   validator.py           - 校验模块 (位置/mask质量)
#   position_selector.py   - 位置选择 (多种策略)
#   mask_generator.py      - Mask生成 (椭球 + 弹性形变 + 后处理)
#   main.py                - 主入口 (批量生成)
```

---

## 二、Python 包机制说明

### 2.1 为什么需要 `__init__.py`？

```
没有 __init__.py:                   有 __init__.py:
──────────────────────────          ──────────────────────────
src/                                src/
  utils.py          ← 普通目录       __init__.py  ← Python包
  data_loader.py                       utils.py
                                     data_loader.py

import src.utils     ✗ 失败         from src import utils  ✓ 成功
```

### 2.2 包导入路径

```
本项目中的导入约定:
────────────────────────────────────────────────
# main.py 中导入同包模块:
from src.utils import clip_hu, voxel_to_mm
from src.data_loader import load_ct, load_organ_mask
from src.position_selector import select_location
from src.mask_generator import create_mask, mask_to_nifti
from src.validator import validate_sample

# 测试文件中导入:
from src.utils import erode_mask, compute_ellipsoid_dist
from src.mask_generator import create_ellipsoid
```

---

## 三、项目结构注释

文件中的注释块充当了**模块索引**的角色。当开发者在 IDE 中打开 `src/__init__.py` 时，可以快速了解：

| 模块文件 | 职责 | 依赖 |
|----------|------|------|
| `utils.py` | 坐标变换、形态学操作、HU处理、弹性形变 | 无 |
| `data_loader.py` | CT与器官mask的加载、校验、索引 | utils |
| `validator.py` | 位置校验、mask质量检查 | utils |
| `position_selector.py` | 在器官有效区域内选取肿瘤中心位置 | utils, data_loader, validator |
| `mask_generator.py` | 椭球生成、弹性形变、后处理管线、NIfTI输出 | utils, data_loader |
| `main.py` | 配置加载、批量循环、统计汇总 | 全部模块 |

---

## 四、后续可能的扩展

### 4.1 添加包级别导入

当所有模块就绪后，可在 `__init__.py` 中添加便捷导入：

```python
# 便捷导入: from src import create_mask
from src.mask_generator import create_mask, mask_to_nifti
from src.data_loader import load_ct, load_organ_mask
from src.position_selector import select_location
from src.validator import validate_sample
```

### 4.2 添加包版本信息

```python
__version__ = "1.0.0"
__author__ = "Tumor Mask Generator Team"
```

---

## 五、与其他文件的关系

```
generation_config.json          ← 配置数据（Step 0 产物，位于 Step0/config/）
        │
        │ 被 Step6/src/main.py 读取
        ▼
Step0/src/__init__.py           ← 本文件：包标识 + 模块索引
        │
        ├── Step1/src/utils.py        ← Step 1
        ├── Step2/src/data_loader.py  ← Step 2
        ├── Step3/src/validator.py    ← Step 3
        ├── Step4/src/position_selector.py ← Step 4
        ├── Step5/src/mask_generator.py    ← Step 5
        └── Step6/src/main.py         ← Step 6
```
