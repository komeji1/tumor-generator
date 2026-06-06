# Step0/tests/__init__.py — 详细解读

> **所属步骤**: Step 0 — 项目初始化  
> **文件路径**: `Step0/tests/__init__.py`  
> **运行时路径**: 项目运行时会复制/链接到 `tests/__init__.py` 作为测试包入口  
> **文件类型**: Python 测试包初始化文件

---

## 一、文件功能概述

`tests/__init__.py` 将 `tests/` 目录标记为 Python 包，使得测试发现工具（如 `pytest`、`unittest`）可以正确识别和运行测试。

### 当前内容

```python
# Tests for Tumor Mask Generator
```

---

## 二、测试框架选择

本项目使用 **pytest** 作为测试框架。`tests/__init__.py` 的存在确保 pytest 能正确导入测试模块。

### 测试运行方式

```bash
# 运行全部测试
pytest tests/

# 运行单个测试文件
pytest tests/test_utils.py

# 运行单个测试函数
pytest tests/test_utils.py::test_clip_hu

# 带详细输出
pytest tests/ -v

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

---

## 三、测试文件规划

```
tests/
├── __init__.py                  ← 本文件
├── test_utils.py                ← Step 1 完成后创建
├── test_data_loader.py          ← Step 2 完成后创建
├── test_validator.py            ← Step 3 完成后创建
├── test_position_selector.py    ← Step 4 完成后创建
├── test_mask_generator.py       ← Step 5 完成后创建
└── test_integration.py          ← Step 6 完成后创建（端到端测试）
```

---

## 四、测试原则

| 原则 | 说明 |
|------|------|
| **一个模块一个测试文件** | `test_<模块名>.py` 对应 `src/<模块名>.py` |
| **依赖最小化** | 测试只依赖被测试模块 + numpy，不依赖其他src模块 |
| **独立可运行** | 每个测试文件可单独执行，不依赖执行顺序 |
| **快速** | 单元测试使用合成的numpy数组，不读取真实.nii.gz文件 |
| **确定性** | 固定 `np.random.seed(42)`，确保结果可复现 |

---

## 五、测试数据策略

```
┌─────────────────────────────────────────────────────────────────┐
│  测试阶段      │  数据来源           │  说明                      │
│  ─────────────────────────────────────────────────────────────  │
│  单元测试      │  numpy 合成数组     │  无需真实CT文件             │
│  集成测试      │  小型合成 .nii.gz   │  使用 nibabel 创建测试数据  │
│  真实验证      │  真实 AbdomenAtlas  │  最终验证阶段手动运行       │
└─────────────────────────────────────────────────────────────────┘
```

### 合成测试数据示例

```python
# 在 test_mask_generator.py 中创建假CT
import numpy as np
import nibabel as nib

def make_fake_ct(shape=(64, 64, 32)):
    """创建合成CT用于测试"""
    data = np.random.randint(-175, 250, shape, dtype=np.int16)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm isotropic
    return data, affine

def make_fake_organ_mask(shape=(64, 64, 32), organ_radius=20):
    """创建球形器官mask用于测试"""
    Z, Y, X = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) // 2
    dist = np.sqrt((Z-center[0])**2 + (Y-center[1])**2 + (X-center[2])**2)
    return (dist <= organ_radius).astype(np.uint8)
```

---

## 六、与其他文件的关系

```
Step0/tests/__init__.py                     ← 本文件：测试包标识
        │
        ├── test_utils.py                   ← 测试 Step1/src/utils.py
        ├── test_data_loader.py             ← 测试 Step2/src/data_loader.py
        ├── test_validator.py               ← 测试 Step3/src/validator.py
        ├── test_position_selector.py       ← 测试 Step4/src/position_selector.py
        ├── test_mask_generator.py          ← 测试 Step5/src/mask_generator.py
        └── test_integration.py             ← 端到端集成测试
```
