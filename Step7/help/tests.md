# 测试套件 — Step 7

> **运行**: `pytest Step7/tests/ -v`  
> **结果**: 62 passed, 0 failed

## 测试文件

| 文件 | 测试类 | 用例数 | 覆盖模块 |
|------|--------|--------|----------|
| `test_utils.py` | 8 class | 18 | Step 1 全部函数 |
| `test_data_loader.py` | 5 class | 9 | Step 2 CTVolume/OrganMask/Sample + 加载/校验/manifest |
| `test_validator.py` | 6 class | 13 | Step 3 全部 6 校验函数 |
| `test_position_selector.py` | 4 class | 8 | Step 4 margin/采样/选择/异常 |
| `test_mask_generator.py` | 4 class | 9 | Step 5 半径/椭球/管线/NIfTI |
| `test_integration.py` | 1 class | 5 | 端到端 + 确定性 + 配置验证 |

## 测试策略

| 层级 | 内容 | 数据 |
|------|------|------|
| 单元测试 | 每个函数的正确性、边界值、异常 | numpy 合成数组 |
| 集成测试 | 完整 e2e 管线 + 确定性验证 | nibabel 合成 .nii.gz |
