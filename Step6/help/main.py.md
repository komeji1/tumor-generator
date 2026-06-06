# main.py — 详细解读

> **所属步骤**: Step 6 — 主入口  
> **文件路径**: `Step6/src/main.py`  
> **依赖**: 全部 Step0~Step5 模块  
> **被依赖**: 无（顶层入口）

---

## 目录

1. [文件功能概述](#一文件功能概述)
2. [批量生成流程](#二批量生成流程)
3. [函数详解](#三函数详解)
4. [CLI 用法](#四cli-用法)
5. [自检说明](#五自检说明)

---

## 一、文件功能概述

`main.py` 是项目的**顶层胶水模块**，串联全部 5 个底层模块实现批量生成。

```
main.py
  ├── load_config()            → 读取 JSON
  ├── sample_size_category()   → 4:2:1 权重采样 (Scaling §4.1)
  ├── sample_radius()          → 范围内均匀采样
  ├── generate_one()           → 单样本完整流程
  │     ├── data_loader        → 加载 CT + organ mask
  │     ├── 距离变换预检        → 预先过滤不可行尺寸
  │     ├── sample_size/radius → 尺寸采样 + tiny回退
  │     ├── position_selector  → 选择位置
  │     ├── mask_generator     → 生成 mask
  │     ├── 器官边界裁剪        → 裁剪肿瘤到器官内
  │     ├── 体积检测            → 拒绝零/极小体积
  │     ├── validator          → 校验
  │     └── mask_to_nifti      → 保存 .nii.gz
  ├── generate_batch()         → 循环直至成功数达标
  └── compute_statistics()     → 汇总统计
```

### 新增逻辑

前 5 个步骤实现了独立模块，`main.py` 新增的算法包括：
- 尺寸分类的 4:2:1 权重采样 + tiny自动回退（小器官）
- 距离变换预检：计算器官distance map，预先过滤不可行尺寸
- 器官边界裁剪：弹性变形后将肿瘤裁剪回器官内部
- 体积检测：拒绝生成后体积<3或裁剪损失>20%的mask
- tiny肿瘤禁用弹性变形：r<5mm时关闭变形避免边界溢出
- 批量循环：持续尝试直至成功数达标（非固定次数）

---

## 二、批量生成流程

```
generate_batch(config, rng_seed=42)

Step 1/4: 构建样本索引
  └── build_manifest(ct_dir, label_dir, organs)
      → 扫描数据目录 → 列出可用样本

Step 2/4: 生成Mask (循环尝试至成功数达标)
  └── for organ in organs:          (6 types, 50 each)
        while success < target:
          ├── 预检器官是否容纳肿瘤
          ├── sample_size_category  → 尺寸 + tiny回退
          ├── sample_radius         → radius_mm
          ├── generate_one()        → metadata
          ├── 体积检测 (拒绝0体积)
          └── 打印进度

Step 3/4: 统计汇总
  └── compute_statistics(results)
      → 成功率 / 尺寸分布 / 器官分布

Step 4/4: 保存日志
  ├── generation_log.json  → 逐样本详细记录
  └── statistics.json      → 汇总统计
```

### 单样本容错

单个样本失败不影响批量——错误被捕获并记录到 metadata，循环继续。

---

## 三、函数详解

### `load_config(config_path) → dict`

搜索顺序: 给定路径 → `Step0/config/generation_config.json`

### `sample_size_category(size_config, rng) → str`

按 4:2:1 权重从 `small/medium/large` 中采样。`tiny` 的 weight=0 不参与。

### `sample_radius(category, size_config, rng) → float`

在 `[r_min, r_max]` 内 uniform 采样。

### `generate_one(ct_path, organ_path, organ_type, ...) → dict`

单样本完整流程，异常被 try/except 捕获，返回 `{success: bool, ...}`。

### `generate_batch(config, rng_seed) → List[dict]`

批量主循环。打印 4 步进度条。

### `compute_statistics(results, config) → dict`

对比实际分布 vs 目标 4:2:1 分布。

### `main()` — CLI 入口

argparse 驱动。

---

## 四、CLI 用法

```bash
# 默认运行
python main.py

# 指定配置
python main.py --config path/to/config.json

# 预览模式（不实际生成）
python main.py --dry-run

# 只生成单器官
python main.py --organ liver_lesion

# 指定随机种子
python main.py --seed 123
```

---

## 五、自检说明

| 测试 | 内容 | 验证 |
|------|------|------|
| [1] sample_size_category | 20次采样分布 | 无 tiny, 接近 4:2:1 |
| [2] sample_radius | 三类各一次 | 均在 [r_min, r_max] 内 |
| [3] generate_one | 单样本完整流程 | metadata 完整 |
| [4] generate_batch | 3个样本批量生成 | 3/3 成功 |
| [5] compute_statistics | 汇总 | total/size_dist 正确 |

**运行方式**: `python Step6/src/main.py`
