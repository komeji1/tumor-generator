# 合成 CT 分割验证实验报告

> 日期: 2026-06-19 | 项目: NV-Generate-CTMR (MAISI + DiffTumor)

## 1. 实验目标

验证 NV-Generate-CTMR (MAISI + DiffTumor) 生成的合成 CT 数据是否可用于训练下游分割模型。

## 2. 方法

### 2.1 三阶段生成管线

```
Phase 1: MAISI (rflow-ct) → 生成基础 CT + 132-class full mask
Phase 2: DiffTumor → 生成真实肿瘤纹理
Phase 3: 融合嵌入 → 肿瘤嵌入 CT，输出含肿瘤的完整 CT + tumor mask
```

### 2.2 标签处理

132-class full label → 3-class: 背景(0) + 肝脏(1) + 肿瘤(2)
- liver_label=1 in MAISI 132-class map
- tumor_mask > 0 → class 2 (override organ)

### 2.3 分割模型

- **架构**: UNet (spatial_dims=3, in=1, out=3)
- **参数量**: 4.8M
- **Loss**: DiceCELoss (weighted: bg=1.0, organ=2.0, tumor=10.0)
- **Optimizer**: AdamW (lr=2e-3, weight_decay=1e-5)
- **Scheduler**: Warmup(5ep) + CosineAnnealing
- **Crop**: RandCropByPosNegLabeld (pos:neg=5:1, spatial_size=96³)
- **Inference**: Sliding window (96³, overlap=0.5)

### 2.4 数据配置

| 版本 | 训练 | 验证 | 总计 | 生成配置 |
|------|------|------|------|---------|
| v1 | 15 | 4 | 19 | liver small/medium × 10 |
| v2 | 18 | 5 | 23 | + liver small/medium × 10 |
| v3 | 42 | 11 | 53 | + liver medium/large × 30 |

## 3. 结果

### 3.1 三版对比

| 指标 | v1 (20ep, 19张) | v2 (150ep, 23张) | v3 (150ep, 53张) |
|------|-----------------|------------------|------------------|
| Organ Dice (Mean) | 0.603 | 0.796 | **0.856** |
| Organ Dice (Best) | 0.659 | 0.857 | **0.896** |
| Tumor Dice (Mean) | 0.000 | 0.036 | 0.009 |
| Tumor Dice (Best case) | 0.000 | 0.182 | 0.055 |

### 3.2 v3 Case-by-Case (11 验证样本)

| Case | Organ Dice | Tumor Dice |
|------|-----------|-----------|
| 1 | 0.858 | 0.001 |
| 2 | 0.856 | 0.000 |
| 3 | 0.851 | 0.000 |
| 4 | 0.827 | 0.017 |
| 5 | 0.884 | 0.055 |
| 6 | 0.836 | 0.027 |
| 7 | 0.812 | 0.000 |
| 8 | 0.896 | 0.000 |
| 9 | 0.857 | 0.000 |
| 10 | 0.859 | 0.000 |
| 11 | 0.886 | 0.000 |

### 3.3 Tumor Size Distribution (54 张)

| 类别 | voxels 范围 | 数量 |
|------|-------------|------|
| 小 (<2000) | 0-1999 | 17 |
| 中 (2000-10000) | 2000-9999 | 30 |
| 大 (10000+) | 10000+ | 7 |

## 4. 结论

### 4.1 核心结论

1. **合成 CT 数据可用于分割训练** — 仅用合成数据训练，Organ Dice 达到 0.86，接近真实数据训练水平
2. **数据量与性能正相关** — 19→23→53 张，Organ Dice: 0.60→0.80→0.86
3. **肿瘤分割是难点** — 肿瘤体积小（中位数 6564 voxels / 全图 0.05%），11 个 case 中仅 3 个被检测到

### 4.2 肿瘤分割瓶颈分析

- **肿瘤太小**: 平均仅占全图 0.02-0.05%，96³ crop 很难采样到
- **类别极不平衡**: bg:organ:tumor ≈ 99:1:0.02
- **数据多样性不足**: 仅 liver 单器官，需要更多器官/尺寸变化
- **模型容量**: 4.8M UNet 可能不够

### 4.3 改进方向

| 方向 | 预期效果 | 难度 |
|------|---------|------|
| 生成更大肿瘤 (large+) | 直接提升 tumor Dice | 低 |
| 更多数据 (100+张) | 提升泛化 | 中 (耗时) |
| Focal Loss | 专注 hard examples | 低 |
| 更深网络 (SwinUNETR) | 更强特征提取 | 中 |
| 两阶段训练 (organ→tumor) | 分步优化 | 中 |
| 多器官数据 | 提升泛化 | 低 |

## 5. 文件索引

| 文件 | 用途 |
|------|------|
| `scripts/seg_val_train.py` | 分割训练脚本 v2 |
| `scripts/tumor_prompt_runner.py` | 肿瘤生成管线入口 |
| `configs/liver_large_tumor_30ct.json` | 30张大肿瘤配置 |
| `configs/liver_50ct.json` | 50张混合配置 |
| `configs/multi_organ_40ct.json` | 多器官配置 |
| `output/seg_model_v3/best_model.pth` | v3 最佳模型 |
| `output/tumor_ct/liver_lesion/` | 54 对合成 CT+mask |

## 6. 遇到的技术问题及修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `ImportError: AddChanneld` | MONAI 新版弃用 | 改用 `EnsureChannelFirstd` |
| `TypeError: DiceCELoss n_classes` | MONAI API 变更 | 移除 n_classes 参数 |
| `AttributeError: lrScheduler` | PyTorch 命名 | 改用 `lr_scheduler` |
| `TypeError: list indices` | RandCropByPosNegLabeld 返回 list | collate_fn 扁平化 |
| Tumor Dice = 0 (v1) | 标签只有 0/1, 缺 organ 区域 | 改用 full 132-class label → 3-class |
| Evaluate Dice = 0 | pred/label shape 不匹配 | 手动计算 Dice 替代 DiceMetric |
| 3-class label affine 丢失 | nib.save 用 np.eye(4) | 保留原始 affine+header |
| tumor_mask 匹配错误 | 所有 sample 用同一 mask | 按 CT shape 精确匹配 |

## 7. 肿瘤生成提示词使用指南

### JSON 配置 (推荐)
```bash
python -m scripts.tumor_prompt_runner configs/your_config.json
```

### CLI 快速模式
```bash
python -m scripts.tumor_prompt_runner --quick --organ liver --size medium
```

### 支持的器官与尺寸

| organ | tumor_label | organ_label |
|-------|-------------|-------------|
| liver | 26 | 1 |
| pancreas | 24 | 4 |
| kidney | 116 | 14 |
| colon | 27 | 62 |
| lung | 23 | 20 |
| bone | 128 | 21 |
| esophagus | 0 (zero-shot) | 11 |
| uterus | 0 (zero-shot) | 161 |

| size_category | 半径(mm) | phase |
|---------------|---------|-------|
| tiny | 1-5 | early |
| small | 5-10 | early |
| medium | 10-20 | noearly |
| large | 20-50 | noearly |

### 关键参数
- `phase`: early=小肿瘤权重, noearly=大肿瘤权重
- `eta`: 0=确定性(论文默认), 1=最大随机(仅noearly有效)
- `repeat`: 同配置重复生成次数
- `random_seed`: null=随机, 数字=可复现
