# FID 桥接前后对比 — 批量生成计划

## 目标

增加含肿瘤 CT 数量（从 8 → ≥50），使 FID 桥接前后对比有统计意义。

## 现状

- 68 张 MAISI step1 无肿瘤 CT（基础图）
- 8 张 MAISI step2 含肿瘤 CT（已有）
- FID(step1 vs step2) ≈ 24（只有 8 张，统计意义弱）

## 管线

```
68张 step1 CT+label → bridge_maisi_mask.py → 50张含肿瘤label → infer_image_from_mask.py → 50张含肿瘤CT
```

## 生成策略

### 第1步：批量画肿瘤 mask

对 50 张 step1 CT 用 bridge_maisi_mask.py 画肿瘤，覆盖 6器官：
- 每张 CT 画 1 个 medium 尺寸肿瘤（半径 10-20mm）
- 6器官轮流：liver(10), pancreas(8), kidney(8), colon(8), lung(8), bone(10) = 50张

### 第2步：批量生成含肿瘤 CT

对 50 张含肿瘤 label，逐个调用 infer_image_from_mask.py：
- 配置：config_infer_8g_256x256x128.json（8GB GPU适用）
- 每张约需 2-5 分钟
- 50张 ≈ 2-4 小时

### 第3步：计算 FID

用 compute_fid_single_gpu.py 计算：
- Real = 50张 step1 无肿瘤 CT
- Synth = 50张 step2 含肿瘤 CT

## 脚本

### Relate/batch_generate_tumor_ct.py

自动执行全部流程：
1. 从 68 张 step1 中选 50 张
2. 为每张分配器官+尺寸
3. 调用 bridge_maisi_mask.py 画肿瘤 mask
4. 逐个调用 infer_image_from_mask.py 生成含肿瘤 CT
5. 收集结果到 output/tumor_ct_batch/

## 时间估算

- 画 tumor mask：50张 × ~1秒 ≈ 1分钟
- MAISI 步骤2：50张 × ~3分钟 ≈ 2.5小时
- FID 计算：~5分钟
- 总计：~3小时
