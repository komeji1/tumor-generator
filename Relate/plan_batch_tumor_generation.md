# 批量生成含肿瘤合成CT — 计划

## 目标
生成 **50例含肿瘤的合成CT**（当前仅8例，FID协方差奇异），要求**肿瘤类型（器官+尺寸）有明确区分**。

## 当前状况
- 53个 CT + label_full 对（132类标签，含完整器官结构）
- 已有8个step2结果（全来自同一个基础CT）
- 器官可用性：liver/pancreas/kidney/colon/esophagus/lung/stomach/adrenal/small_bowel/duodenum/spleen/heart → 100%，gallbladder → 53%，bladder/prostate/thyroid/brain → 0%
- 管线两步：① bridge_maisi_mask.py 画肿瘤标签 → ② MAISI ControlNet 推理生成CT

## 肿瘤类型分配方案

7个主器官 × 3个尺寸档 = 21种组合，每个组合2-3例，共50例：

| 器官 | small | medium | large | 总计 |
|------|-------|--------|-------|------|
| liver | 3 | 4 | 3 | 10 |
| pancreas | 3 | 3 | 2 | 8 |
| kidney | 3 | 3 | 2 | 8 |
| colon | 2 | 3 | 3 | 8 |
| lung | 2 | 3 | 3 | 8 |
| bone | 2 | 3 | 5 | 10 |
| esophagus | 2 | 2 | - | 4 |
| **总计** | **17** | **19** | **14** | **50** |

esophagus 无 large档（器官太小，大肿瘤会溢出严重）。

每个基础CT只画一种肿瘤 → 50个不同的基础CT → 50个独立合成CT → FID统计可靠。

## 实现方案

修改 `Relate/batch_generate_tumor_ct.py`：

1. **更新ASSIGNMENTS** — 从当前的6器官×1尺寸改为7器官×3尺寸的详细分配
2. **确保1 CT = 1 肿瘤** — 每个基础CT只画一种肿瘤，不重复使用
3. **step1_paint_masks** — 循环53个CT，按分配表依次画肿瘤，输出到 `output/tumor_ct_batch/`
4. **step2_generate_cts** — 调用 MAISI infer_image_from_mask.py CLI，逐个生成CT（每张约3-5分钟）
5. **新增 step2_generate_cts_batch** — 可选：用Python直接import infer函数，避免subprocess开销

## 文件变更

| 文件 | 变更 |
|------|------|
| `Relate/batch_generate_tumor_ct.py` | 更新ASSIGNMENTS表，改进step1分配逻辑，保留step2管线 |

## 时间预估
- step1（画肿瘤标签）：~1分钟
- step2（MAISI ControlNet推理）：每张约3分钟，50张 ≈ 2.5小时

## 风险点
1. **8GB VRAM** — MAISI推理需要 ~6GB，RTX 4060 Laptop 可用但需关闭其他GPU进程
2. **单张CT单肿瘤** — 部分器官在特定CT中可能不存在（虽然label_full有，但实际体积可能很小），bridge_single会返回skip
3. **step2耗时** — 2.5小时，建议先跑step1确认成功率，再批量跑step2
