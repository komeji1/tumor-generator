# FID 评估实现计划

## 目标

对 MAISI 合成 CT 做两个维度的 FID 评估：

| 评估 | 真实参考集 | 合成待评集 | 目的 |
|---|---|---|---|
| **对标论文** | autoPET23 50例 (data/autopet_ct50/) | MAISI 68张 step1 CT | 和 MAISI 论文 FID=5.124 对标 |
| **桥接前后** | MAISI 68张 step1 CT (无肿瘤) | MAISI 8张 step2 CT (含肿瘤) | 评估画肿瘤后质量变化 |

## 核心问题

原始 `scripts/compute_fid_2-5d_ct.py` 依赖 `torch.distributed` + NCCL，**Windows 不支持 NCCL**。需要写一个单 GPU 版本。

## 方案：新建 `Relate/compute_fid_single_gpu.py`

从原始脚本移植核心逻辑，去掉所有 `dist.*` 调用：

1. **特征提取**：复用 `get_features_2p5d()` + `radimagenet_intensity_normalisation()` 等函数
2. **预处理管线**：复用 MONAI transforms (LoadImage → ChannelFirst → RAS → Spacing → Pad → Crop → ScaleIntensity)
3. **FID 计算**：复用 `monai.metrics.FIDMetric`
4. **单 GPU**：去掉 `dist.init_process_group` / `dist.all_gather` / `dist.destroy_process_group`
5. **特征缓存**：保留 `.pt` 文件缓存机制，避免重复计算

### 预处理参数

| 参数 | MAISI 合成 CT | autoPET 真实 CT | LIDC 真实 CT |
|---|---|---|---|
| 原始 shape | 256×256×256 | 400×400×284-588 | 133-561×512×512 |
| 原始 spacing | 1.5×1.5×2.0 | 2.0×2.0×3.0 | 0.66-0.82×0.66-0.82×1.25-2.5 |
| HU 范围 | [-1000, 1000] | [-1499, 3770] | [-3024, 5892] |

论文配置：`resampling=1.0×1.0×1.0, padding+center_crop=512×512×512, ScaleIntensity[-1000,1000]`

但 512³ 在 RTX 4060 Laptop (8GB VRAM) 上可能 OOM。需要调整：
- `target_shape=256x256x128` — 匹配 MAISI 输出分辨率，VRAM 友好
- 或 `target_shape=256x256x256` — 稍大但可控
- `resampling_spacing` 保持 1.0×1.0×1.0 对齐论文

### 特征网络

优先 RadImageNet ResNet50（需联网下载 ~100MB），失败时 fallback 到 SqueezeNet 1.1（已缓存）。

## 脚本结构

```python
# Relate/compute_fid_single_gpu.py

def main(
    real_dataset_root,    # 真实CT目录
    real_filelist,        # 文件列表.txt
    synth_dataset_root,   # 合成CT目录  
    synth_filelist,       # 文件列表.txt
    model_name,           # "radimagenet_resnet50" | "squeezenet1_1"
    target_shape,         # "256x256x128"
    resampling_spacing,   # "1.0x1.0x1.0"
    center_slices_ratio,  # 0.4 (只用中心40%切片)
    output_root,          # 特征缓存目录
    ignore_existing,      # 是否重算已有特征
    num_images,           # 最大图像数
):
    # 1. 加载特征网络 (RadImageNet or SqueezeNet)
    # 2. 构建 MONAI transforms
    # 3. 提取真实CT特征 (缓存.pt)
    # 4. 提取合成CT特征 (缓存.pt)
    # 5. 计算 FID: XY, YZ, ZX 三个平面 + Avg
    # 6. 输出结果
```

## 附属脚本：`Relate/run_fid_eval.bat`

一键运行两个评估：
1. autoPET vs MAISI step1 (对标论文)
2. MAISI step1 vs step2 (桥接前后)

## 文件清单

| 文件 | 说明 |
|---|---|
| `Relate/compute_fid_single_gpu.py` | 单GPU FID计算脚本 (~300行) |
| `Relate/run_fid_eval.bat` | 一键运行脚本 |
| `data/autopet_ct50/filelist.txt` | autoPET 50例文件列表 (自动生成) |
| `data/fid_maisi_step1/filelist.txt` | MAISI step1 68例文件列表 (自动生成) |
| `data/fid_maisi_step2/filelist.txt` | MAISI step2 8例文件列表 (自动生成) |

## 预期结果

对标论文（MAISI 论文用 autoPET 200例，我们50例，数值会偏高）：
- MAISI rflow-ct 论文值: FID Avg ≈ 5.1
- 我们50例预估: FID Avg ≈ 7-10 (样本少会偏高)

桥接前后（首次计算，无参考值）：
- 预期 FID < 5 (同一模型生成，只多了肿瘤)
