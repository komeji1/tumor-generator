# MAISI + DiffTumor 对接技术文档

> 详细记录两个模型的对接接口、数据流转、参数映射、轴约定差异等技术细节

---

## 1. 两个模型概述

### MAISI (NV-Generate-CTMR)

- **功能**: 基于条件扩散模型生成全身 CT + 132-class 分割 mask
- **架构**: VQGAN 自编码器 + Rectified Flow 扩散 UNet + ControlNet
- **输出**: NIfTI 格式 CT (HU 值) + 132-class label mask
- **维度约定**: `(D, H, W)` — nibabel 默认轴顺序
- **体素间距**: 可配置，默认 `(1.7, 1.7, 2.0)` mm
- **HU 范围**: `-1000 ~ 1000`

### DiffTumor (CVPR 2024)

- **功能**: 基于条件扩散模型在健康 CT 区域内生成真实肿瘤纹理
- **架构**: VQGAN 3D 自编码器 + DDPM/DDIM 扩散 UNet
- **输入**: 96³ patch, 1mm³ 各向同性, HU 归一化到 [0,1]
- **输出**: 96³ patch, [-1,1] 范围合成纹理
- **维度约定**: `(B, C, W, D, H)` — DiffTumor 内部轴顺序（与 MAISI 不同）
- **HU 茁剪**: `-175 ~ 250`

---

## 2. 对接架构

### 2.1 核心文件

| 文件 | 角色 |
|------|------|
| `scripts/tumor_adapter.py` | 核心适配器 — 配置翻译 + 纹理注入 + 融合 |
| `scripts/tumor_prompt_runner.py` | 入口 — JSON/CLI → 调用适配器 → 完整管线 |
| `configs/config_tumor_pipeline.json` | 管线默认参数 |
| `configs/tumor_paths.json` | DiffTumor 外部路径（换机器只需改此文件） |
| `configs/label_dict_ctmr.json` | MAISI 132-class 标签字典 |

### 2.2 数据流

```
用户 JSON 配置
    │
    ▼
TumorConfigAdapter.task_to_maisi_params()
    │  翻译: organ → anatomy_list + body_region
    │  翻译: size_category → controllable_anatomy_size
    │  翻译: phase → early/noearly
    │
    ▼
tumor_prompt_runner.generate_maisi_base_ct()
    │  调用 MAISI LDMSampler
    │  输出: CT.nii.gz + label_full.nii.gz
    │  设置: controllable_anatomy_size=[] (空→用候选mask数据库)
    │  设置: save_full_label=True (保存完整132-class mask)
    │
    ▼
TumorTextureInjector.inject_tumor()
    │  ① 从 CT+mask 中裁剪肿瘤中心 ±48mm 区域
    │  ② 重采样到 1mm³, HU茁剪归一化, 居中裁到 96³
    │  ③ build_condition() → 9通道条件向量
    │  ④ generate_texture() → 扩散采样 → 合成纹理
    │  ⑤ blend_texture() → Gaussian alpha融合
    │  ⑥ 重采样回原生空间 + 写回完整CT
    │
    ▼
输出: {organ}_{size}_{phase}_{timestamp}.nii.gz
       + _tumor_mask.nii.gz
```

---

## 3. 关键对接细节

### 3.1 轴约定差异（最重要的坑）

MAISI 和 DiffTumor 使用**不同的轴顺序**：

| 模型 | Tensor 形状 | 轴含义 |
|------|-------------|--------|
| MAISI / nibabel | `(B, C, D, H, W)` | D=前后, H=上下, W=左右 |
| DiffTumor | `(B, C, W, D, H)` | W=左右, D=前后, H=上下 |

**适配方式**: `permute(0, 1, -1, -3, -2)` — 即 `(B,C,D,H,W) → (B,C,W,D,H)`

```python
# tumor_adapter.py: build_condition() 中
masked_volume_p = masked_volume.permute(0, 1, -1, -3, -2)  # → (B,C,W,D,H)
mask_p = mask.permute(0, 1, -1, -3, -2)

# generate_texture() 输出还原
sample_latent = sample_latent.permute(0, 1, -2, -1, -3)  # → (B,C,D,H,W)
```

### 3.2 HU 值范围差异

| 模型 | 范围 | 归一化方式 |
|------|------|-----------|
| MAISI 输出 | `-1000 ~ 1000` | 无归一化，直接输出 HU |
| DiffTumor 输入 | `[0, 1]` | `clip(-175,250) → (HU-HU_MIN)/(HU_MAX-HU_MIN)` |
| DiffTumor 条件编码 | `[-1, 1]` | `volume = ct * 2.0 - 1.0` |
| DiffTumor VQGAN 潜在空间 | 自定义 | `(feat-emb_min)/(emb_max-emb_min)*2-1` |

**关键**: VQGAN 编码后的潜在向量需要用 `emb_min/emb_max` 归一化到 [-1,1]，这两个值从 VQGAN codebook 的 embeddings 中提取：

```python
self.emb_min = self.vqgan.codebook.embeddings.min().detach()
self.emb_max = self.vqgan.codebook.embeddings.max().detach()
```

### 3.3 条件编码构建（DiffTumor 核心接口）

DiffTumor 的条件编码是 9 通道向量，构建步骤如下：

```
① volume = ct_tensor * 2.0 - 1.0        → [-1, 1]
② mask = tumor_mask * 2.0 - 1.0          → {-1, 1}
③ mask_ = 1 - tumor_mask                  → {0, 1}  (健康区域mask)
④ masked_volume = volume * mask_          → 肿瘤区域置0, 健康区域保留
⑤ permute: (B,1,D,H,W) → (B,1,W,D,H)   → 轴重排到DiffTumor约定
⑥ VQGAN encode(masked_volume)             → (B,8,D/4,H/4,W/4)  8通道潜在特征
⑦ 归一化: (feat-emb_min)/(emb_max-emb_min)*2-1  → [-1,1]
⑧ mask 下采样到潜在空间尺寸               → (B,1,D/4,H/4,W/4)
⑨ cond = cat([masked_feat, mask_down], dim=1) → (B,9,D/4,H/4,W/4)
```

**为什么是 9 通道**: 8 通道 VQGAN 编码的健康 CT 特征 + 1 通道下采样肿瘤 mask。mask 隐式编码位置+形状+尺寸, CT 隐式编码器官类型+HU纹理。不需要显式的 organ embedding 或 Fourier position。

### 3.4 扩散采样策略

DiffTumor 提供两种采样策略，对应不同肿瘤大小：

| phase | 算法 | 时间步 | 适用场景 | 权重文件 |
|-------|------|--------|---------|---------|
| early | DDPM | T=4 | 小肿瘤 (tiny/small) | `{organ}_early.pt` |
| noearly | DDIM | S=50 (基于T=200) | 大肿瘤 (medium/large) | `{organ}_noearly.pt` |

**early 模式**:
```python
# 加载权重 → Tester.ema_model
sample_latent = tester.ema_model.sample(cond=cond, batch_size=1)
```

**noearly 模式**:
```python
# 加载权重 → diffusion
ddim_sampler = DDIMSampler(diffusion, schedule="cosine")
samples_ddim, _ = ddim_sampler.sample(
    S=50, conditioning=cond, batch_size=1,
    shape=cond[:, :8].shape[1:], eta=eta  # eta=0确定性, eta=1随机
)
# 反归一化 + VQGAN解码
samples_ddim = ((samples_ddim + 1.0) / 2.0) * (emb_max - emb_min) + emb_min
sample_latent = vqgan.decode(samples_ddim, quantize=True)
```

### 3.5 权重路由

部分器官没有专属 DiffTumor 权重，采用零样本策略：

| organ | 专属权重 | 实际使用 |
|-------|---------|---------|
| liver | ✅ liver_early/noearly.pt | 专属 |
| pancreas | ✅ pancreas_early/noearly.pt | 专属 |
| kidney | ✅ kidney_early/noearly.pt | 专属 |
| colon | ❌ | liver_early.pt (复用) |
| esophagus | ❌ | liver_early.pt (复用) |
| uterus | ❌ | liver_early.pt (复用) |
| lung | ❌ | liver_early.pt (复用) |
| bone | ❌ | liver_early.pt (复用) |

**零样本器官**（esophagus/uterus）的处理：tumor_label=0 → MAISI mask 中无肿瘤区域 → 在器官内部用椭球合成肿瘤 mask → 再用 DiffTumor 注入纹理。

### 3.6 纹理融合策略

DiffTumor 的融合方式取决于器官类型：

| organ_type | 融合方式 | sigma |
|-----------|---------|-------|
| liver, kidney, bone, lung | Gaussian alpha blending | σ ~ U(0, 4) |
| pancreas, esophagus | 直接替换 | — |

**Gaussian alpha blending 公式**:
```python
mask_blurred = gaussian_filter(tumor_mask, sigma=σ)
final = (1 - mask_blurred) * orig_CT + mask_blurred * synthetic_texture
```

σ 越大 → 边界越平滑过渡；σ=0 → 硬边界替换。

---

## 4. MAISI 132-class 标签体系

### 4.1 与肿瘤相关的标签

| label_id | 名称 | 对应器官 |
|----------|------|---------|
| 1 | liver | liver 器官 |
| 4 | pancreas | pancreas 器官 |
| 14 | left kidney | kidney 器官 |
| 20 | lung | lung 器官 |
| 21 | bone | bone 器官 |
| 11 | esophagus | esophagus 器官 |
| 161 | uterocervix | uterus 器官 |
| 62 | colon | colon 器官 |
| 23 | lung tumor | lung 肿瘤 |
| 24 | pancreatic tumor | pancreas 肿瘤 |
| 26 | hepatic tumor | liver 肿瘤 |
| 27 | colon cancer primaries | colon 肿瘤 |
| 116 | left kidney cyst | kidney "肿瘤" |
| 128 | bone lesion | bone 肿瘤 |

### 4.2 分割验证中的标签映射

训练分割模型时，132-class → 3-class 映射：

```
132-class full_label:
  label == organ_label (如 1=liver) → class 1 (器官)
  tumor_mask > 0 → class 2 (肿瘤，覆盖器官)
  其他 → class 0 (背景)
```

**注意**: tumor_mask 是单独的二值文件（0/1），不是 132-class 中的肿瘤标签。这是因为 DiffTumor 生成的肿瘤区域可能与 MAISI 原始 mask 中的肿瘤标签不完全对应。

---

## 5. DiffTumor 模块加载（绕过 __init__.py）

DiffTumor 的 `TumorGeneration/__init__.py` 会导入 `elasticdeform`（我们不需要），直接 import 会报错。解决方案：

```python
import types

# 手动注册 TumorGeneration 为包，阻止 __init__.py 自动执行
tg_mod = types.ModuleType("TumorGeneration")
tg_mod.__path__ = [os.path.join(diffumor_repo_dir, "TumorGeneration")]
tg_mod.__package__ = "TumorGeneration"
sys.modules["TumorGeneration"] = tg_mod

# 注册 ldm 子包
ldm_mod = types.ModuleType("TumorGeneration.ldm")
ldm_mod.__path__ = [os.path.join(diffumor_repo_dir, "TumorGeneration", "ldm")]
ldm_mod.__package__ = "TumorGeneration.ldm"
sys.modules["TumorGeneration.ldm"] = ldm_mod

# 然后正常导入
from TumorGeneration.ldm.ddpm import Unet3D, GaussianDiffusion, Tester
from TumorGeneration.ldm.ddpm.ddim import DDIMSampler
from TumorGeneration.ldm.vq_gan_3d.model.vqgan import VQGAN
```

---

## 6. 配置文件说明

### 6.1 tumor_paths.json（换机器只需改此文件）

```json
{
  "diffumor_repo_dir": "DiffTumor仓库的STEP3.SegmentationModel子目录路径",
  "vqgan_ckpt_path": "VQGAN权重路径 (AutoencoderModel.ckpt)",
  "diffusion_ckpt_dir": "扩散模型权重目录 (包含各器官_early/noearly.pt)",
  "mask_library_dir": "可选: 肿瘤mask库目录"
}
```

**DiffTumor 权重文件**:
- `AutoencoderModel.ckpt` (~243MB) — VQGAN 自编码器，所有器官共用
- `liver_early.pt` — liver DDPM 权重 (T=4)
- `liver_noearly.pt` — liver DDIM 权重 (T=200)
- `pancreas_early/noearly.pt` — pancreas 权重
- `kidney_early/noearly.pt` — kidney 权重

### 6.2 任务配置 JSON（用户编写）

```json
{
  "tasks": [
    {
      "organ": "liver",
      "size_category": "medium",
      "phase": "noearly",      // null=自动
      "output": "both",         // full_ct / patch_96 / both
      "repeat": 5,              // 重复次数
      "eta": 0.0                // DDIM随机性 (仅noearly)
    }
  ],
  "maisi": {
    "generate_version": "rflow-ct",
    "output_size": [256, 256, 128],
    "spacing": [1.7, 1.7, 2.0],
    "num_output_samples": 1,
    "random_seed": null         // null=随机, 数字=固定
  },
  "global": {
    "device": "cuda"
  }
}
```

### 6.3 MAISI 参数覆盖（关键决策）

`generate_maisi_base_ct()` 中有几个重要的参数覆盖：

```python
# 不用 controllable_anatomy_size 控制器官大小
# 原因: 该参数触发稀疏mask生成，无法为DiffTumor提供足够器官边界
args.controllable_anatomy_size = []  # 空 = 使用候选mask数据库

# 保存完整132-class mask (不过滤)
ldm_sampler.save_full_label = True

# body_region + anatomy_list 从 ORGAN_TO_MAISI 映射获取
args.body_region = organ_info["body_region"]
args.anatomy_list = organ_info["anatomy_list"]
```

**为什么不用 controllable_anatomy_size**: MAISI 的 `controllable_anatomy_size` 触发 mask 生成模式，产出的 mask 只包含目标器官，过于稀疏（周围没有其他器官边界），DiffTumor 的 VQGAN 编码器需要完整的解剖结构来理解器官类型和位置。

---

## 7. 裁剪 + 重采样流程

肿瘤纹理注入需要将完整 CT 裁剪到 96³ patch，再重采样到 DiffTumor 的输入格式：

```
完整 CT (D,H,W, spacing=1.7/1.7/2.0)
  │
  │ ① 找肿瘤中心 → 裁剪 ±48mm 区域 (物理尺寸)
  │   half_vox = ceil(48mm / spacing[i])
  │
  ▼
局部 CT crop (可能 ≠ 96³)
  │
  │ ② 如果物理尺寸 < 96mm → padding到96mm
  │
  ▼
局部 CT crop_padded (≥96mm物理)
  │
  │ ③ 保存临时NIfTI → SimpleITK重采样到1mm³
  │
  ▼
1mm³ crop (≈96³)
  │
  │ ④ 居中裁剪到精确 96³
  │
  ▼
96³ patch (1mm³, [0,1]) → DiffTumor 输入
  │
  │ ⑤ DiffTumor生成 + 融合
  │
  ▼
96³ blended patch (1mm³, HU值)
  │
  │ ⑥ SimpleITK重采样回原生spacing
  │
  ▼
原生空间 blended → 写回完整CT的裁剪位置
```

---

## 8. GPU 显存管理

两个模型都很大，连续任务需要显存管理：

```python
# tumor_prompt_runner.py 中每个任务完成后
del autoencoder, diffusion_unet, controlnet, ldm_sampler
del mask_generation_autoencoder, mask_generation_diffusion_unet
gc.collect()
torch.cuda.empty_cache()

# DiffTumor injector 同理
del injector
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()  # 确保GPU操作完成
```

**TumorTextureInjector 的缓存优化**: 扩散引擎按 `{organ}_{phase}` 缓存，同一配置的连续任务不重复加载权重。

---

## 9. 已知限制与注意事项

1. **colon/lung/bone/esophagus/uterus 无专属权重** — 用 liver_early.pt 零样本，纹理可能不够真实
2. **kidney tumor_label=116 (左肾囊肿)** — MAISI 无专用肾癌标签，语义不完全匹配
3. **DDIM eta 仅对 noearly 有效** — early 模式 (DDPM T=4) 不支持 eta 参数
4. **肿瘤太小导致 mask 体素 <10** → 自动跳过该任务
5. **MAISI mask 可能缺少肿瘤标签** → 自动在器官内创建合成椭球肿瘤区域
6. **93³ crop 的 pos:neg=5:1** — 对小肿瘤采样仍有困难，肿瘤体积占比太小
