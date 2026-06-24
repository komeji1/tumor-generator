# MAISI 工程分析总结 — 与 Mask / Tumor 工程关联

> **来源**: `work/MAISI` — NVIDIA MAISI (Medical AI for Synthetic Imaging)
> **关联工程**: `work/Mask` (肿瘤Mask生成), `work/Tumor` (DiffTumor肿瘤纹理生成)
> **日期**: 2026-06-23

---

## 一、MAISI 是什么

MAISI 是 NVIDIA 开发的**完整合成CT生成系统**。它从零生成腹部/胸部 CT 扫描 + 132 类分割标签，支持器官大小可控和可变体素间距。核心能力：

1. **从空气生成完整 CT** — 不需要宿主 CT 作为输入
2. **同步产出解剖标签** — 132 类分割 mask（器官+肿瘤），一步到位
3. **可以控制器官尺寸** — 10 维 `anatomy_size` 向量调整各器官相对大小

## 二、MAISI 架构

```
┌──────────────────────────────────────────────────────┐
│                   MAISI 两阶段管线                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│  阶段①: Mask 生成 (解剖蓝图)                           │
│  ─────────────────────────────                       │
│  路径A (数据库): 从 ~4000 个真实 mask 中查找 + 弹性增强  │
│  路径B (生成式): DDPM条件生成, 10维anatomy_size控制      │
│  输出: 132类整数标签 mask (H,W,D), 每个voxel=器官编号    │
│                                                      │
│  阶段②: 图像生成 (纹理填充)                             │
│  ─────────────────────────────                       │
│  1. binarize_labels() → mask拆成8个二值通道            │
│  2. ControlNet 以8通道为条件                           │
│  3. Rectified Flow 扩散模型 (30步推荐)                  │
│  4. AE解码 → CT在[-1000,1000] HU范围                   │
│  5. 背景(mask=0)设-1000 HU                            │
│                                                      │
│  阶段③ (可选): DiffTumor肿瘤纹理替换                     │
│  ─────────────────────────────                       │
│  tumor_adapter.py 调用DiffTumor, 和我们的Tumor项目完全同构│
│                                                      │
└──────────────────────────────────────────────────────┘
```

## 三、与我们的 Mask + Tumor 工程逐项对比

### 3.1 全局对标

| 维度 | Mask 项目 (我们) | Tumor 项目 (我们) | MAISI |
|------|:--:|:--:|:--:|
| **输入** | 30张正常CT + 器官mask | 肿瘤mask文件 | 空气（从无到有） |
| **Mask格式** | 二值 (0/1) ×6器官文件夹 | 同上 | 132类整数标签 (1个文件含全部解剖) |
| **Mask生成方式** | 椭球+弹性变形+裁剪 | 不生成 | 数据库查找(~4000个) 或 DDPM条件生成 |
| **CT生成方式** | 不生成（用宿主CT） | 不生成CT，只换肿瘤纹理 | ControlNet + Rectified Flow 从mask生成整张CT |
| **肿瘤纹理** | 不涉及 | DiffTumor DDPM/DDIM | DiffTumor（通过tumor_adapter.py桥接，**与我们的实现完全同构**） |
| **条件编码** | 无 | concat([VQGAN(masked_CT), mask↓]) | ① ControlNet: 8通道二进制mask → ② DiffTumor: 同上 |
| **尺寸控制** | tiny/small/medium/large | size_category筛选 + radius_mm精确过滤 | 10维 anatomy_size 向量 (0~1, -1=不控制) |
| **位置控制** | 器官内随机 | position字段 L2距离筛选 | 由mask决定 |
| **输出** | mask .nii.gz | full-CT嵌入 + 96³ patch | 完整CT + 132类mask + (可选)肿瘤增强CT |

### 3.2 关键同构：tumor_adapter.py 就是我们的 embed_to_full_ct.py

MAISI 的 `scripts/tumor_adapter.py` 中的 `TumorTextureInjector` 类，和我们 `src/embed_to_full_ct.py` 的管线**逐步骤一致**：

```
MAISI tumor_adapter              我们的 embed_to_full_ct
─────────────────────            ──────────────────────
CT + mask 加载                   CT + mask 加载
↓                                ↓
96mm区域裁剪 + padding            96mm区域裁剪 + padding
↓                                ↓
重采样到1mm³ isotropic            重采样到1mm³ isotropic
↓                                ↓
HU裁剪[-175,250]→[0,1]           HU裁剪[-175,250]→[0,1]
↓                                ↓
VQGAN编码: volume*2-1→permute→encode  VQGAN编码 (完全一致)
↓                                ↓
归一化到[-1,1]                   归一化到[-1,1]
↓                                ↓
cond=c([masked_feat, mask↓])     cond=c([masked_feat, mask↓])
↓                                ↓
DDPM(T=4) 或 DDIM(S=50)          DDPM(T=4) 或 DDIM(S=50)
↓                                ↓
Gaussian alpha blending          Gaussian alpha blending
↓                                ↓
Resample回原生spacing             Resample回原生spacing
↓                                ↓
嵌入全CT                         嵌入全CT
```

**两者是独立实现、功能同构的同一算法。** MAISI 多了 `TumorConfigAdapter` 类来翻译配置参数，我们多了 `resolve_mask` 和 `size_category` 过滤。

### 3.3 差异与互补

| 维度 | 我们的优势 | MAISI 的优势 |
|------|------|------|
| **CT来源** | 30张真实CT → 肿瘤嵌入在高保真解剖环境中 | 无限合成CT → 训练数据规模无上限 |
| **Mask精度** | 体素级二值mask，边界由elastic deformation自定义 | 132类多器官标签，解剖关系更完整 |
| **输入门槛** | 需要宿主CT + 器官分割 | 零输入 |
| **权重需求** | DiffTumor权重 (~1.7GB) | MAISI全量权重 (~40GB: AE, DM, ControlNet, Mask AE, Mask DM) |
| **GPU需求** | CPU可跑 | 最低16GB显存 |
| **肿瘤纹理质量** | 同等（使用完全相同的DiffTumor权重） | 同等 |
| **配置灵活性** | 更高——size精确过滤, position筛选, eta控制, mask_file指定 | 较粗——靠anatomy_size和body_region |
| **工程成熟度** | prompt_runner + mask_config 双JSON, 自动路由, 尺寸降级 | tumor_prompt_runner 单JSON, 集成在大型MAISI框架内 |

## 四、MAISI 关键技术细节

### 4.1 ControlNet 条件编码：8通道二进制mask

MAISI 的图像扩散模型使用 ControlNet，其条件输入来自 mask 的二进制分解：

```
132类整数mask (1, H, W, D)
     ↓  binarize_labels()
8通道二值张量 (8, H, W, D)
     ↓  每个bit独立作为一个通道
ControlNet 在每个去噪步处理
```

这和我们的 DiffTumor 条件编码**完全不同**：
- DiffTumor: `concat([VQGAN_encode(healthy_CT), downsampled_mask])` → 9通道，潜在空间
- MAISI ControlNet: `binarize_labels(132_class_mask)` → 8通道，像素空间

### 4.2 Mask 数据库 vs Mask 生成

MAISI 有两种 mask 获取方式：

| 方式 | 条件 | 优点 | 缺点 |
|------|------|------|------|
| **数据库查找** (find_masks.py) | `controllable_anatomy_size=[]` | 真实解剖结构，4000个预存 | 弹性变形有限，无法精确控制单一器官大小 |
| **DDPM条件生成** (sample_mask.py) | `controllable_anatomy_size` 非空 | 精细控制，10维anatomy_size | 质量依赖mask DDPM训练水平 |

MAISI 的 `tumor_prompt_runner.py` 默认使用数据库模式（`body_region` + `anatomy_list` 而非 `controllable_anatomy_size`），因为这能获取更真实的解剖结构。

### 4.3 anatomy_size 向量（10维）

```
[gallbladder, liver, stomach, pancreas, colon,
 lung_tumor, pancreatic_tumor, hepatic_tumor, colon_cancer, bone_lesion]
 
-1 = 不控制 (默认)
 0 = 最小尺寸
 1 = 最大尺寸
```

这个向量作为 mask DDPM 的条件，控制合成 mask 中各器官/肿瘤的比例。

### 4.4 132类标签系统

MAISI 使用单一整数标签文件编码全部解剖结构（不像我们的6个独立二值文件）。关键肿瘤标签：

| 标签ID | 结构 |
|:--:|------|
| 23 | 肺肿瘤 |
| 24 | 胰腺肿瘤 |
| 26 | 肝脏肿瘤 |
| 27 | 结肠癌 |
| 116 | 肾囊肿 |
| 128 | 骨病变 |

食管和子宫在MAISI中没有独立的肿瘤标签——tumor_adapter.py中对它们使用 `tumor_label=0`（zero-shot），和我们的策略一致。

## 五、三者关系总图

```
┌─ Mask 项目 (我们) ─────────────────────────────────────┐
│ 30张正常CT → 器官mask → 椭球+弹性变形 → 301个肿瘤mask     │
│ 输出: 二值 mask 文件 (每器官独立文件夹)                    │
└──────────────────────┬──────────────────────────────────┘
                       │ 肿瘤mask
                       ▼
┌─ Tumor 项目 (我们) ─────────────────────────────────────┐
│ mask → DiffTumor (VQGAN+UNet DDPM/DDIM) → 合成纹理      │
│ 输出: full-CT嵌入 + 96³ patch (1mm³ isotropic)           │
│ 配置: prompts.json + mask_config.json                    │
└──────────────────────────────────────────────────────────┘

┌─ MAISI (NVIDIA) ───────────────────────────────────────┐
│ 空气 → Mask生成(数据库/DDPM) → 132类mask                  │
│      → ControlNet+RectifiedFlow → 完整合成CT + mask       │
│      → (可选)DiffTumor → 肿瘤纹理增强                     │
│ 输出: 完整CT + 132类mask + 可选肿瘤增强CT                 │
│ 配置: JSON配置 + tumor_paths.json                        │
└──────────────────────────────────────────────────────────┘

三者交集:
  Mask项目 + MAISI = 都做mask生成，一个用几何方法(简单精确)，一个用扩散模型(复杂全面)
  Tumor项目 + MAISI tumor_adapter = 功能同构，都用DiffTumor在mask区域生成纹理
  Mask+MAISI可以互补: MAISI生成解剖底图 → Mask项目生成肿瘤mask → Tumor注入纹理
```

## 六、工程启发

### 6.1 可以借鉴MAISI的

1. **类封装**: `TumorTextureInjector` 比我们的函数式更好——器官/phase的扩散引擎缓存值得学习
2. **ConfigAdapter**: `TumorConfigAdapter` 统一了配置翻译逻辑，避免散落在各函数中
3. **output_size/spacing控制**: MAISI允许指定输出体积尺寸和体素间距，我们的全固定为96³
4. **10维anatomy_size**: 比4档size_category更精细，但需要重新训练mask DDPM

### 6.2 我们比MAISI更好的

1. **CPU可用**: MAISI最低16GB GPU显存，我们的Tumor项目可以在CPU上跑
2. **轻量级**: 我们的体重 ~2GB vs MAISI ~40GB
3. **精确mask控制**: size_category过滤、radius_mm ±25%、position L2筛选——MAISI没有这些
4. **可移植性**: config.py + paths.json vs MAISI的硬编码引用
5. **不依赖宿主CT**: 我们的Tumor需要宿主CT，但Mask项目已经在30张真实CT上运行——保证了解剖真实性

### 6.3 可能的整合方向

1. **MAISI作为上游**: MAISI生成CT+132类mask → 提取器官mask和肿瘤区域 → 我们的Mask项目生成更精确的肿瘤边界 → Tumor项目注入纹理
2. **Mask项目作为MAISI插件**: 将我们的mask生成算法替换MAISI的mask DDPM（数据库模式已类似）
3. **独立保持**: 两个系统目标不同——MAISI追求大规模数据生成，我们追求精细的肿瘤形态控制

## 七、最终方案确认（基于聊天记录）

聊天记录明确了最终技术路线：

```
参与方:
  MAISI    → 全部图像生成（器官CT + 肿瘤纹理），使用ControlNet
  Mask项目  → 提供精确肿瘤mask（惠祥峰生成的301个结果直接用）
  Tumor项目 → 不参与生成。仅保留其mask提示词设计理念（prompt格式）

流程:
  1. MAISI 生成器官mask (132类, 不含肿瘤label)
  2. Mask项目生成的肿瘤mask叠加到器官mask上 (把对应voxel改成肿瘤label)
  3. 合并的132类mask输入MAISI ControlNet
  4. MAISI从头生成完整CT (器官+肿瘤纹理均由MAISI产出)
```

这个方案意味着：Tumor 工程中 DiffTumor 相关代码（`diffusion_engine.py`、`condition_builder.py`、`texture_blender.py`）在最终管线中不执行——但它们积累的配置经验（`prompts.json` 字段设计、`size_category` 物理过滤、`radius_mm` 精确筛选）保留为 mask 生成的接口规范。

## 八、结论

MAISI 和我们 Mask + Tumor 工程是**互补而非替代**的关系。MAISI 解决"从零造CT"的问题，我们解决"在真实CT上精准放置肿瘤"的问题。肿瘤纹理注入（DiffTumor）部分两者功能完全同构，但我们的配置系统更灵活，MAISI的生成能力更全面。

建议：保持独立运作，肿瘤纹理注入的核心逻辑（embed_to_full_ct / TumorTextureInjector）参考MAISI的类封装改进，但不必迁移到MAISI框架中——代价大于收益。
