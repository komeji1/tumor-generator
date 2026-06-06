# Tumor Mask Generator — 项目概览

> **项目名称**: 肿瘤位置Mask自动生成器  
> **创建日期**: 2026-06-04  
> **最后更新**: 2026-06-04  
> **工作目录**: `Mask/`

---

## 目录

1. [工作目标](#一工作目标)
2. [工作流程](#二工作流程)
3. [工作模块](#三工作模块)
4. [方法分析](#四方法分析)
5. [目录结构规划](#五目录结构规划)
6. [里程碑与排期](#六里程碑与排期)

---

## 一、工作目标

### 1.1 一句话定义

> **在正常CT的器官区域内，按策略自动生成肿瘤位置的二值Mask (.nii.gz, 值域{0,1})，作为下游DiffTumor生成模型的"位置约束条件"输入。**

### 1.2 核心交付

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  交付物        │  规格                                          │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  肿瘤Mask文件  │  .nii.gz 格式，uint8，值域 {0, 1}              │
│  覆盖肿瘤类型  │  6类: liver_lesion, pancreatic_lesion,         │
│                │       kidney_lesion, colon_lesion,              │
│                │       esophagus_tumor, endometrioma_tumor       │
│  每类数量      │  50 张                                         │
│  总量          │  6 × 50 = 300 个mask                           │
│  维度          │  逐帧2D切片，与CT的z轴对齐                      │
│  空间信息      │  与原CT共享 affine matrix，体素一一对应         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **命名依据**: 6类肿瘤命名严格遵循 AbdomenAtlas2.0 官方class_map（class 27-32），参见 `d:\.vscode\nature-skills\AbdomenAtlas2.0\data\README.md`。

### 1.3 定位边界

```
我们做的 ←────────────边界────────────→ 不做的（下游/董开明负责）
───────────────────────────────────────────────────────────────────
                                                                  
  ✅ 正常CT上选定肿瘤位置             ✗ 生成肿瘤的CT纹理           
  ✅ 生成 0/1 二值位置Mask            ✗ 训练或运行DiffTumor模型     
  ✅ 控制肿瘤尺寸分布(4:2:1)          ✗ 生成完整的合成CT影像        
  ✅ 控制肿瘤在器官内的位置            ✗ 手动标注肿瘤               
  ✅ 输出.nii.gz格式mask              ✗ 完整的CT影像生成            
                                                                  
───────────────────────────────────────────────────────────────────
        本项目的职责范围
```

---

## 二、工作流程

### 2.1 总体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Tumor Mask Generator — 总流程                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ ① 加载   │───→│ ② 位置   │───→│ ③ Mask   │───→│ ④ 输出   │          │
│  │ 数据     │    │ 选择     │    │ 生成     │    │ 保存     │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       │               │               │               │                │
│       ▼               ▼               ▼               ▼                │
│  正常CT +        在器官mask     以中心为原点     保存为.nii.gz          │
│  器官分割        有效区域内     按尺寸生成       与CT对齐               │
│                  采样位置       椭球Mask +                              │
│                                 弹性形变                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 单张Mask生成流程（详细）

```
输入:
├── ct.nii.gz          ← 正常CT扫描 (提供空间参考)
├── organ_mask.nii.gz  ← 靶器官的二值分割mask
├── organ_type         ← "liver_lesion" | "pancreatic_lesion" | ...
├── size_category      ← "tiny" | "small" | "medium" | "large"
└── strategy           ← "uniform" | "distance_weighted" | ...

                      │
                      ▼

Step 1: 加载与预处理
├── 读取 CT → 获取 affine, shape, spacing
├── 读取 organ_mask → 二值数组 (D,H,W)
├── HU值裁剪到 [-175, 250] —— 依据: DiffTumor §E.2 (P21)
└── 验证: mask非空, 器官体积足够大

                      │
                      ▼

Step 2: 确定尺寸参数
├── 从 size_category 对应的半径范围中随机采样
│      tiny:   r ≤ 5mm          —— 依据: Scaling §3.2 (P5)
│      small:  5 < r ≤ 10mm    —— 依据: Scaling §3.2 (P5)
│      medium: 10 < r ≤ 20mm   —— 依据: Scaling §3.2 (P5)
│      large:  r > 20mm         —— 依据: Scaling §3.2 (P5)
├── 将物理半径转为体素单位: r_voxel = r_mm / spacing
└── 添加随机扰动(±20%) → 避免所有同类别肿瘤完全一样大

                      │
                      ▼

Step 3: 选择放置位置
├── 计算有效区域: valid = erode(organ_mask, margin)
│      margin = r_voxel + feather(3mm) + safety(5mm)
├── 按策略采样 center_zyx
│      uniform:       所有valid体素等概率
│      distance:      按到边界距离加权
│      subregion:     先选解剖亚区, 再区内均匀
├── 校验: 肿瘤球体 ⊆ 器官mask
└── 失败则重试(最多50次)

                      │
                      ▼

Step 4: 生成Mask (椭球体 + 弹性形变管线)
│
│  —— 依据: DiffTumor §3.3 (P5): "generate realistic tumor-like
│     shapes using ellipsoids" + §F.1 (P22): Hu et al.管线描述
│
├── 4a. 创建椭球体
│   ├── 生成基础椭球: dist = √((z/r_z)² + (y/r_y)² + (x/r_x)²) ≤ 1
│   └── 各轴比例 ±20% 随机扰动
│
├── 4b. 弹性形变 (Elastic Deformation) —— 依据: §F.1 (P22)
│   ├── 生成随机位移场 (使用低通滤波控制变形程度)
│   └── 对椭球体素坐标施加位移
│
├── 4c. 噪声添加与后处理 (可选，按Hu et al.管线)
│   ├── Salt-noise生成
│   ├── Gaussian滤波
│   ├── 缩放
│   └── Clipping
│
└── 置球内体素为 1

                      │
                      ▼

Step 5: 保存
├── NIfTI 格式: mask.nii.gz
├── uint8 编码, 值域 {0, 1}
├── 使用 CT 的 affine matrix → 保证空间对齐
└── 命名: {organ_type}_t{序号}__{CT编号}.nii.gz (如 liver_lesion_t00__BDMAP_00000012.nii.gz)

                      │
                      ▼

输出: mask.nii.gz (D,H,W), uint8, {0,1}
```

### 2.3 批量生成流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        批量生成循环                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  加载配置 JSON                                                           │
│      │                                                                  │
│      ▼                                                                  │
│  for each organ_type in [liver_lesion, pancreatic_lesion,               │
│     kidney_lesion, colon_lesion, esophagus_tumor, endometrioma_tumor]:  │
│      │                                                                  │
│      ▼                                                                  │
│  for sample_id in range(20):                                            │
│      │                                                                  │
│      ├── ① 加载该器官的正常CT + 器官mask                                │
│      │                                                                  │
│      ├── ② 按4:2:1分布采样 size_category                                │
│      │      small(4) / medium(2) / large(1)                             │
│      │      —— 依据: Scaling §4.1 (P7)                                  │
│      │                                                                  │
│      ├── ③ 执行 Step 1-5 生成mask                                       │
│      │                                                                  │
│      ├── ④ 保存 mask_{organ_type}_{sample_id:03d}.nii.gz               │
│      │                                                                  │
│      └── ⑤ 记录元数据 (位置/尺寸/类型) 到 manifest.csv                  │
│                                                                         │
│  总计: 6类 × 20张 = 300 个mask文件                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、工作模块

### 3.1 模块架构图

```
Mask/
│
├── config/                         # ← 配置模块
│   └── generation_config.json      #    肿瘤类型、尺寸分布、策略、输出路径
│
├── src/                            # ← 源码模块
│   ├── main.py                     #    主入口: 批量生成脚本
│   ├── data_loader.py              #    CT与器官mask加载模块
│   ├── position_selector.py        #    位置选择模块
│   ├── mask_generator.py           #    Mask生成模块 (椭球 + 弹性形变)
│   ├── validator.py                #    校验模块 (边界/重叠检查)
│   └── utils.py                    #    工具函数 (HU处理/坐标变换)
│
├── data/                           # ← 数据模块
│   ├── ct/                         #    正常CT扫描
│   ├── organ_labels/               #    器官分割mask
│   └── manifest.csv                #    样本索引表
│
├── output/                         # ← 输出模块
│   ├── liver_lesion/               #    肝脏肿瘤mask (50个)
│   ├── pancreatic_lesion/          #    胰腺肿瘤mask (50个)
│   ├── kidney_lesion/              #    肾脏肿瘤mask (50个)
│   ├── colon_lesion/               #    结肠肿瘤mask (50个)
│   ├── esophagus_tumor/            #    食管肿瘤mask (50个)
│   ├── endometrioma_tumor/         #    子宫肿瘤mask (50个)
│   └── generation_log.json         #    生成日志 (含所有生成参数)
│
└── PROJECT_OVERVIEW.md             # ← 本文件
```

### 3.2 各模块职责

```
┌─────────────────────────────────────────────────────────────────────────┐
│  模块                  │  职责                       │  关键函数           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  config/               │  定义所有生成参数            │  加载JSON配置      │
│  generation_config.json│  肿瘤类型/尺寸/策略/路径     │  参数校验          │
│                                                                         │
│  data_loader.py        │  加载CT和器官mask            │  load_ct()         │
│                        │  提取spacing/affine/shape    │  load_organ_mask() │
│                        │  器官mask → 二值提取         │  get_organ_bbox()  │
│                                                                         │
│  position_selector.py  │  在器官有效区域内选位置      │  compute_valid()   │
│                        │  多种策略实现                │  sample_uniform()  │
│                        │  位置可行性校验              │  sample_distance() │
│                        │                              │  validate_place()  │
│                                                                         │
│  mask_generator.py     │  以位置为中心生成mask        │  create_ellipsoid()│
│                        │  椭球体生成                  │  apply_elastic()   │
│                        │  弹性形变 (Hu et al.管线)    │  apply_noise_filter│
│                        │  噪声/滤波/缩放后处理       │  to_nifti()        │
│                                                                         │
│  validator.py          │  校验mask质量                │  check_in_organ()  │
│                        │  边界/重叠/尺寸合理性        │  check_size_dist() │
│                        │                              │  check_overlap()   │
│                                                                         │
│  utils.py              │  HU值裁剪/归一化              │  clip_hu()         │
│                        │  坐标变换/voxel↔mm           │  voxel_to_mm()     │
│                        │  形态学操作                   │  erode_mask()      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 模块依赖关系

```
main.py
  │
  ├──→ Step0/config/generation_config.json   (读配置)
  │
  ├──→ data_loader.py                  (加载数据)
  │      └──→ utils.py                 (HU处理)
  │
  ├──→ position_selector.py            (选位置)
  │      ├──→ utils.py                 (形态学腐蚀)
  │      └──→ validator.py             (校验位置)
  │
  ├──→ mask_generator.py               (生成mask: 椭球+弹性形变+后处理)
  │      └──→ validator.py             (校验mask)
  │
  └──→ output/                         (保存结果)
```

---

## 四、方法分析

> **设计原则**: 本节所有关键设计决策均标注原始论文出处。论文明确写的内容标注为"依据"，论文未明确写但由其明确约束推导的标注为"推导"，论文完全未涉及的标注"—"。

### 4.0 文献来源总览

本项目的方法设计依据以下两篇核心论文：

| 论文 | 标题 | 发表 | 与本项目关系 |
|------|------|------|-------------|
| **DiffTumor** (Chen et al.) | Towards Generalizable Tumor Synthesis | CVPR 2024 | 肿瘤Mask形状设计、位置选择约束、CT预处理标准 |
| **Scaling Tumor** (Chen et al.) | Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data | ICCV 2025 | 尺寸分类标准、尺寸分布比例、6类器官定义 |

> **重要区分**: DiffTumor仅覆盖3种器官（liver/pancreas/kidney），其核心贡献是肿瘤生成框架。Scaling Tumor扩展至6种器官（增加colon/esophagus/uterus），其核心贡献是数据规模法则研究。**我们的6类肿瘤列表来源于Scaling Tumor + AbdomenAtlas2.0，肿瘤形状生成方法来源于DiffTumor。**

### 4.1 Mask形状生成

#### 论文依据（两重证据）

**证据1 — DiffTumor 正文章节 (§3.3, P5 左侧)**

> "Following Hu et al. [37], we generate realistic tumor-like shapes using **ellipsoids** and refine them with expert radiologist feedback for clinical plausibility (implementation details in Appendix E)."

**证据2 — DiffTumor 附录 (§F.1, P22 左侧)** — Hu et al. [37]的完整管线描述

> "In recent works, Hu et al. [37] have synthesized tumors in the liver using a model-based approach. This approach, guided by radiologists, involves several image-processing operations such as **ellipse generation, elastic deformation, salt-noise generation, Gaussian filtering, scaling, and clipping**."

**证据3 — DiffTumor 附录 (§F.2, P22 右侧)** — 早期肿瘤形态的临床佐证

> "early-stage tumors originating from parenchymal organs typically exhibit a **round or oval shape**."

#### 形状生成管线（基于论文的综合实现）

```
┌─────────────────────────────────────────────────────────────────────────┐
│        Mask形状生成管线 (综合 DiffTumor §3.3 + §F.1 的完整描述)           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: 椭球生成 (Ellipse Generation)                                   │
│  ├── 依据: DiffTumor §3.3 (P5): "using ellipsoids"                     │
│  ├── 方法: 以采样中心为原点生成各向异性椭球                                │
│  │    dist = √((z/r_z)² + (y/r_y)² + (x/r_x)²) ≤ 1                     │
│  ├── 三轴随机 ±20% (体积守恒)                                            │
│  └── 早期肿瘤呈"round or oval shape" (§F.2, P22右侧) → 椭球合理         │
│                                                                         │
│  Step 2: 弹性形变 (Elastic Deformation)                                  │
│  ├── 依据: DiffTumor §F.1 (P22): "elastic deformation"                  │
│  ├── 方法: 生成低频随机位移场, 对椭球表面施加变形                         │
│  └── 目的: 使规则椭球呈现自然的不规则边界                                 │
│                                                                         │
│  Step 3: 噪声添加 (Salt-Noise Generation)                                │
│  ├── 依据: DiffTumor §F.1 (P22): "salt-noise generation"                │
│  └── 方法: 在mask内部随机翻转少量体素 (模拟内部纹理不规则)                │
│                                                                         │
│  Step 4: 高斯滤波 (Gaussian Filtering)                                   │
│  ├── 依据: DiffTumor §F.1 (P22): "Gaussian filtering"                   │
│  └── 方法: 对mask边界做高斯平滑, 实现边界平滑过渡                          │
│                                                                         │
│  Step 5: 缩放与裁剪 (Scaling & Clipping)                                 │
│  ├── 依据: DiffTumor §F.1 (P22): "scaling, and clipping"                │
│  └── 方法: 调整整体大小, 裁剪至{0,1}范围                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

> **注意**: `create_ellipsoid` 函数名是本项目基于论文描述的**派生实现**。论文原文（§3.3, P5）说的是 "using ellipsoids"，§E.2（P22）说 "please refer to the provided code"。论文**没有**以伪代码形式给出 `create_ellipsoid(center, size)` 这样的函数签名。Hu et al. [37]的管线描述出现在§F.1，是DiffTumor对前人工作的对比讨论，但DiffTumor自身确实使用了这一管线（§3.3明确写了"Following Hu et al. [37]"）。

#### 椭球 vs 球体 vs 复杂形状

```
椭球体 (ellipsoid) ← 论文选择
├── 各向异性: r_x ≠ r_y ≠ r_z
├── 优势: 可贴合器官的自然形态（肝左右>前后>上下，胰腺水平延伸）
├── 数学定义简单: x²/a² + y²/b² + z²/c² ≤ 1，无歧义
├── 参数可控: 3个轴半径即可覆盖所有形态变体
└── 论文依据:
    ├── DiffTumor §3.3 (P5): "using ellipsoids" 
    ├── DiffTumor §F.2 (P22右侧): "early-stage tumors typically exhibit 
    │    a round or oval shape"
    └── 扩散模型精修纹理，初始mask只需大体位置和范围

球体 (sphere) — 论文未选用
├── 各向同性: r_x = r_y = r_z
└── 问题: 腹部器官呈各向异性，完美球体在解剖上不自然

复杂扰动形状 (Perlin noise / statistical shape models) — 论文未选用
└── 因为扩散模型本身会为椭球"添加纹理和边界不规则性"
    初始mask只需提供大体位置和范围，纹理细节由扩散模型负责
```

### 4.2 尺寸分类与分布

#### 尺寸分类标准

**依据: Scaling Tumor (ICCV 2025) §3.2 (P5 右侧)**

> "We group them into four categories: **tiny (r ≤ 5 mm), small (5 < r ≤ 10 mm), medium (10 < r ≤ 20 mm), and large (r > 20 mm)**."

```
类别        半径(mm)        论文出处
──────────────────────────────────────────
tiny         r ≤ 5       Scaling §3.2 (P5 右侧)
small     5 < r ≤ 10     Scaling §3.2 (P5 右侧)
medium   10 < r ≤ 20     Scaling §3.2 (P5 右侧)
large      r > 20        Scaling §3.2 (P5 右侧)
```

#### 尺寸分布比例

**依据: Scaling Tumor (ICCV 2025) §4.1 (P7 左侧)**

> "we employ DiffTumor [12] to generate synthetic tumors, with a **ratio of 4:2:1 for small, medium, and large tumors**, respectively."

```
类别        比例        每类20张期望       说明
─────────────────────────────────────────────────
small        4          ~11 张            论文4:2:1中的"4"
medium       2          ~6 张             论文4:2:1中的"2"
large        1          ~3 张             论文4:2:1中的"1"
tiny         —          附加              论文§3.2单列以强调早期小肿瘤
```

> **注意**: 4:2:1比例直接对应small:medium:large三类。tiny在§3.2中单独列出以强调早期小肿瘤的重要性，tiny不属于4:2:1显式列出的三类之一。

#### 采样流程

```
① 按4:2:1比例选择 size_category (small/medium/large)，tiny作为附加类别
② 在类别对应的半径范围内 uniform 采样具体值
③ 各轴应用 ±20% 随机扰动 (保持体积守恒)
```

### 4.3 器官覆盖范围

#### 论文覆盖的器官

| 论文 | 覆盖器官 | 出处 |
|------|---------|------|
| **DiffTumor** (CVPR 2024) | **3类**: liver, pancreas, kidney | P1 摘要: "whether they originate in the liver, pancreas, or kidneys"; P5 §4 实验: LiTS/MSD-Pancreas/KiTS |
| **Scaling Tumor** (ICCV 2025) | **6类**: liver, pancreas, kidney, colon, esophagus, uterus | P1 摘要: "six organs (pancreas, liver, kidney, colon, esophagus, and uterus)" |

#### 我们的6类肿瘤命名

| # | 官方名称 (AbdomenAtlas2.0) | Class ID | 中文 |
|---|---------------------------|----------|------|
| 1 | `liver_lesion` | 27 | 肝脏肿瘤 |
| 2 | `pancreatic_lesion` | 28 | 胰腺肿瘤 |
| 3 | `kidney_lesion` | 29 | 肾脏肿瘤 |
| 4 | `colon_lesion` | 30 | 结肠肿瘤 |
| 5 | `endometrioma_tumor` | 31 | 子宫内膜瘤（子宫肿瘤） |
| 6 | `esophagus_tumor` | 32 | 食管肿瘤 |

> **命名来源**: AbdomenAtlas2.0 官方class_map（class 27-32），参见 `d:\.vscode\nature-skills\AbdomenAtlas2.0\data\README.md`。

### 4.4 位置选择策略

#### 论文描述

DiffTumor §3.3 (P5) 中关于位置选择的约束描述：

1. **Mask指示位置**: Diffusion Model conditioned on "a tumor mask that indicates the shape and location of tumors" (P5 左侧)
2. **器官内**: mask必须是器官内部的区域（从"healthy region" `zhealthy := (1−m)⊙z0` 的定义可知，m必须覆盖在器官上）
3. **无具体采样策略**: 论文**没有指定**位置选择的具体算法（如均匀随机、距离加权等）

#### 我们的策略选择

由于论文未规定具体采样策略，我们采用**最自然且最大化多样性**的方式：

| 优先级 | 策略 | 说明 | 选择理由 |
|--------|------|------|----------|
| ★ 默认 | **Uniform Random** | 器官有效区域内，每个体素等概率 | 最大位置多样性，不引入未被论文验证的偏置假设 |
| 备选 | Distance-Weighted | prob ∝ dist^α (α>0偏中心) | 仅在有其他文献支持或补充实验时启用 |
| 备选 | Anatomic Subregion | 先选解剖亚区，再区内均匀 | 仅在肿瘤类型有明确位置偏好时启用 |
| 备选 | Clinical Distribution | 匹配临床肿瘤位置分布 | 需要临床统计数据支持 |

> **原则**: 论文有什么我们就用什么，论文没说的不自行添加。避免引入未经实验验证的偏置。默认使用Uniform Random。

### 4.5 有效区域计算

```
为什么需要有效区域:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  organ_mask (原始)              valid_region (腐蚀后)
  ┌──────────────────┐          ┌──────────────────┐
  │██████████████████│          │░░░░░░░░░░░░░░░░░░│
  │██████████████████│          │░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░│
  │██████████████████│  ──→     │░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░│  ← 仅▓可采样
  │██████████████████│          │░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░│
  │██████████████████│          │░░░░░░░░░░░░░░░░░░│
  └──────────────────┘          └──────────────────┘

  margin = max_tumor_radius_voxels + feather(3mm) + safety(5mm)
  
  示例: radius=20mm, spacing=1mm, feather=3mm, safety=5mm
       → margin = 20 + 3 + 5 = 28 voxels
       
  保证: 即使肿瘤中心取在valid_region的最外层体素,
        肿瘤球体也不会超出器官边界
```

### 4.6 关键设计决策汇总（含精确论文出处）

| # | 决策点 | 我们的选择 | 论文依据 | 出处 | 依据类型 |
|---|--------|-----------|----------|------|----------|
| 1 | **Mask形状** | 椭球体 (ellipsoid) | "generate realistic tumor-like shapes using **ellipsoids**" | DiffTumor §3.3 (P5左侧) | 依据 |
| 2 | **弹性形变** | 必须添加 | "**elastic deformation**" in Hu et al. pipeline | DiffTumor §F.1 (P22左侧) | 依据 |
| 3 | **后处理管线** | 噪声→滤波→缩放→裁剪 | "**salt-noise generation, Gaussian filtering, scaling, and clipping**" | DiffTumor §F.1 (P22左侧) | 依据 |
| 4 | **早期肿瘤形状** | round or oval | "**round or oval shape**" | DiffTumor §F.2 (P22右侧) | 依据 |
| 5 | **尺寸分类** | tiny≤5/small(5-10]/medium(10-20]/large(>20)mm | "four categories: tiny(r≤5), small(5<r≤10), medium(10<r≤20), large(r>20)" | Scaling §3.2 (P5右侧) | 依据 |
| 6 | **尺寸比例** | 4:2:1 (small:medium:large) | "a ratio of 4:2:1 for small, medium, and large tumors" | Scaling §4.1 (P7左侧) | 依据 |
| 7 | **DiffTumor器官** | 3类: liver/pancreas/kidney | "whether they originate in the liver, pancreas, or kidneys" | DiffTumor P1摘要 | 依据 |
| 8 | **6类器官** | liver/pancreas/kidney/colon/esophagus/uterus | "six organs (pancreas, liver, kidney, colon, esophagus, and uterus)" | Scaling P1摘要 | 依据 |
| 9 | **器官命名** | 使用AbdomenAtlas2.0命名 | class_map 27-32 | AbdomenAtlas2.0 README | 依据 |
| 10 | **位置策略** | 器官mask内均匀随机 | 论文mask条件约束，但未指定具体采样算法 | DiffTumor §3.3 (P5) | 推导 |
| 11 | **HU值域** | clip [-175, 250] | "intensity in each scan is truncated to the range [−175, 250]" | DiffTumor §E.2 (P21) | 依据 |
| 12 | **有效区域** | 器官mask腐蚀 margin | 由mask必须在器官内的约束推导 | — | 推导 |
| 13 | **轴比例扰动** | ±20% 各轴独立随机 | 论文尺寸档内均匀采样精神的延伸 | — | 推导 |
| 14 | **性能饱和** | ~1500 real scans plateau | "performance plateaued after 1,500 scans" | Scaling P1摘要 | 依据 |
| 15 | **合成数据效率** | 500 real + synth = 1500 pure real | "reached the same performance using only 500 real scans" | Scaling P1摘要 | 依据 |

> **标注说明**:  
> - **"依据"**: 论文原文明确描述 → 直接引用原句 + 精确页码  
> - **"推导"**: 论文未明确写，但由其明确约束逻辑推导 → 标注"推导"  
> - **"—"**: 论文完全未涉及，属于工程实现决策 → 标注"—"  
>  
> **特别提醒**: `create_ellipsoid(center, size)` 这样的函数签名**并非**论文伪代码。论文说"using ellipsoids"（§3.3）+ "refer to the provided code"（§E.2）。函数名是本项目的派生实现。

---

## 五、目录结构规划

```
Mask/
│
├── PROJECT_OVERVIEW.md                # ← 本文件 (项目概览)
│
├── config/
│   └── generation_config.json         # 生成参数配置
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # 主入口
│   ├── data_loader.py                 # 数据加载
│   ├── position_selector.py           # 位置选择
│   ├── mask_generator.py              # Mask生成 (椭球+弹性形变+后处理)
│   ├── validator.py                   # 校验
│   └── utils.py                       # 工具函数
│
├── data/
│   ├── ct/                            # 正常CT (符号链接或复制)
│   │   ├── BDMAP_00000001/ct.nii.gz
│   │   └── ...
│   ├── organ_labels/                  # 器官分割
│   │   ├── BDMAP_00000001/
│   │   │   └── segmentations/
│   │   │       ├── liver.nii.gz
│   │   │       ├── pancreas.nii.gz
│   │   │       └── ...
│   │   └── ...
│   └── manifest.csv                   # CT-器官-肿瘤映射表
│
├── output/
│   ├── liver_lesion/                  # 肝脏肿瘤mask × 20
│   ├── pancreatic_lesion/             # 胰腺肿瘤mask × 20
│   ├── kidney_lesion/                 # 肾脏肿瘤mask × 20
│   ├── colon_lesion/                  # 结肠肿瘤mask × 20
│   ├── esophagus_tumor/               # 食管肿瘤mask × 20
│   ├── endometrioma_tumor/            # 子宫肿瘤mask × 20
│   ├── generation_log.json            # 生成记录
│   └── statistics.json                # 统计信息
│
├── tests/
│   ├── test_position_selector.py
│   ├── test_mask_generator.py
│   └── test_validator.py
│
└── README.md                          # 使用说明
```

---

## 六、里程碑与排期

```
Phase 1: 核心工具开发
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  □ 1.1 项目初始化 — 创建目录结构、配置文件
  □ 1.2 数据加载模块 — load_ct / load_organ_mask
  □ 1.3 位置选择模块 — uniform + 备选策略
  □ 1.4 Mask生成模块 — 椭球体 + 弹性形变 + 后处理 + .nii.gz输出
  □ 1.5 校验模块 — 边界检查、尺寸检查
  □ 1.6 主脚本 — main.py 串联全流程


Phase 2: 批量生成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  □ 2.1 批量循环 — 6类 × 20张
  □ 2.2 尺寸分布采样 — 4:2:1 控制 (Scaling §4.1)
  □ 2.3 生成日志 — 记录所有参数
  □ 2.4 统计报告 — 实际分布 vs 目标分布


Phase 3: 验证与交付
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  □ 3.1 可视化检查 — overlay 在CT上检查
  □ 3.2 质量统计 — 位置分布、尺寸分布、器官覆盖
  □ 3.3 格式确认 — 与董开明确认输出规格
  □ 3.4 交付 — 150个mask文件 + 生成日志
```

---

## 附录：与下游的接口约定

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    我们的输出 = 下游(DiffTumor)的输入                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  我们输出                                    下游(DiffTumor)接收         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  mask.nii.gz                              输入①: 肿瘤位置约束            │
│  ├── shape: (D, H, W)                     ├── 值域 {0, 1}               │
│  ├── dtype: uint8                         ├── 确定"在哪里生成"           │
│  ├── values: {0, 1}                       └── 与CT逐体素对齐             │
│  └── affine: 同CT                                                       │
│                                                                         │
│  manifest.csv / generation_log.json       输入②: 生成条件参数            │
│  ├── organ_type                           ├── 肿瘤类型 → Embedding      │
│  ├── size_mm                              ├── 肿瘤尺寸 → Fourier编码    │
│  ├── center_zyx                           ├── 位置信息 → Position编码   │
│  └── hu_stats                             └── CT统计 → 物理约束         │
│                                                                         │
│  normal_ct (引用路径)                     输入③: 正常CT (宿主图像)       │
│  └── 我们不改动, 仅引用                   └── 下游直接读取原始CT         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 附录B：论文依据速查表

| 页码 | 论文 | 位置 | 关键原文 |
|------|------|------|---------|
| P1 | DiffTumor | 摘要 | Early-stage tumors "present small, round, or oval shapes" |
| P1 | DiffTumor | 摘要 | Organs: "liver, pancreas, or kidneys" |
| P3 | DiffTumor | 左侧 | "they originate in the liver, pancreas, or kidneys" |
| P5 | DiffTumor | 左侧 (§3.3) | "generate realistic tumor-like shapes using **ellipsoids**" |
| P5 | DiffTumor | 左侧 (§3.3) | "Following Hu et al. [37]" + "implementation details in Appendix E" |
| P21 | DiffTumor | 左侧 (§E.2) | "intensity truncated to the range **[−175, 250]**" |
| P22 | DiffTumor | 左侧 (§F.1) | "**ellipse generation, elastic deformation, salt-noise generation, Gaussian filtering, scaling, and clipping**" |
| P22 | DiffTumor | 右侧 (§F.2) | "early-stage tumors... typically exhibit a **round or oval shape**" |
| P1 | Scaling | 摘要 | "six organs (pancreas, liver, kidney, colon, esophagus, and uterus)" |
| P1 | Scaling | 摘要 | "performance stopped improving after **1,500 scans**" |
| P1 | Scaling | 摘要 | "reached same performance using only **500 real scans**" |
| P5 | Scaling | 右侧 (§3.2) | "**tiny (r ≤ 5 mm), small (5 < r ≤ 10 mm), medium (10 < r ≤ 20 mm), and large (r > 20 mm)**" |
| P7 | Scaling | 左侧 (§4.1) | "ratio of **4:2:1 for small, medium, and large tumors**" |

---

> **下一步**: 开始 Phase 1.1 — 创建目录结构和配置文件 `generation_config.json`
