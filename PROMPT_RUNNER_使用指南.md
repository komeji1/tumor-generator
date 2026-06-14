# Prompt Runner — 使用指南 (JSON 配置)

> **文件**: `Tumor/src/prompt_runner.py`  
> **依据**: `AI肿瘤生成提示词设计分析.md` + DiffTumor 实际条件编码  
> **更新日期**: 2026-06-14

---

## 一、快速开始

只需两步：

**1. 编辑配置文件** `prompts.json`：

```json
{
  "tasks": [
    {
      "organ": "liver",
      "size_category": "large",
      "host_ct": "BDMAP_00000012",
      "mask_index": 0,
      "phase": "noearly",
      "output": "full_ct"
    }
  ],
  "global": {
    "device": "cpu"
  }
}
```

**2. 运行**：

```bash
# cmd
run prompts.json

# PowerShell
.\run.bat prompts.json
```

输出在 `Tumor/output/full_ct/{organ}/` 下，一个完整 CT (含嵌入的合成肿瘤) + 一个对应的肿瘤 mask。

---

## 二、配置文件完整参考

### 2.1 顶层结构

```json
{
  "tasks": [ ... ],     // 必填: 任务列表，1-N 个
  "global": { ... }     // 可选: 全局默认值
}
```

### 2.2 任务字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:--:|------|------|
| `organ` | string | ✅ | — | 器官类型 |
| `size_category` | string | — | `"small"` | 肿瘤尺寸档位 |
| `phase` | string / null | — | `null` (自动) | 扩散采样策略 |
| `host_ct` | string / null | — | `null` (随机) | 宿主 CT 的 BDMAP ID |
| `mask_index` | int | — | `0` | 同器官/CT 下的第 N 个 mask |
| `mask_file` | string | — | `null` | 直接指定 mask 文件名 (最高优先级) |
| `repeat` | int | — | `1` | 同一任务重复次数（每次用不同 mask_index） |
| `output` | string | — | `"both"` | 输出格式 |
| `eta` | float | — | `0.0` | DDIM 随机性: 0=确定性(论文默认), 1=最大随机。仅 noearly 生效 |
| `output_name` | string | — | `null` (自动) | 自定义输出文件名 |

### 2.3 字段可选值

**organ** — 器官类型：

| 值 | 器官 | 早期权重 | 中晚期权重 |
|------|------|------|------|
| `liver` | 肝脏 | liver_early.pt | liver_noearly.pt |
| `pancreas` | 胰腺 | pancreas_early.pt | pancreas_noearly.pt |
| `kidney` | 肾脏 | kidney_early.pt | kidney_noearly.pt |
| `colon` | 结肠 | colon_early.pt* | colon_early.pt* |
| `esophagus` | 食管 | liver_early.pt‡ | liver_early.pt‡ |
| `uterus` | 子宫 | liver_early.pt‡ | liver_early.pt‡ |

> \* 训练中 | ‡ zero-shot 跨器官推理

**size_category** — 肿瘤尺寸档位：

| 值 | 半径范围 | 自动 phase |
|------|------|:--:|
| `tiny` | 1-5mm | early |
| `small` | 5-10mm | early |
| `medium` | 10-20mm | noearly |
| `large` | 20-50mm | noearly |

**phase** — 手动覆盖权重 (设为 `null` 则自动按上表选择)：

| 值 | 采样策略 | 适用于 |
|------|------|------|
| `early` | DDPM T=4 步 | r ≤ 10mm 的早期小肿瘤 |
| `noearly` | DDIM S=50 步 | r > 10mm 的中晚期肿瘤 |
| `null` | 自动 | 推荐，让系统按 size 自动选 |

**output** — 输出格式：

| 值 | 产出 |
|------|------|
| `full_ct` | 完整 CT + 嵌入的合成肿瘤 (肿瘤区改性，其余不变) |
| `patch_96` | 96³ 各向同性 patch (仅肿瘤区域) |
| `both` | 以上两种都输出 |

### 2.4 全局配置

```json
"global": {
  "device": "cpu"       // "cpu" 或 "cuda"
}
```

### 2.5 配置示例

```json
{
  "_description": "参考: 5个器官各一个任务",
  "_guide": {
    "organ": "liver | pancreas | kidney | colon | esophagus | uterus",
    "size_category": "tiny | small | medium | large",
    "phase": "early | noearly | null (auto)"
  },

  "tasks": [
    {
      "_comment": "肝脏大肿瘤 — 手动指定 noearly",
      "organ": "liver",
      "size_category": "large",
      "host_ct": "BDMAP_00000012",
      "phase": "noearly",
      "output": "full_ct"
    },
    {
      "_comment": "胰腺小肿瘤 — phase=null 自动选 early",
      "organ": "pancreas",
      "size_category": "small",
      "host_ct": "BDMAP_00000019",
      "phase": null,
      "output": "full_ct"
    },
    {
      "_comment": "食管微小肿瘤 — zero-shot, 不指定宿主CT",
      "organ": "esophagus",
      "size_category": "tiny",
      "host_ct": null,
      "phase": "early",
      "output": "full_ct"
    },
    {
      "_comment": "肾脏 — 指定第3个mask, 自定义输出名",
      "organ": "kidney",
      "size_category": "medium",
      "host_ct": "BDMAP_00000019",
      "mask_index": 2,
      "output_name": "kidney_test_case",
      "output": "full_ct"
    },
    {
      "_comment": "子宫 — 全自动, 随机选CT和mask",
      "organ": "uterus",
      "size_category": "small",
      "output": "full_ct"
    }
  ],

  "global": {
    "device": "cpu"
  }
}
```

---

## 三、工作文件管理

| 文件 | 用途 | 改它？ |
|------|------|:--:|
| `example_prompts.json` | 📖 参考模板 (5个示例任务) | ❌ 不动 |
| `prompts.json` | ✏️ 你的工作配置 | ✅ 随便改 |

```
run prompts.json            # 执行你的配置
run example_prompts.json    # 跑一遍参考示例
```

`prompts.json` 可在 IDE 中直接编辑，修改完保存 → 终端 `run prompts.json` 即可。

---

## 四、输出

所有输出保存在 `Tumor/output/full_ct/{organ}/`：

| 文件 | 说明 |
|------|------|
| `{organ}_s{NN}__BDMAP_XXXXXXXX.nii.gz` | 完整 CT + 嵌入的合成肿瘤 |
| `{organ}_s{NN}__BDMAP_XXXXXXXX_mask.nii.gz` | 对应的肿瘤 mask |

- 文件名中 `t`→`s` 表示 tumor → synthetic
- Mask 与原 CT 精确对齐
- 非肿瘤区域与原始 CT 一致 (max diff = 0.00HU)
- 肿瘤边界通过软器官过渡 (Gaussian sigma=8/10)

---

## 五、生成新 Mask（可选）

`mask_config.json` 控制是否调用 Mask 工程生成新的肿瘤 mask。默认 `"generate": false`，不生成。

```bash
# 1. 编辑 mask_config.json，设 "generate": true
# 2. 运行
.\run mask_config.json
```

新 mask 保存到 `Mask/output/real_ct/{organ}/`，之后在 `prompts.json` 中通过 `mask_file` 字段引用。

## 六、辅助命令

```bash
# 查看可用资源 (mask数量、CT分布、权重状态)
run --list

# 查看配置模板 (在终端打印完整示例)
run --example

# CLI 快速模式 (不用配置文件，适合单次测试)
run --quick --organ liver --size medium --host BDMAP_00000012
```

---

## 七、原理简述

DiffTumor 的实际条件编码是 `cond = concat([z_healthy, mask_downsampled])`。Mask **隐式编码了位置+形状+尺寸+边界**，CT 上下文**隐式编码了器官类型+HU 纹理**。JSON 中的高层语义字段 (organ/size/host) 通过 Prompt Runner 映射到这些底层输入——不需要用户理解 VQGAN 或 DDIM：

```
JSON 字段                Prompt Runner 映射              DiffTumor 输入
──────────              ────────────────────          ──────────────────
organ: "liver"      →  权重路由 liver_early.pt    →  DiffusionEngine
size: "medium"      →  phase=noearly              →  T=200 DDIM S=50
host: "BDMAP_12"    →  CT路径解析                 →  ct.nii.gz
mask_index: 0       →  Mask文件选取               →  tumor_mask.nii.gz
```

---

## 八、可移植性

换机器只需修改 `paths.json` 中的 4 行外部路径，内部路径 (`checkpoints/`, `output/`) 自动相对项目根目录解析。

```
1. 复制整个 Tumor/ 目录
2. 修改 paths.json 的 4 个外部路径
3. pip install -r requirements.txt
4. run --list 验证
```
