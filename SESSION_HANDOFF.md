# 会话交接文档 — Tumor Mask Generator

> **日期**: 2026-06-04 ~ 2026-06-06
> **目的**: 向新会话解释项目现状、遇到的问题、解决过程、所有注意事项

---

## 一、项目概述

在真实腹部CT数据（AbdomenAtlas2.0, 30个扫描）上，基于 **DiffTumor (CVPR 2024)** 和 **Scaling Tumor (ICCV 2025)** 论文，为6种器官自动生成肿瘤位置二值Mask (.nii.gz, {0,1})。

**最终成果**:
- **301个肿瘤mask** (6器官 × 50个，endometrioma多了1个)
- **18个预览视频** (每器官3种：mask/CT/叠加)
- **30个CT + 器官分割 + 肿瘤mask** 全部已上传GitHub

---

## 二、关键技术决策

### 2.1 数据来源
- **CT数据**: AbdomenAtlas2.0Mini (HuggingFace), BDMAP_00000001~00000030
- **器官分割**: TotalSegmentator --fast CPU模式 (5器官: liver, pancreas, kidney_left, colon, esophagus)
- **子宫**: TotalSegmentator不包含uterus → 方案B：合成子宫mask（骨盆区椭球体，body-relative positioning）

### 2.2 肿瘤生成算法（核心变更）

**原始设计**（Step1~7模块）:
1. 椭球体创建 → 弹性变形 → Salt噪声 → 高斯滤波 → 裁剪

**实际优化后**:
1. **距离变换替代球形腐蚀** — 位置选择中的 `erode_mask` 从 `binary_erosion` 改为 `distance_transform_edt`，O(N)恒定时间
2. **距离变换预检** — 在尝试位置选择前，先计算器官的distance map，过滤掉不可能容纳的肿瘤尺寸
3. **裁剪区域弹性变形** — 不再在全CT体积上生成位移场，而是裁剪到肿瘤周围区域
4. **器官边界裁剪** — 生成后 `mask = mask & organ_mask`，超出部分裁剪掉
5. **体积检测** — 裁剪后体积<3或裁剪损失>20%则拒绝重新生成
6. **tiny肿瘤禁弹性变形** — r<5mm时关闭弹性变形，避免小肿瘤全部溢出边界
7. **尺寸回退** — large→medium→small→tiny 依次尝试

### 2.3 配置关键参数
```json
{
  "organs[].count": 50,    // 每器官50个（非20）
  "margin": {
    "feather_mm": 0.5,     // 极小化以适应小器官
    "safety_mm": 1
  },
  "size_categories": {
    "weight": [0,4,2,1]    // tiny:0 small:4 medium:2 large:1
  },
  "naming_pattern": "{organ_type}_{sample_id}.nii.gz"
}
```

---

## 三、主要问题与解决过程

### 问题1: 弹性变形导致肿瘤超出器官边界
**发现**: 295/300肿瘤有体素超出器官mask（验证时发现）
**原因**: 弹性变形和高斯滤波扩展了边界，位置选择只保证中心在有效区域内
**解决**: 
- `main.py` 第335行: `mask_3d = ((mask_3d > 0) & (organ_mask.array > 0)).astype(np.uint8)` — 裁剪到器官边界
- 裁剪损失>20%则拒绝

### 问题2: 小器官（食管、胰腺）生成成功率极低
**发现**: esophagus只有2/50成功，pancreatic只有~4/50
**原因**: TotalSegmentator对食管的分割极薄（1-4K体素），margin侵蚀后无剩余
**解决**:
- tiny肿瘤回退（r=1-5mm），对tiny自动禁用弹性变形
- 距离变换预检，只尝试可行尺寸
- 每器官30个CT中约29个是viable的

### 问题3: 大量零体积mask被保存
**发现**: 56/68个pancreatic mask体积为0
**原因**: 裁剪后体积=0，但 `clip_loss = (0-0)/max(1,0) = 0 < 0.20` 不触发拒绝
**解决**: 添加 `if final_vol < 3: raise RuntimeError` 显式检测

### 问题4: shutil.move覆盖文件导致数据丢失
**发现**: 重新编号后文件从304减少到190
**原因**: validate_and_renumber.py的 `shutil.move` 在Windows上直接覆盖同名目标
**解决**: 两阶段重命名——先全部移到临时名(.tmpNNN)，再统一移到最终名(.tNN)

### 问题5: Git LFS导致IDE显示异常
**发现**: 设置 `*.nii.gz` 为LFS追踪后，689个已提交的nii.gz在IDE变绿色(Modified)
**原因**: Git LFS将实际文件替换为指针文件，IDE检测为修改
**解决**: 回退LFS迁移，只对CT (`data/ct/**/*.nii.gz`) 和视频 (`**/*.mp4`) 使用LFS，器官标签和肿瘤mask作为普通Git文件

### 问题6: 生成速度慢
**发现**: 单mask需要37-90秒（位置选择中的二进制腐蚀）
**解决**: 距离变换替代球形腐蚀 → 37s→4.5s；裁剪区域弹性变形 → 避免全CT位移场生成

---

## 四、文件结构（重要！）

```
Mask/
├── README.md                    ← 项目文档（含命名规则、参考文献）
├── PROJECT_OVERVIEW.md          ← 工程概览
├── IMPLEMENTATION_PLAN.md       ← 实现计划
├── SESSION_HANDOFF.md           ← 本文档
│
├── data/
│   ├── ct/                      ← 30个CT扫描 (Git LFS, 总共1.4GB)
│   │   └── BDMAP_000000XX/ct.nii.gz
│   ├── organ_labels/            ← 器官分割 (普通Git, ~97MB)
│   │   └── BDMAP_000000XX/segmentations/{organ}.nii.gz
│   └── manifest.csv             ← 样本索引
│
├── output/real_ct/              ← 生成的肿瘤mask (普通Git)
│   ├── liver_lesion/            (50个: t00~t49)
│   ├── pancreatic_lesion/       (50个: t00~t49)
│   ├── kidney_lesion/           (50个: t00~t49)
│   ├── colon_lesion/            (50个: t00~t49)
│   ├── esophagus_tumor/         (50个: t00~t49)
│   ├── endometrioma_tumor/      (51个: t00~t50)
│   └── */video/                 ← 视频 (Git LFS)
│
├── Step0~Step7/                 ← 模块化实现
├── .gitignore                   ← 排除: data/tmp/, __pycache__, .pytest_cache
├── .gitattributes               ← LFS: data/ct/**/*.nii.gz, **/*.mp4
│
├── supplement_masks.py          ← 补充生成脚本（运行多轮达到目标数量）
├── validate_and_renumber.py     ← 验证+重新编号（两阶段rename避免覆盖）
├── generate_videos.py           ← 视频生成
├── generate_uterus_masks.py     ← 子宫mask生成
└── run_generation.py            ← 批量生成入口
```

**命名规则**: `{organ}_t{序号}__{CT编号}.nii.gz`
- 例: `liver_lesion_t00__BDMAP_00000012.nii.gz`
- t00=该器官体积最大的，t49=最小的
- `__` 后面的CT名通过shape匹配自动添加

---

## 五、关键源码修改位置

| 文件 | 行号/函数 | 修改内容 |
|------|----------|----------|
| `Step6/src/main.py` | `generate_one()` L205-260 | 距离变换预检 + 尺寸回退 |
| `Step6/src/main.py` | `generate_one()` L308-312 | tiny肿瘤禁用弹性变形 |
| `Step6/src/main.py` | `generate_one()` L319-323 | 生成后立即检查体积=0 |
| `Step6/src/main.py` | `generate_one()` L335-355 | 器官边界裁剪 + 体积<3拒绝 + 裁剪损失>20%拒绝 |
| `Step6/src/main.py` | `generate_batch()` L365-400 | 循环至成功数达标 |
| `Step5/src/mask_generator.py` | `create_mask()` L280-320 | 裁剪区域弹性变形 |
| `Step1/src/utils.py` | `erode_mask()` L206-241 | 距离变换替代球形腐蚀 |
| `Step1/src/utils.py` | `compute_valid_region()` L331-400 | 裁剪到器官BB再腐蚀 |
| `supplement_masks.py` | L85-110 | 预过滤可行CT + 随机化顺序 + 体积>=5检查 |

---

## 六、常用命令

```bash
# 生成更多肿瘤mask（修改config中的count后）
cd Mask && python -u supplement_masks.py

# 验证质量并重新编号
python validate_and_renumber.py

# 生成视频
python -u generate_videos.py

# Git操作
git status                    # 检查变更（应始终clean）
git push origin master        # 推送（LFS文件需网络稳定）
```

---

## 七、注意事项（给新会话）

1. **使用 `python` 不是 `python3`** — Windows上python3有TerminateProcess bug (exit code 49)

2. **不要删除 output/real_ct/*/video/ 和 data/ct/** — 这些用Git LFS追踪，重新生成需要运行时环境

3. **Git LFS只追踪CT和视频** — .gitattributes中不应有 `*.nii.gz` 通配符，只追踪 `data/ct/**/*.nii.gz` 和 `**/*.mp4`

4. **命名规则**: 肿瘤mask文件名是 `{organ}_t{序号}__{CT名}.nii.gz`，`__` 后面是通过shape匹配自动附加的源CT标识

5. **generate_videos.py的 `find_ct`** — 已改为通过shape匹配找CT，不依赖文件名解析

6. **validate_and_renumber.py** — 使用两阶段重命名（→.tmpNNN→.tNN），避免shutil.move覆盖

7. **supplement_masks.py** — 预过滤viable CTs，随机化顺序，80×需要数量的最大尝试

8. **margin极小化** — feather=0.5 safety=1，是为适应小器官，修改需谨慎

9. **器官mask体积差异巨大** — liver 765K vox vs esophagus 1.7K vox，这是TotalSegmentator的特性

---

## 八、如果要从头开始

```bash
# 1. 安装依赖
pip install nibabel numpy scipy matplotlib imageio

# 2. 下载CT数据
# 从 AbdomenAtlas2.0Mini 下载 tar.gz，解压到 data/ct/

# 3. 运行TotalSegmentator获取器官mask
python run_totalsegmentator.py

# 4. 生成子宫mask
python generate_uterus_masks.py

# 5. 生成肿瘤mask (清理旧数据后)
rm -f output/real_ct/*/*.nii.gz
python -u supplement_masks.py  # 可能需要运行多轮直到每器官50个

# 6. 验证并重编号
python validate_and_renumber.py

# 7. 生成视频
python -u generate_videos.py
```
