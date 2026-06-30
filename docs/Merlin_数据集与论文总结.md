# Merlin 腹部CT数据集 & 论文总结

> 来源: Nature 2026 — [Merlin: a computed tomography vision–language foundation model and dataset](https://www.nature.com/articles/s41586-026-10181-8)
> 数据集链接: [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)
> GitHub: [StanfordMIMI/Merlin](https://github.com/StanfordMIMI/Merlin)

## 论文核心

Merlin 是首个 **3D CT视觉-语言基础模型**, 能同时处理整个3D体积数据, 而非切片式2D处理.

- **训练数据**: 15,331次腹部CT (>6M图像) + 1.8M条EHR诊断码 + 6M tokens放射学报告
- **训练成本**: 单NVIDIA A6000 GPU, 约160小时
- **验证规模**: 内部5,137 CT + 外部44,098 CT (3个独立站点)
- **数据集**: 释放25,494对CT扫描+放射学报告 (18,317名患者)
- **模型**: ResNet152 + I3D膨胀 + Clinical Longformer文本编码器

## 6类评估任务

| 任务 | 子任务数 | 关键指标 |
|------|---------|---------|
| 零样本Findings分类 | 30 findings | F1 0.741 (内部), 0.647 (外部) |
| 表型分类 (PheWAS) | 692 phenotypes | AUROC 0.812 |
| 零样本跨模态检索 | Image↔Findings/Impressions | Recall@1 0.780 |
| 5年慢性病预测 | 6 diseases | AUROC 0.757 |
| 放射学报告生成 | 分解剖区段生成 | 优于RadFM |
| 3D语义分割 | 20 organs | 10%数据下优于nnUNet |

## 30个零样本Findings (腹部CT)

按解剖区域分类:

### 肝脏/胆道
- **Hepatic Steatosis** — 肝脂肪变性
- **Hepatomegaly** — 肝肿大
- **Biliary Ductal Dilation** — 胆管扩张
- **Gallstones** — 胆结石
- **Surgically Absent Gallbladder** — 手术切除胆囊

### 胰腺/脾脏
- **Pancreatic Atrophy** — 胰腺萎缩
- **Splenomegaly** — 脾肿大

### 肾脏
- **Renal Cyst** — 肾囊肿
- **Renal Hypodensity** — 肾脏低密度灶
- **Hydronephrosis** — 肾积水

### 胃肠道
- **Bowel Obstruction** — 肠梗阻
- **Submucosal Edema** — 黏膜下水肿
- **Appendicitis** — 阑尾炎
- **Free Air** — 游离气体 (穿孔标志)
- **Hiatal Hernia** — 食管裂孔疝

### 腹膜/盆腔
- **Ascites** — 腹水
- **Anasarca** — 全身水肿
- **Prostatomegaly** — 前列腺肥大

### 胸腔
- **Pleural Effusion** — 胸腔积液
- **Atelectasis** — 肺不张

### 血管系统
- **Thrombosis** — 血栓形成
- **Atherosclerosis** — 动脉粥样硬化
- **Abdominal Aortic Aneurysm** — 腹主动脉瘤
- **Aortic Valve Calcification** — 主动脉瓣钙化
- **Coronary Calcification** — 冠状动脉钙化
- **Cardiomegaly** — 心脏增大
- **Lymphadenopathy** — 淋巴结肿大

### 肌肉骨骼
- **Osteopenia** — 骨量减少
- **Fracture** — 骨折
- **Metastatic Disease** — ⚠️ 转移性疾病 (唯一直接肿瘤相关finding)

### 5年慢性病预测
- Chronic Kidney Disease — 慢性肾病
- Osteoporosis — 骨质疏松
- Cardiovascular Disease — 心血管疾病
- Ischemic Heart Disease — 缺血性心脏病
- Hypertension — 高血压
- Diabetes — 糖尿病

## PheWAS表型中的肿瘤类型 (119条)

### 腹部恶性肿瘤 (与CT扫描区域直接相关)

| PheCode | 肿瘤类型 |
|---------|---------|
| 155 | 肝癌和肝内胆管癌 |
| 155.1 | 肝原发性恶性肿瘤 |
| 157 | 胰腺癌 |
| 151 | 胃癌 |
| 153 | 结直肠癌 |
| 153.2 | 结肠癌 |
| 153.3 | 直肠恶性肿瘤 |
| 189.1 | 肾癌和肾盂癌 |
| 189.11 | 肾恶性肿瘤 (除外肾盂) |
| 159.3 | 胆囊和肝外胆管恶性肿瘤 |
| 159.2 | 小肠恶性肿瘤 (含十二指肠) |
| 150 | 食管癌 |
| 159.4 | 腹膜后和腹膜恶性肿瘤 |

### 其他恶性肿瘤

| PheCode | 肿瘤类型 |
|---------|---------|
| 174 | 乳腺癌 |
| 185 | 前列腺癌 |
| 184.11 | 卵巢癌 |
| 193 | 甲状腺癌 |
| 165.1 | 肺癌 |
| 170.1 | 骨癌 |
| 202.2 | 非霍奇金淋巴瘤 |
| 204.1 | 淋巴细胞白血病 |
| 209 | 神经内分泌肿瘤 |

### 转移性肿瘤

| PheCode | 转移部位 |
|---------|---------|
| 198 | 继发性恶性肿瘤 (总) |
| 198.1 | 淋巴结转移 |
| 198.2 | 呼吸系统转移 |
| 198.3 | 消化系统转移 |
| 198.4 | 肝转移 |
| 198.5 | 脑/脊髓转移 |
| 198.6 | 骨转移 |
| 198.7 | 皮肤转移 |

### 良性肿瘤 (部分)

| PheCode | 类型 |
|---------|------|
| 208 | 结肠良性肿瘤 |
| 211 | 消化系统良性肿瘤 |
| 223 | 肾脏良性肿瘤 |
| 220 | 卵巢良性肿瘤 |
| 226 | 甲状腺良性肿瘤 |
| 227.1 | 肾上腺良性肿瘤 |
| 228 | 血管瘤/淋巴管瘤 |
| 610.4 | 乳腺良性肿瘤 |

## 对CTMR项目的启示

1. **Merlin不含体素级肿瘤分割标注** — 肿瘤信息以文本形式存在于放射学报告中, 无逐像素标签
2. **零样本层面唯一直接肿瘤finding是 "Metastatic Disease"** — 可判断有无转移, 但不能定位具体肿瘤
3. **EHR表型覆盖几乎所有腹部恶性肿瘤** — 119种肿瘤相关诊断码, 是患者级标签而非图像级标注
4. **与DiffTumor/MAISI管线互补** — Merlin提供宏观诊断级信息 ("这个CT有肝癌"), 我们的项目提供体素级肿瘤合成+分割
5. **数据获取需要签署DUA** — 需在Stanford AIMI网站提交数据使用协议, 审批后通过Azure Blob Storage下载
6. **数据格式**: NIfTI, DICOM中最大切片数的series, 已去标识化
