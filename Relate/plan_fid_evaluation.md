# FID 评估步骤：在现有 NIfTI 上计算 2.5D FID

## 前置条件

| 项目 | 状态 | 说明 |
|------|------|------|
| GPU | ✅ 可用 | CUDA=True |
| `monai.metrics.FIDMetric` | ✅ 已安装 | 核心计算 |
| `fire` | ✅ 已安装 | CLI 参数 |
| RadImageNet ResNet50 | ❌ 需联网下载 | 首次运行自动下载到 `~/.cache/torch/hub/` |
| SqueezeNet 1.1 | ✅ 已缓存 | 备选方案（精度低于 RadImageNet） |
| `torch.distributed` | ⚠️ 需 NCCL | Windows 不支持 NCCL，需修改脚本 |

## 整体流程

```
真实 CT (N张 .nii.gz)          合成 CT (N张 .nii.gz)
      ↓                              ↓
  ┌──────────────────────────────────────────┐
  │  预处理: 重采样1mm³ → 512³ → CT裁剪     │
  └──────────────────────────────────────────┘
      ↓                              ↓
  ┌──────────────────────────────────────────┐
  │  2.5D 切片: 沿 XY/YZ/ZX 三个方向切片    │
  │  单通道→复制3通道→RadImageNet归一化      │
  └──────────────────────────────────────────┘
      ↓                              ↓
  ┌──────────────────────────────────────────┐
  │  特征提取: RadImageNet ResNet50 前向     │
  │  → 特征图 → spatial_average → 1维向量    │
  └──────────────────────────────────────────┘
      ↓                              ↓
  μ₁, Σ₁ (真实)              μ₂, Σ₂ (合成)
      ↓                              ↓
  ┌──────────────────────────────────────────┐
  │  FID = ‖μ₁-μ₂‖² + Tr(Σ₁+Σ₂-2√(Σ₁Σ₂)) │
  │  分别计算 XY/YZ/ZX 三个平面的 FID       │
  │  最终: FID_avg = (FID_xy+FID_yz+FID_zx)/3│
  └──────────────────────────────────────────┘
```

## 步骤 1：准备数据

### 1.1 确认本地 76 张 CT 的来源

output/ 下 76 张 `*_image.nii.gz` 的特征：
- 文件名格式: `sample_*_image.nii.gz`
- 统一 shape: (256,256,128) / (256,256,256) / (256,256,512)
- 统一 spacing: ~1.5×1.5×2.0 mm
- HU 严格裁剪到 [-1000, 1000]

**结论：这 76 张是 MAISI 生成的合成 CT，不是真实临床 CT。不能用作 FID 的真实参考数据集。**

FID 的原理是衡量合成分布与真实分布的距离。如果用合成 CT 当"真实"参考，
算出的 FID 只反映两个合成分布之间的差异，不能说明生成质量。

### 1.2 获取真实 CT 数据集

MAISI 论文 FID 评估使用的真实 CT 来源：**[autoPET 2023](https://www.nature.com/articles/s41597-022-01718-3)**

autoPET 2023 数据集信息：
- 全称: The autoPET 2023 challenge — Large-Scale FDG-PET-CT Lesion Segmentation
- 内容: 全身 FDG-PET/CT 扫描，含 CT 和 PET 两个模态
- 规模: ~1000 例（训练集 900+，测试集 200+）
- 格式: NIfTI (.nii.gz)
- 下载: https://autopet-2023.grand-challenge.org/ 或 HuggingFace
- 许可: CC BY 4.0（学术研究免费）

MAISI 只用了 CT 模态（不需要 PET），且是**未参与训练**的测试子集。

#### 替代数据集（如果 autoPET 不可用）

根据 data/README.md，MAISI 训练数据来自 24+ 个公开 CT 数据集，
其中**未参与训练的**可作为独立测试集：

| 数据集 | 规模 | 下载 | 特点 |
|--------|------|------|------|
| **autoPET 2023** | ~1000例 | grand-challenge.org | MAISI 论文用的，含全身CT |
| MSD Task03 (Liver) | 131例 | http://medicaldecathlon.com/ | 腹部肝CT，**参与过训练** |
| MSD Task07 (Pancreas) | 282例 | http://medicaldecathlon.com/ | 腹部胰腺CT，**参与过训练** |
| MSD Task08 (Hepatic Vessel) | 443例 | http://medicaldecathlon.com/ | 腹部肝血管CT，**参与过训练** |
| NLST | ~3100例 | TCIA | 胸部CT，**参与过训练** |

⚠️ **重要**：用于 FID 评估的真实 CT 应当是模型**未见过**的数据（hold-out）。
autoPET 2023 MAISI 训练时未使用，是最正确的选择。
MSD/NLST 等已被用于训练，用它们做 FID 会偏高估计质量。

#### 下载 autoPET 2023 CT

```bash
# 方法1: 通过 grand-challenge 平台
# 访问 https://autopet-2023.grand-challenge.org/ 注册并下载数据

# 方法2: 通过 HuggingFace (如果可用)
# pip install huggingface_hub
# python -c "from huggingface_hub import snapshot_download; snapshot_download('username/autopet2023')"

# 下载后，提取 CT 图像（不需要 PET 和 label）
# autoPET 目录结构: autopet_2023/xxxx_xxxx/ct.nii.gz
# 需要整理到统一目录
```

### 1.3 准备合成 CT（待评估数据集）

需要**经过 `infer_image_from_mask.py` 生成的含肿瘤 CT**，不是 mask 文件。
当前已有 8 张含肿瘤 CT：
```bash
mkdir -p data/fid_synth
cd output
ls *tumor*image*.nii.gz > ../data/fid_synth/filelist.txt
for f in $(cat ../data/fid_synth/filelist.txt); do
    cp "$f" ../data/fid_synth/
done
```

⚠️ **注意**：8 张太少，FID 统计意义不足。建议：
- 每种器官+尺寸组合至少生成 10 张含肿瘤 CT
- 总计至少 50 张以上

### 1.4 文件列表格式

`filelist.txt` 每行一个文件名（相对路径或纯文件名）：
```
case001_image.nii.gz
case002_image.nii.gz
...
```

## 步骤 2：解决 Windows 兼容性

`compute_fid_2-5d_ct.py` 使用 `torch.distributed` + NCCL，**Windows 不支持 NCCL**。
有两种方案：

### 方案 A：修改脚本为单 GPU 版本（推荐，简单）

在脚本末尾添加一个非分布式的入口函数，跳过 `dist.init_process_group`：

```python
def main_single_gpu(
    real_dataset_root, real_filelist, real_features_dir,
    synth_dataset_root, synth_filelist, synth_features_dir,
    enable_center_slices_ratio=None,
    enable_padding=True, enable_center_cropping=True,
    enable_resampling_spacing=None,
    ignore_existing=False, model_name="radimagenet_resnet50",
    num_images=100, output_root="./features",
    target_shape="512x512x512",
):
    """单 GPU 版本，无需 torchrun/NCCL。"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载特征网络
    if model_name == "radimagenet_resnet50":
        feature_network = torch.hub.load(
            "Warvito/radimagenet-models", model="radimagenet_resnet50",
            verbose=True, trust_repo=True)
    else:
        import torchvision
        feature_network = torchvision.models.squeezenet1_1(pretrained=True)
    feature_network.to(device).eval()

    # ... 后续逻辑同 main()，但去掉所有 dist 相关调用
    # 把 dist.all_gather 替换为直接在本地计算 FID
```

### 方案 B：使用 WSL2 或 Linux 环境运行原始脚本

```bash
# 在 WSL2 中
torchrun --nproc_per_node=1 scripts/compute_fid_2-5d_ct.py \
  --model_name "radimagenet_resnet50" \
  ...
```

## 步骤 3：下载 RadImageNet ResNet50

首次运行需联网下载（约 100MB）：
```bash
python -c "
import torch
model = torch.hub.load('Warvito/radimagenet-models', model='radimagenet_resnet50', trust_repo=True)
print('下载成功')
"
```

如果无法联网，可用本地已缓存的 SqueezeNet 1.1 作为替代（精度较低）：
```bash
--model_name "squeezenet1_1"
```

## 步骤 4：运行 FID 计算

```bash
# 单 GPU 版本（推荐，Windows 兼容）
python scripts/compute_fid_2-5d_ct.py \
  --model_name "radimagenet_resnet50" \
  --real_dataset_root "data/fid_real" \
  --real_filelist "data/fid_real/filelist.txt" \
  --real_features_dir "real" \
  --synth_dataset_root "data/fid_synth" \
  --synth_filelist "data/fid_synth/filelist.txt" \
  --synth_features_dir "synth" \
  --enable_center_slices_ratio 0.4 \
  --enable_padding True \
  --enable_center_cropping True \
  --enable_resampling_spacing "1.0x1.0x1.0" \
  --ignore_existing True \
  --num_images 50 \
  --output_root "./features" \
  --target_shape "512x512x512"
```

## 步骤 5：解读结果

输出示例：
```
FID XY: 5.124
FID YZ: 4.723
FID ZX: 7.963
FID Avg: 5.937
```

参考值（来自 MAISI 论文，对比 autoPET 2023）：

| 方法 | FID Avg |
|------|---------|
| DDPM | 22.608 |
| LDM | 12.379 |
| HA-GAN | 13.757 |
| MAISI ddpm-ct | 6.083 |
| MAISI rflow-ct | **5.124** |

- FID < 10：合成质量接近真实
- FID 10-20：有明显差异但可接受
- FID > 20：合成质量较差

## 关键注意事项

1. **FID 是两组分布的距离**，不能对单张图计算，至少需要每组 50 张以上
2. **输入必须是 CT 图像**（`*_image.nii.gz`），不是 label mask
3. **两组数据的预处理必须一致**（相同的重采样、裁剪、归一化）
4. **RadImageNet vs SqueezeNet**：RadImageNet 在医学影像上更准确，但需联网下载
5. **Windows 限制**：原始脚本依赖 NCCL 分布式，需改单 GPU 版或用 Linux
6. **2.5D 的意义**：3D CT 无法直接用 2D Inception 网络，所以沿三个正交平面切片后分别提取特征
