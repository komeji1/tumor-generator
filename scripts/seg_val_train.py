"""
分割模型快速验证 — 用合成 CT 训练 UNet，验证合成数据可用性

用法:
  python -m scripts.seg_val_train                          # 训练 + 评估
  python -m scripts.seg_val_train --eval-only              # 仅评估已有模型
  python -m scripts.seg_val_train --epochs 50              # 训练 50 epochs

标签: 3-class — 背景(0) + 肝脏(1) + 肿瘤(2)
数据: 使用 MAISI full label (132-class) + tumor_mask，合并为 3-class

改进 (v2):
  - 修复 tumor_mask 匹配逻辑
  - RandCropByPosNegLabeld pos=5, neg=1 (强化肿瘤采样)
  - DiceCELoss 加权: tumor ×10
  - 手动 Dice 计算替代有 bug 的 DiceMetric
  - warmup + cosine scheduler
  - 训练中打印 organ/tumor 分开 Dice
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord,
)
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet


# ---------- Constants ----------

LIVER_LABEL = 1  # in MAISI 132-class map


# ---------- Data Finding ----------

def find_synthetic_data(project_root: str):
    """查找 MAISI 生成的 CT + full_label + tumor_mask，配对成训练数据

    匹配逻辑: output/ 下的 sample_XXX_image/label_full 与
               output/tumor_ct/liver_lesion/ 下的 CT+tumor_mask
               通过 CT 文件的 affine/header 做精确配对
    """
    output_dir = os.path.join(project_root, "output")
    data = []

    if not os.path.isdir(output_dir):
        return data

    # Collect all MAISI samples with full labels
    maisi_samples = {}
    for f in sorted(os.listdir(output_dir)):
        if not f.endswith("_image.nii.gz"):
            continue
        prefix = f.replace("_image.nii.gz", "")
        ct_path = os.path.join(output_dir, f)
        full_label_path = os.path.join(output_dir, f"{prefix}_label_full.nii.gz")
        if not os.path.exists(full_label_path):
            continue
        maisi_samples[prefix] = {
            "image": ct_path,
            "label": full_label_path,
            "name": prefix,
        }

    # Collect all tumor masks from output/tumor_ct/
    tumor_ct_dir = os.path.join(output_dir, "tumor_ct")
    tumor_masks = []
    if os.path.isdir(tumor_ct_dir):
        for subdir in os.listdir(tumor_ct_dir):
            subpath = os.path.join(tumor_ct_dir, subdir)
            if not os.path.isdir(subpath):
                continue
            for tf in os.listdir(subpath):
                if tf.endswith("_tumor_mask.nii.gz"):
                    tumor_masks.append(os.path.join(subpath, tf))

    # Match tumor masks to MAISI samples by checking CT shape
    # Each tumor_mask file has a corresponding CT file in the same subdir
    for tm_path in tumor_masks:
        # The CT paired with this tumor mask
        ct_in_subdir = tm_path.replace("_tumor_mask.nii.gz", ".nii.gz")
        if not os.path.exists(ct_in_subdir):
            continue

        # Load the CT shape to match with MAISI output
        try:
            ct_nib = nib.load(ct_in_subdir)
            ct_shape = ct_nib.shape
        except Exception:
            continue

        # Find MAISI sample with matching shape
        matched = False
        for prefix, sample in maisi_samples.items():
            if "tumor_mask" in sample:
                continue  # already matched
            try:
                maisi_ct_nib = nib.load(sample["image"])
                if maisi_ct_nib.shape == ct_shape:
                    sample["tumor_mask"] = tm_path
                    matched = True
                    break
            except Exception:
                continue

        if not matched:
            # Try matching by checking if the tumor mask shape matches any MAISI sample
            tm_shape = nib.load(tm_path).shape
            for prefix, sample in maisi_samples.items():
                if "tumor_mask" in sample:
                    continue
                try:
                    maisi_ct_nib = nib.load(sample["image"])
                    if maisi_ct_nib.shape == tm_shape:
                        sample["tumor_mask"] = tm_path
                        break
                except Exception:
                    continue

    data = list(maisi_samples.values())
    with_tumor = sum(1 for d in data if "tumor_mask" in d)
    print(f"MAISI samples: {len(data)}, with tumor mask: {with_tumor}")
    return data


# ---------- Dataset ----------

class TumorSegDataset(Dataset):
    """3-class 肿瘤分割数据集: bg=0, organ=1, tumor=2"""

    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = dict(self.data[idx])
        if self.transform:
            d = self.transform(d)
        # RandCropByPosNegLabeld returns list of dicts, take first
        if isinstance(d, list):
            d = d[0]
        return d


# ---------- Transforms ----------

def get_train_transform():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,
                           b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=5, neg=1, num_samples=2,  # 5:1 pos:neg for small tumors
            image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transform():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,
                           b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
    ])


# ---------- Dice Calculation ----------

def compute_dice(pred_argmax, label, n_classes=3):
    """手动计算 per-class Dice

    Args:
        pred_argmax: (B, D, H, W) long tensor, argmax prediction
        label: (B, 1, D, H, W) or (B, D, H, W) tensor, ground truth
        n_classes: number of classes

    Returns:
        dict: {class_idx: dice_score}
    """
    if label.dim() == 5 and label.shape[1] == 1:
        label = label.squeeze(1)
    label = label.long()

    result = {}
    for c in range(n_classes):
        p = (pred_argmax == c).float()
        l = (label == c).float()
        inter = (p * l).sum().item()
        union = p.sum().item() + l.sum().item()
        if union > 0:
            result[c] = 2.0 * inter / union
        else:
            result[c] = float("nan")  # class not present
    return result


# ---------- Training ----------

def train(model, train_loader, val_loader, device, epochs=50, lr=2e-3, save_dir=None):
    """训练分割模型 — 带 tumor 加权 loss + warmup"""
    # Weighted loss: bg=1.0, organ=2.0, tumor=10.0
    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True,
        weight=torch.tensor([1.0, 2.0, 10.0]).to(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Warmup + cosine schedule
    warmup_epochs = min(5, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_dice = 0.0
    best_organ_dice = 0.0
    best_tumor_dice = 0.0
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0

        # Validate
        model.eval()
        all_organ = []
        all_tumor = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = sliding_window_inference(
                    images, (96, 96, 96), 1, model, overlap=0.5,
                )
                pred_argmax = outputs.argmax(dim=1)
                dice_dict = compute_dice(pred_argmax, labels)
                if not np.isnan(dice_dict.get(1, float("nan"))):
                    all_organ.append(dice_dict[1])
                if not np.isnan(dice_dict.get(2, float("nan"))):
                    all_tumor.append(dice_dict[2])

        organ_dice = np.mean(all_organ) if all_organ else 0.0
        tumor_dice = np.mean(all_tumor) if all_tumor else 0.0
        val_dice = (organ_dice + tumor_dice) / 2.0  # combined metric

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            best_organ_dice = organ_dice
            best_tumor_dice = tumor_dice

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
              f"organ={organ_dice:.4f}  tumor={tumor_dice:.4f}  "
              f"combined={val_dice:.4f}  best={best_dice:.4f}  "
              f"lr={lr_now:.1e}  time={dt:.0f}s")

        if is_best and save_dir:
            torch.save(model.state_dict(),
                      os.path.join(save_dir, "best_model.pth"))

    print(f"\nBest: organ={best_organ_dice:.4f}  tumor={best_tumor_dice:.4f}  "
          f"combined={best_dice:.4f}")
    return best_dice


# ---------- Evaluation ----------

def evaluate(model, val_loader, device):
    """评估分割模型 — 分别报告 organ 和 tumor 的 Dice"""
    model.eval()

    per_case = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = sliding_window_inference(
                images, (96, 96, 96), 1, model, overlap=0.5,
            )

            pred_argmax = outputs.argmax(dim=1)
            dice_dict = compute_dice(pred_argmax, labels)
            organ_d = dice_dict.get(1, 0.0)
            tumor_d = dice_dict.get(2, 0.0)
            if np.isnan(organ_d):
                organ_d = 0.0
            if np.isnan(tumor_d):
                tumor_d = 0.0
            per_case.append({"organ": organ_d, "tumor": tumor_d})

    print(f"\n{'='*60}")
    print(f"Evaluation Results (3-class: bg=0, organ=1, tumor=2)")
    print(f"{'='*60}")
    for i, d in enumerate(per_case):
        print(f"  Case {i+1}: Organ Dice = {d['organ']:.4f}, Tumor Dice = {d['tumor']:.4f}")
    mean_organ = np.mean([d["organ"] for d in per_case])
    mean_tumor = np.mean([d["tumor"] for d in per_case])
    print(f"  Mean:   Organ Dice = {mean_organ:.4f}, Tumor Dice = {mean_tumor:.4f}")
    print(f"{'='*60}")
    return mean_organ, mean_tumor


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="分割模型快速验证 v2")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    parser.add_argument("--lr", type=float, default=2e-3, help="学习率")
    parser.add_argument("--batch-size", type=int, default=1, help="批大小")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--eval-only", action="store_true", help="仅评估")
    parser.add_argument("--save-dir", default=None, help="模型保存目录")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.save_dir is None:
        args.save_dir = os.path.join(project_root, "output", "seg_model_v2")

    device = torch.device(args.device)

    # Find data
    data = find_synthetic_data(project_root)
    if not data:
        print("未找到合成 CT 数据。请先运行 tumor_prompt_runner 生成数据。")
        return

    # Merge labels: convert 132-class → 3-class for each sample
    print("Converting 132-class labels to 3-class...")
    for d in data:
        full_nib = nib.load(d["label"])
        full_label = full_nib.get_fdata()
        new_label = np.zeros_like(full_label, dtype=np.float32)
        new_label[full_label == LIVER_LABEL] = 1  # organ (liver)

        if "tumor_mask" in d:
            tumor_nib = nib.load(d["tumor_mask"])
            tumor_mask = tumor_nib.get_fdata()
            new_label[tumor_mask > 0] = 2  # tumor

        # Save the 3-class label — preserve original affine & header
        merged_path = d["label"].replace("_label_full.nii.gz", "_label_3class.nii.gz")
        # Check if already exists and matches
        need_save = True
        if os.path.exists(merged_path):
            existing = nib.load(merged_path).get_fdata()
            if np.array_equal(existing, new_label):
                need_save = False
        if need_save:
            merged_nib = nib.Nifti1Image(new_label, full_nib.affine, full_nib.header)
            nib.save(merged_nib, merged_path)
        d["label"] = merged_path

    # Verify samples
    print(f"Found {len(data)} synthetic CT pairs")
    with_tumor = sum(1 for d in data if "tumor_mask" in d)
    print(f"  With tumor mask: {with_tumor}")

    # Check a sample
    verify = nib.load(data[0]["label"]).get_fdata()
    print(f"3-class label unique: {np.unique(verify).tolist()}")
    for v in np.unique(verify):
        if v > 0:
            cnt = np.sum(verify == v)
            print(f"  class {int(v)}: {cnt} voxels ({cnt/verify.size*100:.2f}%)")

    # Split: 80% train, 20% val
    np.random.seed(42)
    np.random.shuffle(data)
    n_train = max(1, int(len(data) * 0.8))
    train_data = data[:n_train]
    val_data = data[n_train:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_ds = TumorSegDataset(train_data, get_train_transform())
    val_ds = TumorSegDataset(val_data, get_val_transform())

    # Custom collate: RandCropByPosNegLabeld returns list of dicts, flatten them
    def collate_fn(batch):
        from torch.utils.data import default_collate
        flat = []
        for item in batch:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return default_collate(flat)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    # Create model (slightly larger UNet for better tumor detection)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # background + organ + tumor
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNet, {n_params/1e6:.1f}M parameters")

    # Load or train
    model_path = os.path.join(args.save_dir, "best_model.pth")
    if args.eval_only and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        print(f"\nTraining for {args.epochs} epochs...")
        best_dice = train(model, train_loader, val_loader, device,
                         epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)
        print(f"\nTraining complete. Best combined Dice: {best_dice:.4f}")

    # Evaluate
    evaluate(model, val_loader, device)


if __name__ == "__main__":
    main()
