"""
Colon Diffusion Model 训练脚本 v2 (内存安全版)

修复:
  1. CTPreprocessor 移到循环外 (避免每步创建 SimpleITK 对象)
  2. gc.collect() 每 100 步
  3. 显式 del 大变量
  4. 预验证 UNet 参数数量
  5. 更好的进度日志
"""
import sys, os, time, glob, random, warnings, gc
import torch, numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (VQGAN_CKPT, DIFFUSION_DIR as WEIGHT_DIR, TRAINED_DIR as OUT_DIR,
                    COLON_CT_DIR, COLON_LABEL_DIR, COLON_IDX_FILE,
                    DIFFTUMOR_REPO_DIR, TEMP_DIR)

sys.path.insert(0, os.path.dirname(__file__))
from ct_preprocessor import CTPreprocessor
from vqgan.vqgan import VQGAN
warnings.filterwarnings("ignore")

# ─── 导入 DiffTumor UNet + GaussianDiffusion ───
sys.path.insert(0, DIFFTUMOR_REPO_DIR)
from TumorGeneration.ldm.ddpm import Unet3D, GaussianDiffusion

BATCH_SIZE   = 1
LR           = 1e-4
TOTAL_STEPS  = 10000
SAVE_EVERY   = 2000
GC_EVERY     = 100
TIMESTEPS    = 4          # 早期肿瘤 T=4
DEVICE       = "cpu"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. 加载 VQGAN (冻结) ──
    print("[1/5] Loading VQGAN...")
    ckpt = torch.load(VQGAN_CKPT, map_location=DEVICE, weights_only=False)
    vqgan = VQGAN(ckpt["hyper_parameters"]["cfg"]).to(DEVICE)
    vqgan.load_state_dict(ckpt["state_dict"], strict=False)
    vqgan.eval()
    for p in vqgan.parameters():
        p.requires_grad = False

    # ── 2. 构建 UNet + Diffusion ──
    print("[2/5] Building UNet...")
    unet = Unet3D(dim=24, dim_mults=(1,2,4,8), channels=17, out_dim=8).to(DEVICE)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  UNet params: {n_params:,} (expected ~36M)")

    diffusion = GaussianDiffusion(unet, vqgan_ckpt=None,
        image_size=24, num_frames=24, channels=8,
        timesteps=TIMESTEPS, loss_type="l1", device=DEVICE).to(DEVICE)
    diffusion.vqgan = vqgan

    # 从 liver_early.pt 初始化 (加速收敛)
    liver_wt = os.path.join(WEIGHT_DIR, "liver_early.pt")
    if os.path.exists(liver_wt):
        print("  Initializing from liver_early.pt...")
        data = torch.load(liver_wt, map_location=DEVICE, weights_only=False)
        unet_state = {k.replace("denoise_fn.", "", 1): v
                       for k, v in data["ema"].items() if k.startswith("denoise_fn.")}
        missing, unexpected = unet.load_state_dict(unet_state, strict=False)
        print(f"  Loaded {len(unet_state)-len(missing)}/{len(unet_state)} params from liver_early "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR, betas=(0.9, 0.999))
    ema_decay = 0.995
    ema_model = Unet3D(dim=24, dim_mults=(1,2,4,8), channels=17, out_dim=8).to(DEVICE)
    ema_model.load_state_dict(unet.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    # ── 3. 加载训练数据索引 ──
    print("[3/5] Loading data index...")
    pairs = []
    with open(COLON_IDX_FILE) as f:
        for line in f:
            ct_rel, lbl_rel = line.strip().split()
            ct_path = os.path.join(COLON_CT_DIR, os.path.basename(ct_rel))
            lbl_path = os.path.join(COLON_LABEL_DIR, os.path.basename(lbl_rel))
            if os.path.exists(ct_path) and os.path.exists(lbl_path):
                pairs.append((ct_path, lbl_path))
    print(f"  {len(pairs)} training pairs")

    if len(pairs) == 0:
        print("ERROR: No training data found!")
        return

    # ── 4. 创建 CTPreprocessor (循环外, 复用) ──
    print("[4/5] Creating CTPreprocessor...")
    pre = CTPreprocessor(DEVICE)
    tmp = TEMP_DIR
    os.makedirs(tmp, exist_ok=True)

    # ── 5. 训练循环 ──
    print(f"[5/5] Training {TOTAL_STEPS} steps on {DEVICE}...")
    print(f"  Estimated: {TOTAL_STEPS*5/3600:.1f}h (assuming ~5s/step)")

    losses = []
    t_start = time.time()
    step_times = []
    skipped = 0

    for step in range(1, TOTAL_STEPS + 1):
        t0 = time.time()

        # 随机选取一个样本
        ct_path, lbl_path = random.choice(pairs)

        try:
            ct_nii = nib.load(ct_path)
            ct_arr = ct_nii.get_fdata().astype(np.float32)
            lbl_nii = nib.load(lbl_path)
            lbl_arr = lbl_nii.get_fdata()
            spacing = np.array(ct_nii.header.get_zooms()[:3])
        except Exception as e:
            print(f"  WARN: load error step {step}: {e}")
            skipped += 1
            continue

        tumor_mask = (lbl_arr == 2)  # {0=bg, 1=organ, 2=tumor}
        organ_mask = (lbl_arr >= 1)

        if tumor_mask.sum() < 3:
            skipped += 1
            continue

        # 裁剪 96³ 肿瘤区域 (原生空间 → 1mm³)
        tumor_idx = np.argwhere(tumor_mask)
        ctr = tumor_idx.mean(axis=0).astype(int)
        half = [int(np.ceil(48.0 / s)) for s in spacing]
        x0, x1 = max(0, ctr[0]-half[0]), min(ct_arr.shape[0], ctr[0]+half[0])
        y0, y1 = max(0, ctr[1]-half[1]), min(ct_arr.shape[1], ctr[1]+half[1])
        z0, z1 = max(0, ctr[2]-half[2]), min(ct_arr.shape[2], ctr[2]+half[2])

        ct_crop = ct_arr[x0:x1, y0:y1, z0:z1].copy()
        tm_crop = tumor_mask[x0:x1, y0:y1, z0:z1].copy()
        og_crop = organ_mask[x0:x1, y0:y1, z0:z1].copy()

        # Pad to ensure ≥96mm physical extent
        need_phys = [96.0, 96.0, 96.0]
        for i, (s, need) in enumerate(zip(spacing, need_phys)):
            current = ct_crop.shape[i] * s
            if current < need:
                pad_voxels = int(np.ceil((need - current) / s))
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[i] = (0, pad_voxels)
                ct_crop = np.pad(ct_crop, pad_width, mode='constant', constant_values=ct_crop.min())
                tm_crop = np.pad(tm_crop, pad_width, mode='constant', constant_values=0)
                og_crop = np.pad(og_crop, pad_width, mode='constant', constant_values=0)

        # 写临时文件 → CTPreprocessor 重采样到 1mm³
        real_aff = np.diag(list(spacing) + [1.0])
        for name, arr in [("ct", ct_crop), ("org", og_crop.astype(np.int16)), ("tm", tm_crop.astype(np.int16))]:
            nib.save(nib.Nifti1Image(arr.astype(np.float32), real_aff), os.path.join(tmp, f"tr_{name}.nii.gz"))

        try:
            r = pre.process(
                os.path.join(tmp, "tr_ct.nii.gz"),
                os.path.join(tmp, "tr_org.nii.gz"),
                os.path.join(tmp, "tr_tm.nii.gz"), "colon")
        except Exception as e:
            print(f"  WARN: preprocess error step {step}: {e}")
            skipped += 1
            continue

        # 中心裁剪到 96³
        ct_t = r.ct_tensor; tm_t = r.tumor_mask_tensor
        d, h, w = ct_t.shape[2:]
        if d < 96 or h < 96 or w < 96:
            skipped += 1
            continue
        ct_t = ct_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]
        tm_t = tm_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]

        # 前向: DiffTumor 训练格式
        ct_t_scaled = ct_t * 2.0 - 1.0
        mask_t = tm_t.float() * 2.0 - 1.0
        x = torch.cat([ct_t_scaled, mask_t], dim=0).to(DEVICE)

        optimizer.zero_grad()
        loss = diffusion(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), unet.parameters()):
                ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * p.data

        losses.append(loss.item())
        step_times.append(time.time() - t0)

        # 每 100 步显示进度 + 垃圾回收
        if step % GC_EVERY == 0:
            avg_loss = np.mean(losses[-GC_EVERY:])
            avg_time = np.mean(step_times[-GC_EVERY:])
            eta = (TOTAL_STEPS - step) * avg_time / 3600
            print(f"  step {step:>6d}/{TOTAL_STEPS}  loss={avg_loss:.4f}  "
                  f"step_time={avg_time:.1f}s  ETA={eta:.1f}h  skipped={skipped}")
            gc.collect()

        # 保存检查点
        if step % SAVE_EVERY == 0:
            ckpt_file = os.path.join(OUT_DIR, f"colon_early_step{step}.pt")
            n_p = sum(p.numel() for p in unet.parameters())
            torch.save({
                "step": step, "model": unet.state_dict(),
                "ema": ema_model.state_dict(), "loss": avg_loss,
                "n_params": n_p,
            }, ckpt_file)
            print(f"  -> saved {ckpt_file} ({n_p:,} params)")

        # 清理大变量
        del ct_t, tm_t, ct_t_scaled, mask_t, x, loss, r
        del ct_crop, tm_crop, og_crop, ct_arr, lbl_arr

    # 最终保存
    final_file = os.path.join(OUT_DIR, "colon_early.pt")
    torch.save({
        "step": TOTAL_STEPS, "model": unet.state_dict(),
        "ema": ema_model.state_dict(),
        "n_params": sum(p.numel() for p in unet.parameters()),
    }, final_file)
    total_time = (time.time() - t_start) / 3600
    print(f"\nDone! Total: {total_time:.1f}h, Skipped: {skipped}. Saved: {final_file}")


if __name__ == "__main__":
    main()
