"""
端到端诊断: 检查 DiffTumor 生成是否真的在工作

测试点:
  1. VQGAN 编码-解码是否改变图像?
  2. 扩散模型是否在 mask 区域产生了变化?
  3. 纯合成纹理 (无blend) vs 原始 CT 的差异
"""
import sys, os, time
import numpy as np, nibabel as nib, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VQGAN_CKPT, DIFFUSION_DIR

sys.path.insert(0, os.path.dirname(__file__))
from ct_preprocessor import CTPreprocessor
from condition_builder import ConditionBuilder
from diffusion_engine import DiffusionEngine

DIFFTUMOR_ROOT = r"D:\Users\33067\claude-data\DiffTumor\STEP3.SegmentationModel"
sys.path.insert(0, DIFFTUMOR_ROOT)

DEVICE = "cpu"
ORGAN = "liver"

def diagnose():
    ct_path  = r"C:\Users\33067\.claude\work\Mask\data\ct\BDMAP_00000012\ct.nii.gz"
    og_path  = r"C:\Users\33067\.claude\work\Mask\data\organ_labels\BDMAP_00000012\segmentations\liver.nii.gz"
    tm_path  = r"C:\Users\33067\.claude\work\Mask\output\real_ct\liver_lesion\liver_lesion_t00__BDMAP_00000012.nii.gz"

    print("=" * 60)
    print("DIAGNOSTIC: DiffTumor Pipeline End-to-End")
    print("=" * 60)

    # ── 0. Load + preprocess ──
    print("\n[0] Loading & preprocessing...")
    orig_nii = nib.load(ct_path)
    spacing = np.array(orig_nii.header.get_zooms()[:3])
    full_ct = orig_nii.get_fdata().astype(np.float32)

    tm_data = nib.load(tm_path).get_fdata() > 0
    og_data = nib.load(og_path).get_fdata() > 0

    t_idx = np.argwhere(tm_data)
    ctr = t_idx.mean(axis=0).astype(int)
    half = [int(np.ceil(48.0 / s)) for s in spacing]
    x0, x1 = max(0, ctr[0]-half[0]), min(full_ct.shape[0], ctr[0]+half[0])
    y0, y1 = max(0, ctr[1]-half[1]), min(full_ct.shape[1], ctr[1]+half[1])
    z0, z1 = max(0, ctr[2]-half[2]), min(full_ct.shape[2], ctr[2]+half[2])

    ct_crop = full_ct[x0:x1, y0:y1, z0:z1].copy()
    tm_crop = tm_data[x0:x1, y0:y1, z0:z1].copy()
    og_crop = og_data[x0:x1, y0:y1, z0:z1].copy()

    # Pad to 96mm
    need_phys = [96.0, 96.0, 96.0]
    for i, (s, need) in enumerate(zip(spacing, need_phys)):
        current = ct_crop.shape[i] * s
        if current < need:
            pad_voxels = int(np.ceil((need - current) / s))
            pad_width = [(0, 0), (0, 0), (0, 0)]
            pad_width[i] = (0, pad_voxels)
            ct_crop = np.pad(ct_crop, pad_width, constant_values=ct_crop.min())
            tm_crop = np.pad(tm_crop, pad_width, constant_values=0)
            og_crop = np.pad(og_crop, pad_width, constant_values=0)

    tmp = r"D:\Users\33067\claude-data\downloads\_tmp"
    os.makedirs(tmp, exist_ok=True)
    real_aff = np.diag(list(spacing) + [1.0])
    for name, arr in [("ct", ct_crop), ("org", og_crop.astype(np.int16)),
                       ("tm", tm_crop.astype(np.int16))]:
        nib.save(nib.Nifti1Image(arr.astype(np.float32), real_aff),
                 os.path.join(tmp, f"diag_{name}.nii.gz"))

    pre = CTPreprocessor(DEVICE)
    r = pre.process(os.path.join(tmp, "diag_ct.nii.gz"),
                    os.path.join(tmp, "diag_org.nii.gz"),
                    os.path.join(tmp, "diag_tm.nii.gz"), ORGAN)
    ct_t = r.ct_tensor; tm_t = r.tumor_mask_tensor
    d, h, w = ct_t.shape[2:]
    ct_t = ct_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]
    tm_t = tm_t[:, :, (d-96)//2:(d-96)//2+96, (h-96)//2:(h-96)//2+96, (w-96)//2:(w-96)//2+96]

    print(f"  CT tensor shape: {ct_t.shape}, range: [{ct_t.min():.3f}, {ct_t.max():.3f}]")
    print(f"  Mask voxels in 96^3: {tm_t.sum().item():,}")

    # ── 1. VQGAN encode-decode roundtrip (no diffusion) ──
    print("\n[1] VQGAN encode-decode roundtrip (no diffusion)...")
    builder = ConditionBuilder(VQGAN_CKPT, DEVICE)
    vqgan = builder.vqgan
    vqgan.eval()

    # Encode the original CT in DiffTumor format
    volume = ct_t * 2.0 - 1.0  # [-1, 1]
    mask = tm_t.float() * 2.0 - 1.0  # {-1, 1}
    masked_volume = (volume * (1 - tm_t.float())).detach()

    # Permute: (B,C,D,H,W) → (B,C,W,D,H) DiffTumor convention
    volume_p = volume.permute(0, 1, -1, -3, -2)
    masked_p = masked_volume.permute(0, 1, -1, -3, -2)

    with torch.no_grad():
        # Encode healthy tissue
        masked_feat = vqgan.encode(masked_p, quantize=False, include_embeddings=True)
        # Also encode full volume for comparison
        full_feat = vqgan.encode(volume_p, quantize=False, include_embeddings=True)

        # Decode both
        masked_recon = vqgan.decode(masked_feat, quantize=True)
        full_recon = vqgan.decode(full_feat, quantize=True)

    # Permute back: (B,C,W,D,H) → (B,C,D,H,W)
    masked_recon = masked_recon.permute(0, 1, -2, -1, -3)
    full_recon = full_recon.permute(0, 1, -2, -1, -3)

    # Clamp to [-1,1] → [0,1]
    masked_recon_01 = torch.clamp((masked_recon + 1.0) / 2.0, 0, 1)
    full_recon_01 = torch.clamp((full_recon + 1.0) / 2.0, 0, 1)

    # Compare: how much does VQGAN change the original?
    ct_01 = ct_t  # already [0,1]
    vqgan_diff = (full_recon_01 - ct_01).abs()
    vqgan_diff_tumor = vqgan_diff[tm_t.bool()]
    vqgan_diff_healthy = vqgan_diff[~tm_t.bool()]

    print(f"  VQGAN full reconstruction error:")
    print(f"    Tumor area:   mean={vqgan_diff_tumor.mean().item():.4f} max={vqgan_diff_tumor.max().item():.4f}")
    print(f"    Healthy area: mean={vqgan_diff_healthy.mean().item():.4f} max={vqgan_diff_healthy.max().item():.4f}")

    # ── 2. Diffusion generation ──
    print("\n[2] Diffusion generation (early, T=4 DDPM)...")
    cond = builder.build(ct_t, tm_t)
    engine = DiffusionEngine(VQGAN_CKPT, DIFFUSION_DIR, ORGAN, "early", DEVICE)
    synthetic = engine.generate(cond)
    print(f"  Synthetic shape: {synthetic.shape}, range: [{synthetic.min().item():.3f}, {synthetic.max().item():.3f}]")

    # ── 3. Compare synthetic vs original ──
    print("\n[3] Synthetic vs Original comparison...")
    synthetic_01 = torch.clamp((synthetic + 1.0) / 2.0, 0, 1)
    # synthetic is (B,1,W,D,H) → permute to (B,1,D,H,W)
    synthetic_01 = synthetic_01.permute(0, 1, -2, -1, -3)

    diff_abs = (synthetic_01 - ct_01).abs()
    diff_tumor = diff_abs[tm_t.bool()]
    diff_healthy = diff_abs[~tm_t.bool()]

    tumor_orig_hu = ct_01[tm_t.bool()] * 425 - 175
    tumor_synth_hu = synthetic_01[tm_t.bool()] * 425 - 175

    print(f"  Tumor area:")
    print(f"    Original HU:  mean={tumor_orig_hu.mean().item():.0f} std={tumor_orig_hu.std().item():.0f}")
    print(f"    Synthetic HU: mean={tumor_synth_hu.mean().item():.0f} std={tumor_synth_hu.std().item():.0f}")
    print(f"    |diff|:       mean={diff_tumor.mean().item():.4f} max={diff_tumor.max().item():.4f}")
    print(f"    >0.05 (≈20HU): {100*diff_tumor.gt(0.05).float().mean().item():.1f}%")
    print(f"    >0.10 (≈40HU): {100*diff_tumor.gt(0.10).float().mean().item():.1f}%")
    print(f"  Healthy area:")
    print(f"    |diff|:       mean={diff_healthy.mean().item():.4f} max={diff_healthy.max().item():.4f}")

    # ── 4. Check if diffusion is actually doing anything ──
    print("\n[4] Sanity check: is diffusion model producing output different from VQGAN reconstruction?")
    # Encode synthetic back through VQGAN encode to compare at latent level
    synth_for_encode = synthetic_01 * 2.0 - 1.0
    synth_for_encode = synth_for_encode.permute(0, 1, -1, -3, -2)  # (B,C,W,D,H)

    with torch.no_grad():
        synth_feat = vqgan.encode(synth_for_encode, quantize=False, include_embeddings=True)

    # Compare latent features: masked_feat (healthy encoding) vs synth_feat (generated)
    latent_diff = (synth_feat - masked_feat).abs()
    print(f"  Latent feature |diff| (masked_feat vs synth_feat):")
    print(f"    mean={latent_diff.mean().item():.4f} max={latent_diff.max().item():.4f}")

    # Compare: synth_feat vs full_feat (healthy area should match full_feat)
    latent_diff_full = (synth_feat - full_feat).abs()
    print(f"  Latent feature |diff| (full_feat vs synth_feat):")
    print(f"    mean={latent_diff_full.mean().item():.4f} max={latent_diff_full.max().item():.4f}")

    # ── 5. Final verdict ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    tumor_hu_diff = diff_tumor.mean().item() * 425  # Convert [0,1] diff to HU diff
    pct_visible = diff_tumor.gt(0.05).float().mean().item() * 100

    if tumor_hu_diff < 5:
        print(f"  WARNING: Tumor area mean diff = {tumor_hu_diff:.1f} HU (very subtle)")
        print(f"  Only {pct_visible:.0f}% of tumor voxels changed by >20 HU")
    else:
        print(f"  OK: Tumor area mean diff = {tumor_hu_diff:.1f} HU")

    # Check if diffusion actually changes the masked region
    if latent_diff.mean().item() < 0.001:
        print("  CRITICAL: Diffusion model may not be working!")
        print("  Latent features unchanged between masked input and synthetic output.")
    else:
        print(f"  Latent change magnitude: {latent_diff.mean().item():.4f} (nonzero = model is working)")

    print(f"\n  Interpretation:")
    print(f"  - VQGAN reconstruction error: {vqgan_diff_tumor.mean().item():.4f} [0-1 scale]")
    print(f"  - Diffusion-introduced change:  {diff_tumor.mean().item():.4f} [0-1 scale]")
    print(f"  - If diffusion change ≈ VQGAN error, the model is barely modifying the tissue.")

if __name__ == "__main__":
    diagnose()
