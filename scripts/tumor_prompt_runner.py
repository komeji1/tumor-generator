"""
肿瘤生成提示词执行器 (Tumor Prompt Runner)

双层接口:
  1. JSON 配置文件 → 批量精确控制 (推荐)
  2. CLI 命令行 → 快速单次生成

完整管线:
  Phase 1: MAISI 生成基础 CT + 132-class mask
  Phase 2: DiffTumor 生成真实肿瘤纹理
  Phase 3: 融合嵌入 → 输出含真实肿瘤纹理的完整 CT

用法:
  python -m scripts.tumor_prompt_runner tumor_config.json     # JSON 配置
  python -m scripts.tumor_prompt_runner --quick --organ liver # CLI 快速模式
  python -m scripts.tumor_prompt_runner --list                # 列出资源
  python -m scripts.tumor_prompt_runner --example             # 打印示例配置

设计依据:
  - tumor-generator (komeji1) 的 prompts.json 格式
  - DiffTumor (CVPR 2024) 条件编码: cond = concat([z_healthy, mask_downsampled])
  - MAISI (NV-Generate-CTMR) 132 类分割标签体系
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time

import numpy as np

from .tumor_adapter import (
    ORGAN_TO_MAISI,
    SIZE_PHASE,
    TumorConfigAdapter,
    TumorPipelineResult,
    TumorTask,
    run_tumor_pipeline_from_files,
)

logger = logging.getLogger("maisi.tumor_prompt_runner")


# ═══════════════════════════════════════════════════════════════
#  示例配置
# ═══════════════════════════════════════════════════════════════

EXAMPLE_CONFIG = {
    "_description": "肿瘤生成配置 — 三阶段管线: MAISI基础CT → DiffTumor纹理 → 融合",
    "_guide": {
        "organ": "liver | pancreas | kidney | colon | lung | bone | esophagus | uterus",
        "size_category": "tiny | small | medium | large",
        "phase": "early | noearly | null (自动: tiny/small→early, medium/large→noearly)",
        "eta": "0=确定性(论文默认) | 1=最大随机性 (仅noearly有效)",
    },
    "tasks": [
        {
            "organ": "liver",
            "size_category": "medium",
            "phase": "noearly",
            "output": "both",
        },
        {
            "organ": "pancreas",
            "size_category": "small",
            "phase": None,
            "output": "full_ct",
        },
        {
            "organ": "kidney",
            "size_category": "large",
            "eta": 1.0,
            "output": "both",
        },
    ],
    "maisi": {
        "generate_version": "rflow-ct",
        "output_size": [256, 256, 128],
        "spacing": [1.7, 1.7, 2.0],
        "num_output_samples": 1,
        "random_seed": 0,
    },
    "global": {
        "device": "cuda",
    },
}


# ═══════════════════════════════════════════════════════════════
#  MAISI 基础 CT 生成
# ═══════════════════════════════════════════════════════════════

def generate_maisi_base_ct(
    task: TumorTask,
    maisi_config: dict,
    device: str = "cpu",
) -> tuple:
    """调用 MAISI 生成基础 CT + mask。

    Args:
        task: 肿瘤任务
        maisi_config: MAISI 配置 (generate_version, output_size, spacing, ...)
        device: 计算设备

    Returns:
        (ct_path, mask_path): 生成的 CT 和 mask 文件路径
    """
    import torch
    from monai.utils import set_determinism

    from .download_model_data import download_model_data
    from .sample import LDMSampler, check_input_ct
    from .utils import define_instance

    adapter = TumorConfigAdapter()
    maisi_params = adapter.task_to_maisi_params(task)

    generate_version = maisi_config.get("generate_version", "rflow-ct")
    output_size = tuple(maisi_config.get("output_size", [256, 256, 128]))
    spacing = tuple(maisi_config.get("spacing", [1.7, 1.7, 2.0]))
    num_samples = maisi_config.get("num_output_samples", 1)
    random_seed = maisi_config.get("random_seed", 0)

    if random_seed is not None:
        set_determinism(seed=random_seed)

    # 设置环境
    import tempfile
    root_dir = os.environ.get("MONAI_DATA_DIRECTORY", tempfile.mkdtemp())
    os.makedirs(root_dir, exist_ok=True)

    # 下载模型数据（跳过数据集，本地已有完整 mask 数据库）
    download_model_data(generate_version, root_dir, model_only=True)

    # 加载配置
    network = "rflow"
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", f"config_network_{network}.json"
    )
    env_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", f"environment_{generate_version}.json"
    )
    infer_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", f"config_infer_16g_{output_size[0]}x{output_size[1]}x{output_size[2]}.json"
    )

    if not os.path.exists(infer_file):
        # 使用默认推理配置
        infer_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "config_infer_16g_256x256x128.json"
        )

    # 加载环境配置
    with open(env_file) as f:
        env_dict = json.load(f)
    for k, v in env_dict.items():
        val = v if "datasets/" not in v else os.path.join(root_dir, v)
        # 设置到 args namespace

    # 加载网络配置
    with open(config_file) as f:
        config_dict = json.load(f)

    # 加载推理配置
    with open(infer_file) as f:
        infer_dict = json.load(f)

    # 构建 args
    args = argparse.Namespace()
    for k, v in env_dict.items():
        setattr(args, k, v if "datasets/" not in v else os.path.join(root_dir, v))
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in infer_dict.items():
        setattr(args, k, v)

    # 覆盖参数 — 使用 body_region + anatomy_list 而非 controllable_anatomy_size
    # 原因: controllable_anatomy_size 触发 mask 生成, 产出的 mask 过于稀疏,
    # 无法为 DiffTumor 提供足够的器官边界。改用候选 mask 数据库生成完整 132-class mask。
    organ_info = ORGAN_TO_MAISI[task.organ]
    args.body_region = organ_info["body_region"]
    args.anatomy_list = organ_info["anatomy_list"]
    args.output_size = list(output_size)
    args.spacing = list(spacing)
    args.controllable_anatomy_size = []  # 空 = 使用候选 mask 数据库
    args.modality = maisi_params.get("modality", 1)
    args.num_output_samples = num_samples

    if not hasattr(args, "cfg_guidance_scale"):
        args.cfg_guidance_scale = 0.0

    # 验证输入 (controllable_anatomy_size 为空, 跳过 anatomy_size 相关验证)
    try:
        check_input_ct(
            args.body_region,
            args.anatomy_list,
            args.label_dict_json,
            args.output_size,
            args.spacing,
            args.controllable_anatomy_size,
        )
    except ValueError as e:
        # 原始 check_input_ct 要求 controllable_anatomy_size 非空时才严格验证
        # 我们设为空, 只验证 body_region + anatomy_list
        if "controllable_anatomy_size" not in str(e):
            raise

    latent_shape = [
        args.latent_channels,
        args.output_size[0] // 4,
        args.output_size[1] // 4,
        args.output_size[2] // 4,
    ]

    # 加载模型
    device_obj = torch.device(device)
    noise_scheduler = define_instance(args, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

    autoencoder = define_instance(args, "autoencoder_def").to(device_obj)
    ckpt = torch.load(args.trained_autoencoder_path, weights_only=False)
    if "unet_state_dict" in ckpt:
        ckpt = ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ckpt)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device_obj)
    ckpt_dm = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(ckpt_dm["unet_state_dict"], strict=False)
    scale_factor = ckpt_dm["scale_factor"].to(device_obj)

    controlnet = define_instance(args, "controlnet_def").to(device_obj)
    ckpt_cn = torch.load(args.trained_controlnet_path, weights_only=False)
    import monai
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(ckpt_cn["controlnet_state_dict"], strict=False)

    mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder").to(device_obj)
    ckpt_mae = torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True)
    mask_generation_autoencoder.load_state_dict(ckpt_mae)

    mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion").to(device_obj)
    ckpt_mdm = torch.load(args.trained_mask_generation_diffusion_path, weights_only=False)
    mask_generation_diffusion_unet.load_state_dict(ckpt_mdm["unet_state_dict"])
    mask_generation_scale_factor = ckpt_mdm["scale_factor"]

    # 创建 sampler 并生成
    ldm_sampler = LDMSampler(
        body_region=args.body_region,
        anatomy_list=args.anatomy_list,
        all_mask_files_json=args.all_mask_files_json,
        all_anatomy_size_conditions_json=args.all_anatomy_size_conditions_json,
        all_mask_files_base_dir=args.all_mask_files_base_dir,
        label_dict_json=args.label_dict_json,
        label_dict_remap_json=args.label_dict_remap_json,
        autoencoder=autoencoder,
        diffusion_unet=diffusion_unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        mask_generation_autoencoder=mask_generation_autoencoder,
        mask_generation_diffusion_unet=mask_generation_diffusion_unet,
        mask_generation_scale_factor=mask_generation_scale_factor,
        mask_generation_noise_scheduler=mask_generation_noise_scheduler,
        device=device_obj,
        latent_shape=latent_shape,
        mask_generation_latent_shape=args.mask_generation_latent_shape,
        output_size=args.output_size,
        output_dir=args.output_dir,
        controllable_anatomy_size=args.controllable_anatomy_size,
        spacing=args.spacing,
        modality=args.modality,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=random_seed,
        autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
        cfg_guidance_scale=args.cfg_guidance_scale,
    )
    # 保存完整 132-class mask (不过滤), 供 DiffTumor 注入使用
    ldm_sampler.save_full_label = True

    # 生成
    output_filenames = ldm_sampler.sample_multiple_images(num_samples)

    # 释放 GPU 显存 (模型权重等不再需要)
    del autoencoder, diffusion_unet, controlnet, ldm_sampler
    del mask_generation_autoencoder, mask_generation_diffusion_unet
    del noise_scheduler, mask_generation_noise_scheduler
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回最后一个生成的文件对 (image, label_full)
    if output_filenames:
        last = output_filenames[-1]
        ct_path = last[0]
        # 使用完整 132-class mask (如果存在), 否则用过滤后的 mask
        mask_path = last[2] if len(last) > 2 else last[1]
        return ct_path, mask_path
    else:
        raise RuntimeError("MAISI 生成失败: 未产出任何文件")


# ═══════════════════════════════════════════════════════════════
#  JSON 配置接口
# ═══════════════════════════════════════════════════════════════

def run_config(config: dict) -> list:
    """执行 JSON 配置文件中的所有任务"""
    tasks_raw = config.get("tasks", [])
    if not tasks_raw:
        print("错误: 配置中没有任务")
        return []

    maisi_config = config.get("maisi", {})
    global_cfg = config.get("global", {})
    device = global_cfg.get("device", "cuda")

    print(f"{'='*60}")
    print(f"肿瘤生成管线: {len(tasks_raw)} 个任务")
    print(f"设备: {device}")
    print(f"{'='*60}\n")

    results: list[TumorPipelineResult] = []
    ok = skip = fail = 0
    t_total = time.time()

    # 展开 repeat 字段
    expanded = []
    for task in tasks_raw:
        n = max(1, task.get("repeat", 1))
        user_set_idx = "mask_index" in task
        for j in range(n):
            t = dict(task)
            if user_set_idx:
                t["mask_index"] = task["mask_index"] + j
            t.pop("repeat", None)
            expanded.append(t)

    for i, task_dict in enumerate(expanded):
        task = TumorTask(
            organ=task_dict.get("organ", "liver"),
            size_category=task_dict.get("size_category", "small"),
            phase=task_dict.get("phase"),
            modality=task_dict.get("modality", "ct"),
            output=task_dict.get("output", "both"),
            eta=task_dict.get("eta", 0.0),
            mask_file=task_dict.get("mask_file"),
            repeat=1,
            output_name=task_dict.get("output_name"),
            radius_mm=task_dict.get("radius_mm"),
            position=task_dict.get("position"),
            output_size=tuple(maisi_config.get("output_size", [256, 256, 128])),
            spacing=tuple(maisi_config.get("spacing", [1.7, 1.7, 2.0])),
        )

        adapter = TumorConfigAdapter()
        phase = adapter.resolve_phase(task.organ, task.size_category, task.phase)

        print(f"[{i+1}/{len(expanded)}] {task.organ}/{task.size_category}  "
              f"phase={phase}  device={device}")

        try:
            # Phase 1: MAISI 生成基础 CT
            print(f"  Phase 1: MAISI 生成基础 CT...")
            ct_path, mask_path = generate_maisi_base_ct(task, maisi_config, device)
            print(f"  MAISI 输出: CT={ct_path}, Mask={mask_path}")

            # Phase 2+3: DiffTumor 纹理注入
            print(f"  Phase 2: DiffTumor 纹理注入...")
            r = run_tumor_pipeline_from_files(
                ct_path=ct_path,
                mask_path=mask_path,
                task=task,
                device=device,
            )
            results.append(r)

            if r.status == "ok":
                ok += 1
                print(f"  [OK] 完成  HU={r.tumor_hu_mean:.0f}+-{r.tumor_hu_std:.0f}  "
                      f"耗时={r.time_s:.0f}s")
            else:
                fail += 1
                print(f"  [FAIL] 失败: {r.error}")

        except Exception as e:
            fail += 1
            results.append(TumorPipelineResult(
                status="fail", organ=task.organ, error=str(e)
            ))
            print(f"  [FAIL] 失败: {e}")

        # 释放 GPU 显存, 防止连续任务 OOM
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    total_t = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"完成: {total_t/60:.1f}分钟 | 成功={ok} 跳过={skip} 失败={fail}")
    return results


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="肿瘤生成提示词执行器 — MAISI基础CT + DiffTumor纹理注入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例:
          python -m scripts.tumor_prompt_runner tumor_config.json
          python -m scripts.tumor_prompt_runner --example
          python -m scripts.tumor_prompt_runner --list
          python -m scripts.tumor_prompt_runner --quick --organ liver --size medium
        """)
    )
    parser.add_argument("config", nargs="?", help="JSON 配置文件路径")
    parser.add_argument("--quick", action="store_true", help="CLI 快速模式")
    parser.add_argument("--example", action="store_true", help="输出示例 JSON 配置")
    parser.add_argument("--list", action="store_true", help="列出可用资源和配置状态")
    parser.add_argument("--organ", choices=list(ORGAN_TO_MAISI.keys()), help="器官类型")
    parser.add_argument("--size", choices=list(SIZE_PHASE.keys()), default="small",
                        help="肿瘤尺寸类别 (默认: small)")
    parser.add_argument("--phase", choices=["early", "noearly"],
                        help="权重阶段 (默认: 按尺寸自动选择)")
    parser.add_argument("--output", choices=["full_ct", "patch_96", "both"], default="both",
                        help="输出格式 (默认: both)")
    parser.add_argument("--device", default="cuda", help="计算设备 (默认: cuda)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM 随机性: 0=确定性, 1=最大随机 (默认: 0)")

    args = parser.parse_args()

    if args.example:
        print(json.dumps(EXAMPLE_CONFIG, indent=2, ensure_ascii=False))

    elif args.list:
        _list_resources()

    elif args.quick:
        if not args.organ:
            parser.error("--quick 模式需要 --organ 参数")
        config = {
            "tasks": [{
                "organ": args.organ,
                "size_category": args.size,
                "phase": args.phase,
                "output": args.output,
                "eta": args.eta,
            }],
            "maisi": {
                "generate_version": "rflow-ct",
                "output_size": [256, 256, 128],
                "spacing": [1.7, 1.7, 2.0],
                "num_output_samples": 1,
            },
            "global": {
                "device": args.device,
            },
        }
        run_config(config)

    elif args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return
        run_config(config)

    else:
        parser.print_help()


def _list_resources():
    """列出可用资源和配置状态"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Tumor Pipeline - Resource Check")
    print("=" * 60)

    # Check MAISI models
    print("\n[MAISI Models]")
    root_dir = os.environ.get("MONAI_DATA_DIRECTORY", project_root)
    for name in ["rflow-ct", "rflow-mr", "rflow-mr-brain", "ddpm-ct"]:
        env_file = os.path.join(project_root, "configs", f"environment_{name}.json")
        if os.path.exists(env_file):
            with open(env_file) as f:
                env = json.load(f)
            # Check actual model file existence, not just model_dir
            autoencoder_path = env.get("trained_autoencoder_path", "")
            if "datasets/" in autoencoder_path:
                autoencoder_full = os.path.join(root_dir, autoencoder_path)
            else:
                autoencoder_full = os.path.join(project_root, autoencoder_path)
            diffusion_path = env.get("trained_diffusion_path", "")
            if "datasets/" in diffusion_path:
                diffusion_full = os.path.join(root_dir, diffusion_path)
            else:
                diffusion_full = os.path.join(project_root, diffusion_path)
            ae_ok = os.path.exists(autoencoder_full)
            dm_ok = os.path.exists(diffusion_full)
            if ae_ok and dm_ok:
                print(f"  [OK]  {name}: autoencoder + diffusion model found")
            elif ae_ok:
                print(f"  [~~]  {name}: autoencoder found, diffusion model missing")
            else:
                print(f"  [~~]  {name}: models not found at expected paths")
                print(f"        looked for: {autoencoder_full}")
        else:
            print(f"  [X]   {name}: config missing")

    # Check DiffTumor resources
    print("\n[DiffTumor Resources]")
    tumor_paths_file = os.path.join(project_root, "configs", "tumor_paths.json")
    if os.path.exists(tumor_paths_file):
        with open(tumor_paths_file, "r", encoding="utf-8") as f:
            tp = json.load(f)

        for key, label in [
            ("diffumor_repo_dir", "DiffTumor source"),
            ("vqgan_ckpt_path", "VQGAN checkpoint"),
            ("diffusion_ckpt_dir", "Diffusion weights"),
        ]:
            path = tp.get(key, "")
            if path and os.path.exists(path):
                print(f"  [OK]  {label}: {path}")
            elif path:
                print(f"  [X]   {label}: configured but path invalid ({path})")
            else:
                print(f"  [~~]  {label}: not configured")
    else:
        print(f"  [X]   tumor_paths.json not found")

    # List supported organs
    print("\n[Supported Organs]")
    for organ, info in ORGAN_TO_MAISI.items():
        tumor_label = info["tumor_label"]
        tumor_name = f"label={tumor_label}" if tumor_label else "zero-shot"
        print(f"  {organ:<12} {info['organ_type']:<25} {tumor_name}")

    print()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
