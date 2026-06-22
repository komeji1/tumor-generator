"""MAISI Inference Tutorial converted from notebook.

This script reproduces the inference workflow from inference_tutorial.ipynb:
- set up environment and imports
- load configuration files
- initialize models and schedulers
- run image/mask generation
- visualize generated results

Note: This script assumes the model has been trained on the LIDC-IDRI
Chest CT dataset. Run the training tutorials first to obtain LIDC-trained weights.

Requirements:
  pip install monai-weekly[nibabel,tqdm] matplotlib scikit-image einops
"""

import argparse
import json
import os
import tempfile

import monai
import torch
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism
from scripts.sample import LDMSampler, check_input_ct
from scripts.utils import define_instance
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image
from scripts.diff_model_setting import setup_logging
from scripts.download_model_data import download_model_data


def main():
    print_config()

    logger = setup_logging("inference_tutorial")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    generate_version = "rflow-ct"
    if generate_version == "ddpm-ct":
        model_def_path = "./configs/config_network_ddpm.json"
    elif generate_version == "rflow-ct":
        model_def_path = "./configs/config_network_rflow.json"
    else:
        raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct'], yet got {generate_version}.")

    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    include_body_region = model_def["include_body_region"]
    logger.info(f"Models are {generate_version}, whether to use body_region is {include_body_region}")

    os.environ["MONAI_DATA_DIRECTORY"] = "./temp_work_dir"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory

    download_model_data(generate_version, root_dir)

    args = argparse.Namespace()

    if generate_version == "ddpm-ct":
        environment_file = "./configs/environment_ddpm-ct.json"
    elif generate_version == "rflow-ct":
        environment_file = "./configs/environment_rflow-ct.json"
    else:
        raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct'], yet got {generate_version}.")

    # Load default env config first (has mask generation fields like label_dict_json, etc.)
    with open(environment_file, "r") as f:
        default_env_dict = json.load(f)

    # Prefer LIDC-trained model config for model weights, but keep default for missing fields
    lidc_env_config = os.path.join(root_dir, "environment_maisi_diff_model.json")
    if os.path.exists(lidc_env_config):
        with open(lidc_env_config, "r") as f:
            lidc_env_dict = json.load(f)
        # Merge: LIDC config overrides default, but default fills in missing keys
        merged_env = {**default_env_dict, **lidc_env_dict}
        logger.info(f"Using LIDC-trained model config merged with default: {lidc_env_config}")
    else:
        merged_env = default_env_dict
        logger.info("Using default pre-trained model config.")

    env_dict = merged_env

    with open(environment_file, "r") as f:
        env_dict = json.load(f)
    for k, v in env_dict.items():
        val = v if "datasets/" not in v else os.path.join(root_dir, v)
        setattr(args, k, val)
        logger.info(f"{k}: {val}")
    logger.info("Global config variables have been loaded.")

    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    for k, v in model_def.items():
        setattr(args, k, v)

    config_infer_file = "./configs/config_infer.json"
    with open(config_infer_file, "r") as f:
        config_infer_dict = json.load(f)
    for k, v in config_infer_dict.items():
        setattr(args, k, v)
        logger.info(f"{k}: {v}")

    if generate_version == "ddpm-ct":
        args.num_inference_steps = 1000
        logger.warning(
            f"For DDPM, num_inference_steps must be: {args.num_inference_steps}. And it has been changed to {args.num_inference_steps}."
        )

    if generate_version in ["ddpm-ct", "rflow-ct"]:
        check_input_ct(
            args.body_region,
            args.anatomy_list,
            args.label_dict_json,
            args.output_size,
            args.spacing,
            args.controllable_anatomy_size,
        )

    latent_shape = [
        args.latent_channels,
        args.output_size[0] // 4,
        args.output_size[1] // 4,
        args.output_size[2] // 4,
    ]
    logger.info("Network definition and inference inputs have been loaded.")

    set_determinism(seed=0)
    args.random_seed = 0

    noise_scheduler = define_instance(args, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

    device = torch.device("cuda")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path)
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=False)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=False)

    mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
    checkpoint_mask_generation_autoencoder = torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True)
    mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

    mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
    checkpoint_mask_generation_diffusion_unet = torch.load(
        args.trained_mask_generation_diffusion_path, weights_only=True
    )
    mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_diffusion_unet["unet_state_dict"])
    mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]

    logger.info("All the trained model weights have been loaded.")

    ldm_sampler = LDMSampler(
        args.body_region,
        args.anatomy_list,
        args.all_mask_files_json,
        args.all_anatomy_size_conditions_json,
        args.all_mask_files_base_dir,
        args.label_dict_json,
        args.label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        args.mask_generation_latent_shape,
        args.output_size,
        args.output_dir,
        args.controllable_anatomy_size,
        image_output_ext=args.image_output_ext,
        label_output_ext=args.label_output_ext,
        spacing=args.spacing,
        modality=args.modality,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=args.random_seed,
        autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
        cfg_guidance_scale=args.cfg_guidance_scale,
    )

    logger.info(f"The generated image/mask pairs will be saved in {args.output_dir}.")
    output_filenames = ldm_sampler.sample_multiple_images(args.num_output_samples)
    logger.info("MAISI image/mask generation finished")

    visualize_image_filename = output_filenames[0][0]
    visualize_mask_filename = output_filenames[0][1]
    logger.info(f"Visualizing {visualize_image_filename} and {visualize_mask_filename}...")

    loader = LoadImage(image_only=True, ensure_channel_first=True)
    orientation = Orientation(axcodes="RAS")
    image_volume = orientation(loader(visualize_image_filename))
    mask_volume = orientation(loader(visualize_mask_filename)).to(torch.uint8)

    image_volume = torch.clip(image_volume, -1000, 300)
    image_volume = image_volume - torch.min(image_volume)
    image_volume = image_volume / torch.max(image_volume)

    colorize = torch.clip(
        torch.cat([torch.zeros(3, 1, 1, 1), torch.randn(3, 200, 1, 1)], 1), 0, 1
    )
    target_class_index = 1

    center_loc_axis = find_label_center_loc(torch.flip(mask_volume[0, ...] == target_class_index, [-3, -2, -1]))

    vis_mask = get_xyz_plot(
        mask_volume,
        center_loc_axis,
        mask_bool=True,
        n_label=201,
        colorize=colorize,
        target_class_index=target_class_index,
    )
    show_image(vis_mask, title="mask")

    vis_image = get_xyz_plot(image_volume, center_loc_axis, mask_bool=False)
    show_image(vis_image, title="image")


if __name__ == "__main__":
    main()
