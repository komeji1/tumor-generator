"""Train 3D latent diffusion model based on train_diff_unet_tutorial.ipynb.

This script reproduces the notebook flow:
1. choose MAISI version
2. load LIDC chest CT dataset as training data
3. build config files for training and inference
4. create training data embeddings
5. train the diffusion model
6. run inference and visualize results

Requirements:
  pip install monai-weekly[pillow,tqdm] nibabel numpy torch
"""

import copy
import glob
import json
import os
import tempfile

import nibabel as nib
import numpy as np
import torch
from monai.config import print_config
from monai.transforms import LoadImage, Orientation

from scripts.diff_model_setting import setup_logging
from scripts.download_model_data import download_model_data
from scripts.diff_model_create_training_data import diff_model_create_training_data
from scripts.diff_model_train import diff_model_train
from scripts.diff_model_infer import diff_model_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image


def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files


def create_json_files(gz_files, modality, include_body_region, logger):
    """Create .json files for each .gz file with the specified metadata."""
    for gz_file in gz_files:
        img = nib.load(gz_file)
        dimensions = img.shape[:3]
        spacing = [float(z) for z in img.header.get_zooms()[:3]]

        data = {"dim": dimensions, "spacing": spacing, "modality": modality}
        if include_body_region:
            data["top_region_index"] = [0, 1, 0, 0]
            data["bottom_region_index"] = [0, 0, 1, 0]
        logger.info(f"data: {data}.")

        json_filename = gz_file + ".json"
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Save json file to {json_filename}")


def main():
    print_config()
    logger = setup_logging("train_diff_unet_tutorial")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    generate_version = "rflow-ct"
    if generate_version == "ddpm-ct":
        model_def_path = "./configs/config_network_ddpm.json"
    elif generate_version in ["rflow-ct", "rflow-mr"]:
        model_def_path = "./configs/config_network_rflow.json"
    else:
        raise ValueError(
            f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr'], yet got {generate_version}."
        )

    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    include_body_region = model_def["include_body_region"]
    logger.info(f"Models are {generate_version}, whether to use body_region is {include_body_region}")

    os.environ["MONAI_DATA_DIRECTORY"] = "./temp_work_dir"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory

    download_model_data(generate_version, root_dir, model_only=True)

    # ------------------------------------------------------------------
    # Load LIDC Chest CT dataset (downloaded via download_lidc_hf.py)
    # ------------------------------------------------------------------
    lidc_dir = os.path.join(root_dir, "demo_train_datasets", "LIDC")
    lidc_nii = sorted(glob.glob(os.path.join(lidc_dir, "*.nii.gz")))

    if not lidc_nii:
        raise FileNotFoundError(
            f"No LIDC NIfTI files found in {lidc_dir}. "
            "Please run: python download_lidc_hf.py --num_scans 450"
        )

    logger.info(f"Found {len(lidc_nii)} LIDC NIfTI files.")

    # Build data dicts with relative paths (data_base_dir will be lidc_dir)
    demo_train_data_rootdir = lidc_dir
    demo_train_data_modality = "ct"

    data_dicts = [
        {"image": os.path.relpath(f, demo_train_data_rootdir), "modality": demo_train_data_modality}
        for f in lidc_nii
    ]
    len_train = int(0.95 * len(data_dicts))
    train_files = data_dicts[:len_train]
    val_files = data_dicts[len_train:]
    logger.info(f"Training: {len(train_files)}, Validation: {len(val_files)}")

    dataset_list = {"training": train_files, "testing": val_files}
    datalist_file = os.path.join(root_dir, "lidc_diff_model_datalist.json")
    with open(datalist_file, "w") as f:
        json.dump(dataset_list, f)

    # Visualize one sample
    visualize_image_filename = os.path.join(demo_train_data_rootdir, dataset_list["training"][0]["image"])
    logger.info(f"Visualize training image {visualize_image_filename}")
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    orientation = Orientation(axcodes="RAS")
    image_volume = orientation(loader(visualize_image_filename))
    image_volume = image_volume - torch.min(image_volume)
    image_volume = image_volume / torch.max(image_volume)
    logger.info(f"Train image shape {image_volume.shape}")
    center_loc_axis = find_label_center_loc(image_volume.squeeze(0))
    vis_image = get_xyz_plot(image_volume, center_loc_axis, mask_bool=False)
    show_image(vis_image, title="training image")

    env_config_path = f"./configs/environment_maisi_diff_model_{generate_version}.json"
    model_config_path = f"./configs/config_maisi_diff_model_{generate_version}.json"

    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    env_config_out = copy.deepcopy(env_config)
    model_config_out = copy.deepcopy(model_config)
    model_def_out = copy.deepcopy(model_def)

    env_config_out["data_base_dir"] = demo_train_data_rootdir
    env_config_out["embedding_base_dir"] = os.path.join(root_dir, env_config_out["embedding_base_dir"])
    env_config_out["json_data_list"] = datalist_file
    env_config_out["model_dir"] = os.path.join(root_dir, env_config_out["model_dir"])
    env_config_out["output_dir"] = os.path.join(root_dir, env_config_out["output_dir"])
    # Force all CT volumes to be resized to a fixed dimension before encoding.
    # This ensures consistent embedding sizes and avoids OOM during diffusion training.
    # 256x256x128 → latent [4, 64, 64, 32], fits in 8GB+ VRAM.
    env_config_out["target_dim"] = [256, 256, 128]
    trained_autoencoder_path = env_config_out["trained_autoencoder_path"]

    os.makedirs(env_config_out["embedding_base_dir"], exist_ok=True)
    os.makedirs(env_config_out["model_dir"], exist_ok=True)
    os.makedirs(env_config_out["output_dir"], exist_ok=True)

    env_config_filepath = os.path.join(root_dir, "environment_maisi_diff_model.json")
    with open(env_config_filepath, "w") as f:
        json.dump(env_config_out, f, sort_keys=True, indent=4)

    max_epochs = 100
    model_config_out["diffusion_unet_train"]["n_epochs"] = max_epochs

    model_config_filepath = os.path.join(root_dir, "config_maisi_diff_model.json")
    with open(model_config_filepath, "w") as f:
        json.dump(model_config_out, f, sort_keys=True, indent=4)

    model_def_out["autoencoder_def"]["num_splits"] = 2
    model_def_filepath = os.path.join(root_dir, "config_maisi.json")
    with open(model_def_filepath, "w") as f:
        json.dump(model_def_out, f, sort_keys=True, indent=4)

    logger.info(f"files and folders under root_dir: {os.listdir(root_dir)}.")
    num_gpus = 1
    logger.info(f"number of GPUs: {num_gpus}.")

    logger.info("Creating training data...")
    module_args = {
        "env_config_path": env_config_filepath,
        "model_config_path": model_config_filepath,
        "model_def_path": model_def_filepath,
        "num_gpus": num_gpus,
    }
    logger.info(module_args)
    diff_model_create_training_data(**module_args)

    # Embedding files are saved directly in embedding_base_dir (not in a subdirectory)
    folder_path = env_config_out["embedding_base_dir"]
    gz_files = list_gz_files(folder_path)
    create_json_files(gz_files, demo_train_data_modality, include_body_region, logger)
    logger.info("Completed creating .json files for all embedding files.")

    logger.info("Training the model...")
    module_args = {
        "env_config_path": env_config_filepath,
        "model_config_path": model_config_filepath,
        "model_def_path": model_def_filepath,
        "num_gpus": num_gpus,
        "amp": True,
    }
    diff_model_train(**module_args)

    logger.info("Running inference...")
    module_args = {
        "env_config_path": env_config_filepath,
        "model_config_path": model_config_filepath,
        "model_def_path": model_def_filepath,
        "num_gpus": num_gpus,
    }
    saved_filepath = diff_model_infer(**module_args)

    logger.info(f"Visualizing {saved_filepath[0]}...")
    visualize_image_filename = saved_filepath[0]
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    orientation = Orientation(axcodes="RAS")
    image_volume = orientation(loader(visualize_image_filename))

    if generate_version == "rflow-mr":
        image_volume = torch.clip(image_volume, 0, 1000)
    else:
        image_volume = torch.clip(image_volume, -1000, 300)
    image_volume = image_volume - torch.min(image_volume)
    image_volume = image_volume / torch.max(image_volume)

    center_loc_axis = find_label_center_loc(torch.flip(image_volume[0, ...], [-3, -2, -1]))
    vis_image = get_xyz_plot(image_volume, center_loc_axis, mask_bool=False)
    show_image(vis_image, title="image")

    logger.info("Completed all steps.")


if __name__ == "__main__":
    main()
