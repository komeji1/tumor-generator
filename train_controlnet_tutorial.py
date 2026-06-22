"""Train 3D ControlNet based on the notebook train_controlnet_tutorial.ipynb.

This script reproduces the notebook flow:
1. select MAISI version
2. load LIDC chest CT dataset and generate body masks
3. create training data embeddings (via autoencoder)
4. prepare environment and model config files
5. train ControlNet
6. run inference

Requirements:
  pip install monai-weekly[pillow,tqdm] nibabel numpy torch
"""

import copy
import glob
import json
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np
import torch
from monai.config import print_config
from scripts.diff_model_setting import setup_logging
from scripts.diff_model_create_training_data import diff_model_create_training_data
from scripts.train_controlnet import train_controlnet
from scripts.infer_image_from_mask_batch import infer_image_from_mask_batch
from scripts.download_model_data import download_model_data


def generate_body_mask_from_ct(ct_nii_path, output_path, target_dim=None, body_label=200, hu_threshold=-500):
    """Generate a simple body mask from a CT volume using HU thresholding.

    Voxels above hu_threshold are considered body and assigned body_label.
    All other voxels are background (0).

    If target_dim is provided, the CT is first resized to target_dim before
    generating the mask, ensuring the mask spatial size matches the embeddings.

    Args:
        ct_nii_path: Path to the CT NIfTI file.
        output_path: Path to save the generated mask NIfTI file.
        target_dim: Optional target spatial size (e.g. [256, 256, 128]) to resize CT before masking.
        body_label: Label value for body region (default 200, matching MAISI convention).
        hu_threshold: HU threshold below which voxels are considered air/background.

    Returns:
        tuple: (dimensions, spacing) of the (resized) volume.
    """
    from monai.transforms import Resize

    img = nib.load(ct_nii_path)
    ct_data = img.get_fdata().astype(np.float32)
    affine = img.affine

    if target_dim is not None:
        # Resize CT to target_dim so mask matches embedding spatial size
        ct_tensor = torch.from_numpy(ct_data).unsqueeze(0)  # [1, H, W, D]
        resizer = Resize(spatial_size=target_dim, mode="trilinear", align_corners=False)
        ct_resized = resizer(ct_tensor).squeeze(0).numpy()  # [H, W, D]
        # Compute new spacing
        orig_spacing = [float(z) for z in img.header.get_zooms()[:3]]
        new_spacing = [orig_spacing[i] * img.shape[i] / target_dim[i] for i in range(3)]
        dimensions = tuple(target_dim)
        spacing = new_spacing
        ct_data = ct_resized
    else:
        dimensions = img.shape[:3]
        spacing = [float(z) for z in img.header.get_zooms()[:3]]

    mask_data = np.zeros_like(ct_data, dtype=np.int16)
    mask_data[ct_data > hu_threshold] = body_label
    nib.save(nib.Nifti1Image(mask_data, affine), output_path)
    return dimensions, spacing


def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files


def create_json_files_for_embeddings(gz_files, modality, include_body_region, logger):
    """Create .json sidecar files for each embedding .gz file with metadata."""
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

    logger = setup_logging("train_controlnet_tutorial")

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

    sim_modality = "mri_t2" if "mr" in generate_version else "ct"

    with open("./configs/modality_mapping.json", "r") as f:
        modality_mapping = json.load(f)
    if sim_modality not in modality_mapping:
        raise ValueError(
            f"sim_modality has to be chosen from {list(modality_mapping.keys())}. Yet got sim_modality={sim_modality}."
        )

    # ------------------------------------------------------------------
    # Load LIDC Chest CT dataset (downloaded via download_lidc_hf.py)
    # ------------------------------------------------------------------
    os.environ["MONAI_DATA_DIRECTORY"] = "./temp_work_dir"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    work_dir = tempfile.mkdtemp() if directory is None else directory

    download_model_data(generate_version, work_dir, model_only=True)

    lidc_dir = os.path.join(work_dir, "demo_train_datasets", "LIDC")
    lidc_nii = sorted(glob.glob(os.path.join(lidc_dir, "*_img.nii.gz")))

    if not lidc_nii:
        raise FileNotFoundError(
            f"No LIDC NIfTI files found in {lidc_dir}. "
            "Please run: python download_lidc_hf.py --num_scans 450"
        )

    logger.info(f"Found {len(lidc_nii)} LIDC NIfTI files.")

    # ------------------------------------------------------------------
    # Step 1: Prepare raw CT datalist and create embeddings via autoencoder
    # ------------------------------------------------------------------
    # Build datalist for diff_model_create_training_data (raw CT → embeddings)
    ct_data_rootdir = lidc_dir
    dataset_name = "LIDC"

    raw_data_dicts = [
        {"image": os.path.relpath(f, ct_data_rootdir), "modality": sim_modality}
        for f in lidc_nii
    ]
    len_train = int(0.95 * len(raw_data_dicts))
    train_files = raw_data_dicts[:len_train]
    val_files = raw_data_dicts[len_train:]

    raw_datalist = {"training": train_files, "testing": val_files}
    raw_datalist_file = os.path.join(work_dir, "lidc_controlnet_raw_datalist.json")
    with open(raw_datalist_file, "w") as f:
        json.dump(raw_datalist, f)
    logger.info(f"Raw CT datalist saved to {raw_datalist_file}")

    # Load env/config templates for embedding creation
    diff_env_config_path = f"./configs/environment_maisi_diff_model_{generate_version}.json"
    diff_model_config_path = f"./configs/config_maisi_diff_model_{generate_version}.json"

    with open(diff_env_config_path, "r") as f:
        diff_env_config = json.load(f)
    with open(diff_model_config_path, "r") as f:
        diff_model_config = json.load(f)

    diff_env_out = copy.deepcopy(diff_env_config)
    diff_model_out = copy.deepcopy(diff_model_config)
    diff_model_def_out = copy.deepcopy(model_def)

    diff_env_out["data_base_dir"] = ct_data_rootdir
    diff_env_out["embedding_base_dir"] = os.path.join(work_dir, diff_env_out.get("embedding_base_dir", "embeddings"))
    diff_env_out["json_data_list"] = raw_datalist_file
    diff_env_out["model_dir"] = os.path.join(work_dir, diff_env_out.get("model_dir", "models"))
    diff_env_out["output_dir"] = os.path.join(work_dir, diff_env_out.get("output_dir", "output"))

    os.makedirs(diff_env_out["embedding_base_dir"], exist_ok=True)
    os.makedirs(diff_env_out["model_dir"], exist_ok=True)
    os.makedirs(diff_env_out["output_dir"], exist_ok=True)

    diff_env_filepath = os.path.join(work_dir, "environment_maisi_diff_model_controlnet.json")
    with open(diff_env_filepath, "w") as f:
        json.dump(diff_env_out, f, sort_keys=True, indent=4)

    diff_model_def_out["autoencoder_def"]["num_splits"] = 4
    diff_model_def_filepath = os.path.join(work_dir, "config_maisi_controlnet.json")
    with open(diff_model_def_filepath, "w") as f:
        json.dump(diff_model_def_out, f, sort_keys=True, indent=4)

    diff_model_filepath = os.path.join(work_dir, "config_maisi_diff_model_controlnet.json")
    with open(diff_model_filepath, "w") as f:
        json.dump(diff_model_out, f, sort_keys=True, indent=4)

    num_gpus = 1

    # Create embeddings
    logger.info("Creating training data embeddings from LIDC CT...")
    emb_module_args = {
        "env_config_path": diff_env_filepath,
        "model_config_path": diff_model_filepath,
        "model_def_path": diff_model_def_filepath,
        "num_gpus": num_gpus,
    }
    diff_model_create_training_data(**emb_module_args)

    # Create .json sidecar files for each embedding file
    embedding_folder = diff_env_out["embedding_base_dir"]
    emb_gz_files = list_gz_files(embedding_folder)
    if emb_gz_files:
        create_json_files_for_embeddings(emb_gz_files, sim_modality, include_body_region, logger)
        logger.info(f"Created .json sidecar files for {len(emb_gz_files)} embeddings.")
    else:
        logger.warning(f"No embedding files found in {embedding_folder}.")

    # ------------------------------------------------------------------
    # Step 2: Generate body masks and build ControlNet datalist
    # ------------------------------------------------------------------
    dataroot_dir = os.path.join(work_dir, "sim_controlnet_datasets")
    os.makedirs(dataroot_dir, exist_ok=True)

    controlnet_datalist = {"training": []}

    for idx, ct_path in enumerate(lidc_nii):
        basename = os.path.basename(ct_path)
        # Generate corresponding mask filename
        mask_basename = basename.replace("_img.nii.gz", "_mask.nii.gz")
        mask_path = os.path.join(dataroot_dir, mask_basename)

        # Generate body mask from CT if not already exists
        # Mask should be 256x256x128 (same as CT input to autoencoder).
        # ControlNet's controlnet_cond_embedding internally downsamples 4x (stride-2 conv x2)
        # to match the latent spatial size [64, 64, 32].
        target_dim = [256, 256, 128]
        if not os.path.isfile(mask_path):
            dimensions, spacing = generate_body_mask_from_ct(
                ct_path, mask_path, target_dim=target_dim, body_label=200, hu_threshold=-500
            )
            logger.info(f"Generated body mask: {mask_basename}")
        else:
            # Mask already exists, use the target dimensions (resized)
            dimensions = tuple(target_dim)
            img = nib.load(ct_path)
            orig_spacing = [float(z) for z in img.header.get_zooms()[:3]]
            spacing = [orig_spacing[i] * img.shape[i] / target_dim[i] for i in range(3)]

        # The "image" key in ControlNet datalist must point to the EMBEDDING file,
        # not the raw CT. Embedding filename pattern: <original>_emb.nii.gz
        emb_basename = basename.replace("_img.nii.gz", "_img_emb.nii.gz")

        # Skip samples where the embedding file does not exist
        # (some LIDC IDs have gaps and may not have been encoded)
        emb_check_path = os.path.join(embedding_folder, emb_basename)
        if not os.path.isfile(emb_check_path):
            logger.warning(f"Skipping {basename}: embedding not found at {emb_check_path}")
            # Also remove the mask we just generated for this missing embedding
            if os.path.isfile(mask_path):
                os.remove(mask_path)
            continue

        entry = {
            "image": emb_basename,
            "label": mask_basename,
            "fold": 0 if idx % 5 != 0 else 1,  # 80/20 train/val split by fold
            "dim": list(dimensions),
            "spacing": spacing,
            "modality": sim_modality,
        }
        if include_body_region:
            # LIDC is chest CT → thorax region
            entry["top_region_index"] = [0, 1, 0, 0]
            entry["bottom_region_index"] = [0, 0, 1, 0]

        controlnet_datalist["training"].append(entry)

    # Copy embedding files and masks into dataroot_dir
    # First, copy the embeddings from the autoencoder output directory
    for idx, ct_path in enumerate(lidc_nii):
        basename = os.path.basename(ct_path)
        emb_basename = basename.replace("_img.nii.gz", "_img_emb.nii.gz")

        # Look for the embedding file in the autoencoder output directory
        emb_src = os.path.join(embedding_folder, emb_basename)
        emb_dst = os.path.join(dataroot_dir, emb_basename)

        if os.path.isfile(emb_src) and not os.path.isfile(emb_dst):
            shutil.copy2(emb_src, emb_dst)

        # Also copy the sidecar .json file if it exists
        json_src = emb_src + ".json"
        json_dst = emb_dst + ".json"
        if os.path.isfile(json_src) and not os.path.isfile(json_dst):
            shutil.copy2(json_src, json_dst)

    logger.info(f"Prepared {len(controlnet_datalist['training'])} LIDC samples for ControlNet training.")

    datalist_file = os.path.join(work_dir, "sim_controlnet_datalist.json")
    with open(datalist_file, "w") as f:
        json.dump(controlnet_datalist, f, indent=4)
    logger.info(f"Save data list json file to {datalist_file}")

    # ------------------------------------------------------------------
    # Step 3: Train ControlNet
    # ------------------------------------------------------------------
    env_config_path = f"./configs/environment_maisi_controlnet_train_{generate_version}.json"
    train_config_path = f"./configs/config_maisi_controlnet_train_{generate_version}.json"

    final_env_config_path = os.path.join(work_dir, "environment_maisi_controlnet_train.json")
    final_train_config_path = os.path.join(work_dir, "config_maisi_controlnet_train.json")

    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    with open(train_config_path, "r") as f:
        train_config = json.load(f)
    with open(model_def_path, "r") as f:
        model_def = json.load(f)

    env_config_out = copy.deepcopy(env_config)
    train_config_out = copy.deepcopy(train_config)
    model_def_out = copy.deepcopy(model_def)

    env_config_out["data_base_dir"] = dataroot_dir
    env_config_out["json_data_list"] = datalist_file
    env_config_out["model_dir"] = os.path.join(work_dir, env_config_out["model_dir"])
    env_config_out["output_dir"] = os.path.join(work_dir, env_config_out["output_dir"])
    env_config_out["tfevent_path"] = os.path.join(work_dir, env_config_out["tfevent_path"])
    env_config_out["exp_name"] = "lidc_controlnet_training"
    env_config_out["trained_controlnet_path"] = f"{env_config_out['model_dir']}/{env_config_out['exp_name']}_current.pt"

    os.makedirs(env_config_out["model_dir"], exist_ok=True)
    os.makedirs(env_config_out["output_dir"], exist_ok=True)
    os.makedirs(env_config_out["tfevent_path"], exist_ok=True)

    with open(final_env_config_path, "w") as f:
        json.dump(env_config_out, f, sort_keys=True, indent=4)

    max_epochs = 50
    train_config_out["controlnet_train"]["n_epochs"] = max_epochs
    # LIDC body masks only have label 200 (body), no tumors to weight
    train_config_out["controlnet_train"]["weighted_loss"] = 1
    train_config_out["controlnet_train"]["weighted_loss_label"] = [None]
    train_config_out["controlnet_infer"]["num_inference_steps"] = 1

    with open(final_train_config_path, "w") as f:
        json.dump(train_config_out, f, sort_keys=True, indent=4)

    model_def_out["autoencoder_def"]["num_splits"] = 4
    model_def_filepath = os.path.join(work_dir, "config_maisi.json")
    with open(model_def_filepath, "w") as f:
        json.dump(model_def_out, f, sort_keys=True, indent=4)

    logger.info(f"files and folders under work_dir: {os.listdir(work_dir)}.")
    logger.info(f"number of GPUs: {num_gpus}.")

    logger.info("Training ControlNet...")
    module_args = {
        "env_config_path": final_env_config_path,
        "model_config_path": final_train_config_path,
        "model_def_path": model_def_filepath,
        "num_gpus": num_gpus,
    }
    logger.info(module_args)
    train_controlnet(**module_args)

    logger.info("Inference...")
    infer_image_from_mask_batch(**module_args)


if __name__ == "__main__":
    main()
