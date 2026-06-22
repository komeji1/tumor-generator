# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MAISI Inference Tutorial (trained on LIDC)

This tutorial illustrates how to use trained NV-Generate-CTMR model and codebase
to generate synthetic 3D images and paired masks.

Note: This script assumes the diffusion model has been trained on the LIDC-IDRI
Chest CT dataset using train_diff_unet_tutorial.py.
"""

# %% Setup environment
# Run the following commands if packages are not installed:
# python -c "import monai" || python -m pip install -q "monai-weekly[nibabel, tqdm]"
# python -c "import matplotlib" || python -m pip install -q matplotlib
# python -c "import skimage" || python -m pip install -U scikit-image
# python -c "import einops" || python -m pip install -U einops

import argparse
import json
import os
import tempfile

import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism
from scripts.utils_plot import find_label_center_loc, get_xyz_plot
from scripts.diff_model_setting import setup_logging
from scripts.download_model_data import download_model_data
from scripts.diff_model_infer import diff_model_infer

print_config()

logger = setup_logging("notebook")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
num_gpus = 1

# %% Set up the MAISI version
# Choose among 'ddpm-ct', 'rflow-ct', 'rflow-mr', 'rflow-mr-brain'. The differences are:
# - The version 'ddpm-ct' and 'rflow-ct' generate CT images, while 'rflow-mr' generate MR images,
#   'rflow-mr-brain' generate MR brain images.
# - The version 'ddpm-ct' uses basic noise scheduler DDPM. 'rflow-ct', 'rflow-mr' and 'rflow-mr-brain'
#   use Rectified Flow scheduler, can be 33 times faster during inference.
# - The version 'ddpm-ct' requires training images to be labeled with body region ("top_region_index"
#   and "bottom_region_index"), while 'rflow-ct' does not have such requirement. In other words,
#   it is easier to prepare training data.
# - For the released model weights, 'rflow-ct' can generate images with better quality for head region
#   and small output volumes, and comparable quality for other cases compared with 'ddpm-ct'.

generate_version = "rflow-ct"
if generate_version == "ddpm-ct":
    model_def_path = "./configs/config_network_ddpm.json"
elif generate_version in ['rflow-ct', 'rflow-mr', 'rflow-mr-brain']:
    model_def_path = "./configs/config_network_rflow.json"
else:
    raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr', 'rflow-mr-brain'], yet got {generate_version}.")

with open(model_def_path, "r") as f:
    model_def = json.load(f)
include_body_region = model_def["include_body_region"]
logger.info(f"Models are {generate_version}, whether to use body_region is {include_body_region}")

# %% Setup data directory
# You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.
# This allows you to save results and reuse downloads.
# If not specified a temporary directory will be used.

os.environ["MONAI_DATA_DIRECTORY"] = "./temp_work_dir"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory

download_model_data(generate_version, root_dir, model_only=True)

# %% Read in environment setting, including data directory, model directory, and output directory
# The information for data directory, model directory, and output directory are saved in
# ./configs/environment.json

args = argparse.Namespace()

environment_file = f"./configs/environment_maisi_diff_model_{generate_version}.json"
config_infer_file = f"./configs/config_maisi_diff_model_{generate_version}.json"

# Prefer LIDC-trained model config if available, otherwise fall back to default
lidc_env_config = os.path.join(root_dir, "environment_maisi_diff_model.json")
if os.path.exists(lidc_env_config):
    environment_file = lidc_env_config
    config_infer_file = os.path.join(root_dir, "config_maisi_diff_model.json")
    logger.info(f"Using LIDC-trained model config: {environment_file}")
else:
    logger.warning(
        f"LIDC-trained model config not found at {lidc_env_config}. "
        "Using default pre-trained model. Run train_diff_unet_tutorial.py first to train on LIDC."
    )

with open(environment_file, "r") as f:
    env_dict = json.load(f)
for k, v in env_dict.items():
    logger.info(f"{k}: {v}")
logger.info("Global config variables have been loaded.")

# %% Read in configuration setting, including network definition, body region and anatomy to generate, etc.
# The information for the inference input, like body region and anatomy to generate, is stored in
# "./configs/config_infer.json". Please refer to README.md for the details.

# check the format of inference inputs

# %% Set deterministic training for reproducibility
set_determinism(seed=0)
args.random_seed = 0

# %% Perform the inference
# This process will generate 3D images with specified top/bottom body regions, spacing, and dimensions.

logger.info("Running inference...")
logger.info("Note: This uses model weights trained/fine-tuned on LIDC-IDRI Chest CT dataset.")

# Define the arguments for torchrun
module_args = {
    "env_config_path": environment_file,
    "model_config_path": config_infer_file,
    "model_def_path": model_def_path,
    "num_gpus": num_gpus
}

saved_filepath = diff_model_infer(**module_args)

logger.info("Completed all steps.")
logger.info(saved_filepath)

# %% Visualize the results
visualize_image_filename = saved_filepath[0]
logger.info(f"Visualizing {visualize_image_filename}...")

# load image/mask pairs
loader = LoadImage(image_only=True, ensure_channel_first=True)
orientation = Orientation(axcodes="RAS")
image_volume = orientation(loader(visualize_image_filename))

# visualize for CT HU intensity between [-1000, 300] (LIDC chest CT range)
if generate_version == "rflow-mr" or generate_version == "rflow-mr-brain":
    image_volume = torch.clip(image_volume, 0, 1000)
    logger.info("clipping image_volume to 0-1000")
else:
    image_volume = torch.clip(image_volume, -1000, 300)
    logger.info("clipping image_volume to -1000-300")
image_volume = image_volume - torch.min(image_volume)
image_volume = image_volume / torch.max(image_volume)

# find center voxel location for 2D slice visualization
center_loc_axis = find_label_center_loc(torch.flip(image_volume[0,...], [-3, -2, -1]))
print(center_loc_axis)

# visualization
vis_image = get_xyz_plot(image_volume, center_loc_axis, mask_bool=False)
# Save visualization to file instead of showing (for non-interactive environments)
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)
vis_output_path = os.path.join(output_dir, "lidc_inference_result.png")
plt.figure("check", (24, 12))
plt.subplot(1, 2, 1)
plt.title("LIDC-generated image")
plt.imshow(vis_image)
plt.savefig(vis_output_path, bbox_inches='tight', dpi=150)
plt.close()
logger.info(f"Visualization saved to {vis_output_path}")