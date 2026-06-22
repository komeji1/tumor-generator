"""Train MAISI VAE based on train_vae_tutorial.ipynb.

This script reproduces the notebook flow:
1. prepare environment and load LIDC chest CT dataset
2. load training and model configs
3. build training/validation dataloaders
4. initialize VAE and discriminator
5. train with adversarial and perceptual losses
6. validate and save checkpoints

Requirements:
  pip install monai-weekly[nibabel,tqdm] matplotlib tensorboard lpips
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import glob
import json
import os
import tempfile
import warnings
from pathlib import Path

import torch
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.inferers.inferer import SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image
from scripts.download_model_data import download_model_data


warnings.filterwarnings("ignore")


def main():
    print_config()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["MONAI_DATA_DIRECTORY"] = "./temp_work_dir"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory

    generate_version = "rflow-ct"
    download_model_data(generate_version, root_dir, model_only=True)

    # ------------------------------------------------------------------
    # Load LIDC Chest CT dataset (downloaded via download_lidc_hf.py)
    # ------------------------------------------------------------------
    lidc_dir = os.path.join(root_dir, "demo_train_datasets", "LIDC")
    lidc_datalist_path = os.path.join(lidc_dir, "lidc_datalist.json")

    if os.path.exists(lidc_datalist_path):
        print(f"Loading LIDC dataset from: {lidc_datalist_path}")
        with open(lidc_datalist_path, "r") as f:
            lidc_datalist = json.load(f)
        train_files_ct = lidc_datalist.get("training", [])
        val_files_ct = lidc_datalist.get("validation", [])
        print(f"  LIDC training scans: {len(train_files_ct)}")
        print(f"  LIDC validation scans: {len(val_files_ct)}")
    else:
        # Fallback: scan directory for NIfTI files and split 95/5
        print(f"LIDC datalist not found, scanning {lidc_dir} for NIfTI files...")
        all_nii = sorted(glob.glob(os.path.join(lidc_dir, "*.nii.gz")))
        if not all_nii:
            raise FileNotFoundError(
                f"No LIDC NIfTI files found in {lidc_dir}. "
                "Please run: python download_lidc_hf.py --num_scans 450"
            )
        n_train = int(0.95 * len(all_nii))
        train_files_ct = [{"image": f, "class": "ct"} for f in all_nii[:n_train]]
        val_files_ct = [{"image": f, "class": "ct"} for f in all_nii[n_train:]]
        print(f"  LIDC training scans: {len(train_files_ct)}")
        print(f"  LIDC validation scans: {len(val_files_ct)}")

    datasets = {
        1: {
            "data_name": "LIDC-IDRI Chest CT",
            "train_files": train_files_ct,
            "val_files": val_files_ct,
            "modality": "ct",
        },
    }

    environment_file = "./configs/environment_maisi_vae_train.json"
    with open(environment_file, "r") as f:
        env_dict = json.load(f)

    args = argparse.Namespace()
    for k, v in env_dict.items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    print(f"Trained model will be saved as {trained_g_path} and {trained_d_path}.")

    Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
    tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
    Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_path)
    print(f"Tensorboard event will be saved as {tensorboard_path}.")
    print(f"Whether load pretrained model and finetune on it: {args.finetune}")

    config_file = "./configs/config_network_rflow.json"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    for k, v in config_dict.items():
        setattr(args, k, v)

    config_train_file = "./configs/config_maisi_vae_train.json"
    with open(config_train_file, "r") as f:
        config_train_dict = json.load(f)

    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    print("Network definition and training hyperparameters have been loaded.")

    set_determinism(seed=0)

    train_files = {"ct": [], "mri": []}
    val_files = {"ct": [], "mri": []}

    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    for _, dataset in datasets.items():
        train_files_i = dataset["train_files"]
        val_files_i = dataset["val_files"]
        print(f"{dataset['data_name']}: number of training data is {len(train_files_i)}.")
        print(f"{dataset['data_name']}: number of val data is {len(val_files_i)}.")

        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
        val_files[modality] += add_assigned_class_to_datalist(val_files_i, modality)

    for modality in train_files.keys():
        print(f"Total number of training data for {modality} is {len(train_files[modality])}.")
        print(f"Total number of val data for {modality} is {len(val_files[modality])}.")

    train_files_combined = train_files["ct"] + train_files["mri"]
    val_files_combined = val_files["ct"] + val_files["mri"]

    train_transform = VAE_Transform(
        is_train=True,
        random_aug=args.random_aug,
        k=4,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float16,
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float16,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )

    print(f"Total number of training data is {len(train_files_combined)}.")
    dataset_train = CacheDataset(data=train_files_combined, transform=train_transform, cache_rate=args.cache, num_workers=0)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

    print(f"Total number of validation data is {len(val_files_combined)}.")
    dataset_val = CacheDataset(data=val_files_combined, transform=val_transform, cache_rate=args.cache, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=0, shuffle=False)

    example_vis_img = dataset_train[0]["image"]
    print(f"Train image shape {example_vis_img.shape}")
    center_loc_axis = find_label_center_loc(example_vis_img.squeeze(0))
    vis_image = get_xyz_plot(example_vis_img, center_loc_axis, mask_bool=False)
    show_image(vis_image, title="training image")

    example_vis_img = dataset_val[0]["image"]
    print(f"Val image shape {example_vis_img.shape}")
    center_loc_axis = find_label_center_loc(example_vis_img.squeeze(0))
    vis_image = get_xyz_plot(example_vis_img, center_loc_axis, mask_bool=False)
    show_image(vis_image, title="validation image")

    device = torch.device("cuda")

    args.autoencoder_def["num_splits"] = 1
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)

    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
        print("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        print("Use l1 loss")

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = (
        PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        .eval()
        .to(device)
    )

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)

    def warmup_rule(epoch):
        # Linear warmup: from 0.01 to 1.0 over 30 epochs, no sudden jumps
        if epoch < 30:
            return 0.01 + 0.99 * epoch / 30.0
        else:
            return 1.0

    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    scaler_g = None
    scaler_d = None
    if args.amp:
        scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
        scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)

    if args.finetune:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path)
        if "unet_state_dict" in checkpoint_autoencoder.keys():
            checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
        autoencoder.load_state_dict(checkpoint_autoencoder)
        print(f"Finetune on pretrained model {args.trained_autoencoder_path}")
    else:
        print("Train from scratch!")

    val_interval = args.val_interval
    best_val_recon_epoch_loss = 1e10
    total_step = 0
    start_epoch = 0
    max_epochs = args.n_epochs

    val_inferer = SlidingWindowInferer(roi_size=args.val_sliding_window_patch_size, sw_batch_size=1, overlap=0.25)

    def loss_weighted_sum(losses):
        return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]

    for epoch in range(start_epoch, max_epochs):
        print("lr:", scheduler_g.get_last_lr())
        autoencoder.train()
        discriminator.train()
        train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

        for batch in dataloader_train:
            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                losses = {
                    "recons_loss": intensity_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "p_loss": loss_perceptual(reconstruction.float(), images.float()),
                }
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

                if args.amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                if args.amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()

            total_step += 1
            for loss_name, loss_value in losses.items():
                tensorboard_writer.add_scalar(f"train_{loss_name}_iter", loss_value.item(), total_step)
                train_epoch_losses[loss_name] += loss_value.item()
            tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
            tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
            tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

        scheduler_g.step()
        scheduler_d.step()
        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)

        print(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")
        for loss_name, loss_value in train_epoch_losses.items():
            tensorboard_writer.add_scalar(f"train_{loss_name}_epoch", loss_value, epoch)

        torch.save(autoencoder.state_dict(), trained_g_path)
        torch.save(discriminator.state_dict(), trained_d_path)
        print("Save trained autoencoder to", trained_g_path)
        print("Save trained discriminator to", trained_d_path)

        if epoch % val_interval == 0:
            autoencoder.eval()
            val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
            for batch in dataloader_val:
                with torch.no_grad():
                    with autocast("cuda", enabled=args.amp):
                        images = batch["image"].to(device)
                        reconstruction, z_mu, z_sigma = dynamic_infer(val_inferer, autoencoder, images)
                        val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images).item()
                        val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                        val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images).item()

            for key in val_epoch_losses:
                val_epoch_losses[key] /= len(dataloader_val)

            val_loss_g = loss_weighted_sum(val_epoch_losses)
            print(f"Epoch {epoch} val_vae_loss {val_loss_g}: {val_epoch_losses}.")

            if val_loss_g < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_loss_g
                trained_g_path_epoch = f"{trained_g_path[:-3]}_epoch{epoch}.pt"
                torch.save(autoencoder.state_dict(), trained_g_path_epoch)
                print("Got best val vae loss.")
                print("Save trained autoencoder to", trained_g_path_epoch)

            for loss_name, loss_value in val_epoch_losses.items():
                tensorboard_writer.add_scalar(loss_name, loss_value, epoch)

            scale_factor_sample = 1.0 / z_mu.flatten().std()
            tensorboard_writer.add_scalar("val_one_sample_scale_factor", scale_factor_sample, epoch)

            center_loc_axis = find_label_center_loc(images[0, 0, ...])
            vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
            vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)

            tensorboard_writer.add_image("val_orig_img", vis_image.transpose([2, 0, 1]), epoch)
            tensorboard_writer.add_image("val_recon_img", vis_recon_image.transpose([2, 0, 1]), epoch)

            show_image(vis_image, title="val image")
            show_image(vis_recon_image, title="val recon result")


if __name__ == "__main__":
    main()
