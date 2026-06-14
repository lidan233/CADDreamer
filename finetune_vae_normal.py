import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
# from diffusers import AutoencoderKL
from automl.AutoencoderKL import AutoencoderKL
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, CLIPProcessor, CLIPVisionModel
from diffusers import StableDiffusionPipeline
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.abc_dataset_vae_normal import ABCDataset as MVDiffusionDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from einops import rearrange
from omegaconf import OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from data.DeepCadDatasetBF import DeepCadDatasetBF_ST

logger = get_logger(__name__)
import os, argparse

from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, cv2, glob


def latents2img(latents, vae):
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().numpy()
    images = (images * 255).round().astype("uint8")
    return images


def inputs2img(input):
    target_images = (input / 2 + 0.5).clamp(0, 1)
    target_images = target_images.detach().cpu().numpy()
    target_images = (target_images * 255).round().astype("uint8")
    return target_images


def visualize_dp(im, dp):
    im = im.transpose((1, 2, 0))
    hsv = np.zeros(im.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    dp = dp.cpu().detach().numpy()
    mag, ang = cv2.cartToPolar(dp[0], dp[1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    bgr = bgr.transpose((2, 0, 1))
    return bgr


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def show_image(pred_images):
    images = (pred_images.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().numpy()
    images = (images * 255).round().astype("uint8")
    from skimage import io
    io.imshow(images[0])
    io.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def generate_distinct_colors(num_colors):
    base_colors = plt.cm.tab10.colors
    colors = []
    for i in range(num_colors):
        color = to_rgba(base_colors[i % len(base_colors)])
        colors.append(tuple(int(c * 255) for c in color[:3]))
    return colors


from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    revision: Optional[str]
    train_dataset: Dict
    validation_dataset: Dict
    validation_train_dataset: Dict
    output_dir: str
    seed: Optional[int]
    train_batch_size: int
    validation_batch_size: int
    validation_train_batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    snr_gamma: Optional[float]
    use_8bit_adam: bool
    allow_tf32: bool
    use_ema: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    logging_dir: str
    vis_dir: str
    mixed_precision: Optional[str]
    report_to: Optional[str]
    local_rank: int
    checkpointing_steps: int
    checkpoints_total_limit: Optional[int]
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_sanity_check: bool
    tracker_project_name: str
    trainable_modules: Optional[list]
    use_classifier_free_guidance: bool
    condition_drop_rate: float
    scale_input_latents: bool
    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float
    num_views: int
    camera_embedding_type: str
    pred_type: str
    drop_type: str
    run_name: str


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    writer = SummaryWriter(f'results/logs/{args.run_name}')

    log_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=log_config
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.

    if args.seed is not None:
        set_seed(args.seed)
    # NOTE: the training block below is intentionally always executed (seed only sets the RNG).
    if True:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        vae.load_state_dict(torch.load(os.path.join(args.pretrained_model_name_or_path, "vae", "diffusion_pytorch_model.bin")))

        vae.requires_grad_(False)        # freeze all, then unfreeze decoder only (below)
        vae_trainable_params = []
        for name, param in vae.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
                vae_trainable_params.append(param)

        print(f"VAE total params = {len(list(vae.named_parameters()))}, trainable params = {len(vae_trainable_params)}")
        if args.scale_lr:
            args.learning_rate = (
                    args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            vae_trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


        # Get the training dataset
        train_dataset = MVDiffusionDataset(
            **args.train_dataset
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        )
        # validation set (test h5)
        validation_dataset = MVDiffusionDataset(
            **args.validation_dataset
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.validation_batch_size, shuffle=False, num_workers=args.dataloader_num_workers,
        )

        lr_scheduler = get_scheduler(
            'constant',
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

        vae, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            vae, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
        weight_dtype = torch.float32
        vae.to(accelerator.device, dtype=weight_dtype)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        if accelerator.is_main_process:
            # tracker_config = dict(vars(args))
            tracker_config = {}
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(vae):
                    imgs_out = batch['imgs_out_normal']
                    bnm, Nv = imgs_out.shape[0], imgs_out.shape[1]
                    imgs_out = rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")
                    imgs_out = imgs_out.to(weight_dtype)
                    latents = vae.encode(imgs_out * 2.0 - 1.0).latent_dist.sample()
                    pred_images, pred_images_feature = vae.decode(latents, return_dict=False)[0]


                    loss = F.mse_loss(pred_images.float(), (imgs_out * 2.0 - 1.0).clamp(-1, 1).float(),
                                      reduction="mean")
                    loss1 = F.l1_loss(pred_images.float(), (imgs_out * 2.0 - 1.0).clamp(-1, 1).float(),
                                      reduction="mean")
                    loss = loss + loss1
                    print("sum loss:", loss,  "l1 rgb loss:", loss1)
                    # from skimage import io
                    # io.imshow(input_images)
                    # io.show()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                # write to tensorboard
                writer.add_scalar("loss/train", loss.detach().item(), global_step)

                # write to tensorboard
                if global_step % 10 == 0:
                    # Draw VAE decoder weights
                    weights = vae.decoder.conv_out.weight.cpu().detach().numpy()
                    weights = np.sum(weights, axis=0)
                    weights = weights.flatten()
                    plt.figure()
                    plt.plot(range(len(weights)), weights)
                    plt.title(f"VAE Decoder Weights = {np.mean(weights)}")
                    writer.add_figure('decoder_weights', plt.gcf(), global_step=global_step)

                    # Draw VAE encoder weights
                    weights = vae.encoder.conv_out.weight.cpu().detach().numpy()
                    weights = np.sum(weights, axis=0)
                    weights = weights.flatten()
                    plt.figure()
                    plt.plot(range(len(weights)), weights)
                    plt.title(f"Fixed VAE Encoder Weights= {np.mean(weights)}")
                    writer.add_figure('encoder_weights', plt.gcf(), global_step=global_step)

                if global_step == 1 or global_step % 50 == 0:
                    with torch.no_grad():
                        pred_images = inputs2img(pred_images)
                        target = inputs2img(imgs_out)
                        viz = np.concatenate([pred_images[0], target[0]], axis=2)
                        writer.add_image(f'train/pred_img', viz, global_step=global_step)

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # ---- validation on the test set ----
                if global_step % args.validation_steps == 0:
                    vae.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vbatch in validation_dataloader:
                            v_out = rearrange(vbatch['imgs_out_normal'], "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
                            v_lat = vae.encode(v_out * 2.0 - 1.0).latent_dist.sample()
                            v_pred, _ = vae.decode(v_lat, return_dict=False)[0]
                            v_imgs = (v_out * 2.0 - 1.0).clamp(-1, 1)
                            v_l = F.mse_loss(v_pred.float(), v_imgs.float()) + F.l1_loss(v_pred.float(), v_imgs.float())
                            val_losses.append(v_l.item())
                    if len(val_losses):
                        val_loss = float(np.mean(val_losses))
                        print(f"[val] step {global_step}  test loss = {val_loss:.4f}")
                        writer.add_scalar("loss/val", val_loss, global_step)
                        accelerator.log({"val_loss": val_loss}, step=global_step)
                    vae.train()

                if global_step >= args.max_train_steps:
                    break

                # save model
                if global_step % 500 == 0:
                    model_path = args.output_dir + f'/vae_normal_{epoch}.pth'
                    torch.save(vae.state_dict(), model_path)
            accelerator.wait_for_everyone()

        # save model
        if accelerator.is_main_process:
            print("Saving final model to ", args.output_dir)
            model_path = args.output_dir + f'/vae_normal_{epoch}.pth'
            torch.save(vae.state_dict(), model_path)

        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)