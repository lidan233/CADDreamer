import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
from packaging import version
import shutil
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset
# from mvdiffusion.data.objaverse_dataset import ObjaverseDataset as MVDiffusionDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from einops import rearrange
from rembg import remove
import pdb
from automl.AutoencoderKL import AutoencoderKL,ClassifyMLP
weight_dtype = torch.float16
from skimage import io
from utils.util import *
# from neus.launch import  main as neus_main
from instant_nsr_pl.launch import  main as neus_main

@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool


def log_validation(dataloader, pipeline, cfg: TestConfig, weight_dtype, name, save_dir):


    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
    
    images_cond, images_pred = [], defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader)):
        # (B, Nv, 3, H, W)
        imgs_in = batch['imgs_in']
        alphas = batch['alphas']
        # (B, Nv, Nce)
        camera_embeddings = batch['camera_embeddings']
        filename = batch['filename']

        bsz, num_views = imgs_in.shape[0], imgs_in.shape[1]
        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        alphas = rearrange(alphas, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)

        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
                cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}")

                # pdb.set_trace()
                for i in range(bsz):
                    scene = os.path.basename(filename[i])
                    print(scene)
                    scene_dir = os.path.join(cur_dir, scene)
                    outs_dir = os.path.join(scene_dir, "outs")
                    masked_outs_dir = os.path.join(scene_dir, "masked_outs")
                    os.makedirs(outs_dir, exist_ok=True)
                    os.makedirs(masked_outs_dir, exist_ok=True)
                    img_in = imgs_in[i*num_views]
                    alpha = alphas[i*num_views]
                    img_in = torch.cat([img_in, alpha], dim=0)
                    save_image(img_in, os.path.join(scene_dir, scene+".png"))
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j
                        pred = out[idx]

                        # pdb.set_trace()
                        out_filename = f"{cfg.pred_type}_000_{view}.png"
                        pred = save_image(pred, os.path.join(outs_dir, out_filename))

                        rm_pred = remove(pred)

                        save_image_numpy(rm_pred, os.path.join(scene_dir, out_filename))
    torch.cuda.empty_cache()



def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

def log_validation_joint(dataloader, pipeline, cfg: TestConfig, weight_dtype, name, save_dir):

    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
    
    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader)):
        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)

        # filename = batch['filename']
        filename = [cfg.revision]
        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)
        num_views = len(VIEWS)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                imgs_in = F.interpolate(imgs_in, size=(256, 256), mode='nearest')

                outs = pipeline(
                    imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                )
                out = outs.images
                labels = outs.labels_flags
                normals = outs.normal_images


                bsz = out.shape[0] // 2
                normals_pred = out[:bsz]
                real_normal_pred = normals[:bsz] 
                images_pred = out[bsz:]
                labels_pred = labels[bsz:]
                labels = torch.argmax(labels_pred, dim=-1)



                cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}")
                all_normals = []
                all_colors = []

                for i in range(bsz//num_views):
                    scene = filename[i]
                    scene_dir = os.path.join(cur_dir, scene)
                    normal_dir = os.path.join(scene_dir, "normals")
                    masked_colors_dir = os.path.join(scene_dir, "masked_colors")
                    os.makedirs(normal_dir, exist_ok=True)
                    os.makedirs(masked_colors_dir, exist_ok=True)
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j
                        # normal = normals_pred[idx]
                        real_normal = real_normal_pred[idx]
                        color = images_pred[idx]
                        label = labels[idx].cpu().numpy()

                        normal_filename = f"normals_000_{view}.png"
                        rgb_filename = f"rgb_000_{view}.png"
                        label_filename = f'label_000_{view}.png'
                        real_normal_filename = f'real_normal_000_{view}.png'
                        mask_filename = f'masked_colors_{view}.png'

                        # normal = normal.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        color = color.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        real_normal = real_normal.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        
                        real_mask = np.where(label == 0)
                        color[real_mask] = 0
                        real_normal[real_mask] = 0 
                        alpha_channel = np.ones((real_normal.shape[0], real_normal.shape[1], 1), dtype = 'uint8') * 255
                        alpha_channel[real_mask] = 0 
                        rm_normal = np.concatenate([real_normal, alpha_channel], axis=-1)
                        rm_color = np.concatenate([color, alpha_channel], axis=-1)
                        rm_normal = np.array(rm_normal)
                        rm_color = np.array(rm_color)
                        rm_normal[real_mask] = 0
                        rm_color[real_mask] = 0


                        # save_image_numpy(normal, os.path.join(normal_dir, normal_filename))
                        save_image_numpy(color, os.path.join(scene_dir, rgb_filename))
                        save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
                        io.imsave(os.path.join(scene_dir, mask_filename), (label*50).astype(np.uint8))
                        
                        
                        
                        save_image_numpy(rm_normal, os.path.join(normal_dir, normal_filename))
                        save_image_numpy(rm_color, os.path.join(masked_colors_dir, rgb_filename))
                        save_cache(labels[idx], os.path.join(scene_dir, label_filename))

                        all_normals.append(rm_normal)
                        all_colors.append(rm_color)
                    all_normal_image = np.concatenate(all_normals, axis=1)
                    all_color_image = np.concatenate(all_colors, axis=1)
                    save_image_numpy(all_normal_image, os.path.join(scene_dir, "all_normal.png"))
                    save_image_numpy(all_color_image, os.path.join(scene_dir, "all_colors.png"))
    torch.cuda.empty_cache()


def load_wonder3d_pipeline(cfg):

    pipeline = MVDiffusionImagePipeline.from_pretrained(
    cfg.pretrained_model_name_or_path,
    torch_dtype=weight_dtype
    )

    pipeline.mask_vae =  AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    pipeline.mask_vae =  pipeline.mask_vae.to(torch.float32)
    pipeline.mask_vae.load_state_dict(torch.load("./finetuning_vae_normal/outputs/finetune_vae/vae_16.pth"))
    pipeline.mask_vae = pipeline.mask_vae.to("cuda")
    pipeline.mlp = ClassifyMLP(128, hidden_size=256, num_classes=11)
    pipeline.mlp.load_state_dict(torch.load("./finetuning_vae_normal/outputs/finetune_vae/mlp_16.pth"))
    pipeline.mlp = pipeline.mlp.to("cuda")

    pipeline.normal_vae =  AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    pipeline.normal_vae =  pipeline.normal_vae.to(torch.float32)
    pipeline.normal_vae.load_state_dict(torch.load("./finetuning_vae_normal/outputs/finetuning_vae_normal/vae_normal_2.pth"))
    pipeline.normal_vae = pipeline.normal_vae.to("cuda")
    pipeline.unet.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline


def main(
    cfg: TestConfig
):

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    pipeline = load_wonder3d_pipeline(cfg)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print("use xformers.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the  dataset
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )


    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )


    os.makedirs(cfg.save_dir, exist_ok=True)
    log_validation_joint(
                    validation_dataloader,
                    pipeline,
                    cfg,
                    weight_dtype,
                    'validation',
                    cfg.save_dir
                    )



def maintest(
        cfg: TestConfig,
        pipeline
):
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Get the  dataset
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )

    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    # log_validation(
    #     validation_dataloader,
    #     pipeline,
    #     cfg,
    #     weight_dtype,
    #     'validation',
    #     cfg.save_dir
    # )

    log_validation_joint(
        validation_dataloader,
        pipeline,
        cfg,
        weight_dtype,
        'validation',
        cfg.save_dir
    )


if __name__ == '__main__':
    root_path = "./"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--prefix", type=str, default="deepcad")

    args, extras = parser.parse_known_args()
    from utils.misc import load_config    



    cfg = load_config(args.config, cli_args=extras)
    print(cfg)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)



    pipeline = load_wonder3d_pipeline(cfg)
    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print("use xformers.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    revision = args.idx
    cfg.revision = str(revision)+'_0'
    cfg.validation_dataset.filepath = "testnormal_"+str(cfg.revision)+".png"
    cfg.validation_dataset.root_dir = "./our_inputs/test_real_images"
    potential_image_path = os.listdir(cfg.validation_dataset.root_dir)
    gpu = str(args.gpu)
    cfg.revision += args.prefix
    if True:
        if cfg.num_views == 6:
            VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        elif cfg.num_views == 4:
            VIEWS = ['front', 'right', 'back', 'left']
        if  not os.path.exists( os.path.join(cfg.save_dir,
                                            f"cropsize-{cfg.validation_dataset.crop_size}-cfg{1.0:.1f}",
                                        str(cfg.revision))):
            maintest(cfg, pipeline)
        config_path_for_neus = os.path.join(root_path, "neus/configs/neuralangelo-ortho-wmask.yaml")
        input_root_dir = os.path.join(root_path, "test_outputs2")
        project_root = os.path.dirname(os.path.abspath(__file__))
        input_root_dir = os.path.join(project_root, "test_outputs")
        input_dir_for_neus = os.path.join(input_root_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{1.0:.1f}", str(cfg.revision))
        if  os.path.exists(os.path.join(input_dir_for_neus,  "False_mm.obj")):
            print(os.path.join(input_dir_for_neus,  "False_mm.obj"), " is existing. ")
        merge_cfg = {"gpu" : gpu, "config" : config_path_for_neus, "dataset": [
            "dataset.root_dir="+str(input_root_dir),
            "dataset.scene="+input_dir_for_neus
        ]}
        print(merge_cfg)
        neus_main(merge_cfg)

       