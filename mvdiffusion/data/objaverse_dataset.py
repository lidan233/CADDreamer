import glob
from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math
import trimesh 

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb


import h5py
import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb
import itertools


class ObjaverseDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 num_views: int,
                 bg_color: Any,
                 img_wh: Tuple[int, int],
                 groups_num: int = 1,
                 validation: bool = False,
                 data_view_num: int = 6,
                 num_validation_samples: int = 1000,
                 num_samples: Optional[int] = None,
                 trans_norm_system: bool = True,  # if True, transform all normals map into the cam system of front view
                 augment_data: bool = False,
                 read_normal: bool = True,
                 read_color: bool = False,
                 read_depth: bool = False,
                 read_mask: bool = True,
                 mix_color_normal: bool = False,
                 suffix: str = 'png',
                 subscene_tag: int = 3,
                 crop_size: int=256 ,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.bg_color = bg_color
        self.validation = validation
        self.num_samples = num_samples
        self.trans_norm_system = trans_norm_system
        self.augment_data = augment_data
        self.groups_num = groups_num
        print("augment data: ", self.augment_data)
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_cmask = read_color
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.mix_color_normal = mix_color_normal  # mix load color and normal maps
        self.suffix = suffix
        self.subscene_tag = subscene_tag

        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"

        self.fix_cam_poses = self.load_fixed_poses()
        from tqdm import tqdm 

        self.file_objs = [h5py.File(os.path.join(self.root_dir, file), 'r') for file in tqdm(os.listdir(self.root_dir))]
        file_objs_sizes = [self.file_objs[i][key].shape[0] for i in tqdm(range(len(self.file_objs))) for key in
                           self.file_objs[i].keys() if 'normals_000_back' in key]
        self.file_objs_keys = [key.split('normals')[0] for i in tqdm(range(len(self.file_objs))) for key in
                               self.file_objs[i].keys() if 'normals_000_back' in key]
        self.file_objs_input = [self.file_objs[i] for i in tqdm(range(len(self.file_objs))) for key in
                                self.file_objs[i].keys() if 'normals_000_back' in key]
        self.file_objs_sizes_cum = [0] + list(itertools.accumulate(file_objs_sizes))

        self.objects = np.zeros(self.file_objs_sizes_cum[-1], dtype=np.int32)
        for i in range(len(self.file_objs_sizes_cum) - 1):
            self.objects[self.file_objs_sizes_cum[i]:self.file_objs_sizes_cum[i + 1]] = i

        self.all_objects = list(self.objects)
        if not validation:
            self.all_objects = self.all_objects[:-num_validation_samples]
            self.start_index = 0
        else:
            self.all_objects = self.all_objects[-num_validation_samples:]
            self.start_index = self.file_objs_sizes_cum[self.all_objects[-num_validation_samples]]
        if num_samples is not None:
            self.all_objects = self.all_objects[:num_samples]
            self.start_index = 0
        print("loading ", len(self.all_objects), " objects in the dataset")
        if self.mix_color_normal:
            self.backup_data = self.__getitem_mix__(0)
        else:
            self.backup_data = self.__getitem_joint__(0)

    def __len__(self):
        return len(self.objects) * self.total_view

    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(os.path.join(self.fix_cam_pose_dir, '%03d_%s_RT.txt' % (0, face)))
            poses[face] = RT
        return poses

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T  # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_mask(self, img_data, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(img_data.resize(self.img_wh))
        img = np.float32(img > 0)

        assert len(np.shape(img)) == 2

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_image(self, img_data, bg_color, alpha, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(img_data.resize(self.img_wh, resample=Image.NEAREST))
        img = img.astype(np.float32) / 255.  # [0, 1]
        assert img.shape[-1] == 3 or img.shape[-1] == 4  # RGB or RGBA

        if alpha is None and img.shape[-1] == 4:
            alpha = img[:, :, 3:]
            img = img[:, :, :3]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_depth(self, img_path, bg_color, alpha, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 65535.  # [0, 1]

        img[img > 0.4] = 0
        img = img / 0.4

        assert img.ndim == 2  # depth
        img = np.stack([img] * 3, axis=-1)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        # print(np.max(img[:, :, 0]))

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_normal(self, img_data, bg_color, alpha, RT_w2c=None, RT_w2c_cond=None, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(img_data.resize(self.img_wh, resample=Image.NEAREST))

        assert normal.shape[-1] == 3 or normal.shape[-1] == 4  # RGB or RGBA

        if normal.shape[-1] == 4:
            alpha = normal[:, :, 3:] / 255.
            normal = normal[:, :, :3]

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)

        img = (normal * 0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        # from skimage import io
        # io.imshow(img.cpu().numpy())
        # io.show()
        return img

    def __len__(self):
        return len(self.all_objects)

    def __getitem_mix__(self, index, debug_object=None, choose_normal_only=False, choose_mask_only=False):
        if debug_object is not None:
            object_name = debug_object  #
            set_idx = random.sample(range(0, self.groups_num), 1)[0]  # without replacement
        else:
            object_name = self.all_objects[index % len(self.all_objects)]
            set_idx = 0

        if self.augment_data:
            cond_view = random.sample(self.view_types, k=1)[0]
        else:
            cond_view = 'front'


        view_types = self.view_types

        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in view_types]

        elevations = []
        azimuths = []

        # get the bg color
        bg_color = self.get_bg_color()

        current_file_obj = self.file_objs_input[self.all_objects[index]]
        current_file_obj_index = index - self.file_objs_sizes_cum[self.all_objects[index]] + self.start_index
        current_prefix = self.file_objs_keys[self.all_objects[index]]
        if self.read_mask:
            mask_key = "normals_%03d_%s.%s" % (set_idx, cond_view, self.suffix)
            real_key = current_prefix + mask_key
            mask_data = current_file_obj[real_key][current_file_obj_index][:, :, -1]
        

            pil_image = Image.fromarray(mask_data)
            cond_alpha = self.load_mask(
                pil_image,
                return_type='np')
        else:
            cond_alpha = None

        normal_key = "normals_%03d_%s.%s" % (set_idx, cond_view, self.suffix)
        normal_key = current_prefix + normal_key
        normal_data = current_file_obj[normal_key][current_file_obj_index]
        pil_image = Image.fromarray(normal_data)

        # print("input data begin process")
        img_tensors_in = [
                             self.load_normal(pil_image,
                                              bg_color,
                                              cond_alpha,
                                              RT_w2c=self.fix_cam_poses[cond_view], RT_w2c_cond=cond_w2c,
                                              return_type='pt').permute(2, 0, 1)
                         ] * self.num_views
        img_tensors_normal_out = []
        img_tensors_mask_out = []
        # print("output data begin process")

        current_name = current_file_obj[current_prefix+'name'][current_file_obj_index]
        current_vertices = current_file_obj[current_file_obj[current_prefix+'name'][current_file_obj_index][0].decode('utf-8') + '_vertices']
        current_faces = current_file_obj[current_file_obj[current_prefix+'name'][current_file_obj_index][0].decode('utf-8') + '_faces']
        current_mesh = trimesh.Trimesh(vertices=current_vertices[0], faces=current_faces[0], process=False)
        current_rotate = current_file_obj[current_prefix+'rotate.txt'][current_file_obj_index]

        if index == 10:
            print("asdf")
        for view, tgt_w2c in zip(view_types, tgt_w2cs):
            normal_key = "normals_%03d_%s.%s" % (set_idx, view, self.suffix)
            normal_key = current_prefix + normal_key
            mask_key = normal_key


            cmask_key = "cmask_%03d_%s.%s" % (set_idx, view, self.suffix)
            cmask_key = current_prefix + cmask_key


            mask_data = current_file_obj[mask_key][current_file_obj_index][:, :, -1]
            pil_image = Image.fromarray(mask_data)
            alpha = self.load_mask(pil_image, return_type='np')


            cmask_data = current_file_obj[cmask_key][current_file_obj_index]
            pil_image = Image.fromarray(cmask_data)
            img_tensor = self.load_image(pil_image, bg_color, alpha, return_type="pt")
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensors_mask_out.append(img_tensor)

            normal_data = current_file_obj[normal_key][current_file_obj_index]
            pil_image = Image.fromarray(normal_data)
            normal_tensor = self.load_normal(pil_image, bg_color, alpha, RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c,
                                             return_type="pt").permute(2, 0, 1)
            img_tensors_normal_out.append(normal_tensor)
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 4 views to train
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1)  # (Nv, 3)


        img_tensors_mask_out =    torch.stack(img_tensors_mask_out, dim=0).float()  # (Nv, 3, H, W)
        img_tensors_normal_out =    torch.stack(img_tensors_normal_out, dim=0).float()  # (Nv, 3, H, W)
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class] * self.num_views, dim=0)  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_views, dim=0)  # (Nv, 2)


        meta_data = {
            'step_name': current_name,
            'step_mesh': current_mesh, 
            'step_rotate': current_rotate,
        }


        if choose_normal_only:
            return {
                'elevations_cond': elevations_cond,
                'elevations_cond_deg': torch.rad2deg(elevations_cond),
                'elevations': elevations,
                'azimuths': azimuths,
                'elevations_deg': torch.rad2deg(elevations),
                'azimuths_deg': torch.rad2deg(azimuths),
                'imgs_in': img_tensors_in,
                'imgs_out': img_tensors_normal_out,
                'camera_embeddings': camera_embeddings,
                'task_embeddings': normal_task_embeddings,
                'meta_data': meta_data
            }

        if choose_mask_only:
            return {
                'elevations_cond': elevations_cond,
                'elevations_cond_deg': torch.rad2deg(elevations_cond),
                'elevations': elevations,
                'azimuths': azimuths,
                'elevations_deg': torch.rad2deg(elevations),
                'azimuths_deg': torch.rad2deg(azimuths),
                'imgs_in': img_tensors_in,
                'imgs_out': img_tensors_mask_out,
                'camera_embeddings': camera_embeddings,
                'task_embeddings': color_task_embeddings,
                'meta_data': meta_data
            }

        if random.random() < 0.7:
            return {
                'elevations_cond': elevations_cond,
                'elevations_cond_deg': torch.rad2deg(elevations_cond),
                'elevations': elevations,
                'azimuths': azimuths,
                'elevations_deg': torch.rad2deg(elevations),
                'azimuths_deg': torch.rad2deg(azimuths),
                'imgs_in': img_tensors_in,
                'imgs_out': img_tensors_mask_out,
                'camera_embeddings': camera_embeddings,
                'task_embeddings': color_task_embeddings,
                'meta_data': meta_data
            }
        else:
            return {
                'elevations_cond': elevations_cond,
                'elevations_cond_deg': torch.rad2deg(elevations_cond),
                'elevations': elevations,
                'azimuths': azimuths,
                'elevations_deg': torch.rad2deg(elevations),
                'azimuths_deg': torch.rad2deg(azimuths),
                'imgs_in': img_tensors_in,
                'imgs_out': img_tensors_normal_out,
                'camera_embeddings': camera_embeddings,
                'task_embeddings': normal_task_embeddings,
                'meta_data': meta_data
            }


    def __getitem_joint__(self, index, debug_object=None):
        if debug_object is not None:
            object_name = debug_object  #
            set_idx = random.sample(range(0, self.groups_num), 1)[0]  # without replacement
        else:
            object_name = self.all_objects[index % len(self.all_objects)]
            set_idx = 0

        if self.augment_data:
            cond_view = random.sample(self.view_types, k=1)[0]
        else:
            cond_view = 'front'

        view_types = self.view_types
        cond_w2c = self.fix_cam_poses[cond_view]
        tgt_w2cs = [self.fix_cam_poses[view] for view in view_types]
        elevations = []
        azimuths = []

        # get the bg color
        bg_color = self.get_bg_color()

        current_file_obj = self.file_objs_input[self.all_objects[index]]
        current_file_obj_index = index - self.file_objs_sizes_cum[self.all_objects[index]] + self.start_index
        current_prefix = self.file_objs_keys[self.all_objects[index]]

        mask_key = "normals_%03d_%s.%s" % (set_idx, cond_view, self.suffix)
        real_key = current_prefix + mask_key
        mask_data = current_file_obj[real_key][current_file_obj_index][:, :, -1]
        pil_image = Image.fromarray(mask_data)
        cond_alpha = self.load_mask(
                pil_image,
                return_type='np')


        normal_key = "normals_%03d_%s.%s" % (set_idx, cond_view, self.suffix)
        real_key = current_prefix + normal_key
        normal_data = current_file_obj[real_key][current_file_obj_index]
        pil_image = Image.fromarray(normal_data)
        img_tensors_in = [
                             self.load_normal(pil_image,
                                              bg_color,
                                              cond_alpha,
                                              RT_w2c=self.fix_cam_poses[cond_view], RT_w2c_cond=cond_w2c,
                                              return_type='pt').permute(2, 0, 1)
                         ] * self.num_views


        img_tensors_out = []
        normal_tensors_out = []

        for view, tgt_w2c in zip(view_types, tgt_w2cs):
            normal_key = "normals_%03d_%s.%s" % (set_idx, view, self.suffix)
            normal_key = current_prefix + normal_key
            mask_key = normal_key
            cmask_key = "cmask_%03d_%s.%s" % (set_idx, view, self.suffix)
            cmask_key = current_prefix + cmask_key


            mask_data = current_file_obj[mask_key][current_file_obj_index][:, :, -1]
            pil_image = Image.fromarray(mask_data)
            alpha = self.load_mask(pil_image, return_type='np')


            cmask_data = current_file_obj[cmask_key][current_file_obj_index]
            pil_image = Image.fromarray(cmask_data)
            img_tensor = self.load_image(pil_image, bg_color, alpha, return_type="pt")
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensors_out.append(img_tensor)


            normal_data = current_file_obj[normal_key][current_file_obj_index]
            pil_image = Image.fromarray(normal_data)
            normal_tensor = self.load_normal(pil_image, bg_color, alpha, RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c,
                                                 return_type="pt").permute(2, 0, 1)
            normal_tensors_out.append(normal_tensor)

            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float()  # (Nv, 3, H, W)
        normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float()  # (Nv, 3, H, W)

        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 4 views to train

        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1)  # (Nv, 3)
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class] * self.num_views, dim=0)  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_views, dim=0)  # (Nv, 2)


        return {
            'elevations_cond': elevations_cond,
            'elevations_cond_deg': torch.rad2deg(elevations_cond),
            'elevations': elevations,
            'azimuths': azimuths,
            'elevations_deg': torch.rad2deg(elevations),
            'azimuths_deg': torch.rad2deg(azimuths),
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'normals_out': normal_tensors_out,
            'camera_embeddings': camera_embeddings,
            'normal_task_embeddings': normal_task_embeddings,
            'color_task_embeddings': color_task_embeddings,
            "filename" : str(index)
        }

    def __getitem__(self, index):
        try:
            if self.mix_color_normal:
                data = self.__getitem_mix__(index)
            else:
                data = self.__getitem_joint__(index)
            return data
        except:
            print("load error ", self.all_objects[index % len(self.all_objects)])
            return self.backup_data


if __name__ == "__main__":
    train_dataset = ObjaverseDataset(
        root_dir="/ghome/l5/xxlong/.objaverse/hf-objaverse-v1/renderings",
        size=(128, 128),
        ext="hdf5",
        default_trans=torch.zeros(3),
        return_paths=False,
        total_view=8,
        validation=False,
        object_list=None,
        views_mode='fourviews'
    )
    data0 = train_dataset[0]
    data1 = train_dataset[50]
    # print(data)
