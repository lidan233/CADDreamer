# import sys
# sys.path.append("/neus")
# import os

# import json
# import math
# import numpy as np
# from PIL import Image
# import cv2

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, IterableDataset
# import torchvision.transforms.functional as TF

# import pytorch_lightning as pl

# import datasets
# from models.ray_utils import get_ortho_ray_directions_origins, get_ortho_rays, get_ray_directions
# from utils.misc import get_rank

# from glob import glob
# import PIL.Image


# def camNormal2worldNormal(rot_c2w, camNormal):
#     H,W,_ = camNormal.shape
#     normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

#     return normal_img

# def worldNormal2camNormal(rot_w2c, worldNormal):
#     H,W,_ = worldNormal.shape
#     normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

#     return normal_img

# def trans_normal(normal, RT_w2c, RT_w2c_target):

#     normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
#     normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

#     return normal_target_cam

# def img2normal(img):
#     return (img/255.)*2-1

# def normal2img(normal):
#     return np.uint8((normal*0.5+0.5)*255)

# def norm_normalize(normal, dim=-1):

#     normal = normal/(np.linalg.norm(normal, axis=dim, keepdims=True)+1e-6)

#     return normal

# def RT_opengl2opencv(RT):
#      # Build the coordinate transform matrix from world to computer vision camera
#     # R_world2cv = R_bcam2cv@R_world2bcam
#     # T_world2cv = R_bcam2cv@T_world2bcam

#     R = RT[:3, :3]
#     t = RT[:3, 3]

#     R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

#     R_world2cv = R_bcam2cv @ R
#     t_world2cv = R_bcam2cv @ t

#     RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

#     return RT

# def normal_opengl2opencv(normal):
#     H,W,C = np.shape(normal)
#     # normal_img = np.reshape(normal, (H*W,C))
#     R_bcam2cv = np.array([1, -1, -1], np.float32)
#     normal_cv = normal * R_bcam2cv[None, None, :]

#     print(np.shape(normal_cv))

#     return normal_cv

# def inv_RT(RT):
#     RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
#     RT_inv = np.linalg.inv(RT_h)

#     return RT_inv[:3, :]


# def calculate_probabilities_along_dimension(distances, axis):
#     neg_distances = -distances
#     exp_distances = np.exp(neg_distances)
#     sum_exp_distances = np.sum(exp_distances, axis=axis, keepdims=True)
#     probabilities = exp_distances / (sum_exp_distances + 1e-5)
#     return probabilities


# from skimage import io
# # def show_label_mask(mask, label):
# #     img = mask == label
# #     io.imshow(img.astype(np.uint8))
# #     io.show()
# def connect_components_analysis(all_masks, all_semantics):
#     components = []
#     mask_comps = []

#     for i in range(len(all_semantics)):
#         keys, values = [], []
#         record_bound = []
#         prob = all_semantics[i]
#         img = prob.argmax(axis=-1)
#         from skimage import io
#         # io.imshow(img)
#         # io.show()

#         cmask = all_masks[i] == 0
#         img[np.where(cmask)] = 255
#         img = img.astype(np.uint8)

#         instance = set(img.reshape(-1))
#         instance.remove(255)


#         for ins in instance:
#             num_labels, labeled_img = cv2.connectedComponents((img == ins).astype(np.uint8))
#             delete_labels = set(labeled_img[np.where(cmask)].tolist())
#             conn_comp = set(labeled_img.reshape(-1)) - delete_labels

#             for comp in conn_comp:
#                 comp_single_img = (labeled_img == comp).astype(np.uint8)
#                 contours, _ = cv2.findContours(comp_single_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#                 # show_label_mask(comp_single_img, 1)
#                 perimeter = [cv2.arcLength(contour, closed=True) for contour in contours]
#                 area =  [comp_single_img.sum()]

#                 if area[0] < 100:
#                     img[np.where(labeled_img == comp)] = 8
#                     continue

#                 weight = np.sum(perimeter) / np.sum(area)
#                 if weight > 0.3:
#                     img[np.where(labeled_img == comp)] = 8
#                     continue
#                 print(ins, comp, perimeter, area, weight)
#                 keys.append(weight)
#                 values.append([i, ins, comp, labeled_img, contours])

#         new_keys = [keys[t] for t in np.argsort(keys).tolist()]
#         new_values = [values[t] for t in np.argsort(keys).tolist()]

#         for weight, value in zip(new_keys, new_values):
#             i, ins, comp, labeled_img, contours = value
#             from scipy import stats
#             mode_result = stats.mode(img[np.where(labeled_img == comp)])[0]
#             label_group = img[np.where(labeled_img == comp)]
#             count_edge = np.count_nonzero(label_group == 8)


#             if count_edge / len(label_group) < 0.1:
#                 cv2.drawContours(img, contours, -1, (8,) , 1)
#             else:
#                 img[np.where(labeled_img == comp)] = 8

#         components.append(img)

#         gt = np.copy(img)
#         gt[np.where(gt == 255)] = 5
#         from skimage import io
#         # io.imshow(gt )
#         # io.show()


#         int_mask = all_masks[i]
#         int_mask[np.where(img == 8)] = True
#         mask_comps.append(int_mask)


#     return components, mask_comps







# def connect_components_analysis_single(all_masks, all_semantics):
#     components = []
#     mask_comps = []

#     for i in range(len(all_semantics)):
#         keys, values = [], []
#         record_bound = []
#         img = all_semantics[i]
#         from skimage import io
#         # io.imshow(img)
#         # io.show()

#         cmask = all_masks[i] == 0
#         img[np.where(cmask)] = 255
#         img = img.astype(np.uint8)

#         instance = set(img.reshape(-1))
#         instance.remove(255)


#         for ins in instance:
#             num_labels, labeled_img = cv2.connectedComponents((img == ins).astype(np.uint8))
#             delete_labels = set(labeled_img[np.where(cmask)].tolist())
#             conn_comp = set(labeled_img.reshape(-1)) - delete_labels

#             for comp in conn_comp:
#                 comp_single_img = (labeled_img == comp).astype(np.uint8)
#                 contours, _ = cv2.findContours(comp_single_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#                 # show_label_mask(comp_single_img, 1)
#                 perimeter = [cv2.arcLength(contour, closed=True) for contour in contours]
#                 area =  [comp_single_img.sum()]

#                 # if area[0] < 100:
#                 #     img[np.where(labeled_img == comp)] = 8
#                 #     continue

#                 weight = np.sum(perimeter) / np.sum(area)
#                 if weight > 0.3:
#                     img[np.where(labeled_img == comp)] = 8
#                     continue
#                 print(ins, comp, perimeter, area, weight)
#                 keys.append(weight)
#                 values.append([i, ins, comp, labeled_img, contours])

#         new_keys = [keys[t] for t in np.argsort(keys).tolist()]
#         new_values = [values[t] for t in np.argsort(keys).tolist()]

#         for weight, value in zip(new_keys, new_values):
#             i, ins, comp, labeled_img, contours = value
#             from scipy import stats
#             mode_result = stats.mode(img[np.where(labeled_img == comp)])[0]
#             label_group = img[np.where(labeled_img == comp)]
#             count_edge = np.count_nonzero(label_group == 8)

#             cv2.drawContours(img, contours, -1, (8,), 1)
#             # if count_edge / len(label_group) < 0.1:
#             #     cv2.drawContours(img, contours, -1, (8,) , 8)
#             # else:
#             #     img[np.where(labeled_img == comp)] = 8

#         components.append(img)

#         gt = np.copy(img)
#         gt[np.where(gt == 255)] = 5
#         from skimage import io
#         # io.imshow(gt )
#         # io.show()


#         # int_mask = all_masks[i]
#         # int_mask[np.where(img == 8)] = True
#         # mask_comps.append(int_mask)


#     return components, mask_comps







# import dill
# import pickle
# def save_cache(obj, path):
#     with open(path, "wb") as f:
#         pickle.dump(obj, f)

# def save_cache_dill(obj, path):
#     with open(path,'wb') as f:
#         dill.dump(obj,f)

# def load_cache_dill( path):
#     with open(path, 'rb') as f:
#         return dill.load(f)



# def load_a_prediction(root_dir, test_object, imSize, view_types, load_color=False, cam_pose_dir=None,
#                          normal_system='front', erode_mask=True, camera_type='ortho', cam_params=None):

#     all_images = []
#     all_normals = []
#     all_normals_world = []
#     all_masks = []
#     all_color_masks = []
#     all_poses = []
#     all_w2cs = []
#     directions = []
#     ray_origins = []

#     RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt'%( 'front')))[0])   # world2cam matrix
#     RT_front_cv = RT_opengl2opencv(RT_front)   # convert normal from opengl to opencv
#     for idx, view in enumerate(view_types):
#         print(os.path.join(root_dir,test_object))
#         normal_filepath = os.path.join(root_dir, test_object, 'normals_000_%s.png'%( view))
#         # Load key frame
#         if load_color:  # use bgr
#             image =np.array(PIL.Image.open(normal_filepath.replace("normals", "rgb")).resize(imSize))[:, :, :3]

#         normal = np.array(PIL.Image.open(normal_filepath).resize(imSize))
#         mask = normal[:, :, 3]
#         normal = normal[:, :, :3]

#         color_mask = np.array(PIL.Image.open(os.path.join(root_dir,test_object, 'masked_colors/rgb_000_%s.png'%( view))).resize(imSize))[:, :, 3]
#         invalid_color_mask = color_mask < 255*0.5
#         threshold =  np.ones_like(image[:, :, 0]) * 250
#         invalid_white_mask = (image[:, :, 0] > threshold) & (image[:, :, 1] > threshold) & (image[:, :, 2] > threshold)
#         invalid_color_mask_final = invalid_color_mask & invalid_white_mask
#         color_mask = (1 - invalid_color_mask_final) > 0

#         # if erode_mask:
#         #     kernel = np.ones((3, 3), np.uint8)
#         #     mask = cv2.erode(mask, kernel, iterations=1)

#         RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt'%( view)))  # world2cam matrix

#         normal = img2normal(normal)

#         normal[mask==0] = [0,0,0]
#         mask = mask> (0.5*255)
#         if load_color:
#             all_images.append(image)
        
#         all_masks.append(mask)
#         all_color_masks.append(color_mask)
#         RT_cv = RT_opengl2opencv(RT)   # convert normal from opengl to opencv
#         all_poses.append(inv_RT(RT_cv))   # cam2world
#         all_w2cs.append(RT_cv)

#         # whether to 
#         normal_cam_cv = normal_opengl2opencv(normal)

#         if normal_system == 'front':
#             print("the loaded normals are defined in the system of front view")
#             normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], normal_cam_cv)
#         elif normal_system == 'self':
#             print("the loaded normals are in their independent camera systems")
#             normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)
#         all_normals.append(normal_cam_cv)
#         all_normals_world.append(normal_world)

#         if camera_type == 'ortho':
#             origins, dirs = get_ortho_ray_directions_origins(W=imSize[0], H=imSize[1])
#         elif camera_type == 'pinhole':
#             dirs = get_ray_directions(W=imSize[0], H=imSize[1],
#                                                  fx=cam_params[0], fy=cam_params[1], cx=cam_params[2], cy=cam_params[3])
#             origins = dirs # occupy a position
#         else:
#             raise Exception("not support camera type")
#         ray_origins.append(origins)
#         directions.append(dirs)
        
        
#         if not load_color:
#             all_images = [normal2img(x) for x in all_normals_world]


#     return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), \
#         np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs), np.stack(ray_origins), np.stack(directions), np.stack(all_color_masks)


# # #
# # def load_a_prediction(root_dir, test_object, imSize, view_types, load_color=False, cam_pose_dir=None,
# #                          normal_system='front', erode_mask=True, camera_type='ortho', cam_params=None):

# #     all_images = []
# #     all_normals = []
# #     all_normals_world = []
# #     all_masks = []
# #     all_color_masks = []
# #     all_poses = []
# #     all_w2cs = []
# #     all_semantics = []
# #     temp_semantics = []
# #     all_semantics_imgs = []

# #     directions = []
# #     ray_origins = []

# #     mask_colors = generate_distinct_colors(7)
# #     mask_colors.append((0, 255, 0))


# #     RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt'%( 'front')))[0])   # world2cam matrix
# #     RT_front_cv = RT_opengl2opencv(RT_front)   # convert normal from opengl to opencv
# #     for idx, view in enumerate(view_types):
# #         mask_filepath = os.path.join(root_dir, test_object, 'label_000_%s.png' % (view))
# #         semantic_mask = np.array(PIL.Image.fromarray(load_cache_dill(mask_filepath).cpu().numpy().astype(np.uint8)).resize(imSize, resample=PIL.Image.Resampling.NEAREST))
# #         temp_semantics.append(semantic_mask)

# #         normal_filepath = os.path.join(root_dir, test_object, 'normals_000_%s.png'%( view))
# #         if load_color:  # use bgr
# #             image =np.array(PIL.Image.open(normal_filepath.replace("normals", "rgb")).resize(imSize, resample =PIL.Image.Resampling.NEAREST))[:, :, :3]


# #         normal = np.array(PIL.Image.open(normal_filepath).resize(imSize, resample =PIL.Image.Resampling.NEAREST))
# #         mask = normal[:, :, 3]
# #         normal = normal[:, :, :3]
# #         color_mask = (semantic_mask!=0) & (mask>0)

# #         mask = np.zeros_like(color_mask).astype(np.uint8)
# #         mask[np.where(color_mask)] = 255
# #         mask = mask.astype(np.uint8) > 0
# #         color_mask  = mask

# #         RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt'%( view)))  # world2cam matrix
# #         normal = img2normal(normal)
# #         normal[mask==0] = [0,0,0]
# #         if load_color:
# #             all_images.append(image)



# #         distances = []
# #         for i in range(len(mask_colors)):
# #             colordis  = image.astype(np.float32) - np.array(mask_colors[i]).reshape(1,1,-1).astype(np.float32)
# #             distances.append(colordis)
# #         abs_dis = [np.abs(dis).mean(axis=-1).astype(np.uint8) for dis in distances]

# #         # ttc = 0
# #         # fig, axs = plt.subplots(2, 4, figsize=(12, 6))
# #         # titles = ['boundary', 'bspline', 'boundary', 'Cone', 'torus', 'sphere', 'Cylinder', 'Plane']
# #         # for color, adis in zip(mask_colors[::-1], abs_dis[::-1]):
# #         #     # plt.imshow(adis, cmap='RdYlGn', interpolation='nearest',  vmin=0, vmax=255)
# #         #     # plt.savefig('/mnt/c/Users/Admin/Desktop/'+str(ttc)+'.png')
# #         #     # ttc += 1
# #         #     axs.flat[ttc].imshow(adis, cmap='RdYlGn', interpolation='nearest',  vmin=0, vmax=255)
# #         #     axs.flat[ttc].axis('off')  # 关闭坐标轴
# #         #     axs.flat[ttc].set_title(titles[ttc])  # 设置标题
# #         #     ttc += 1
# #         # plt.tight_layout()
# #         # plt.show()

# #         probs = calculate_probabilities_along_dimension(np.stack(abs_dis).astype(np.float32), 0)
# #         probs = probs.transpose(1,2,0)
# #         all_semantics.append(probs)



# #         all_masks.append(mask)
# #         all_color_masks.append(color_mask)
# #         RT_cv = RT_opengl2opencv(RT)   # convert normal from opengl to opencv
# #         all_poses.append(inv_RT(RT_cv))   # cam2world
# #         all_w2cs.append(RT_cv)

# #         # whether to
# #         normal_cam_cv = normal_opengl2opencv(normal)

# #         if normal_system == 'front':
# #             print("the loaded normals are defined in the system of front view")
# #             normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], normal_cam_cv)
# #         elif normal_system == 'self':
# #             print("the loaded normals are in their independent camera systems")
# #             normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)
# #         all_normals.append(normal_cam_cv)
# #         all_normals_world.append(normal_world)


# #         if camera_type == 'ortho':
# #             origins, dirs = get_ortho_ray_directions_origins(W=imSize[0], H=imSize[1])
# #         elif camera_type == 'pinhole':
# #             dirs = get_ray_directions(W=imSize[0], H=imSize[1],
# #                                                  fx=cam_params[0], fy=cam_params[1], cx=cam_params[2], cy=cam_params[3])
# #             origins = dirs # occupy a position
# #         else:
# #             raise Exception("not support camera type")
# #         ray_origins.append(origins)
# #         directions.append(dirs)


# #     if not load_color:
# #         all_images = [normal2img(x) for x in all_normals_world]
# #         # all_gts, all_masks = connect_components_analysis(all_masks, all_semantics)
# #     all_gts, _ = connect_components_analysis_single(all_masks, temp_semantics)
# #     # all_color_masks = all_masks


# #     return np.stack(all_normals), np.stack(all_masks), np.stack(all_normals), \
# #         np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs), \
# #            np.stack(ray_origins), np.stack(directions), np.stack(all_color_masks), \
# #         np.stack(all_gts)



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import to_rgba
# def generate_distinct_colors(num_colors):
#     base_colors = plt.cm.tab10.colors
#     colors = []
#     for i in range(num_colors):
#         color = to_rgba(base_colors[i % len(base_colors)])
#         colors.append(tuple(int(c * 255) for c in color[:3]))
#     return colors



# class OrthoDatasetBase():
#     def setup(self, config, split):
#         self.config = config
#         self.split = split
#         self.rank = get_rank()

#         self.data_dir = self.config.root_dir
#         self.object_name = self.config.scene
#         self.scene = self.config.scene
#         self.imSize = self.config.imSize
#         self.load_color = True
#         self.img_wh = [self.imSize[0], self.imSize[1]]
#         self.w = self.img_wh[0]
#         self.h = self.img_wh[1]
#         self.camera_type = self.config.camera_type
#         self.camera_params = self.config.camera_params  # [fx, fy, cx, cy]
        
#         self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

#         self.view_weights = torch.from_numpy(np.array(self.config.view_weights)).float().to(self.rank).view(-1)
#         self.view_weights = self.view_weights.view(-1,1,1).repeat(1, self.h, self.w)

#         if self.config.cam_pose_dir is None:
#             self.cam_pose_dir = "./neus/datasets/fixed_poses"
#         else:
#             self.cam_pose_dir = self.config.cam_pose_dir
            
#         self.images_np, self.masks_np, self.normals_cam_np, self.normals_world_np, \
#             self.pose_all_np, self.w2c_all_np, self.origins_np, self.directions_np, self.rgb_masks_np, self.images_semantic = load_a_prediction(
#                 self.data_dir, self.object_name, self.imSize, self.view_types,
#                 self.load_color, self.cam_pose_dir, normal_system='front', 
#                 camera_type=self.camera_type, cam_params=self.camera_params)

#         self.has_mask = True
#         self.apply_mask = self.config.apply_mask

#         self.all_c2w = torch.from_numpy(self.pose_all_np)
#         self.all_images = torch.from_numpy(self.images_np) / 255.
#         self.all_fg_masks = torch.from_numpy(self.masks_np)
#         self.all_rgb_masks = torch.from_numpy(self.rgb_masks_np)
#         self.all_semantics = torch.from_numpy(self.images_semantic)
#         self.all_normals_world = torch.from_numpy(self.normals_world_np)
#         self.origins = torch.from_numpy(self.origins_np)
#         self.directions = torch.from_numpy(self.directions_np)

#         self.directions = self.directions.float().to(self.rank)
#         self.origins = self.origins.float().to(self.rank)
#         self.all_rgb_masks = self.all_rgb_masks.float().to(self.rank)
#         self.all_c2w, self.all_images, self.all_fg_masks, self.all_normals_world = \
#             self.all_c2w.float().to(self.rank), \
#             self.all_images.float().to(self.rank), \
#             self.all_fg_masks.float().to(self.rank), \
#             self.all_normals_world.float().to(self.rank)
#         self.all_semantics = self.all_semantics.float().to(self.rank)

        

# class OrthoDataset(Dataset, OrthoDatasetBase):
#     def __init__(self, config, split):
#         self.setup(config, split)

#     def __len__(self):
#         return len(self.all_images)
    
#     def __getitem__(self, index):
#         return {
#             'index': index
#         }


# class OrthoIterableDataset(IterableDataset, OrthoDatasetBase):
#     def __init__(self, config, split):
#         self.setup(config, split)

#     def __iter__(self):
#         while True:
#             yield {}


# @datasets.register('ortho')
# class OrthoDataModule(pl.LightningDataModule):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
    
#     def setup(self, stage=None):
#         if stage in [None, 'fit']:
#             self.train_dataset = OrthoIterableDataset(self.config, 'train')
#         if stage in [None, 'fit', 'validate']:
#             self.val_dataset = OrthoDataset(self.config, self.config.get('val_split', 'train'))
#         if stage in [None, 'test']:
#             self.test_dataset = OrthoDataset(self.config, self.config.get('test_split', 'test'))
#         if stage in [None, 'predict']:
#             self.predict_dataset = OrthoDataset(self.config, 'train')    

#     def prepare_data(self):
#         pass
    
#     def general_loader(self, dataset, batch_size):
#         sampler = None
#         return DataLoader(
#             dataset, 
#             num_workers=os.cpu_count(), 
#             batch_size=batch_size,
#             pin_memory=True,
#             sampler=sampler
#         )
    
#     def train_dataloader(self):
#         return self.general_loader(self.train_dataset, batch_size=1)

#     def val_dataloader(self):
#         return self.general_loader(self.val_dataset, batch_size=1)

#     def test_dataloader(self):
#         return self.general_loader(self.test_dataset, batch_size=1) 

#     def predict_dataloader(self):
#         return self.general_loader(self.predict_dataset, batch_size=1)       
import os
import json
import math
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ortho_ray_directions_origins, get_ortho_rays, get_ray_directions
from utils.misc import get_rank

from glob import glob
import PIL.Image


def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def worldNormal2camNormal(rot_w2c, worldNormal):
    H,W,_ = worldNormal.shape
    normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def trans_normal(normal, RT_w2c, RT_w2c_target):

    normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
    normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

    return normal_target_cam

def img2normal(img):
    return (img/255.)*2-1

def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)

def norm_normalize(normal, dim=-1):

    normal = normal/(np.linalg.norm(normal, axis=dim, keepdims=True)+1e-6)

    return normal

def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def normal_opengl2opencv(normal):
    H,W,C = np.shape(normal)
    # normal_img = np.reshape(normal, (H*W,C))
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]

    print(np.shape(normal_cv))

    return normal_cv

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]


def load_a_prediction(root_dir, test_object, imSize, view_types, load_color=False, cam_pose_dir=None,
                         normal_system='front', erode_mask=True, camera_type='ortho', cam_params=None):

    all_images = []
    all_normals = []
    all_normals_world = []
    all_masks = []
    all_color_masks = []
    all_poses = []
    all_w2cs = []
    directions = []
    ray_origins = []
    cam_pose_dir = "./instant_nsr_pl/datasets/fixed_poses"
    RT_front = np.loadtxt(glob(os.path.join( cam_pose_dir, '*_%s_RT.txt'%( 'front')))[0])   # world2cam matrix
    RT_front_cv = RT_opengl2opencv(RT_front)   # convert normal from opengl to opencv
    for idx, view in enumerate(view_types):
        print(os.path.join(root_dir,test_object))
        normal_filepath = os.path.join(root_dir, test_object, 'normals_000_%s.png'%( view))
        # Load key frame
        if load_color:  # use bgr
            image =np.array(PIL.Image.open(normal_filepath.replace("normals", "rgb")).resize(imSize))[:, :, :3]

        normal = np.array(PIL.Image.open(normal_filepath).resize(imSize))
        mask = normal[:, :, 3]
        normal = normal[:, :, :3]

        color_mask = np.array(PIL.Image.open(os.path.join(root_dir,test_object, 'masked_colors/rgb_000_%s.png'%( view))).resize(imSize))[:, :, 3]
        invalid_color_mask = color_mask < 255*0.5
        threshold =  np.ones_like(image[:, :, 0]) * 250
        invalid_white_mask = (image[:, :, 0] > threshold) & (image[:, :, 1] > threshold) & (image[:, :, 2] > threshold)
        invalid_color_mask_final = invalid_color_mask & invalid_white_mask
        color_mask = (1 - invalid_color_mask_final) > 0

        # if erode_mask:
        #     kernel = np.ones((3, 3), np.uint8)
        #     mask = cv2.erode(mask, kernel, iterations=1)

        RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt'%( view)))  # world2cam matrix

        normal = img2normal(normal)

        normal[mask==0] = [0,0,0]
        mask = mask> (0.5*255)
        if load_color:
            all_images.append(image)
        
        all_masks.append(mask)
        all_color_masks.append(color_mask)
        RT_cv = RT_opengl2opencv(RT)   # convert normal from opengl to opencv
        all_poses.append(inv_RT(RT_cv))   # cam2world
        all_w2cs.append(RT_cv)

        # whether to 
        normal_cam_cv = normal_opengl2opencv(normal)

        if normal_system == 'front':
            print("the loaded normals are defined in the system of front view")
            normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], normal_cam_cv)
        elif normal_system == 'self':
            print("the loaded normals are in their independent camera systems")
            normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)
        all_normals.append(normal_cam_cv)
        all_normals_world.append(normal_world)

        if camera_type == 'ortho':
            origins, dirs = get_ortho_ray_directions_origins(W=imSize[0], H=imSize[1])
        elif camera_type == 'pinhole':
            dirs = get_ray_directions(W=imSize[0], H=imSize[1],
                                                 fx=cam_params[0], fy=cam_params[1], cx=cam_params[2], cy=cam_params[3])
            origins = dirs # occupy a position
        else:
            raise Exception("not support camera type")
        ray_origins.append(origins)
        directions.append(dirs)
        
        
        if not load_color:
            all_images = [normal2img(x) for x in all_normals_world]


    return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), \
        np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs), np.stack(ray_origins), np.stack(directions), np.stack(all_color_masks)


class OrthoDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.data_dir = self.config.root_dir
        self.object_name = self.config.scene
        self.scene = self.config.scene
        self.imSize = self.config.imSize
        self.load_color = True
        self.img_wh = [self.imSize[0], self.imSize[1]]
        self.w = self.img_wh[0]
        self.h = self.img_wh[1]
        self.camera_type = self.config.camera_type
        self.camera_params = self.config.camera_params  # [fx, fy, cx, cy]
        
        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

        self.view_weights = torch.from_numpy(np.array(self.config.view_weights)).float().to(self.rank).view(-1)
        self.view_weights = self.view_weights.view(-1,1,1).repeat(1, self.h, self.w)

        if self.config.cam_pose_dir is None:
            self.cam_pose_dir = "./datasets/fixed_poses"
        else:
            self.cam_pose_dir = self.config.cam_pose_dir
            
        self.images_np, self.masks_np, self.normals_cam_np, self.normals_world_np, \
            self.pose_all_np, self.w2c_all_np, self.origins_np, self.directions_np, self.rgb_masks_np = load_a_prediction(
                self.data_dir, self.object_name, self.imSize, self.view_types,
                self.load_color, self.cam_pose_dir, normal_system='front', 
                camera_type=self.camera_type, cam_params=self.camera_params)

        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        self.all_c2w = torch.from_numpy(self.pose_all_np)
        self.all_images = torch.from_numpy(self.images_np) / 255.
        self.all_fg_masks = torch.from_numpy(self.masks_np)
        self.all_rgb_masks = torch.from_numpy(self.rgb_masks_np)
        self.all_normals_world = torch.from_numpy(self.normals_world_np)
        self.origins = torch.from_numpy(self.origins_np)
        self.directions = torch.from_numpy(self.directions_np)

        self.directions = self.directions.float().to(self.rank)
        self.origins = self.origins.float().to(self.rank)
        self.all_rgb_masks = self.all_rgb_masks.float().to(self.rank)
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_normals_world = \
            self.all_c2w.float().to(self.rank), \
            self.all_images.float().to(self.rank), \
            self.all_fg_masks.float().to(self.rank), \
            self.all_normals_world.float().to(self.rank)
        

class OrthoDataset(Dataset, OrthoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class OrthoIterableDataset(IterableDataset, OrthoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('ortho')
class OrthoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = OrthoIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = OrthoDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = OrthoDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = OrthoDataset(self.config, 'train')    

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       