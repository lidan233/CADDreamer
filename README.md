# CADDreamer: CAD Object Generation from Single-view Images

[Project Page](https://lidan233.github.io/caddreamer/) | [Paper](https://arxiv.org/pdf/2502.20732)
![CADDreamer Generation Process](cached_output/generate_results.gif)

## Overview

CADDreamer is a novel approach for generating high-quality boundary representations (B-rep) of CAD objects from single-view RGB images. Unlike existing 3D generative models that produce dense and unstructured meshes, CADDreamer creates compact, structured, and sharply-edged CAD models that are comparable to those created by human designers.


## Installation
Our installation process consists of three steps. The first step is to download the pretrained model to the specified ckpts directory.

### First, Download pretrained model
Here are the links:

* Ckpts directory: [save as ckpts directory](https://utdallas.box.com/s/di899wojtlvx0g3v6h575o1xxkxb11ex)
* Finetuned Vae directory: [save as finetuned_vae_normal directory](https://utdallas.box.com/s/tkczrjm0pxcfxdwzelg1ilzoteadyvc0)
* Check the input here: [check the input example](https://utdallas.box.com/s/jf1805d0n0x8w49h9z36tpgdlixo9rg4)
* Check the output here: [check the output example (segmentation results)](https://utdallas.box.com/s/r6bsob98az6z0o5p5qtoh4dojhyafrv0)

Then, modify the corresponding paths in the load_wonder3d_pipeline function of test_mvdiffusion_seq.py. Here is the final structure: 
```bash
CADDreamer
└── ckpts
    └── wonder3d-v1.0
        ├── feature_extractor
        │   └── preprocessor_config.json
        ├── image_encoder
        │   ├── config.json
        │   └── pytorch_model.bin
        ├── model_index.json
        ├── README.md
        ├── scheduler
        │   └── scheduler_config.json
        ├── unet
        │   ├── config.json
        │   └── diffusion_pytorch_model.bin
        └── vae
            ├── config.json
            └── diffusion_pytorch_model.bin
└── finetuning_vae_normal
    └── outputs
        ├── finetune_vae
        │   ├── mlp_16.pth
        │   └── vae_16.pth
        └── finetuning_vae_normal
            └── vae_normal_2.pth

```

### Second, Compile and install CAD-related libraries
The main task is to compile and install OpenCascade and FreeCAD's Python bindings. During installation, please pay special attention to the compatibility between different versions of FreeCAD and OpenCascade. On my Ubuntu 22.04 host, I installed OCC version 7.8.1. The FreeCAD version information is as follows:
```bash
'0', '22', '0', '38495 (Git)', 'git://github.com/FreeCAD/FreeCAD.git main', '2024/08/19 16:34:53', 'main', '131956e201dc5033de69ed09c28db332feb081c1'
```
Using incompatible versions may lead to strange, unknown errors.

### Third, Install Python environment
Create a Python environment using conda and install the corresponding environment using pip. 
```bash
    bash setup.sh
```


## Evaluation
The evaluation consists of two parts: testing on real images and testing on synthetic images.

### Real Images
#### 1. Generate Multi-view Images and 3D Files
In the first step, we generate multi-view normal semantic maps from processed normal images and reconstruct 3D meshes using NEUS.  
The following command loads all processed normal images from the ./test_real_images folder and selects two images from index 0 to 2 for generation:
 ```bash
 python3 test_mvdiffusion_seq.py  --config configs/train/testing_4090_stage_1_cad_6views-lvis.yaml --idx 0  --gpu 0
 ```
 The successful generation file structure can be found in ./cached_output/cropsize-256-cfg1.0/0_0deepcad.

#### 2. Segmentation and CAD Reconstruction
 ```bash
 python3 test_real_images.py  --config ./cached_output/cropsize-256-cfg1.0/0_0deepcad --review False
 ```
This step will cache segmentation results in the neus/temp_mid_results folder and generate a STEP file. 
Example STEP files can be found in ./cached_output/cropsize-256-cfg1.0/0_0deepcad.

### Synthetic Images
For convenient parallel processing, our testing is divided into 3 main steps.

#### 1. Generate Multi-view Images and 3D Files
In the first step, we generate multi-view normal semantic maps from processed normal images and reconstruct 3D meshes using NEUS.  
The following command loads all processed normal images from the ./test_real_images folder and selects two images from index 0 to 2 for generation:
 ```bash
 python3 test_mvdiffusion_seq.py  --config configs/train/testing_4090_stage_1_cad_6views-lvis.yaml --idx 0  --gpu 0
 ```

#### 2. Execute Segmentation
In the second step, we perform segmentation based on the reconstructed mesh and multi-view images. 
The following command executes the second stage segmentation task based on the generated results:
 ```bash
 python3 test_syne_images_stage_2_segmentation.py  --config_dir ./cached_output/cropsize-256-cfg1.0-syne/0_0deepcad --review True
 ```
 This task will cache segmentation results in the neus/temp_mid_results folder.

#### 3. Reconstruct STEP File Using Intersection Strategy and Primitive Stitching
Since primitive stitching is a time-consuming operation, we recommend skipping it if normal fitting can successfully reconstruct the CAD model. 
 ```bash
 python3 test_syne_images_stage_3_generate_step.py  --config_dir ./cached_output/cropsize-256-cfg1.0-syne/0_0deepcad --review True
 ```
More generation results can be found in the cover GIF. Please ignore the content in directory ./neus/exp/*. 

## Todo List
I will do my best to open source the code and dataset before the conference.
Please stay tuned. If you have any question, please contact me via email: Li.Yuan@utdallas.edu. 
- [x] Release `Tools` code, including `pyransac`, `neus`, and so on.
- [x] Release `Multi-view diffusion` code.
- [x] Release `Primitive Stitching` code.
- [x] Release `Testing` code. Released the test code in real normal images.
- [x] Release `Testing` code. Released the test code in synthetic normal images.
- [x] Release `Testing` dataset. Please refer to this [ link](https://utdallas.box.com/s/adl19p7k6pb2wwqdivfl5334n6ntwixa) to check the testing cases.
- [ ] Release ABC Dataset Scripts (filtering and rendering). I've been getting a lot of emails asking about the ABC dataset filter and rendering script. Thanks for your interest! I'll be releasing both scripts in the next two weeks.
- [ ] Release `Training` code.
- [ ] Release `Training Datasets`.

## Related Projects
- [Wonder3D: Single Image to 3D using Cross-Domain Diffusion](https://github.com/xxlong0/Wonder3D)
- [RANSAC: Efficient RANSAC for Point Cloud Shape Detection](https://github.com/alessandro-gentilini/Efficient-RANSAC-for-Point-Cloud-Shape-Detection)
- [SyncDreamer: Generating Multiview-consistent Images from a Single-view Image](https://github.com/liuyuan-pal/SyncDreamer)



## Citation

If you find this work helpful, please cite our paper:
```bibtex
@inproceedings{yuan2025CADDreamer,
    author    = {Yuan Li and Cheng Lin and Yuan Liu and Xiaoxiao Long and Chenxu Zhang and Ningna Wang and Xin Li and Wenping Wang and Xiaohu Guo},
    title     = {CADDreamer: CAD Object Generation from Single-view Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
    publisher = {IEEE},
}
```



