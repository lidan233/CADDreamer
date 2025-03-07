# CADDreamer: CAD Object Generation from Single-view Images

[Project Page](https://lidan233.github.io/caddreamer/) | [Paper](https://arxiv.org/pdf/2502.20732)

## Overview

CADDreamer is a novel approach for generating high-quality boundary representations (B-rep) of CAD objects from single-view RGB images. Unlike existing 3D generative models that produce dense and unstructured meshes, CADDreamer creates compact, structured, and sharply-edged CAD models that are comparable to those created by human designers.


## Installation 


## Todo List
We are focusing on the revision of the paper based on the reviewers' comments.
Code  and dataset will be released before the conference. 
Please stay tuned. 
- [ ] Release `Tools` code, including `pyransac`, `neus`, and so on. 
- [ ] Release `Multi-view diffusion` code.
- [ ] Release `Primitive Stitching` code.
- [ ] Release `Inference` code.
- [ ] Release `Training` code.
- [ ] Release `Datasets`.

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