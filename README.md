# CADDreamer: CAD Object Generation from Single-view Images

[Project Page](https://lidan233.github.io/caddreamer/) | [Paper](https://arxiv.org/pdf/2502.20732)

## Overview

CADDreamer is a novel approach for generating high-quality boundary representations (B-rep) of CAD objects from single-view RGB images. Unlike existing 3D generative models that produce dense and unstructured meshes, CADDreamer creates compact, structured, and sharply-edged CAD models that are comparable to those created by human designers.


## Installation
TO DO

## Download pretrained model
Here are the links:

* Ckpts directory: [save as ckpts directory](https://utdallas.box.com/s/6rwdqoyhgu38udh2cfsf2kympe70i5pu)
* Finetuned Vae directory: [save as finetuned_vae_normal directory](https://utdallas.box.com/s/gpvwli8evucmfjd7hzl2y4f4odjm0dfg)
* Check the input here: [check the input example](https://utdallas.box.com/s/2cnqyv5b9wun5nptp61y7x8hej0ejve4)
* Check the output here: [check the output example (segmentation results)](https://utdallas.box.com/s/jpkb2h0n0frr3svrd25305txlnmsmd3m)

## Todo List
Code and dataset will be released before the conference.
Please stay tuned. If you have any question, please contact me via email: Li.Yuan@utdallas.edu. 
- [x] Release `Tools` code, including `pyransac`, `neus`, and so on.
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