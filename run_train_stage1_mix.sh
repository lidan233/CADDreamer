# Wonder3D multi-view diffusion - stage 1 (mix: color + normal)
accelerate launch --config_file 4gpu.yaml train_mvdiffusion_image.py --config configs/train/cad_wonder3D_1_stage.yaml
