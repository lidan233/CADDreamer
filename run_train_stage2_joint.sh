# Wonder3D multi-view diffusion - stage 2 (joint, continues from stage-1 checkpoint)
accelerate launch --config_file 4gpu.yaml train_mvdiffusion_joint.py --config configs/train/cad_wonder3D_2_stage.yaml
