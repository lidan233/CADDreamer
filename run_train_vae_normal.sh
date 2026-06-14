# train the normal VAE
accelerate launch --config_file 4gpu.yaml finetune_vae_normal.py --config configs/train/cad_vae_normal.yaml
