# train the color / semantic VAE (+ MLP classifier)
accelerate launch --config_file 4gpu.yaml finetune_vae_semantic.py --config configs/train/cad_vae_semantic.yaml
