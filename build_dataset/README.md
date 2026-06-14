# Build Dataset & Training

Scripts to build the datasets used to train CADDreamer (**Step 1 → Step 2 → Step 3**), followed by the
**Training** of the two VAEs and the two-stage multi-view diffusion model.

All commands use the `cad` conda env (the one with `accelerate / diffusers / OCC / pytorch3d / blenderproc`):

```bash
conda activate <cad-env>
```

---

## Step 1: filter the ABC dataset + select train/test

The raw ABC dataset contains many thin / elongated shapes, shapes too large to be covered by the
multi-view cameras, and shapes containing splines. Training on these actually makes the model hard
to converge, so we filter them out and then select the train/test sets.

Folder: `step1_abc_filter/`

- `filter_abc_data.py` — filter raw ABC STEP by 5 criteria, `abc -> abc-filtered`:
  - a. must be composed entirely of primitives (plane/cylinder/cone/sphere/torus; reject any free-form surface)
  - b. reject if number of faces > 1000
  - c. must be readable by OpenCASCADE
  - d. reject overly elongated shapes (bounding-box aspect ratio too large)
  - e. reject shapes containing very thin face patches
  - with a per-file timeout (`--timeout`, default 30s) and parallelism (`--jobs`, default 30)
- `select_train_test.py` — select train/test from the filtered set:
  - bucket by complexity (face count), with hard-coded quotas (total 31000)
  - within a bucket, take 1/5 of the quota per primitive type (independent sampling, overlap allowed, dedup union)
  - then shuffle the whole selected set and split train/test by `TEST_RATIO`
- `run_filter_pipeline.sh` — one-shot: download → extract → filter → select (idempotent)

```bash
cd step1_abc_filter
bash run_filter_pipeline.sh
```

---

## Step 2: DSINE training data (textured CAD images + normal images) (optional)

Folder: `step2_dsine_dataset/`

On the synthetic dataset we train a normal predictor. For prediction on real images, for the sake of
generalization, we use Era3D.

Here we provide the data-preparation recipe used to train DSINE. You may also just use Era3D directly.

The code below renders, for each CAD model, a textured (HDR-lit) RGB image and the matching normal
image, used to train the DSINE normal predictor.

- `build_dsine_dataset.py` — main entry: build the RGB+normal training pairs
- `blenderProc_ortho_texture_hdr.py` / `blenderProc_ortho_texture_hdr_mesh.py` — BlenderProc
  orthographic rendering with texture + HDR lighting

Training code: https://github.com/baegwangbin/DSINE.
Config: see the DSINE repo.
Train: `CUDA_VISIBLE_DEVICES=1 python3 train.py projects/dsine/experiments/exp001_cvpr2024/dsine.txt`

---

## Step 3: ABC multi-view normal + semantic maps → h5

Folder: `step3_multiview_normal_semantic_h5/`

For each CAD object, render 6 views of normal and semantic maps (primitive labels), and pack them into
**two separate h5 files** based on step-1's `train_list.json` / `test_list.json`, ready to be loaded
directly by the Wonder3D multi-view diffusion training.

1. Render multi-view (normal + semantic together, no rgb)
   - `BlenderProc_ortho_all.py` — multi-view orthographic render entry (reads a STEP, computes the
     per-face instance / primitive-type labels internally, renders `normals/cmask/onlymask/mask`;
     run with the `cad` env python directly)
2. Pack into h5
   - `MergeH5.py` — merge each object's renders, in chunks, into `save_h5/<split>/alldata.h5py`
3. One-shot
   - `run_multiview_pipeline.sh` — render + chunked-pack train / test separately, producing
     `save_h5/train/alldata.h5py` and `save_h5/test/alldata.h5py` (chunking avoids OOM at 31000)

Training loader (reads the h5): `ABCDataset` in `mvdiffusion/data/abc_dataset.py`.
Training config (4-GPU): `configs/train/cad_wonder3D_1_stage.yaml` → train→`./save_h5/train/`,
validation→`./save_h5/test/`, `num_views: 6`, `read_normal/read_color/mix_color_normal: true`, `img_wh: [256,256]`

---

# Training

CADDreamer training has **two VAEs** and a **two-stage** multi-view diffusion model.
The accelerate launcher config is `4gpu.yaml` (or `1gpu.yaml` for a single GPU). All training configs
read the h5 datasets produced by Step 3, with train / test kept in **separate** dirs
(`./save_h5/train/` and `./save_h5/test/`). The scripts/configs below live at the repo root.

## Data prerequisites

Build the datasets first (Steps 1–3 above), then make them visible at the repo root:

```bash
ln -s <path>/abc_dataset/save_h5 save_h5     # save_h5/train , save_h5/test
ln -s <path>/wonder3d-v1.0 ckpts             # pretrained base model under ckpts/wonder3d-v1.0
```

## 1. Two VAEs

- **semantic (color) VAE (+ primitive-type MLP classifier)** — `finetune_vae_semantic.py`
  The input AND output are the color (color-coded semantic) image: the VAE reconstructs it
  (mse + l1, decoder-only finetune), while the `ClassifyMLP` (num_classes = 11) maps the decoded
  features to a discrete primitive label map (cross-entropy). Outputs `vae_*.pth` and `mlp_*.pth`.
  ```bash
  bash run_train_vae_semantic.sh   # finetune_vae_semantic.py + configs/train/cad_vae_semantic.yaml
  ```
- **normal VAE** — `finetune_vae_normal.py`. Reconstructs the normal image (decoder-only finetune).
  Outputs `vae_normal_*.pth`.
  ```bash
  bash run_train_vae_normal.sh     # finetune_vae_normal.py + configs/train/cad_vae_normal.yaml
  ```

## 2. Two-stage multi-view diffusion (Wonder3D)

- **Stage 1 — mix** (predict color + normal): `train_mvdiffusion_image.py`
  ```bash
  bash run_train_stage1_mix.sh     # + configs/train/cad_wonder3D_1_stage.yaml  -> outputs/wonder3D-mix-gpus
  ```
- **Stage 2 — joint** (continues from stage-1 checkpoint): `train_mvdiffusion_joint.py`
  Its validation also decodes semantics + normal, so it loads the finetuned VAEs/MLP above
  (`./finetune_vae/vae_0.pth`, `./finetune_vae/mlp_0.pth`, `./finetune_vae_normal/vae_normal_0.pth`;
  override via `mask_vae_ckpt` / `mlp_ckpt` / `normal_vae_ckpt` in the config). The released unet is this joint model.
  ```bash
  bash run_train_stage2_joint.sh   # + configs/train/cad_wonder3D_2_stage.yaml  -> outputs/wonder3D-joint
  ```

## Training order

```
finetune_vae  +  finetune_vae_normal   ->  stage1 (mix)  ->  stage2 (joint)
```

---

## One-line flow

Step 1 `filter_abc_data.py` → clean STEP set, `select_train_test.py` → train/test lists
Step 2 `build_dsine_dataset.py` → DSINE "texture + normal" training pairs (optional)
Step 3 `BlenderProc_ortho_all.py` (6-view normal+semantic) → `MergeH5.py`
→ `save_h5/train` and `save_h5/test` (two separate h5) → multi-view diffusion training (`cad_wonder3D_1_stage.yaml`)
