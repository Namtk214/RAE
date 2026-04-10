## RAE-JAX: Diffusion Transformers with Representation Autoencoders <br><sub>JAX/Flax NNX Implementation for TPU</sub>

### [Paper](https://arxiv.org/abs/2510.11690) | [Project Page](https://rae-dit.github.io/) | [PyTorch](https://github.com/bytetriper/RAE)

JAX/Flax NNX port of [RAE](https://github.com/bytetriper/RAE), optimised for **TPU v5e-8** data-parallel training.  
YAML config file is optional — all parameters can be passed directly as CLI flags.

---

## Setup

### 1. Clone repositories

```bash
# Create working folder
mkdir rae_jax && cd rae_jax

# Main codebase
git clone https://github.com/Namtk214/RAE.git

# TFDS custom builders (required for CelebAHQ256 and other custom datasets)
git clone https://github.com/Namtk214/tfds_builders.git
```

> `tfds_builders/` và `RAE/` sẽ nằm cùng cấp bên trong `rae_jax/`.  
> Truyền đường dẫn của nó qua `--tfds-builder-dir` khi dùng dataset TFDS custom.

### 2. Install dependencies

```bash
cd RAE
pip install -r requirements.txt
```

```

### 4. WandB (optional)

```bash
export WANDB_KEY="your_wandb_api_key"
export ENTITY="your_wandb_entity"
```

---

## Project Structure

```
rae_jax/
├── configs/
│   └── decoder/ViTXL/config.json   # ViT-XL decoder architecture (read-only)
├── stage1/                          # RAE Autoencoder
├── stage2/                          # DiT Diffusion Model
├── disc/                            # Discriminator (Stage 1)
├── eval/                            # FID & IQA utilities
├── utils/                           # Device, Optimizer, Checkpoint helpers
├── train_stage1.py                  # Stage 1 training  ← main entry point
├── train.py                         # Stage 2 training  ← main entry point
├── calculate_stat.py                # Latent normalization statistics
├── extract_decoder.py               # Extract decoder weights from Stage 1 ckpt
├── sample.py                        # Single-device sampling
├── sample_ddp.py                    # Distributed sampling
├── stage1_sample.py                 # Single-image Stage 1 reconstruction
├── stage1_sample_ddp.py             # Distributed Stage 1 reconstruction
└── data.py                          # tf.data / ImageFolder pipeline
```

---

## Data Preparation

### ImageNet-1K

```bash
# Organize as ImageFolder:
#   data/imagenet/train/<class_id>/*.JPEG
#   data/imagenet/val/<class_id>/*.JPEG
```

### CelebA-HQ 256 via TFDS Builder

The `tfds_builders` repo contains custom [TensorFlow Datasets](https://www.tensorflow.org/datasets) builders that auto-download and prepare datasets. CelebAHQ256 pulls from HuggingFace `mattymchen/celeba-hq`.

**Step 1 — Clone the builders repo** (nếu chưa clone ở bước Setup):

```bash
git clone https://github.com/Namtk214/tfds_builders.git
```

**Step 2 — Install dependencies cho builder:**

```bash
pip install tensorflow tensorflow-datasets datasets pillow

# Kiểm tra:
python -c "import tensorflow; import tensorflow_datasets; print('OK')"
```

**Step 3 — Build the dataset:**

```bash
cd tfds_builders/celebahq256

# Option A: tfds CLI
tfds build

# Option B: nếu tfds CLI không có trong PATH, dùng Python trực tiếp
python -c "
import celebahq256_dataset_builder
builder = celebahq256_dataset_builder.Builder(data_dir='$HOME/tensorflow_datasets')
builder.download_and_prepare()
print('Done!')
"

# Chỉ định thư mục data cụ thể:
tfds build --data_dir /path/to/tensorflow_datasets
```

> Quá trình build sẽ tải ~30k images từ HuggingFace, resize về 256×256, và lưu dưới dạng TFRecord tại `~/tensorflow_datasets/celebahq256/1.0.0/`. Mất khoảng 5-10 phút tùy mạng.

**Step 4 — Truyền vào training script:**

```bash
# Stage 1
python train_stage1.py \
  --data-dir ~/tensorflow_datasets \
  --data-source tfds \
  --dataset-name celebahq256 \
  --num-train-samples 30000 \
  ...

# Stage 2 (with config)
python train.py \
  --config configs/stage2/training/CelebAHQ256/DiTDH-S_DINOv2-B.yaml \
  --data-path ~/tensorflow_datasets \
  --dataset-type tfds \
  ...

# Stage 2 (without config)
python train.py \
  --data-path ~/tensorflow_datasets \
  --dataset-type tfds \
  --tfds-name celebahq256 \
  --tfds-builder-dir ~/tensorflow_datasets/celebahq256/1.0.0 \
  --num-train-samples 30000 \
  ...
```

> **Lưu ý:** `--tfds-builder-dir` chỉ cần khi **không dùng `--config`**. Nếu dùng YAML config, đường dẫn builder đã được khai báo trong `data.tfds_builder_dir`.

**Các dataset có sẵn trong `tfds_builders/`:**

| Builder | Mô tả | Splits |
|---------|--------|--------|
| `celebahq256` | CelebA-HQ 256×256 | train (30k), validation |
| `celebahq` | CelebA-HQ full resolution | train, validation |
| `celebahq64` | CelebA-HQ 64×64 | train, validation |
| `cifar10` | CIFAR-10 | train, test |
| `imagenet2012` | ImageNet-1K (cần tải thủ công) | train, validation |

### CelebA-HQ 256 via ImageFolder (thay thế)

Nếu không muốn dùng TFDS, có thể dùng ImageFolder:

```bash
# Cấu trúc:
#   data/celebahq256/train/0/*.png
#   data/celebahq256/val/0/*.png
# Dùng flag: --dataset-type imagefolder
```

## Stage 1: Representation Autoencoder

Stage 1 trains a ViT-XL decoder to reconstruct images from frozen DINOv2 encoder latents.

### Step 1 — Compute Latent Normalization Stats

Run once before training to compute encoder mean/variance used to normalize latents:

```bash
# TFDS (CelebAHQ256 đã build)
python calculate_stat.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path ~/tensorflow_datasets \
  --output-dir models/stats/dinov2 \
  --image-size 256 \
  --batch-size 32 \
  --num-samples 30000 \
  --dataset-type tfds \
  --tfds-name celebahq256

# ImageFolder
python calculate_stat.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path data/celebahq256/train \
  --output-dir models/stats/dinov2 \
  --image-size 256 \
  --batch-size 32 \
  --num-samples 30000 \
  --dataset-type imagefolder
```

Output: `models/stats/dinov2/normalization_stats.npz` (keys: `mean`, `var`)


---

### Step 2 — Train the Decoder

```bash
python train_stage1.py \
  --data-dir ~/tensorflow_datasets \
  --data-source tfds \
  --dataset-name celebahq256 \
  --num-train-samples 30000 \
  --results-dir ckpts/stage1 \
  --experiment-name celebahq256_dinov2b_decXL \
  --epochs 16 \
  --global-batch-size 128 \
  --lr 2e-4 \
  --wandb \
  --wandb-project rae-jax-stage1
```

**ImageNet** variant:

```bash
python train_stage1.py \
  --data-dir /data/imagenet \
  --data-source imagefolder \
  --dataset-name imagenet2012 \
  --num-train-samples 1281167 \
  --results-dir ckpts/stage1 \
  --experiment-name imagenet_dinov2b_decXL \
  --epochs 16 \
  --global-batch-size 512 \
  --lr 2e-4 \
  --wandb
```

Training auto-resumes from the latest checkpoint if `results-dir/experiment-name/checkpoints/` already exists.

**Hardcoded defaults** (edit at the top of `train_stage1.py` to change):

| Constant | Value | Description |
|----------|-------|-------------|
| `_EMA_DECAY` | 0.9978 | EMA decay rate |
| `_SCHEDULE_TYPE` | cosine | LR schedule |
| `_WARMUP_EPOCHS` | 1 | Warmup epochs |
| `_FINAL_LR` | 2e-5 | Final learning rate |
| `_DISC_START` | 8 | Epoch to activate GAN loss |
| `_DISC_UPD_START` | 6 | Epoch to start updating discriminator |
| `_LPIPS_START` | 0 | Epoch to activate LPIPS |
| `_DISC_WEIGHT` | 0.75 | GAN loss weight |
| `_PERCEPTUAL_WEIGHT` | 1.0 | LPIPS weight |
| `_ENCODER_CLS` | `Dinov2withNorm` | Encoder architecture |
| `_DECODER_CONFIG_PATH` | `configs/decoder/ViTXL` | Decoder config |

---

### Step 3 — Reconstruct Images

**Single image:**

```bash
python stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --image assets/test.png \
  --output recon.png \
  --image-size 256 \
  --ckpt-dir ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints \
  --use-ema
```

**Distributed batch reconstruction (all TPU cores):**

```bash
python stage1_sample_ddp.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path data/celebahq256/val \
  --output-dir samples/recon_val \
  --batch-size 8 \
  --num-samples 30000 \
  --image-size 256 \
  --dataset-type imagefolder \
  --ckpt-dir ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints \
  --use-ema
```

---

### Step 4 — Extract Decoder Weights

Extract a standalone decoder `.npz` from a Stage 1 checkpoint for use in Stage 2:

```bash
python extract_decoder.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints/ckpt_0000016.pkl \
  --use-ema \
  --out models/decoders/dinov2/wReg_base/ViTXL_n08/model.npz
```

---

## Stage 2: Latent Diffusion Transformer

Stage 2 trains a flow-matching DiT on RAE latent space.

`train.py` supports **two modes**:
- **With `--config`:** loads all model/training params from YAML (original behavior)
- **Without `--config`:** uses hardcoded defaults + CLI flags (no YAML needed)

### CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | None | YAML config file (optional) |
| `--data-path` | *required* | Dataset root or TFDS data_dir |
| `--results-dir` | `ckpts` | Root directory for checkpoints |
| `--experiment-name` | auto | Sub-folder name inside results-dir |
| `--image-size` | 256 | Input image resolution |
| `--dataset-type` | `tfds` | `imagefolder` or `tfds` |
| `--tfds-name` | None | TFDS dataset name (e.g. `celebahq256`) |
| `--tfds-builder-dir` | None | Path to custom TFDS builder directory |
| `--rae-checkpoint` | None | Stage 1 checkpoint `.pkl` for decoder weights |
| `--normalization-stat-path` | None | Latent normalization `.npz` |
| `--pretrained-decoder-path` | None | Pretrained decoder weights |
| `--reference-npz-path` | None | Pre-computed FID reference `.npz` |
| `--epochs` | 200 | Number of training epochs |
| `--global-batch-size` | 128 | Total batch across all TPU devices |
| `--lr` | 2e-4 | Learning rate |
| `--num-train-samples` | 30000 | Dataset size (for steps/epoch calculation) |
| `--num-classes` | 1 | Number of classes (1=unconditional, 1000=ImageNet) |
| `--precision` | `bf16` | `fp32` or `bf16` |
| `--global-seed` | 42 | Random seed |
| `--eval-fid-every` | 0 | Evaluate FID every N steps (0=disabled) |
| `--num-fid-samples` | 50000 | Number of samples for FID |
| `--wandb` | off | Enable WandB logging |
| `--wandb-project` | `rae-jax-stage2` | WandB project name |
| `--wandb-entity` | `""` | WandB entity |

### Step 1 — Train DiT

**CelebAHQ-256 (DiT-S, ~130M params) — with config:**

```bash
python train.py \
  --config configs/stage2/training/CelebAHQ256/DiTDH-S_DINOv2-B.yaml \
  --data-path ~/tensorflow_datasets \
  --dataset-type tfds \
  --rae-checkpoint ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints/ckpt_last.pkl \
  --results-dir ckpts/stage2 \
  --wandb
```

**CelebAHQ-256 — without config (CLI flags only):**

```bash
python train.py \
  --data-path ~/tensorflow_datasets \
  --dataset-type tfds \
  --tfds-name celebahq256 \
  --tfds-builder-dir ~/tensorflow_datasets/celebahq256/1.0.0 \
  --num-train-samples 30000 \
  --num-classes 1 \
  --rae-checkpoint ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints/ckpt_last.pkl \
  --normalization-stat-path models/stats/dinov2/normalization_stats.npz \
  --results-dir ckpts/stage2 \
  --experiment-name celebahq256_s \
  --epochs 200 \
  --global-batch-size 128 \
  --lr 2e-4 \
  --wandb
```

**ImageNet-256 (DiT-XL, ~675M params) — with config:**

```bash
python train.py \
  --config configs/stage2/training/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --data-path /data/imagenet \
  --dataset-type imagefolder \
  --rae-checkpoint ckpts/stage1/imagenet_dinov2b_decXL/checkpoints/ckpt_last.pkl \
  --reference-npz-path data/imagenet/VIRTUAL_imagenet256_labeled.npz \
  --results-dir ckpts/stage2 \
  --eval-fid-every 25000 \
  --wandb
```

Training auto-resumes from the latest checkpoint if `results-dir/experiment-name/checkpoints/` already exists.

**Hardcoded defaults** (used when `--config` is not provided, edit `_DEFAULTS` dict in `train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| DiT model | DiT-S `[384,2048]` | hidden_size, depth=[12,2] |
| num_classes | 1 | Unconditional (CelebAHQ) |
| ema_decay | 0.9995 | EMA decay rate |
| schedule | linear | LR schedule type |
| warmup | 10 epochs | Warmup period |
| clip_grad | 1.0 | Gradient clip norm |
| log_interval | 50 | Log every N steps |
| sample_every | 5000 | Generate samples every N steps |
| checkpoint_interval | 5000 | Save checkpoint every N steps |

---

### Step 2 — Sample Images

**Single-device CFG sampling:**

```bash
python sample.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --seed 42 \
  --cfg-scale 1.5 \
  --class-labels 207 360 387 974 88 979 417 279 \
  --output-dir samples/gen_cfg15 \
  --ckpt-dir ckpts/stage2/imagenet256_xl/checkpoints \
  --rae-ckpt-dir ckpts/stage1/imagenet_dinov2b_decXL/checkpoints \
  --use-ema
```

**Distributed sampling (FID-50k ready):**

```bash
python sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples/fid50k \
  --num-fid-samples 50000 \
  --batch-size 16 \
  --seed 0 \
  --cfg-scale 1.5 \
  --ckpt-dir ckpts/stage2/imagenet256_xl/checkpoints \
  --rae-ckpt-dir ckpts/stage1/imagenet_dinov2b_decXL/checkpoints \
  --use-ema
```

**AutoGuidance sampling:**

```bash
python sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
  --sample-dir samples/gen_ag \
  --num-fid-samples 50000 \
  --batch-size 16 \
  --ckpt-dir ckpts/stage2/imagenet256_xl/checkpoints \
  --use-ema
```

---

## Evaluation

### FID with ADM Suite

```bash
# 1. Generate reference FID stats (one-time)
python eval/create_fid_ref.py \
  --data-path data/imagenet/train \
  --out-path data/imagenet/VIRTUAL_imagenet256_labeled.npz \
  --image-size 256 \
  --batch-size 64 \
  --num-samples 50000 \
  --dataset-type imagefolder \
  --split train

# 2. Or download OpenAI pre-computed reference
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

# 3. Generate 50k samples
python sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples/fid50k_cfg15 \
  --num-fid-samples 50000 \
  --batch-size 16 \
  --cfg-scale 1.5 \
  --ckpt-dir ckpts/stage2/imagenet256_xl/checkpoints \
  --use-ema

# 4. Score
cd guided-diffusion/evaluations
python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples/fid50k_cfg15/samples_*.npz
```

### Built-in Metrics (PSNR / SSIM / LPIPS / rFID)

```bash
python -m eval \
  --ref-img data/imagenet/val_256.npz \
  --rec-img samples/recon.npz \
  --bs 128
```

---

## TPU Parallelism

All training scripts use **mesh-based data parallelism** — no `jax.pmap`:

```
Mesh("data", 8)

  Core 0 … Core 7
  ┌─────┐   ┌─────┐
  │ B/8 │   │ B/8 │   ← data SHARDED  P("data")
  └──┬──┘   └──┬──┘
  ┌──▼──┐   ┌──▼──┐
  │Model│   │Model│   ← params REPLICATED  P()
  └──┬──┘   └──┬──┘
     └────┬────┘
     XLA all-reduce
  (auto gradient sync)
```

- **bfloat16** (`--precision bf16`) for 2× TPU throughput
- **`donate_argnums`** for zero-copy parameter updates
- **Gradient checkpointing** available via `nnx.remat` for large models

### Multi-Host Training (TPU v4-16, v5e-16, etc.)

When training on multi-host TPU pods (e.g., `v4-16` which comprises 2 VMs with 16 TPU cores total), JAX executes programs symmetrically on all workers. 

**Execution Commands:**
Run scripts on all workers simultaneously by passing `--worker=all`. To avoid the console freezing from SSH timeouts, run long processes in the background using `nohup`.

**1. Stage 1 Training:**
```bash
nohup gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=all \
  --command="cd ~/rae_jax/RAE && python train_stage1.py \
    --data-dir ~/tensorflow_datasets \
    --data-source tfds \
    --dataset-name celebahq256 \
    --num-train-samples 30000 \
    --results-dir ckpts/stage1 \
    --experiment-name celebahq256_dinov2b_decXL \
    --epochs 16 \
    --global-batch-size 128 \
    --lr 2e-4 \
    --wandb \
    --wandb-project rae-jax-stage1" > train_stage1_run.log 2>&1 &
```

**2. Extract Decoder from Checkpoint:**
```bash
# This can be run on all workers so both hosts get the model.npz file locally
gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=all \
  --command="cd ~/rae_jax/RAE && mkdir -p models/decoders/dinov2/wReg_base/ViTXL_n08/ && \
  python extract_decoder.py \
    --config configs/stage1/training/DINOv2-B_decXL.yaml \
    --ckpt ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints/ckpt_0003744.pkl \
    --use-ema \
    --out models/decoders/dinov2/wReg_base/ViTXL_n08/model.npz"
```

**3. Calculate Normalization Stats:**
```bash
gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=all \
  --command="cd ~/rae_jax/RAE && python calculate_stat.py \
    --config configs/stage1/pretrained/DINOv2-B.yaml \
    --data-path ~/tensorflow_datasets \
    --output-dir models/stats/dinov2 \
    --image-size 256 \
    --batch-size 32 \
    --num-samples 30000 \
    --dataset-type tfds \
    --tfds-name celebahq256"
```

> **Warning:** `calculate_stat.py` will only save the `.npz` file on **Worker 0**. Before running Stage 2, you MUST copy the `normalization_stats.npz` to all other workers:
> ```bash
> # Copy from Worker 0 to Worker 1
> gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=0 --command="cat ~/rae_jax/RAE/models/stats/dinov2/normalization_stats.npz" | gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=1 --command="mkdir -p ~/rae_jax/RAE/models/stats/dinov2 && cat > ~/rae_jax/RAE/models/stats/dinov2/normalization_stats.npz"
> ```

**4. Stage 2 Training:**
```bash
nohup gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=all \
  --command="cd ~/rae_jax/RAE && python train.py \
    --data-path ~/tensorflow_datasets \
    --dataset-type tfds \
    --tfds-name celebahq256 \
    --tfds-builder-dir ~/tensorflow_datasets/celebahq256/1.0.0 \
    --num-train-samples 30000 \
    --num-classes 1 \
    --rae-checkpoint ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints/ckpt_0003744.pkl \
    --normalization-stat-path models/stats/dinov2/normalization_stats.npz \
    --results-dir ckpts/stage2 \
    --experiment-name celebahq256_s \
    --epochs 200 \
    --global-batch-size 128 \
    --lr 2e-4 \
    --wandb" > train_stage2_run.log 2>&1 &
```

**Syncing Code Across Workers:**
If you edit code locally, you **MUST** sync the python files to all workers before training, otherwise workers will run out-of-sync code and crash/deadlock!
```bash
# Sync specific files that were modified across the slice
gcloud compute tpus tpu-vm scp ~/rae_jax/RAE/train.py ~/rae_jax/RAE/calculate_stat.py ~/rae_jax/RAE/train_stage1.py node-v4-16:~/rae_jax/RAE/ --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm scp ~/rae_jax/RAE/utils/device_utils.py node-v4-16:~/rae_jax/RAE/utils/device_utils.py --zone=us-central2-b --worker=all
```

**Force Killing Deadlocked Jobs:**
If JAX encounters a synchronization mismatch (e.g., Worker 1 executes a JIT block but Worker 0 skipped it due to a logic bug), the processes will hang indefinitely at 0% CPU waiting for network handshakes. You must force kill them across all workers:
```bash
# Force kill stuck processes across all nodes
gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=all --command="pkill -9 -f train_stage1.py"
```

### Multi-Host Deadlock Avoidance (Developer Note)
The framework natively avoids multi-host deadlocks by using robust JAX multi-host data loading patterns:
1. **Per-Host Batching**: Using `batch_size = global_batch_size // jax.process_count()` instead of loading the full tensor. Each worker reads only its unique dataset slice.
2. **`make_array_from_process_local_data`**: Inside `utils/device_utils.py`, local data batches from each host are seamlessly materialized into a global `NamedSharding` array without the hosts needing identical full local instances.
3. **Symmetric Execution**: Network collectives (`jax.jit`, `device_put`) are never gated behind `if jax.process_index() == 0:`. Gating is strictly reserved for pure IO ops (like `print()`, `wandb.log()`, `os.makedirs()`).

---

## Acknowledgements

- [RAE](https://github.com/bytetriper/RAE) — Original PyTorch implementation
- [SiT](https://github.com/willisma/sit) — Flow matching diffusion
- [DDT](https://github.com/MCG-NJU/DDT) — DiT<sup>DH</sup> architecture
- [LightningDiT](https://github.com/hustvl/LightningDiT/) — Single-stream DiT
- [MAE](https://github.com/facebookresearch/mae) — ViT decoder architecture
