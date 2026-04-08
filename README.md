## RAE-JAX: Diffusion Transformers with Representation Autoencoders <br><sub>JAX/Flax NNX Implementation for TPU</sub>

### [Paper](https://arxiv.org/abs/2510.11690) | [Project Page](https://rae-dit.github.io/) | [PyTorch](https://github.com/bytetriper/RAE)

JAX/Flax NNX port of [RAE](https://github.com/bytetriper/RAE), optimized for **TPUv4e-8** data-parallel training.

---

## Environment

### TPU Setup (Kaggle / GCP)

```bash
pip install -r requirements.txt
```

### Verify TPU

```python
import jax
print(jax.devices())  # Should show 8 TPU cores
```

---

## Project Structure

```
rae_jax/
├── configs/
│   ├── decoder/ViTXL/config.json        # ViT-XL decoder architecture
│   ├── stage1/
│   │   ├── pretrained/                   # Inference-only configs
│   │   │   ├── DINOv2-B.yaml
│   │   │   ├── DINOv2-B_512.yaml
│   │   │   └── MAE.yaml
│   │   └── training/
│   │       └── DINOv2-B_decXL.yaml       # Stage 1 training config
│   └── stage2/
│       ├── training/
│       │   ├── ImageNet256/
│       │   │   ├── DiTDH-XL_DINOv2-B.yaml
│       │   │   └── DiTDH-S_DINOv2-B.yaml
│       │   └── ImageNet512/
│       │       └── DiTDH-XL_DINOv2-B.yaml
│       └── sampling/
│           ├── ImageNet256/
│           │   ├── DiTDHXL-DINOv2-B.yaml      # CFG sampling
│           │   └── DiTDHXL-DINOv2-B_AG.yaml   # AutoGuidance
│           └── ImageNet512/
│               └── DiTDH-XL_DINOv2-B_decXL_AG.yaml
├── stage1/                # RAE Autoencoder
├── stage2/                # DiT Diffusion Model
├── disc/                  # Discriminator (Stage 1)
├── eval/                  # Distributed Evaluation
├── utils/                 # Device, Optimizer, Checkpoint utils
├── train_stage1.py        # Stage 1 training
├── train.py               # Stage 2 training
├── calculate_stat.py      # Latent normalization statistics
├── sample.py              # Single-device sampling
├── sample_ddp.py          # Distributed sampling
├── stage1_sample.py       # Stage 1 reconstruction demo
├── stage1_sample_ddp.py   # Distributed Stage 1 reconstruction
├── extract_decoder.py     # Extract decoder weights
└── data.py                # tf.data / ImageFolder pipeline
```

---

## Data Preparation

### ImageNet

```bash
# Download ImageNet-1k train/val splits
# Organize as ImageFolder: data/imagenet/train/<class_id>/*.JPEG
```

### CelebA-HQ 256

```bash
# If using TFDS builder:
# Set data.source=tfds and data.dataset_name=celebahq256 in config

# If using ImageFolder:
# Organize as: data/celebahq256/train/<class>/*.png
```

---

## Stage 1: Representation Autoencoder

### 1. Calculate Encoder Statistics

Before training, compute the mean/variance of encoder outputs for latent normalization:

```bash
python calculate_stat.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path data/imagenet/train \
  --output-dir models/stats/dinov2 \
  --image-size 256 \
  --batch-size 64 \
  --num-samples 50000
```

> **Note**: Statistics should be computed **without** a pre-computed `normalization_stat_path` in the config. The script uses Welford's online algorithm for numerical stability.

### 2. Train the Decoder

Train the ViT-XL decoder while keeping the DINOv2 encoder frozen:

```bash
python train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --wandb
```

**Key training config parameters** (`DINOv2-B_decXL.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `training.epochs` | 16 | Total training epochs |
| `training.global_batch_size` | 512 | Global batch (÷8 = 64/device) |
| `training.ema_decay` | 0.9978 | EMA decay rate |
| `training.optimizer.lr` | 2e-4 | AdamW learning rate |
| `training.optimizer.betas` | [0.9, 0.95] | Adam betas |
| `training.scheduler.type` | cosine | LR schedule |
| `training.scheduler.warmup_epochs` | 1 | Warmup epochs |
| `training.scheduler.final_lr` | 2e-5 | Final LR |
| `stage_1.noise_tau` | 0.8 | Training noise (set 0 at inference) |
| `gan.loss.disc_start` | 8 | Start GAN loss at epoch 8 |
| `gan.loss.lpips_start` | 0 | Start LPIPS at epoch 0 |
| `gan.loss.disc_weight` | 0.75 | GAN loss weight |

**Logging**: Set environment variables for WandB:

```bash
export WANDB_KEY="your_key"
export ENTITY="your_entity"
export PROJECT="rae-jax-stage1"
```

**Resuming**: The training script auto-resumes from the latest checkpoint in the results directory.

### 3. Reconstruct Images

Single image reconstruction:

```bash
python stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --image assets/test.png
```

Distributed batch reconstruction (all 8 TPU cores):

```bash
python stage1_sample_ddp.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path data/imagenet/val \
  --output-dir samples/recon \
  --batch-size 8 \
  --image-size 256
```

### 4. Extract Decoder Weights

Save a standalone decoder checkpoint:

```bash
python extract_decoder.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt ckpts/stage1/checkpoints/ep-0000016 \
  --use-ema \
  --out models/decoders/dinov2/wReg_base/ViTXL_n08/model.npz
```

---

## Stage 2: Latent Diffusion Transformer

### 1. Train DiT<sup>DH</sup>

Train the flow-matching diffusion transformer on RAE latents:

```bash
python train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path data/imagenet/train \
  --results-dir ckpts/stage2 \
  --image-size 256 \
  --precision bf16 \
  --wandb
```

**Key training config parameters** (`DiTDH-XL_DINOv2-B.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `training.epochs` | 1400 | Total training epochs |
| `training.global_batch_size` | 1024 | Global batch (÷8 = 128/device) |
| `training.ema_decay` | 0.9995 | EMA decay rate |
| `training.clip_grad` | 1.0 | Gradient clipping norm |
| `training.optimizer.lr` | 2e-4 | AdamW learning rate |
| `training.scheduler.type` | linear | LR schedule (linear decay) |
| `training.scheduler.warmup_epochs` | 40 | Warmup epochs |
| `training.scheduler.decay_end_epoch` | 800 | Decay end |
| `transport.params.path_type` | Linear | Flow matching path |
| `transport.params.time_dist_type` | logit-normal_0_1 | Time distribution |
| `stage_2.params.hidden_size` | [1152, 2048] | Encoder, Decoder dim |
| `stage_2.params.depth` | [28, 2] | 28 enc + 2 dec blocks |
| `stage_2.params.num_heads` | [16, 16] | Attention heads |

**Smaller model** (DiTDH-S, for distillation / autoguidance):

```bash
python train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml \
  --data-path data/imagenet/train \
  --results-dir ckpts/stage2-small \
  --precision bf16 \
  --wandb
```

**Online evaluation**: Add the `eval` block in config:

```yaml
eval:
  eval_interval: 25000      # Every N training steps
  eval_model: true
  data_path: 'data/imagenet/val/'
  reference_npz_path: 'data/imagenet/VIRTUAL_imagenet256_labeled.npz'
```

### 2. Sample Images

Single-device CFG sampling:

```bash
python sample.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --seed 42 \
  --cfg-scale 1.5
```

Distributed sampling (FID-50k ready):

```bash
python sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples/gen \
  --num-fid-samples 50000
```

**AutoGuidance** sampling (uses a smaller guidance model):

```bash
python sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
  --sample-dir samples/gen_ag
```

---

## TPU Parallelism

All training scripts use **mesh-based data parallelism** for TPU:

```
┌──────────────────────────────────────────────────┐
│ Mesh("data", 8)                                  │
│                                                  │
│  Core 0   Core 1   Core 2  ...  Core 7          │
│  ┌─────┐  ┌─────┐  ┌─────┐     ┌─────┐         │
│  │B/8  │  │B/8  │  │B/8  │     │B/8  │  ← Data  │
│  │     │  │     │  │     │     │     │  sharded  │
│  └──┬──┘  └──┬──┘  └──┬──┘     └──┬──┘         │
│     │        │        │           │              │
│  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐     ┌──▼──┐         │
│  │Model│  │Model│  │Model│     │Model│  ← Params │
│  │Copy │  │Copy │  │Copy │     │Copy │  replicated│
│  └──┬──┘  └──┬──┘  └──┬──┘     └──┬──┘         │
│     │        │        │           │              │
│     └────────┴────────┴───────────┘              │
│              ↓ XLA auto all-reduce               │
│         Synchronized gradients                   │
└──────────────────────────────────────────────────┘
```

**Key design decisions**:

- **No `jax.pmap`**: We use `jax.sharding.Mesh` + `NamedSharding` instead. Data is sharded with `P("data")`, params are replicated with `P()`.
- **Auto gradient reduction**: When computing `grad(loss)` on sharded data w.r.t. replicated params, XLA automatically inserts an all-reduce mean.
- **bfloat16**: Enabled by default (`--precision bf16`) for 2× throughput on TPU.
- **Gradient checkpointing**: Available via `nnx.remat` for large models.

---

## Evaluation

### FID with ADM Suite

```bash
# Download reference stats
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

# Generate 50k samples
python sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
  --sample-dir samples/fid \
  --num-fid-samples 50000

# Score with ADM evaluator
cd guided-diffusion/evaluation
python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
```

### Built-in Metrics

Run standalone metric computation:

```bash
python -m eval \
  --ref-img data/imagenet/val_256.npz \
  --rec-img samples/recon.npz \
  --bs 128
```

This computes **PSNR**, **SSIM**, **LPIPS**, and **rFID**.

---

## Config Reference

### Config Sections

| Section | Stage 1 | Stage 2 | Sampling | Description |
|---------|:-------:|:-------:|:--------:|-------------|
| `stage_1` | ✅ | ✅ | ✅ | RAE encoder + decoder definition |
| `stage_2` | — | ✅ | ✅ | DiT model definition |
| `transport` | — | ✅ | ✅ | Flow matching path & loss |
| `sampler` | — | ✅ | ✅ | ODE/SDE solver settings |
| `guidance` | — | ✅ | ✅ | CFG or AutoGuidance |
| `misc` | — | ✅ | ✅ | Latent size, class count |
| `training` | ✅ | ✅ | — | Epochs, LR, EMA, etc. |
| `eval` | ✅ | ✅ | — | Online eval settings |
| `gan` | ✅ | — | — | Discriminator + LPIPS loss |

### Stage 2 Model Variants

| Model | hidden_size | depth | #Params | Config |
|-------|-------------|-------|---------|--------|
| DiTDH-S | [384, 2048] | [12, 2] | ~130M | `DiTDH-S_DINOv2-B.yaml` |
| DiTDH-XL | [1152, 2048] | [28, 2] | ~675M | `DiTDH-XL_DINOv2-B.yaml` |

---

## Acknowledgement

This code is built upon:

- [RAE](https://github.com/bytetriper/RAE) — Original PyTorch implementation
- [SiT](https://github.com/willisma/sit) — Diffusion implementation
- [DDT](https://github.com/MCG-NJU/DDT) — DiT<sup>DH</sup> architecture
- [LightningDiT](https://github.com/hustvl/LightningDiT/) — Single-stream DiT
- [MAE](https://github.com/facebookresearch/mae) — ViT decoder architecture
