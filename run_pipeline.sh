#!/bin/bash
# =============================================================================
# RAE-JAX Full Training Pipeline
# =============================================================================
# Cách chạy trên TẤT CẢ workers (cần chạy ĐỒNG THỜI trên cả 2):
#
#   gcloud compute tpus tpu-vm ssh node-v4-16 \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="nohup bash ~/rae_jax/RAE/run_pipeline.sh > ~/pipeline.log 2>&1 & echo PID:\$!"
#
# Xem log real-time:
#   gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=0 \
#     --command="tail -f ~/pipeline.log"
#
# Kill toàn bộ nếu cần:
#   gcloud compute tpus tpu-vm ssh node-v4-16 --zone=us-central2-b --worker=all \
#     --command="pkill -9 -f 'python|run_pipeline' 2>/dev/null; echo killed"
# =============================================================================

set -e
cd ~/rae_jax/RAE
export PYTHONUNBUFFERED=1

LOG=~/pipeline.log
ts() { echo "[$(date '+%H:%M:%S')]"; }

# Detect worker ID từ hostname (t1v-n-XXXXX-w-0 hoặc w-1)
HOSTNAME=$(hostname)
IS_MAIN=false
[[ "$HOSTNAME" == *"-w-0"* ]] && IS_MAIN=true

echo "$(ts) ================================================" | tee -a $LOG
echo "$(ts) START PIPELINE | host=$HOSTNAME | is_main=$IS_MAIN" | tee -a $LOG
echo "$(ts) ================================================" | tee -a $LOG

# =============================================================================
# STEP 1: Calculate normalization stats  ← ĐÃ CHẠY XONG, BỎ QUA
# =============================================================================
STAT_PATH=models/stats/dinov2_celebahq256/normalization_stats.npz

# if [ -f "$STAT_PATH" ]; then
#   echo "$(ts) [SKIP] Norm stats đã có: $STAT_PATH" | tee -a $LOG
# else
#   echo "$(ts) === STEP 1: Calculate normalization stats ===" | tee -a $LOG
#   python calculate_stat.py \
#     --config configs/stage1/pretrained/DINOv2-B.yaml \
#     --data-path ~/tensorflow_datasets \
#     --output-dir models/stats/dinov2_celebahq256 \
#     --image-size 256 \
#     --batch-size 32 \
#     --num-samples 30000 \
#     --dataset-type tfds \
#     --tfds-name celebahq256 2>&1 | tee -a $LOG
#   echo "$(ts) === STEP 1 DONE ===" | tee -a $LOG
# fi
echo "$(ts) [SKIP] STEP 1 đã hoàn thành trước đó." | tee -a $LOG

# =============================================================================
# STEP 2: Train Stage 1  (multi-host — cả 2 workers cùng chạy)
# =============================================================================
STAGE1_CKPT_DIR=ckpts/stage1/celebahq256_dinov2b_decXL/checkpoints
STAGE1_LAST=$STAGE1_CKPT_DIR/ckpt_last.pkl

if [ -f "$STAGE1_LAST" ]; then
  echo "$(ts) [SKIP] Stage 1 checkpoint đã có: $STAGE1_LAST" | tee -a $LOG
else
  echo "$(ts) === STEP 2: Train Stage 1 ===" | tee -a $LOG
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
    --wandb-project rae-jax-stage1 2>&1 | tee -a $LOG
  echo "$(ts) === STEP 2 DONE ===" | tee -a $LOG
fi

# =============================================================================
# STEP 3: Extract decoder weights  (single-host — chỉ w-0 chạy, w-1 chờ)
# Lưu ý: extract_decoder KHÔNG dùng JAX multi-host nên PHẢI tách riêng
# =============================================================================
DECODER_PATH=models/decoders/dinov2/celebahq256/ViTXL/model.npz

if $IS_MAIN; then
  if [ -f "$DECODER_PATH" ]; then
    echo "$(ts) [SKIP][w0] Decoder đã có: $DECODER_PATH" | tee -a $LOG
  else
    echo "$(ts) === STEP 3: Extract decoder (w0 only) ===" | tee -a $LOG
    # Chạy với JAX_PLATFORMS=cpu để tránh chiếm TPU lock
    JAX_PLATFORMS=cpu python extract_decoder.py \
      --config configs/stage1/pretrained/DINOv2-B.yaml \
      --ckpt "$STAGE1_CKPT_DIR" \
      --use-ema \
      --out "$DECODER_PATH" 2>&1 | tee -a $LOG
    echo "$(ts) === STEP 3 DONE ===" | tee -a $LOG
  fi
else
  echo "$(ts) [w1] Chờ w0 extract decoder..." | tee -a $LOG
  for i in $(seq 1 120); do
    [ -f "$DECODER_PATH" ] && break
    sleep 5
    [ $((i % 12)) -eq 0 ] && echo "$(ts) [w1] Vẫn đang chờ... ($((i*5))s)" | tee -a $LOG
  done
  if [ ! -f "$DECODER_PATH" ]; then
    echo "$(ts) [w1] WARNING: decoder không xuất hiện sau 10 phút, tiếp tục không có nó." | tee -a $LOG
  else
    echo "$(ts) [w1] Decoder sẵn sàng, tiếp tục." | tee -a $LOG
  fi
fi

# Barrier: đợi cả 2 workers đến đây (dùng flag file)
BARRIER=/tmp/step3_done_$HOSTNAME
touch $BARRIER
if $IS_MAIN; then
  for i in $(seq 1 30); do
    [ -f "/tmp/step3_done_$(hostname | sed 's/w-0/w-1/')" ] 2>/dev/null && break || true
    sleep 3
  done
fi

# =============================================================================
# STEP 4: Train Stage 2  (multi-host — cả 2 workers cùng chạy)
# =============================================================================
echo "$(ts) === STEP 4: Train Stage 2 ===" | tee -a $LOG
python train.py \
  --data-path ~/tensorflow_datasets \
  --dataset-type tfds \
  --tfds-name celebahq256 \
  --num-train-samples 30000 \
  --rae-checkpoint "$STAGE1_LAST" \
  --normalization-stat-path "$STAT_PATH" \
  --results-dir ckpts/stage2 \
  --experiment-name stage2-DiTDH-S \
  --epochs 1400 \
  --global-batch-size 64 \
  --lr 2e-4 \
  --eval-fid-every 1000 \
  --num-fid-samples 10000 \
  --wandb \
  --wandb-project rae-jax-stage2 2>&1 | tee -a $LOG

echo "$(ts) === PIPELINE HOÀN TẤT ===" | tee -a $LOG
