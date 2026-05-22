#!/bin/bash
# RDT LoRA Fine-tuning for single GPU (e.g., RTX 4090 24GB)
# 
# Usage: bash finetune_lora.sh <config_name>
# Example: bash finetune_lora.sh beat_block_hammer
#
# LoRA 大幅减少可训练参数量，使 RDT 模型可以在单张消费级 GPU 上微调。
# 默认配置: rank=8, alpha=16, dropout=0.0
# 显存占用: ~16-20GB (batch_size=1, bf16, gradient_checkpointing可选)

CONFIG_NAME="${1:-beat_block_hammer}"
CONFIG_FILE="model_config/$CONFIG_NAME.yml"

echo "============================================"
echo "  RDT LoRA Fine-tuning (Single GPU)"
echo "  Config: $CONFIG_FILE"
echo "============================================"

### ===============================

# NCCL settings for single GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="../weights/RDT/siglip-so400m-patch14-384"
export WANDB_PROJECT="RDT"
export WANDB_DEFAULT_RUN_NAME="${CONFIG_NAME}_lora"

# ============ LoRA Hyperparameters ============
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16.0}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
# ==============================================

# check if YAML exist 
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file $CONFIG_FILE does not exist!"
  exit 1
fi

PRETRAINED_MODEL_NAME=$(python scripts/read_yaml.py "$CONFIG_FILE" pretrained_model_name_or_path)
TRAIN_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" train_batch_size)
SAMPLE_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_batch_size)
MAX_TRAIN_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" max_train_steps)
CHECKPOINTING_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpointing_period)
SAMPLE_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_period)
CHECKPOINTS_TOTAL_LIMIT=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoints_total_limit)
LR_SCHEDULER=$(python scripts/read_yaml.py "$CONFIG_FILE" lr_scheduler)
LEARNING_RATE=$(python scripts/read_yaml.py "$CONFIG_FILE" learning_rate)
DATALOADER_NUM_WORKERS=$(python scripts/read_yaml.py "$CONFIG_FILE" dataloader_num_workers)
DATASET_TYPE=$(python scripts/read_yaml.py "$CONFIG_FILE" dataset_type)
STATE_NOISE_SNR=$(python scripts/read_yaml.py "$CONFIG_FILE" state_noise_snr)
GRAD_ACCUM_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" gradient_accumulation_steps)
OUTPUT_DIR=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoint_path)
CUDA_USE=$(python scripts/read_yaml.py "$CONFIG_FILE" cuda_visible_device)

PRETRAINED_MODEL_NAME=$(echo "$PRETRAINED_MODEL_NAME" | tr -d '"')
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')

# Append "_lora" to output directory
OUTPUT_DIR="${OUTPUT_DIR}_lora_r${LORA_RANK}"

# create output path
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  echo "Created output directory: $OUTPUT_DIR"
else
  echo "Output directory already exists: $OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=$CUDA_USE

python -m data.compute_dataset_stat_hdf5 --task_name $CONFIG_NAME

# Single GPU training with LoRA
# NOTE: 不使用 deepspeed，单卡直接 accelerate 启动
# 如果显存不够，可以尝试：
#   1. 减小 TRAIN_BATCH_SIZE (在 yaml 配置中)
#   2. 增大 GRAD_ACCUM_STEPS (在 yaml 配置中)  
#   3. 添加 --use_8bit_adam 使用 8-bit 优化器

accelerate launch --main_process_port=28499 --num_processes 1 main.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_period=$CHECKPOINTING_PERIOD \
    --sample_period=$SAMPLE_PERIOD \
    --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
    --lr_scheduler="constant" \
    --learning_rate=$LEARNING_RATE \
    --mixed_precision="bf16" \
    --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=$STATE_NOISE_SNR \
    --load_from_hdf5 \
    --report_to=wandb \
    --precomp_lang_embed \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --model_config_path=$CONFIG_FILE \
    --CONFIG_NAME=$CONFIG_NAME \
    --use_lora \
    --lora_rank=$LORA_RANK \
    --lora_alpha=$LORA_ALPHA \
    --lora_dropout=$LORA_DROPOUT
