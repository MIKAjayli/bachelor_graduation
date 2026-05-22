#!/bin/bash
# LoRA fine-tuning script for pi05 on single RTX 4090 (24GB VRAM)
# Usage: bash finetune_lora.sh <model_name> <gpu_id>
# Example: bash finetune_lora.sh my_task_lora 0

model_name=$1
gpu_use=${2:-0}

export CUDA_VISIBLE_DEVICES=$gpu_use
echo "=== pi05 LoRA Fine-tuning ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model name: $model_name"
echo "Config: pi05_aloha_lora_base"
echo ""

# Use XLA_PYTHON_CLIENT_MEM_FRACTION to limit GPU memory usage
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_lora_base --exp-name=$model_name --overwrite
