#!/bin/bash
# LoRA fine-tuning script for pi0 on single RTX 4090 (24GB VRAM)
# Usage: bash finetune_lora.sh <model_name> <gpu_id>
# Example: bash finetune_lora.sh my_task_lora 0

model_name=$1
gpu_use=${2:-0}

export CUDA_VISIBLE_DEVICES=$gpu_use
echo "=== pi0 LoRA Fine-tuning ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model name: $model_name"
echo "Config: pi0_base_aloha_robotwin_lora"
echo ""

# Use XLA_PYTHON_CLIENT_MEM_FRACTION to limit GPU memory usage
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_base_aloha_robotwin_lora --exp-name=$model_name --overwrite
