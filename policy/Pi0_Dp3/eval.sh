#!/bin/bash
# SG-DP3 Evaluation
# 在 RoboTwin 环境中评估训练好的 SG-DP3 模型
#
# Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
# Example: bash eval.sh beat_block_hammer demo_clean sg_dp3-train 50 42 0

policy_name=Pi0_Dp3
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..  # move to project root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name}
