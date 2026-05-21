#!/bin/bash
# SG-DP3 Training
# 训练 SG-DP3 模型 (自动检测并预处理数据)
#
# Usage: bash train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id>
# Example: bash train.sh beat_block_hammer demo_clean 50 42 0

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

# 如果 zarr 数据不存在，先进行预处理
if [ ! -d "./data/${task_name}/${task_config}/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# 训练配置
exp_name="${task_name}-sg_dp3-${task_config}"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="data/${task_name}/${task_config}/${task_name}-${task_config}-${expert_data_num}.zarr"

echo "[INFO] Task:        ${task_name}"
echo "[INFO] Setting:    ${task_config}"
echo "[INFO] Episodes:   ${expert_data_num}"
echo "[INFO] Seed:       ${seed}"
echo "[INFO] Output:     ${run_dir}"
echo "[INFO] Zarr:       ${zarr_path}"

# 启动训练
cd sg_dp3_workspace

python train_sg_dp3.py \
    --config sg_dp3_workspace/config/sg_dp3.yaml \
    --output_dir "../${run_dir}" \
    --zarr_path "../${zarr_path}" \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed}

cd ..
