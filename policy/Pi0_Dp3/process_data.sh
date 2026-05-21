#!/bin/bash
# SG-DP3 Data Processing
# 将 RoboTwin 采集的原始 HDF5 数据转换为 zarr 格式
#
# Usage: bash process_data.sh <task_name> <task_config> <expert_data_num>
# Example: bash process_data.sh beat_block_hammer demo_clean 50
#
# task_config: demo_clean | demo_randomized

task_name=${1}
task_config=${2}
expert_data_num=${3}

python scripts/process_data.py $task_name $task_config $expert_data_num
