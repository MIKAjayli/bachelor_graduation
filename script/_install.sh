#!/bin/bash

# ============================================================
# 核心修正：强制隔离环境，确保所有包安装在当前 Conda 环境内
# ============================================================
export PYTHONNOUSERSITE=1

# 获取当前 Python 路径，用于二次确认
PYTHON_EXE=$(which python)
echo "Using Python from: $PYTHON_EXE"

echo "Installing the necessary packages ..."
# 使用 python -m pip 确保路径正确
$PYTHON_EXE -m pip install -r script/requirements.txt

echo "Installing pytorch3d ..."
# 注意：从 GitHub 安装可能需要较长时间编译
$PYTHON_EXE -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ============================================================
# 修正 SAPIEN 代码 (处理编码与路径问题)
# ============================================================
echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# 动态获取当前环境中 sapien 的安装位置
SAPIEN_LOCATION=$($PYTHON_EXE -m pip show sapien | grep 'Location' | awk '{print $2}')/sapien

if [ -d "$SAPIEN_LOCATION" ]; then
    URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
    echo "Found URDF loader at: $URDF_LOADER"
    # 给 open() 函数增加 encoding="utf-8" 避免非 UTF-8 环境报错
    # 并将 srdf 后缀补全点号
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER
    sed -i 's/urdf_file\[:-4\] + "srdf"/urdf_file\[:-4\] + ".srdf"/g' $URDF_LOADER
else
    echo "Warning: Sapien not found in environment. Skipping patch."
fi

# ============================================================
# 修正 MPLIB 代码 (移除碰撞检测限制 - RoboTwin 官方要求)
# ============================================================
echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION=$($PYTHON_EXE -m pip show mplib | grep 'Location' | awk '{print $2}')/mplib

if [ -d "$MPLIB_LOCATION" ]; then
    PLANNER=$MPLIB_LOCATION/planner.py
    echo "Found MPLIB planner at: $PLANNER"
    # 核心修改：移除 line 807 的 "or collide"
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER
else
    echo "Warning: MPLIB not found. You might need to install it first."
fi

# ============================================================
# 安装 Curobo (NVIDIA 运动规划加速)
# ============================================================
echo "Installing Curobo ..."
mkdir -p envs
cd envs
if [ ! -d "curobo" ]; then
    git clone https://github.com/NVlabs/curobo.git
fi
cd curobo
# 使用 --no-build-isolation 确保它使用当前环境中的 torch 和 cuda 环境
$PYTHON_EXE -m pip install -e . --no-build-isolation
cd ../..

echo "------------------------------------------------------------"
echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets: bash script/_download_assets.sh"
echo -e "    2. Verify Vulkan: run 'vulkaninfo' in terminal."
echo "------------------------------------------------------------"