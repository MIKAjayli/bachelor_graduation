# SG-DP3 使用指南

> **Semantic-Guided 3D Diffusion Policy** — 融合 Pi0 语义理解 + DP3 3D 几何精度

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 数据准备](#2-数据准备)
- [3. 配置文件说明](#3-配置文件说明)
- [4. 训练](#4-训练)
- [5. 评估](#5-评估)
- [6. 微调](#6-微调)
- [7. 输出结构](#7-输出结构)
- [8. 常见问题](#8-常见问题)

---

## 1. 环境准备

### 1.1 基础依赖

```bash
# PyTorch (根据 CUDA 版本选择)
pip install torch torchvision

# 一键安装所有依赖 (推荐)
pip install -r requirements.txt

# 或手动安装核心依赖
pip install omegaconf einops termcolor tqdm zarr h5py

# 扩散调度器 (推荐)
pip install diffusers

# VLM 模型 (两种模式二选一)
# 轻量模式: 仅需 torch (默认启用, 无需额外安装)
# 完整 PaliGemma 模式:
pip install transformers Pillow
```

### 1.2 验证安装

```bash
cd policy/Pi0_Dp3
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

### 1.3 项目结构速览

```
policy/Pi0_Dp3/
├── __init__.py                        #   导出 deploy_policy
├── deploy_policy.py                   # ← RoboTwin 评估接口 (get_model / eval / reset_model)
├── deploy_policy.yml                  #   评估参数配置
├── process_data.sh                    # ← 数据预处理入口
├── train.sh                           # ← 训练入口
├── eval.sh                            # ← 评估入口 (调用 script/eval_policy.py)
├── scripts/
│   ├── process_data.py                #   数据预处理 Python 实现
│   └── process_data.sh                #   数据预处理 wrapper
├── sg_dp3_workspace/
│   ├── config/
│   │   ├── sg_dp3.yaml               # ← 主配置文件
│   │   └── task/demo_task.yaml       #   任务配置模板
│   ├── dataset/
│   │   └── multimodal_dataset.py     #   多模态数据集
│   ├── model/
│   │   ├── semantic/
│   │   │   ├── pi0_wrapper.py        #   VLM 语义编码器
│   │   │   └── purification.py       #   语义提纯模块
│   │   ├── vision/
│   │   │   └── pointnet.py           #   PointNet 几何编码器
│   │   └── diffusion/
│   │       ├── cascaded_unet.py      #   级联解耦 U-Net
│   │       └── ddim_scheduler.py     #   DDIM 调度器
│   ├── policy/
│   │   └── sg_dp3_policy.py          #   总体策略
│   ├── train_sg_dp3.py               # ← 训练 Python 入口
│   └── eval_sg_dp3.py                # ← 独立评估入口
├── data/                              #   数据目录 (预处理后自动生成)
│   ├── Adjust_Bottle/                  #   按任务名建子目录
│   │   ├── demo_clean/                #   按 task_config 建子目录
│   │   │   └── ...-{num}.zarr         #   zarr 格式训练数据
│   │   └── demo_randomized/
│   │       └── ...-{num}.zarr
│   └── outputs/                       #   训练输出 & checkpoint
└── USAGE_GUIDE.md                     #   本文件
```

---

## 2. 数据准备

### 2.1 原始数据格式

RoboTwin 采集的数据存储在项目根目录下的 `data/` 目录中：

```
data/
├── {task_name}/
│   ├── demo_clean/                    # 干净场景数据
│   │   └── data/
│   │       ├── episode0.hdf5
│   │       ├── episode1.hdf5
│   │       └── ...
│   └── demo_randomized/              # 随机化场景数据
│       └── data/
│           ├── episode0.hdf5
│           └── ...
```

每个 HDF5 文件包含：

| 字段 | 路径 | Shape | 说明 |
|------|------|-------|------|
| 左夹爪 | `/joint_action/left_gripper` | `(T,)` | 左夹爪开合 |
| 左臂 | `/joint_action/left_arm` | `(T, D)` | 左臂关节角 |
| 右夹爪 | `/joint_action/right_gripper` | `(T,)` | 右夹爪开合 |
| 右臂 | `/joint_action/right_arm` | `(T, D)` | 右臂关节角 |
| 向量 | `/joint_action/vector` | `(T, D_state)` | 完整状态向量 |
| 点云 | `/pointcloud` | `(T, N, 3\|6)` | XYZ 或 XYZ+RGB |
| 图像 | `/observation/{cam}/rgb` | `(T, H, W, C)` | 相机 RGB [可选] |

### 2.2 运行数据预处理

将 HDF5 数据转换为 zarr 格式（与 DP3 格式兼容）：

```bash
cd policy/Pi0_Dp3

# 用法
bash process_data.sh <task_name> <task_config> <expert_data_num>

# 示例: 处理 beat_block_hammer 任务, 干净场景, 50 个 episode
bash process_data.sh beat_block_hammer demo_clean 50

# 示例: 处理随机化场景
bash process_data.sh beat_block_hammer demo_randomized 50
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `task_name` | 任务名称 (对应 envs/ 下的文件名) | `beat_block_hammer` |
| `task_config` | 数据设置 (`demo_clean` 或 `demo_randomized`) | `demo_clean` |
| `expert_data_num` | 要处理的 episode 数量 | `50` |

**输出位置:** `policy/Pi0_Dp3/data/{task_name}/{task_config}/{task_name}-{task_config}-{num}.zarr`

输出 zarr 包含:
- `data/point_cloud`: `(total_steps, N, 3)` — 点云
- `data/state`: `(total_steps, D_state)` — 机器人状态
- `data/action`: `(total_steps, D_action)` — 动作
- `data/images/`: [可选] 图像数据
- `meta/episode_ends`: `(num_episodes,)` — episode 边界

### 2.3 跳过预处理

`train.sh` 会自动检测 zarr 数据是否存在，若不存在会先调用 `process_data.sh`。也可以手动跳过：

```bash
# 如果已有 zarr 数据，直接训练即可
bash train.sh beat_block_hammer demo_clean 50 42 0
```

---

## 3. 配置文件说明

### 3.1 主配置文件 `sg_dp3_workspace/config/sg_dp3.yaml`

```yaml
# ===================== 关键参数速查 =====================

# --- 训练 ---
training:
  num_epochs: 600
  batch_size: 64
  lr: 1.0e-4
  seed: 42
  use_ema: true
  ema_decay: 0.999
  checkpoint_every: 50                # 每 N epoch 保存一次
  val_every: 10                       # 每 N epoch 验证一次

# --- 模型 ---
policy:
  shape_meta:
    action:
      shape: [7]                      # 动作维度
    obs:
      point_cloud:
        shape: [1024, 3]              # 点云 (N, 3)
      agent_pos:
        shape: [14]                   # 状态维度

  horizon: 16                         # 动作序列长度
  n_action_steps: 8                   # 实际执行步数
  n_obs_steps: 2                      # 观测步数

  # 语义模块
  use_light_vlm: true                 # true=轻量CNN(推荐), false=完整PaliGemma
  semantic_feature_dim: 256           # 语义条件维度

  # 语义提纯
  purification_num_points: 1024       # 提纯后目标点数 (N_min)
```

### 3.2 评估配置 `deploy_policy.yml`

```yaml
# RoboTwin eval_policy.py 使用的参数
config_name: robot_sg_dp3
checkpoint_num: 3000                  # checkpoint 编号
expert_data_num: null                 # episode 数量
use_rgb: false                        # 是否使用 RGB 点云
use_light_vlm: true                   # VLM 模式
```

### 3.3 自定义新任务的关键参数映射

| 任务属性 | 配置路径 | 示例 |
|----------|----------|------|
| 动作维度 | `policy.shape_meta.action.shape` | `[7]` (单臂), `[14]` (双臂) |
| 状态维度 | `policy.shape_meta.obs.agent_pos.shape` | `[14]` |
| 点云点数 | `policy.shape_meta.obs.point_cloud.shape` | `[1024, 3]` |
| 序列长度 | `policy.horizon` | `16` |
| 图像尺寸 | `policy.image_height/width` | `224` |

---

## 4. 训练

### 4.1 快速开始

```bash
cd policy/Pi0_Dp3

# 用法
bash train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id>

# 示例: 训练 beat_block_hammer, 干净场景, 50 episodes, seed=42, GPU 0
bash train.sh beat_block_hammer demo_clean 50 42 0

# 示例: 训练随机化场景
bash train.sh beat_block_hammer demo_randomized 50 42 0
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `task_name` | 任务名称 | `beat_block_hammer` |
| `task_config` | 数据设置 | `demo_clean` / `demo_randomized` |
| `expert_data_num` | episode 数量 | `50` |
| `seed` | 随机种子 | `42` |
| `gpu_id` | GPU 编号 | `0` |

> ⚡ `train.sh` 会自动检测 zarr 数据是否存在，不存在则先运行 `process_data.sh`

### 4.2 直接运行 Python 脚本

```bash
cd policy/Pi0_Dp3

python sg_dp3_workspace/train_sg_dp3.py \
    --config config/sg_dp3.yaml \
    --output_dir data/outputs/beat_block_hammer-sg_dp3-demo_clean_seed42
```

### 4.3 训练流程详解

```
┌─────────────────────────────────────────────────────────┐
│  每个 training step:                                     │
│                                                          │
│  1. 加载 batch: {obs: {point_cloud, agent_pos, image},   │
│                   action: (B, T, D_act)}                 │
│       ↓                                                  │
│  2. 语义编码 (VLM 冻结):                                 │
│     Image → PaliGemma/CNN → c_sem (B, 256) + 2D Mask    │
│       ↓                                                  │
│  3. 语义提纯:                                            │
│     3D PointCloud + 2D Mask → Purified PointCloud       │
│     (点数不足时有放回重采样, 严禁均匀噪声)                 │
│       ↓                                                  │
│  4. 几何编码 (PointNet):                                  │
│     Purified Points → c_geo (B, 256)                     │
│       ↓                                                  │
│  5. 扩散去噪 (Cascaded U-Net):                           │
│     w_sem = τ/T_max,  w_geo = 1 - w_sem                 │
│     c_stage = w_sem * c_sem + w_geo * c_geo              │
│     → Loss = MSE(predicted_noise, true_noise)            │
│       ↓                                                  │
│  6. 反向传播 → 更新可训练参数 (VLM 冻结)                  │
│  7. EMA 参数更新                                          │
└─────────────────────────────────────────────────────────┘
```

### 4.4 训练输出

```
policy/Pi0_Dp3/data/outputs/{task}-sg_dp3-{setting}_seed{seed}/
├── checkpoints/
│   ├── 50.ckpt
│   ├── 100.ckpt
│   ├── ...
│   └── final.ckpt
└── logs.json.txt
```

### 4.5 断点续训

编辑 `sg_dp3_workspace/config/sg_dp3.yaml`，设置:

```yaml
training:
  resume: true
```

训练脚本会自动在 output_dir/checkpoints/ 中查找最新 checkpoint 并恢复训练。

---

## 5. 评估

### 5.1 通过 RoboTwin 标准评估流程

SG-DP3 实现了与 DP3 相同的 `deploy_policy` 接口，可以直接使用 RoboTwin 的标准评估脚本:

```bash
cd policy/Pi0_Dp3

# 用法
bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>

# 示例
bash eval.sh beat_block_hammer demo_clean sg_dp3-train 50 42 0
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `task_name` | 任务名称 | `beat_block_hammer` |
| `task_config` | 数据设置 | `demo_clean` / `demo_randomized` |
| `ckpt_setting` | checkpoint 设置名 | `sg_dp3-train` |
| `expert_data_num` | episode 数量 | `50` |
| `seed` | 随机种子 | `42` |
| `gpu_id` | GPU 编号 | `0` |

> `eval.sh` 会自动 `cd ../..` 到项目根目录，然后调用 `script/eval_policy.py`

### 5.2 评估内部流程

```
eval.sh
  → cd ../..  (项目根目录)
  → python script/eval_policy.py \
        --config policy/Pi0_Dp3/deploy_policy.yml \
        --overrides \
        --task_name beat_block_hammer \
        --task_config demo_clean \
        --ckpt_setting sg_dp3-train \
        --expert_data_num 50 \
        --seed 42 \
        --policy_name Pi0_Dp3

  eval_policy.py 内部:
    → import policy.Pi0_Dp3  (触发 __init__.py)
    → get_model(usr_args)    (加载模型 + RobotRunner)
    → for each episode:
        reset_model(model)
        eval(TASK_ENV, model, observation)
```

### 5.3 deploy_policy 接口说明

`deploy_policy.py` 实现了三个标准接口函数:

| 函数 | 说明 |
|------|------|
| `get_model(usr_args)` | 加载 SG-DP3 模型和 RobotRunner |
| `eval(TASK_ENV, model, observation)` | 执行一步推理: 观测 → 动作序列 → 逐步执行 |
| `reset_model(model)` | 每个评估 episode 开始时清空观测缓存 |

### 5.4 独立推理 (不依赖 RoboTwin 环境)

```python
import torch
import numpy as np
from omegaconf import OmegaConf

# 加载模型
from policy.Pi0_Dp3.deploy_policy import get_model

usr_args = {
    "task_name": "beat_block_hammer",
    "task_config": "demo_clean",
    "ckpt_setting": "sg_dp3-train",
    "expert_data_num": 50,
    "seed": 42,
    "checkpoint_num": 3000,
    "use_light_vlm": True,
    "use_rgb": False,
}
model = get_model(usr_args)

# 推理
obs = {
    "agent_pos": joint_state,     # numpy array
    "point_cloud": pointcloud,    # numpy array
}
model.update_obs(obs)
actions = model.get_action()      # (n_action_steps, action_dim)
```

---

## 6. 微调

### 6.1 使用场景

- 已有一个预训练好的 SG-DP3 模型
- 需要迁移到新任务或新环境
- 需要在少量新数据上适应

### 6.2 基本用法

```bash
cd policy/Pi0_Dp3

# Step 1: 准备新任务数据
bash process_data.sh <new_task> demo_clean 30

# Step 2: 修改配置，指向预训练 checkpoint
# 编辑 sg_dp3_workspace/config/sg_dp3.yaml:
#   training:
#     resume: true        # 启用断点续训
#     lr: 1.0e-5          # 更小的学习率
#     num_epochs: 100     # 更少的 epoch

# Step 3: 从预训练 checkpoint 开始训练
bash train.sh <new_task> demo_clean 30 42 0
```

### 6.3 微调参数推荐

| 参数 | 从头训练 | 微调（推荐） |
|------|---------|-------------|
| `lr` | `1e-4` | `1e-5` (小 10 倍) |
| `num_epochs` | `600` | `50~100` |
| `batch_size` | `64` | `16~32` |

### 6.4 微调 vs 从头训练对比

```
从头训练:
  所有参数随机初始化 → 600 epoch → 适合数据充足的全新任务

微调:
  加载预训练权重 → 冻结部分层 → 50~100 epoch → 适合数据较少的新任务
  ┌──────────────────────────────────┐
  │ VLM (PaliGemma/CNN)  → 冻结 ❄️   │  (语义理解通用)
  │ SegHead + Projection → 训练 🔥   │  (适配新任务掩码)
  │ Purifier             → 训练 🔥   │  (适配新物体)
  │ PointNet             → 可选冻结   │
  │ Cascaded U-Net       → 训练 🔥   │  (适配新动作分布)
  └──────────────────────────────────┘
```

---

## 7. 输出结构

训练/评估的输出全部保存在 `policy/Pi0_Dp3/` 目录下：

```
policy/Pi0_Dp3/
├── data/
│   ├── Adjust_Bottle/
│   │   ├── demo_clean/
│   │   │   └── Adjust_Bottle-demo_clean-50.zarr
│   │   └── demo_randomized/
│   │       └── Adjust_Bottle-demo_randomized-50.zarr
│   ├── Beat_Block_Hammer/
│   │   ├── demo_clean/
│   │   │   └── Beat_Block_Hammer-demo_clean-50.zarr
│   │   └── demo_randomized/
│   │       └── Beat_Block_Hammer-demo_randomized-50.zarr
│   └── outputs/
│       ├── Beat_Block_Hammer-sg_dp3-demo_clean_seed42/  # 训练输出
│       │   ├── checkpoints/
│       │   │   ├── 50.ckpt
│       │   │   ├── 100.ckpt
│       │   │   └── ...
│       │   └── logs.json.txt
│       └── Beat_Block_Hammer-sg_dp3-demo_randomized_seed42/
│           └── ...
└── eval_result/                                      # 评估结果 (由 RoboTwin 生成)
    └── {task}/Pi0_Dp3/{setting}/{ckpt}/
        └── _result.txt
```

### Checkpoint 内容

```python
checkpoint = torch.load("3000.ckpt")
# 包含:
# checkpoint["model"]          → 模型参数
# checkpoint["ema_model"]      → EMA 模型参数 (推荐用于推理)
# checkpoint["optimizer"]      → 优化器状态
# checkpoint["lr_scheduler"]   → 学习率调度器状态
# checkpoint["epoch"]          → 当前 epoch
# checkpoint["global_step"]    → 全局步数
```

---

## 8. 常见问题

### Q1: 内存不足 (OOM)

```yaml
# 方法 1: 减小 batch_size
training:
  batch_size: 16

# 方法 2: 启用梯度累积 (等效大 batch)
training:
  batch_size: 16
  gradient_accumulate_every: 4   # 等效 batch_size=64

# 方法 3: 减少点云点数
policy:
  purification_num_points: 512
  shape_meta:
    obs:
      point_cloud:
        shape: [512, 3]

# 方法 4: 使用轻量 VLM
policy:
  use_light_vlm: true   # 不加载 PaliGemma (节省 ~6GB 显存)
```

### Q2: 没有图像数据，只有点云

SG-DP3 的核心使用点云数据。如果没有图像，确保 `use_light_vlm: true`（默认值），模型将只使用点云 + 轻量 CNN 语义模块。

### Q3: 如何使用完整 PaliGemma VLM

```yaml
policy:
  use_light_vlm: false                      # 启用完整 PaliGemma
  vlm_model_name: google/paligemma-3b-mix-224
  use_text_condition: true
  max_token_len: 48
```

> ⚠️ 需要约 6GB 额外显存 (VLM 权重冻结，仅前向传播)。首次运行会自动下载模型权重。

### Q4: 与 DP3 的差异

| 维度 | DP3 | SG-DP3 |
|------|-----|--------|
| 语义模块 | 无 | VLM 语义编码 + 语义提纯 |
| 条件注入 | 单一条件 | 级联解耦注入 (w_sem + w_geo) |
| 数据格式 | zarr (point_cloud + state + action) | 相同 (额外可选 images) |
| 评估接口 | deploy_policy.py | deploy_policy.py (接口一致) |
| 训练入口 | Hydra + train_dp3.py | OmegaConf + train_sg_dp3.py |

### Q5: Windows 系统如何运行

Windows 下不支持 `.sh` 脚本，请直接运行 Python 命令：

```powershell
cd policy\Pi0_Dp3

# 数据预处理
python scripts\process_data.py beat_block_hammer demo_clean 50

# 训练
python sg_dp3_workspace\train_sg_dp3.py --config config\sg_dp3.yaml --output_dir data\outputs\my_task --zarr_path data\my_task\demo_clean\my_task-demo_clean-50.zarr

# 评估 (需要 RoboTwin 环境在项目根目录下)
cd ..\..
python script\eval_policy.py --config policy\Pi0_Dp3\deploy_policy.yml --overrides --task_name beat_block_hammer --task_config demo_clean --ckpt_setting sg_dp3-train --expert_data_num 50 --seed 42 --policy_name Pi0_Dp3
```

### Q6: 可训练参数统计

```
模块                    参数量 (约)     是否训练
─────────────────────────────────────────────
VLM (PaliGemma)         ~3B            ❄️ 冻结
SegHead                 ~0.3M          🔥 训练
SemanticProjection      ~0.1M          🔥 训练
Purifier                ~0.02M         🔥 训练
PointNet Encoder        ~0.5M          🔥 训练
GeoProjection           ~0.1M          🔥 训练
Cascaded U-Net          ~5M            🔥 训练
─────────────────────────────────────────────
总计可训练              ~6M
```

---

## 快速命令速查表

```bash
cd policy/Pi0_Dp3

# === 数据预处理 ===
bash process_data.sh <task_name> <task_config> <expert_data_num>
bash process_data.sh beat_block_hammer demo_clean 50

# === 训练 (自动预处理) ===
bash train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id>
bash train.sh beat_block_hammer demo_clean 50 42 0

# === 评估 (RoboTwin 标准流程) ===
bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
bash eval.sh beat_block_hammer demo_clean sg_dp3-train 50 42 0

# === Windows (PowerShell) ===
python scripts\process_data.py beat_block_hammer demo_clean 50
python sg_dp3_workspace\train_sg_dp3.py --config config\sg_dp3.yaml
```
