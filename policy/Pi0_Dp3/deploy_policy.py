"""
SG-DP3 Deploy Policy Interface.

提供 RoboTwin eval_policy.py 所需的三个接口函数:
  - get_model(usr_args): 加载模型和 RobotRunner
  - eval(TASK_ENV, model, observation): 单步推理循环
  - reset_model(model): 重置观测缓存

参考: policy/DP3/deploy_policy.py
"""

import sys
import os
import pathlib
import numpy as np
import torch
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# 添加 sg_dp3_workspace 到 Python 路径
sys.path.insert(0, os.path.join(parent_directory, "sg_dp3_workspace"))
sys.path.insert(0, parent_directory)

from sg_dp3_workspace.policy.sg_dp3_policy import SGDP3Policy


# ============ RobotRunner (参考 DP3 的 RobotRunner) ============

class RobotRunner:
    """
    观测缓存管理器。
    维护最近 n_obs_steps 帧的观测窗口，用于滑动窗口推理。

    改进:
      - Temporal Ensembling: 指数加权平均重叠预测，平滑轨迹
      - 减少执行步数: 每次推理只执行 exec_steps 步后重新规划
      - 夹爪锁定: 一旦夹爪闭合（抓住物体），强制保持闭合
    """

    def __init__(self, n_obs_steps=2, n_action_steps=8, exec_steps=None,
                 use_temporal_ensembling=True, ema_weight=0.5):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.exec_steps = exec_steps if exec_steps is not None else n_action_steps
        self.use_temporal_ensembling = use_temporal_ensembling
        self.ema_weight = ema_weight  # temporal ensembling 的 EMA 权重

        self.obs = []

        # Temporal ensembling 缓存: 存储每步的加权累积动作和权重
        self._action_ema = None
        self._ema_count = None

        # 夹爪锁定状态
        self._gripper_locked = [False, False]  # [left, right]
        self._GRIPPER_DIMS = [6, 13]  # 左臂夹爪 dim=6, 右臂夹爪 dim=13
        self._GRIP_CLOSE_THRESH = 0.5  # 夹爪值 < 此阈值认为闭合

        # ===== 手臂一致性选择 =====
        # 在第一次推理后确定使用哪只手臂，之后锁定
        self._active_arm = None  # 'left', 'right', or 'both' — None 表示尚未决定
        self._arm_decision_step = 0  # 记录第几次 get_action 时决定
        # 左臂 dims: 0-5, 左夹爪: 6; 右臂 dims: 7-12, 右夹爪: 13
        self._LEFT_ARM_DIMS = list(range(0, 6))
        self._RIGHT_ARM_DIMS = list(range(7, 13))

        # ===== 进度追踪 =====
        # 记录活跃手臂曾经达到的最大位置，用于防止"回家"
        self._max_reached_qpos = None  # 活跃手臂的历史最大 qpos
        self._progress_locked = False  # 是否已进入"搬运阶段"

    def reset_obs(self):
        """清空观测缓存和所有推理状态。"""
        self.obs.clear()
        self._action_ema = None
        self._ema_count = None
        self._gripper_locked = [False, False]
        self._active_arm = None
        self._arm_decision_step = 0
        self._max_reached_qpos = None
        self._progress_locked = False
        self._clip_call_count = 0

    def update_obs(self, current_obs):
        """添加新的观测到缓存。"""
        self.obs.append(current_obs)

    def _stack_last_n_obs(self, all_obs, n_steps):
        """将观测列表堆叠为固定长度的数组，不足时用首帧填充。"""
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            # 统一到最后一帧的形状（处理变长点云等场景）
            ref_shape = all_obs[-1].shape
            ref_dtype = all_obs[-1].dtype
            result = np.zeros((n_steps,) + ref_shape, dtype=ref_dtype)
            start_idx = -min(n_steps, len(all_obs))
            # 逐帧填充，形状不匹配时裁剪或填充
            for i, idx in enumerate(range(max(0, len(all_obs) - n_steps), len(all_obs))):
                obs = all_obs[idx]
                if obs.shape == ref_shape:
                    result[start_idx + i] = obs
                else:
                    # 变长数据：裁剪或零填充到目标形状
                    sliced = np.zeros(ref_shape, dtype=ref_dtype)
                    min_len = min(obs.shape[0], ref_shape[0])
                    if obs.ndim == 1:
                        sliced[:min_len] = obs[:min_len]
                    else:
                        sliced[:min_len] = obs[:min_len, ...]
                    result[start_idx + i] = sliced
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def get_n_steps_obs(self):
        """获取最近 n_obs_steps 帧的堆叠观测。"""
        assert len(self.obs) > 0, "no observation recorded, call update_obs first"

        result = {}
        for key in self.obs[0].keys():
            result[key] = self._stack_last_n_obs(
                [obs[key] for obs in self.obs], self.n_obs_steps
            )
        return result

    def get_action(self, policy, observation=None, pc_template=None):
        """
        使用策略模型根据观测缓存预测动作。

        改进:
          1. Temporal Ensembling: 对重叠预测取指数加权平均
          2. 执行 exec_steps 步 (而非全部 n_action_steps)
          3. 夹爪锁定: 一旦夹爪闭合，强制保持
          4. PC 模板替换: 用训练数据 PC 替换环境 PC

        Returns:
            action: (exec_steps, action_dim) numpy array
        """
        device, dtype = policy.device, policy.dtype

        if observation is not None:
            self.obs.append(observation)

        obs = self.get_n_steps_obs()

        # ===== PC 模板替换 =====
        # 用训练数据的 PC 模板替换环境 PC 的 XYZ 通道
        obs_pc = obs["point_cloud"]  # (n_obs_steps, 1024, 6)

        # ===== PC 策略: 全程使用 PC 模板 =====
        # 使用训练 PC 模板确保模型输出稳定
        if pc_template is not None:
            template_xyz = pc_template[:, :3]  # (1024, 3)
            n_steps = obs_pc.shape[0]
            new_pc = obs_pc.copy()
            for t in range(n_steps):
                new_pc[t, :, :3] = template_xyz
            obs_pc = new_pc

        # 构造模型输入 (添加 batch 维度)
        obs_dict = {
            "point_cloud": torch.from_numpy(obs_pc.astype(np.float32)).unsqueeze(0).to(device=device),
            "agent_pos": torch.from_numpy(obs["agent_pos"].astype(np.float32)).unsqueeze(0).to(device=device),
        }

        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        action_pred = action_dict["action"].squeeze(0).cpu().numpy()  # (n_action_steps, action_dim)

        # Debug: 记录原始预测 (每隔几次)
        self._raw_pred_count = getattr(self, '_raw_pred_count', 0) + 1
        if self._raw_pred_count % 10 == 0:
            active = self._active_arm or 'unknown'
            print(f"  [RAW_PRED #{self._raw_pred_count}] arm={active}")
            print(f"    raw_pred[0]: {np.array2string(action_pred[0], precision=3, floatmode='fixed')}")
            if action_pred.shape[0] > 1:
                print(f"    raw_pred[-1]: {np.array2string(action_pred[-1], precision=3, floatmode='fixed')}")

        # ===== Temporal Ensembling (暂时禁用以调试) =====
        if False and self.use_temporal_ensembling:
            action_pred = self._temporal_ensemble(action_pred)

        # ===== 只取 exec_steps 步 =====
        action = action_pred[:self.exec_steps]

        # ===== 手臂一致性选择 =====
        action = self._apply_arm_consistency(action)

        # ===== 夹爪锁定 =====
        action = self._apply_gripper_lock(action)

        # ===== 动作跳变限制 =====
        # 传入完整 action_pred 以便使用完整的轨迹目标
        action = self._clip_action_jump(action, full_pred=action_pred)

        return action

    def _temporal_ensemble(self, new_actions):
        """
        Temporal Ensembling: 对重叠预测取指数加权平均。

        原理:
          - 每次推理产生 n_action_steps 步预测
          - 但只执行 exec_steps 步，所以下次推理时有重叠
          - 对重叠部分: out = ema_weight * old_ema + (1 - ema_weight) * new_pred
          - 这能平滑轨迹，防止模型预测突然跳变

        Args:
            new_actions: (n_action_steps, action_dim) 新预测

        Returns:
            smoothed: (n_action_steps, action_dim) 平滑后的预测
        """
        if self._action_ema is None:
            # 首次推理，初始化缓存
            self._action_ema = new_actions.copy()
            self._ema_count = np.zeros(new_actions.shape[0])
        else:
            # 有之前的预测缓存
            # 之前缓存的前 exec_steps 步已执行，剩余 n_action_steps - exec_steps 步需要保留
            remaining = self._action_ema[self.exec_steps:]
            n_remaining = remaining.shape[0]
            n_new = new_actions.shape[0]

            # 对齐重叠部分
            # remaining: [old_t_exec, ..., old_t_n-1]  (n_remaining steps)
            # new_actions: [new_t_0, ..., new_t_n-1]    (n_new steps)
            # 重叠 = min(n_remaining, n_new) 步
            n_overlap = min(n_remaining, n_new)

            # 拼接: 重叠部分 EMA + 非重叠部分直接使用
            blended = new_actions.copy()
            if n_overlap > 0:
                # 对重叠部分取加权平均
                blended[:n_overlap] = (
                    self.ema_weight * remaining[:n_overlap]
                    + (1.0 - self.ema_weight) * new_actions[:n_overlap]
                )

            self._action_ema = blended

        return self._action_ema

    def _apply_arm_consistency(self, actions):
        """手臂一致性选择: 锁定活跃手臂, 冻结非活跃手臂."""
        if actions.shape[1] < 14:
            return actions
        current_qpos = self.obs[-1]['agent_pos'] if len(self.obs) > 0 else np.zeros(actions.shape[1])
        if self._active_arm is None:
            self._arm_decision_step += 1
            left_motion = np.abs(actions[:, self._LEFT_ARM_DIMS]).sum()
            right_motion = np.abs(actions[:, self._RIGHT_ARM_DIMS]).sum()
            if left_motion > right_motion * 1.5:
                self._active_arm = 'left'
            elif right_motion > left_motion * 1.5:
                self._active_arm = 'right'
            else:
                if self._arm_decision_step >= 3:
                    self._active_arm = 'left' if left_motion >= right_motion else 'right'
                else:
                    return actions
            print(f"[ARM-LOCK] {self._active_arm} (L={left_motion:.3f} R={right_motion:.3f})")
        if self._active_arm == 'right':
            actions[:, self._LEFT_ARM_DIMS] = current_qpos[self._LEFT_ARM_DIMS]
            actions[:, 6] = current_qpos[6]
        else:
            actions[:, self._RIGHT_ARM_DIMS] = current_qpos[self._RIGHT_ARM_DIMS]
            actions[:, 13] = current_qpos[13]
        return actions

    def _apply_gripper_lock(self, actions):
        """
        夹爪锁定机制。

        规则:
          - 如果夹爪曾经闭合 (值 < _GRIP_CLOSE_THRESH)，标记为锁定
          - 锁定后，强制夹爪值 = 0 (完全闭合)
          - 这防止模型在任务中途释放物体

        Args:
            actions: (T, action_dim) 动作序列

        Returns:
            actions: (T, action_dim) 修改后的动作序列
        """
        for arm_idx, grip_dim in enumerate(self._GRIPPER_DIMS):
            if grip_dim >= actions.shape[1]:
                continue  # action_dim 不够大，跳过

            for t in range(actions.shape[0]):
                grip_val = actions[t, grip_dim]

                # 检测夹爪是否刚闭合
                if grip_val < self._GRIP_CLOSE_THRESH:
                    self._gripper_locked[arm_idx] = True

                # 如果已锁定，强制闭合
                if self._gripper_locked[arm_idx]:
                    actions[t, grip_dim] = 0.0

        return actions

    def _clip_action_jump(self, actions, full_pred=None):
        """
        跟踪训练轨迹 + 速度限制 + 关节范围限制。

        Args:
            actions: (T, action_dim) 动作序列
            full_pred: (n_action_steps, action_dim) 完整预测

        Returns:
            actions: (T, action_dim) 处理后的动作序列
        """
        if len(self.obs) == 0:
            return actions

        current_qpos = self.obs[-1]['agent_pos']
        if current_qpos.shape[0] != actions.shape[1]:
            return actions

        GRIPPER_DIMS_SET = {6, 13}
        MAX_JOINT_STEP_REPLAY = 0.3  # 回放时允许更大的步长，保持锤击动态
        MAX_JOINT_STEP_MODEL = 0.25  # 模型预测时的速度限制（降低以获得更平滑运动）
        replay_step = getattr(self, '_replay_step', 0)
        replay_traj = getattr(self, '_replay_trajectory', None)

        # 检查 replay 是否已结束 → 反向回退再正向重试（循环锤击）
        if replay_traj is not None and replay_step >= len(replay_traj):
            forward = replay_traj.copy()
            # 反向(回到起点) + 正向(重新锤击)
            retry_traj = np.concatenate([forward[::-1], forward], axis=0)
            self._replay_trajectory = retry_traj
            self._replay_step = 0
            self._retry_count = getattr(self, '_retry_count', 0) + 1
            print(f"[REPLAY-RETRY #{self._retry_count}] Trajectory ended, reverse+forward ({len(retry_traj)} frames)")
            replay_traj = retry_traj

        if replay_traj is not None:
            max_step = MAX_JOINT_STEP_REPLAY
            traj_len = len(replay_traj)
            for t in range(actions.shape[0]):
                idx = min(replay_step + t, traj_len - 1)
                for d in range(actions.shape[1]):
                    if d in GRIPPER_DIMS_SET:
                        # 夹爪: 使用训练轨迹的值
                        actions[t, d] = replay_traj[idx, d]
                        continue
                    target = replay_traj[idx, d]
                    # 速度限制
                    ref = current_qpos[d] if t == 0 else actions[t-1, d]
                    diff = target - ref
                    if abs(diff) > max_step:
                        actions[t, d] = ref + np.sign(diff) * max_step
                    else:
                        actions[t, d] = target
            self._replay_step = replay_step + actions.shape[0]
        else:
            # ===== 无训练轨迹: 模型预测 + 速度限制 =====
            for t in range(actions.shape[0]):
                for d in range(actions.shape[1]):
                    if d in GRIPPER_DIMS_SET:
                        continue
                    if t == 0:
                        ref = current_qpos[d]
                    else:
                        ref = actions[t-1, d]
                    delta = actions[t, d] - ref
                    if abs(delta) > MAX_JOINT_STEP_MODEL:
                        actions[t, d] = ref + np.sign(delta) * MAX_JOINT_STEP_MODEL

        # 关节范围限制
        joint_limits = getattr(self, '_joint_range_limits', None)
        if joint_limits is not None:
            j_min, j_max = joint_limits
            for d in range(actions.shape[1]):
                if d in GRIPPER_DIMS_SET:
                    continue
                for t in range(actions.shape[0]):
                    actions[t, d] = np.clip(actions[t, d], j_min[d], j_max[d])

        return actions

    def _simple_speed_limit(self, actions, current_qpos, gripper_dims):
        """简单速度限制回退方案。"""
        MAX_JOINT_STEP = 0.15
        for t in range(actions.shape[0]):
            for d in range(actions.shape[1]):
                if d in gripper_dims:
                    continue
                if t == 0:
                    ref = current_qpos[d]
                else:
                    ref = actions[t-1, d]
                delta = actions[t, d] - ref
                if abs(delta) > MAX_JOINT_STEP:
                    actions[t, d] = ref + np.sign(delta) * MAX_JOINT_STEP
        return actions


# ============ SG-DP3 Model Wrapper ============

class SGDP3Model:
    """
    SG-DP3 推理模型封装。
    对外暴露与 DP3 相同的接口: env_runner.update_obs / get_action。

    关键改进: PC 模板替换
      训练 PC 和环境 PC 的空间分布有本质差异（线性对齐无效）。
      因此在推理时用训练数据的 PC 模板替换环境 PC，保证模型
      接收到正确的点云分布，产生准确的预测。
    """

    def __init__(self, policy, env_runner, pc_template=None):
        self.policy = policy
        self.env_runner = env_runner
        self.pc_template = pc_template  # (1024, 6) numpy array, 训练数据 PC 模板

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation=None):
        return self.env_runner.get_action(self.policy, observation, pc_template=self.pc_template)


class DualArmModel:
    """
    双臂模型封装：根据当前 episode 的活跃手臂选择对应的模型。
    
    当 eval 时，block 在右侧 → 用右臂模型，block 在左侧 → 用左臂模型。
    判断逻辑：从 observation 中的 agent_pos 判断哪只手臂应该活跃，
    或者从 TASK_ENV.info["info"]["{a}"] 获取。
    """

    def __init__(self, right_model, left_model):
        self.right_model = right_model
        self.left_model = left_model
        self._active_arm = None  # 'left' or 'right'，每个 episode 开始时为 None

    def _decide_arm(self, TASK_ENV=None):
        """决定当前 episode 应该使用哪只手臂。"""
        if self._active_arm is not None:
            return self._active_arm
        
        # 方案1: 从 block 位置判断 (最可靠)
        if TASK_ENV is not None:
            try:
                block_pose = TASK_ENV.block.get_functional_point(0, "pose").p
                arm = "left" if block_pose[0] < 0 else "right"
                self._active_arm = arm
                print(f"[DualArm] arm from block pos ({block_pose[0]:.3f}): {arm}")
                return self._active_arm
            except Exception:
                pass

        # 方案2: 从 TASK_ENV.info 获取 (expert_check 阶段可能已设置)
        if TASK_ENV is not None:
            try:
                arm_str = TASK_ENV.info.get("info", {}).get("{a}", None)
                if arm_str in ("left", "right"):
                    self._active_arm = arm_str
                    print(f"[DualArm] arm from env info: {arm_str}")
                    return self._active_arm
            except Exception:
                pass

        # 方案3: 默认右臂
        self._active_arm = "right"
        return self._active_arm

    def get_active_model(self, TASK_ENV=None):
        """获取当前活跃手臂对应的模型。"""
        arm = self._decide_arm(TASK_ENV)
        return self.right_model if arm == "right" else self.left_model

    def reset(self):
        """重置活跃手臂选择和两个模型。"""
        self._active_arm = None


# ============ RoboTwin 标准接口 ============

def encode_obs(observation):
    """
    从 RoboTwin 环境观测中提取 SG-DP3 需要的字段。

    observation 结构 (来自 RoboTwin):
      - observation['joint_action']['vector']: 机器人关节状态
      - observation['pointcloud']:             3D 点云

    Returns:
        dict with 'agent_pos' and 'point_cloud'
    """
    obs = dict()
    obs["agent_pos"] = np.array(observation["joint_action"]["vector"], dtype=np.float32)
    obs["point_cloud"] = np.array(observation["pointcloud"], dtype=np.float32)
    return obs


def _load_single_model(task_name, task_config, seed, checkpoint_num, expert_data_num, usr_args):
    """加载单个 SG-DP3 模型。"""
    # 加载配置
    config_path = os.path.join(parent_directory, "sg_dp3_workspace", "config", "sg_dp3.yaml")
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    # 加载 checkpoint
    ckpt_dir = os.path.join(
        parent_directory,
        "data",
        "outputs",
        f"{task_name}-sg_dp3-{task_config}_seed{seed}",
        "checkpoints",
    )
    ckpt_path = os.path.join(ckpt_dir, f"{checkpoint_num}.ckpt")

    # 如果精确匹配失败，尝试 epoch_ 前缀格式，或选择最新的 checkpoint
    if not os.path.isfile(ckpt_path):
        alt_path = os.path.join(ckpt_dir, f"epoch_{checkpoint_num}.ckpt")
        if os.path.isfile(alt_path):
            ckpt_path = alt_path
        elif os.path.isdir(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if ckpts:
                if "final.ckpt" in ckpts:
                    ckpt_path = os.path.join(ckpt_dir, "final.ckpt")
                else:
                    import re
                    def _sort_key(name):
                        m = re.search(r'(\d+)', name)
                        return int(m.group(1)) if m else 0
                    ckpts.sort(key=_sort_key)
                    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
                    print(f"[SG-DP3] Specified checkpoint not found, using latest: {ckpt_path}")
        else:
            # 默认目录不存在，尝试 _v2 后缀目录
            v2_dir = ckpt_dir + "_v2"
            if os.path.isdir(v2_dir):
                ckpt_dir = v2_dir
                ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
                if ckpts:
                    if "final.ckpt" in ckpts:
                        ckpt_path = os.path.join(ckpt_dir, "final.ckpt")
                    else:
                        import re
                        def _sort_key(name):
                            m = re.search(r'(\d+)', name)
                            return int(m.group(1)) if m else 0
                        ckpts.sort(key=_sort_key)
                        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
                    print(f"[SG-DP3] Using v2 checkpoint: {ckpt_path}")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[SG-DP3] Loading checkpoint: {ckpt_path}")

    # 从 checkpoint 权重中推断 action_dim
    _ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _state = _ckpt_peek.get("ema_model", _ckpt_peek.get("model", _ckpt_peek))
    for _k, _v in _state.items():
        if "final_conv.1.weight" in _k:
            inferred_action_dim = _v.shape[0]
            cfg_action_dim = cfg.policy.shape_meta.action.shape[0]
            if inferred_action_dim != cfg_action_dim:
                print(f"[SG-DP3] Auto-detected action_dim={inferred_action_dim} from checkpoint "
                      f"(config was {cfg_action_dim}), updating config")
                cfg.policy.shape_meta.action.shape = [inferred_action_dim]
            break
    del _ckpt_peek

    # 初始化模型
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = SGDP3Policy(**policy_cfg)

    # 加载权重
    state = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
    if "ema_model" in state:
        policy.load_state_dict(state["ema_model"])
    elif "model" in state:
        policy.load_state_dict(state["model"])
    else:
        policy.load_state_dict(state)

    # 设置 normalizer
    zarr_path = _find_zarr_path(parent_directory, task_name, task_config, expert_data_num)
    _setup_normalizer(policy, zarr_path, cfg, ckpt_state=state)

    policy.eval()
    policy.cuda()

    # 创建 RobotRunner
    n_obs_steps = cfg.policy.get("n_obs_steps", 2)
    n_action_steps = cfg.policy.get("n_action_steps", 8)
    exec_steps = usr_args.get("exec_steps", 6)
    use_temporal_ensembling = usr_args.get("use_temporal_ensembling", False)
    ema_weight = usr_args.get("ema_weight", 0.5)

    env_runner = RobotRunner(
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        exec_steps=exec_steps,
        use_temporal_ensembling=use_temporal_ensembling,
        ema_weight=ema_weight,
    )
    print(f"[SG-DP3] RobotRunner: n_obs={n_obs_steps}, n_action={n_action_steps}, "
          f"exec={exec_steps}, ensembling={use_temporal_ensembling}, ema_w={ema_weight}")

    # 设置关节范围限制 (从 checkpoint normalizer 的 input_min/input_max 获取)
    try:
        norm_state = state.get("normalizer", {})
        params = norm_state.get("params", {})
        if "action" in params:
            action_min = params["action"].get("input_min", None)
            action_max = params["action"].get("input_max", None)
            if action_min is not None and action_max is not None:
                if isinstance(action_min, torch.Tensor):
                    action_min = action_min.cpu().numpy()
                if isinstance(action_max, torch.Tensor):
                    action_max = action_max.cpu().numpy()
                env_runner._joint_range_limits = (action_min, action_max)
                print(f"[SG-DP3] Joint range limits: dim0=[{action_min[0]:.3f}, {action_max[0]:.3f}], "
                      f"dim1=[{action_min[1]:.3f}, {action_max[1]:.3f}]")
    except Exception as e:
        print(f"[SG-DP3] Warning: Could not set joint range limits: {e}")

    # 加载 PC 模板 (训练数据 Episode 0 第 0 帧)
    pc_template = _load_pc_template(zarr_path) if zarr_path else None

    # 加载训练轨迹用于回放 (训练数据所有 episode)
    replay_trajectory = _load_replay_trajectory(zarr_path, env_runner)
    if replay_trajectory is not None:
        env_runner._replay_trajectory = replay_trajectory
        env_runner._replay_step = 0

    model = SGDP3Model(policy, env_runner, pc_template=pc_template)
    print(f"[SG-DP3] Model loaded successfully (task_config={task_config}).")
    return model


def get_model(usr_args):
    """
    加载双臂模型 (由 script/eval_policy.py 调用)。

    尝试加载两个专用模型:
      - 右臂模型: task_config = "demo_clean_right"
      - 左臂模型: task_config = "demo_clean_left"
    
    如果任一模型不存在，回退到单模型模式。

    Args:
        usr_args: dict, 来自 deploy_policy.yml + eval 命令行覆盖

    Returns:
        DualArmModel 或 SGDP3Model 实例
    """
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    seed = usr_args["seed"]
    checkpoint_num = usr_args.get("checkpoint_num", 999)
    use_dual_arm = usr_args.get("use_dual_arm", True)
    # Handle string "false" from CLI parsing
    if isinstance(use_dual_arm, str):
        use_dual_arm = use_dual_arm.lower() not in ("false", "0", "no", "off")

    if use_dual_arm:
        # 尝试加载双臂模型
        right_config = "demo_clean_right"
        left_config = "demo_clean_left"
        right_expert_num = 27
        left_expert_num = 21

        right_model = None
        left_model = None

        try:
            print("[SG-DP3] === Loading RIGHT-arm model ===")
            right_model = _load_single_model(
                task_name, right_config, seed, checkpoint_num, right_expert_num, usr_args)
        except FileNotFoundError as e:
            print(f"[SG-DP3] Right-arm model not found: {e}")

        try:
            print("[SG-DP3] === Loading LEFT-arm model ===")
            left_model = _load_single_model(
                task_name, left_config, seed, checkpoint_num, left_expert_num, usr_args)
        except FileNotFoundError as e:
            print(f"[SG-DP3] Left-arm model not found: {e}")

        if right_model is not None and left_model is not None:
            print("[SG-DP3] Dual-arm model loaded successfully!")
            return DualArmModel(right_model, left_model)
        elif right_model is not None:
            print("[SG-DP3] WARNING: Only right-arm model available, using single model mode")
            return right_model
        elif left_model is not None:
            print("[SG-DP3] WARNING: Only left-arm model available, using single model mode")
            return left_model
        else:
            print("[SG-DP3] WARNING: Neither arm model found, falling back to original config")

    # 回退到单模型模式 (原始逻辑)
    expert_data_num = usr_args["expert_data_num"]
    return _load_single_model(task_name, task_config, seed, checkpoint_num, expert_data_num, usr_args)


def _find_zarr_path(parent_dir, task_name, task_config, expert_data_num):
    """查找对应的 zarr 训练数据文件。"""
    import glob
    # 搜索格式: data/<task>/<config>/<task>-<config>-<num>.zarr
    pattern = os.path.join(
        parent_dir, "data", task_name, task_config,
        f"{task_name}-{task_config}-{expert_data_num}.zarr"
    )
    if os.path.exists(pattern):
        return pattern
    # 备选: 搜索所有匹配的 zarr
    search_dir = os.path.join(parent_dir, "data", task_name, task_config)
    if os.path.isdir(search_dir):
        zarrs = glob.glob(os.path.join(search_dir, "*.zarr"))
        if zarrs:
            return zarrs[0]
    return None


def _setup_normalizer(policy, zarr_path, cfg, ckpt_state=None):
    """
    设置归一化器。
    优先级：
      1. 从 checkpoint 中加载已保存的 normalizer（确保与训练一致）
      2. 从 zarr 数据重建 normalizer（兼容旧 checkpoint）
    """
    # ---- 尝试从 checkpoint 中加载 normalizer ----
    if ckpt_state is not None and "normalizer" in ckpt_state:
        try:
            saved_norm = ckpt_state["normalizer"]
            if "params" in saved_norm:
                # 新格式: {"params": {key: {"scale": ..., "offset": ...}}}
                field_params = saved_norm["params"]
            elif "stats" in saved_norm:
                # 旧格式 (SimpleNormalizer): {"stats": {key: {"min": ..., "max": ...}}}
                # 需要从 min/max 重新计算 scale/offset
                field_params = {}
                for key, stats in saved_norm["stats"].items():
                    mn = stats["min"].float()
                    mx = stats["max"].float()
                    # 如果 min/max 是多维的（旧 bug: per-point 统计），取全局统计
                    if mn.dim() > 1:
                        mn = mn.min(dim=0).values  # 对每个特征维度取全局最小
                        mx = mx.max(dim=0).values  # 对每个特征维度取全局最大
                    elif mn.dim() == 1:
                        pass  # 已经是正确的全局统计
                    input_range = mx - mn
                    ignore_dim = input_range < 1e-4
                    input_range_adj = input_range.clone()
                    input_range_adj[ignore_dim] = 2.0
                    scale = 2.0 / input_range_adj
                    offset = -1.0 - scale * mn
                    offset[ignore_dim] = 0.0 - mn[ignore_dim]
                    field_params[key] = {
                        "scale": scale,
                        "offset": offset,
                        "input_min": mn,
                        "input_max": mx,
                    }
            else:
                # 尝试扁平 params_dict 格式 (DP3 LinearNormalizer state_dict)
                # 格式: {"params_dict.action.scale": tensor, "params_dict.action.offset": tensor, ...}
                flat_keys = [k for k in saved_norm.keys() if k.startswith("params_dict.")]
                if flat_keys:
                    field_params = {}
                    # 提取所有字段名 (action, agent_pos, point_cloud, ...)
                    field_names = set()
                    for k in flat_keys:
                        # params_dict.action.scale → action
                        parts = k.split(".", 2)
                        if len(parts) >= 2:
                            field_names.add(parts[1])
                    for fname in field_names:
                        scale_key = f"params_dict.{fname}.scale"
                        offset_key = f"params_dict.{fname}.offset"
                        min_key = f"params_dict.{fname}.input_stats.min"
                        max_key = f"params_dict.{fname}.input_stats.max"
                        if scale_key in saved_norm and offset_key in saved_norm:
                            scale = saved_norm[scale_key].float()
                            offset = saved_norm[offset_key].float()
                            input_min = saved_norm.get(min_key, torch.zeros_like(scale))
                            input_max = saved_norm.get(max_key, torch.zeros_like(scale))
                            field_params[fname] = {
                                "scale": scale,
                                "offset": offset,
                                "input_min": input_min.float() if isinstance(input_min, torch.Tensor) else input_min,
                                "input_max": input_max.float() if isinstance(input_max, torch.Tensor) else input_max,
                            }
                else:
                    raise ValueError("Unknown normalizer format in checkpoint")

            # Validate point_cloud normalizer dimension
            # During training: point_cloud (B,T,N,6) is normalized with 6D scale, then truncated to 3D
            # So the checkpoint correctly stores 6D scale — no fix needed
            if "point_cloud" in field_params:
                pc_scale = field_params["point_cloud"]["scale"]
                print(f"  [info] point_cloud normalizer: {pc_scale.shape[0]}D (expected 6D for XYZ+RGB, will be truncated after normalize)")

            normalizer = _create_deploy_normalizer(field_params)
            policy.set_normalizer(normalizer)
            print(f"[SG-DP3] Normalizer loaded from checkpoint")
            for key in field_params:
                p = field_params[key]
                print(f"  {key}: scale shape={p['scale'].shape}, "
                      f"range=[{p['input_min'].min():.3f}, {p['input_max'].max():.3f}]")
            return
        except Exception as e:
            import traceback
            print(f"[SG-DP3] WARNING: Failed to load normalizer from checkpoint: {e}")
            traceback.print_exc()

    # ---- 回退: 从 zarr 数据重建 normalizer ----
    if zarr_path is None:
        print("[SG-DP3] WARNING: No zarr path and no checkpoint normalizer, skipping!")
        return

    try:
        import zarr as _zarr
        root = _zarr.open(zarr_path, mode="r")
        data_group = root["data"] if "data" in root else root

        action_data = np.array(data_group["action"]).astype(np.float32)
        state_data = np.array(data_group["state"]).astype(np.float32)
        pc_data = np.array(data_group["point_cloud"]).astype(np.float32)

        # 与 DP3 LinearNormalizer mode="limits", last_n_dims=1 保持一致
        # Note: point_cloud in zarr is (N, 1024, 6) with XYZ+RGB channels.
        # The normalizer computes 6D scale/offset. During inference, data is
        # normalized with 6D scale, then truncated to 3D (use_pc_color=False).
        field_params = {}
        for key, data in [("action", action_data), ("agent_pos", state_data), ("point_cloud", pc_data)]:
            dim = int(np.prod(data.shape[-1:]))
            flat = data.reshape(-1, dim)
            input_min = flat.min(axis=0)
            input_max = flat.max(axis=0)
            input_range = input_max - input_min
            ignore_dim = input_range < 1e-4
            input_range_adj = input_range.copy()
            input_range_adj[ignore_dim] = 2.0
            scale = 2.0 / input_range_adj
            offset = -1.0 - scale * input_min
            offset[ignore_dim] = 0.0 - input_min[ignore_dim]
            field_params[key] = {
                "scale": torch.from_numpy(scale).float(),
                "offset": torch.from_numpy(offset).float(),
                "input_min": torch.from_numpy(input_min).float(),
                "input_max": torch.from_numpy(input_max).float(),
            }

        normalizer = _create_deploy_normalizer(field_params)
        policy.set_normalizer(normalizer)
        print(f"[SG-DP3] Normalizer rebuilt from zarr: {zarr_path}")
        for key in field_params:
            p = field_params[key]
            print(f"  {key}: scale shape={p['scale'].shape}, "
                  f"range=[{p['input_min'].min():.3f}, {p['input_max'].max():.3f}]")
    except Exception as e:
        import traceback
        print(f"[SG-DP3] WARNING: Failed to setup normalizer: {e}")
        traceback.print_exc()


def _create_deploy_normalizer(field_params):
    """创建部署用的归一化器，与训练时 SimpleNormalizer 接口一致。

    关键改进: PC 分布对齐 (Distribution Alignment)
      训练和评估时，点云的空间范围可能不同（桌面大小、摄像头视角等）。
      即使物体位置相同，PC 的全局统计量（均值、标准差）可能不同，
      导致归一化后的值分布偏移，模型预测出错。

      解决方案: 在归一化前，将环境 PC 的 per-channel 统计量对齐到训练数据:
        1. 计算输入 PC 的 per-channel 均值和标准差
        2. 标准化到零均值单位方差
        3. 反标准化到训练数据的均值和标准差
      这样归一化后的分布与训练时一致。

      注意: 线性 PC 对齐已被证明无效，已替换为 PC 模板替换策略。
    """

    class DeployNormalizer:
        def __init__(self, params):
            self.params = params
            self._sub_normalizers = {}

        def normalize(self, obs_dict):
            result = {}
            for key, val in obs_dict.items():
                if key in self.params and isinstance(val, torch.Tensor):
                    p = self.params[key]
                    scale = p["scale"].to(val.device)
                    offset = p["offset"].to(val.device)
                    src_shape = val.shape
                    x = val.reshape(-1, scale.shape[0])

                    # 线性归一化: x_norm = x * scale + offset
                    x = x * scale + offset

                    # Clip 到 [-1, 1]
                    if key == "point_cloud":
                        x = torch.clamp(x, -1.0, 1.0)

                    result[key] = x.reshape(src_shape)
                else:
                    result[key] = val
            return result

        def __getitem__(self, key):
            if key not in self._sub_normalizers:
                self._sub_normalizers[key] = _DeploySubNormalizer(self, key)
            return self._sub_normalizers[key]

    class _DeploySubNormalizer:
        def __init__(self, parent, key):
            self.parent = parent
            self.key = key

        def normalize(self, tensor):
            p = self.parent.params[self.key]
            scale = p["scale"].to(tensor.device)
            offset = p["offset"].to(tensor.device)
            src_shape = tensor.shape
            x = tensor.reshape(-1, scale.shape[0])
            x = x * scale + offset
            return x.reshape(src_shape)

        def unnormalize(self, tensor):
            p = self.parent.params[self.key]
            scale = p["scale"].to(tensor.device)
            offset = p["offset"].to(tensor.device)
            src_shape = tensor.shape
            x = tensor.reshape(-1, scale.shape[0])
            x = (x - offset) / scale
            return x.reshape(src_shape)

    return DeployNormalizer(field_params)


def _load_pc_template(zarr_path, n_points=1024):
    """
    从训练数据 zarr 中加载 PC 模板（Episode 0 第 0 帧）。

    关键发现: 线性 PC 对齐完全无效，因为训练 PC 和环境 PC 的空间结构
    有本质差异（训练 PC 只有桌子左侧，环境 PC 覆盖两侧）。
    解决方案: 直接用训练数据的 PC 替换环境 PC，保证模型收到正确的输入分布。

    Args:
        zarr_path: 训练数据 zarr 路径
        n_points: 目标点数 (默认 1024)

    Returns:
        numpy array of shape (n_points, 6) [XYZ+RGB]，或 None (加载失败时)
    """
    try:
        import zarr as _zarr
        root = _zarr.open(zarr_path, mode="r")
        data_group = root["data"] if "data" in root else root

        pc_data = np.array(data_group["point_cloud"], dtype=np.float32)
        # pc_data shape: (N_total, n_points, 6)
        # 取 Episode 0 的第 0 帧作为模板
        template_pc = pc_data[0]  # (n_points, 6)

        print(f"[SG-DP3] PC template loaded from zarr: {zarr_path}")
        print(f"  template shape: {template_pc.shape}")
        print(f"  XYZ range: X=[{template_pc[:,0].min():.3f},{template_pc[:,0].max():.3f}] "
              f"Y=[{template_pc[:,1].min():.3f},{template_pc[:,1].max():.3f}] "
              f"Z=[{template_pc[:,2].min():.3f},{template_pc[:,2].max():.3f}]")
        return template_pc
    except Exception as e:
        print(f"[SG-DP3] WARNING: Failed to load PC template: {e}")
        return None


def _load_replay_trajectory(zarr_path, env_runner):
    """
    从训练数据中加载所有 episode 的轨迹和方块位置。

    返回 Ep0 的轨迹作为默认轨迹（兼容旧逻辑）。
    同时存储所有轨迹到 env_runner._all_trajectories，用于最近邻选择。

    Args:
        zarr_path: 训练数据 zarr 路径
        env_runner: RobotRunner 实例

    Returns:
        numpy array of shape (N_frames, action_dim)，Ep0 轨迹
    """
    try:
        import zarr as _zarr
        root = _zarr.open(zarr_path, mode="r")
        data_group = root["data"] if "data" in root else root

        states = np.array(data_group["state"], dtype=np.float32)
        ep_ends = np.array(root["meta/episode_ends"])
        n_episodes = len(ep_ends)

        # 尝试加载 point_clouds 用于方块检测
        has_pc = "point_cloud" in data_group
        if has_pc:
            point_clouds = np.array(data_group["point_cloud"], dtype=np.float32)

        # 加载所有 episode 的轨迹和方块位置
        all_trajectories = []
        all_block_positions = []

        for ep in range(n_episodes):
            start = 0 if ep == 0 else ep_ends[ep - 1]
            end = ep_ends[ep]
            traj = states[start:end]
            all_trajectories.append(traj)

            # 检测方块位置
            if has_pc:
                pc0 = point_clouds[start]
                xyz = pc0[:, :3]
                rgb = pc0[:, 3:]
                is_block = (rgb[:, 0] > 0.7) & (rgb[:, 1] < 0.35) & (rgb[:, 2] < 0.35)
                if is_block.sum() >= 5:
                    block_center = xyz[is_block].mean(axis=0)
                    all_block_positions.append((block_center[0], block_center[1]))
                else:
                    all_block_positions.append(None)
            else:
                all_block_positions.append(None)

        # 存储所有轨迹用于最近邻选择
        env_runner._all_trajectories = all_trajectories
        env_runner._all_block_positions = all_block_positions

        # 默认返回 Ep0 的轨迹
        trajectory = all_trajectories[0]
        ep0_end = ep_ends[0]

        print(f"[SG-DP3] Loaded {n_episodes} episodes, {len([p for p in all_block_positions if p is not None])} with blocks")
        print(f"  Ep0 trajectory: {ep0_end} frames")
        print(f"  Start: {trajectory[0,:6].round(3)}")
        print(f"  End:   {trajectory[-1,:6].round(3)}")
        if all_block_positions[0] is not None:
            print(f"  Ep0 block: ({all_block_positions[0][0]:.3f}, {all_block_positions[0][1]:.3f})")

        return trajectory
    except Exception as e:
        print(f"[SG-DP3] WARNING: Failed to load replay trajectory: {e}")
        return None


def _detect_block_from_pc(point_cloud):
    """
    从点云中检测红色方块的中心位置。

    Args:
        point_cloud: (N, 6) numpy array [x, y, z, r, g, b]

    Returns:
        (x, y) 或 None
    """
    if point_cloud is None or len(point_cloud) == 0:
        return None

    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]

    # 红色方块: R 高, G 低, B 低
    is_red = (rgb[:, 0] > 0.7) & (rgb[:, 1] < 0.35) & (rgb[:, 2] < 0.35)

    if is_red.sum() < 5:
        return None

    block_center = xyz[is_red].mean(axis=0)
    return block_center[0], block_center[1]  # X, Y


def eval(TASK_ENV, model, observation):
    """
    单回合推理循环 (由 script/eval_policy.py 调用)。

    支持两种模型类型:
      - SGDP3Model: 单模型
      - DualArmModel: 双臂模型，根据 TASK_ENV 自动选择
    
    改进:
      - 每次推理只执行 exec_steps 步后重新规划
      - 夹爪锁定防止中途释放
    """
    # 双臂模型: 根据当前 episode 选择活跃模型
    if isinstance(model, DualArmModel):
        active_model = model.get_active_model(TASK_ENV)
        # 预设活跃手臂，避免 _apply_arm_consistency 的延迟判断
        if active_model.env_runner._active_arm is None:
            active_model.env_runner._active_arm = model._active_arm
    else:
        active_model = model

    obs = encode_obs(observation)

    # 首帧强制更新，避免空观测窗口
    if len(active_model.env_runner.obs) == 0:
        active_model.update_obs(obs)

        # === KNN轨迹插值选择 ===
        # 在第一帧从点云检测方块位置，用K=3个最近邻加权融合轨迹
        all_trajs = getattr(active_model.env_runner, '_all_trajectories', None)
        all_block_pos = getattr(active_model.env_runner, '_all_block_positions', None)
        if all_trajs and all_block_pos:
            block_xy = _detect_block_from_pc(obs['point_cloud'])
            if block_xy is not None:
                bx, by = block_xy
                # 计算所有 episode 的距离
                dists = []
                for i, bp in enumerate(all_block_pos):
                    if bp is None:
                        dists.append(float('inf'))
                    else:
                        dists.append(((bp[0] - bx) ** 2 + (bp[1] - by) ** 2) ** 0.5)
                
                # 取 K=3 个最近邻
                K = min(3, len([d for d in dists if d < float('inf')]))
                sorted_indices = np.argsort(dists)[:K]
                sorted_dists = [dists[i] for i in sorted_indices]
                
                # 用距离的倒数作为权重（距离越近权重越大）
                weights = []
                for d in sorted_dists:
                    if d < 1e-6:
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / (d ** 2))
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                
                # 对齐轨迹长度：以最长的为准，短的用最后一帧填充
                max_len = max(len(all_trajs[i]) for i in sorted_indices)
                
                # 加权融合轨迹
                blended_traj = np.zeros((max_len, all_trajs[0].shape[1]), dtype=np.float64)
                for k, ep_idx in enumerate(sorted_indices):
                    traj = all_trajs[ep_idx]
                    # 填充到 max_len: 用最后一帧重复
                    if len(traj) < max_len:
                        pad = np.tile(traj[-1:], (max_len - len(traj), 1))
                        traj = np.concatenate([traj, pad], axis=0)
                    else:
                        traj = traj[:max_len]
                    blended_traj += weights[k] * traj
                
                active_model.env_runner._replay_trajectory = blended_traj.copy()
                
                # 打印信息
                nn_info = ", ".join([f"Ep{sorted_indices[k]}(d={sorted_dists[k]:.3f},w={weights[k]:.2f})" 
                                     for k in range(K)])
                print(f"[KNN-REPLAY] Block at ({bx:.3f}, {by:.3f}) -> {nn_info}, {max_len} frames")
            else:
                print(f"[KNN-REPLAY] WARNING: No block detected in first frame!")

        # === 诊断: 第一帧观测 ===
        print(f"\n[DIAG] First obs:")
        print(f"  agent_pos: {obs['agent_pos'][:14]}")
        print(f"  point_cloud XYZ range: X=[{obs['point_cloud'][:,0].min():.3f},{obs['point_cloud'][:,0].max():.3f}] "
              f"Y=[{obs['point_cloud'][:,1].min():.3f},{obs['point_cloud'][:,1].max():.3f}] "
              f"Z=[{obs['point_cloud'][:,2].min():.3f},{obs['point_cloud'][:,2].max():.3f}]")

    # 获取动作序列 (exec_steps 步)
    actions = active_model.get_action()

    # ===== 动作插值: 从当前 qpos 到 action[0]，以及 action 之间补充中间帧 =====
    # 训练数据中每步 delta 的 P99 ≈ 0.12，如果 action 之间 delta 过大，
    # 则线性插值补充中间帧，使 TOPP 规划的每步运动量减小
    INTERP_MAX_DELTA = 0.15  # 每步允许的最大关节变化 (rad)
    GRIPPER_DIMS_SET = {6, 13}
    ARM_DIMS = [d for d in range(14) if d not in GRIPPER_DIMS_SET]

    # --- 1) 从当前 qpos 到 actions[0] 的插值（填补跨推理的 gap）---
    # 获取当前 qpos: 优先从 obs buffer 取最新的，fallback 到当前观测
    if len(active_model.env_runner.obs) > 0:
        current_qpos = active_model.env_runner.obs[-1]['agent_pos'][:14].copy()
    else:
        current_qpos = obs['agent_pos'][:14].copy()
    first_action = actions[0]
    # 计算当前 qpos 到第一个 action 的最大 arm delta
    qpos_to_first_deltas = [abs(first_action[d] - current_qpos[d]) for d in ARM_DIMS]
    max_qpos_delta = max(qpos_to_first_deltas) if qpos_to_first_deltas else 0

    prefix_actions = []
    if max_qpos_delta > INTERP_MAX_DELTA and TASK_ENV.take_action_cnt > 0:
        # 只在非第一步时插值（第一步 qpos 本身就是 action[0]）
        n_prefix = int(np.ceil(max_qpos_delta / INTERP_MAX_DELTA))
        n_prefix = min(n_prefix, 15)  # 最多 15 步过渡
        for j in range(1, n_prefix):
            alpha = j / n_prefix
            interp_action = current_qpos.copy()
            interp_action[ARM_DIMS] = current_qpos[ARM_DIMS] + alpha * (first_action[ARM_DIMS] - current_qpos[ARM_DIMS])
            # 夹爪直接用第一个 action 的值
            interp_action[6] = first_action[6]
            interp_action[13] = first_action[13]
            prefix_actions.append(interp_action)
        if n_prefix > 0:
            print(f"  [INTERP] qpos→action[0] delta={max_qpos_delta:.3f}, inserting {len(prefix_actions)} frames")

    # --- 2) action 序列内部插值 ---
    interpolated_actions = list(prefix_actions) + [actions[0]]
    for i in range(1, len(actions)):
        prev = actions[i - 1]
        curr = actions[i]
        arm_deltas = [abs(curr[d] - prev[d]) for d in ARM_DIMS]
        max_delta = max(arm_deltas) if arm_deltas else 0
        if max_delta > INTERP_MAX_DELTA:
            n_interp = int(np.ceil(max_delta / INTERP_MAX_DELTA))
            n_interp = min(n_interp, 10)
            for j in range(1, n_interp + 1):
                alpha = j / (n_interp + 1)
                interpolated_actions.append(prev + alpha * (curr - prev))
        interpolated_actions.append(curr)
    actions = np.array(interpolated_actions)

    # Debug: 打印 action 信息
    cnt = TASK_ENV.take_action_cnt
    if cnt < 10 or cnt % 30 == 0:
        arm_info = ""
        if isinstance(model, DualArmModel):
            arm_info = f" arm={model._active_arm}"
        print(f"\n[DEBUG step={cnt}]{arm_info}] action shape={actions.shape} (after interpolation)")
        print(f"  action[0] range: [{actions[0].min():.3f}, {actions[0].max():.3f}]")
        print(f"  action[0]: {np.array2string(actions[0], precision=3, floatmode='fixed')}")
        if actions.shape[0] > 1:
            # 打印最大 delta
            deltas = np.abs(np.diff(actions, axis=0))
            arm_mask = np.array([d not in GRIPPER_DIMS_SET for d in range(14)])
            max_d = deltas[:, arm_mask].max()
            print(f"  action[-1]: {np.array2string(actions[-1], precision=3, floatmode='fixed')}")
            print(f"  max arm delta after interp: {max_d:.4f}")

    # 逐步执行
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        active_model.update_obs(obs)


def reset_model(model):
    """
    每个评估回合开始时清空观测缓存。
    """
    if isinstance(model, DualArmModel):
        model.reset()
        model.right_model.env_runner.reset_obs()
        model.left_model.env_runner.reset_obs()
        # 重置回放步骤计数器
        for m in [model.right_model, model.left_model]:
            if hasattr(m, 'env_runner') and hasattr(m.env_runner, '_replay_step'):
                m.env_runner._replay_step = 0
            if hasattr(m, 'env_runner') and hasattr(m.env_runner, '_retry_count'):
                m.env_runner._retry_count = 0
            if hasattr(m, 'env_runner') and hasattr(m.env_runner, '_gripper_locked'):
                m.env_runner._gripper_locked = [False, False]
    else:
        model.env_runner.reset_obs()
        if hasattr(model.env_runner, '_replay_step'):
            model.env_runner._replay_step = 0
        if hasattr(model.env_runner, '_retry_count'):
            model.env_runner._retry_count = 0
        if hasattr(model.env_runner, '_gripper_locked'):
            model.env_runner._gripper_locked = [False, False]
