"""
SG-DP3 Policy: Semantic-Guided 3D Diffusion Policy.

融合 Pi0 的语义理解能力与 DP3 的 3D 几何高精度。

总体架构:
  ┌──────────────────────────────────────────────────────────┐
  │                    SG-DP3 Pipeline                       │
  │                                                          │
  │  Image + Instruction                                     │
  │       │                                                  │
  │       ▼                                                  │
  │  ┌──────────────┐    ┌──────────────┐                    │
  │  │ Pi0Semantic  │───▶│  2D Mask     │                    │
  │  │  Wrapper     │    └──────┬───────┘                    │
  │  └──────┬───────┘           │                            │
  │         │ c_sem             ▼                            │
  │         │         ┌──────────────────┐                   │
  │         │         │ Semantic         │                   │
  │         │         │  Purification    │                   │
  │         │         └────────┬─────────┘                   │
  │         │                  │ purified points              │
  │         │                  ▼                              │
  │         │         ┌──────────────────┐                    │
  │         │         │ PointNet         │                    │
  │         │         │  Encoder         │───▶ c_geo         │
  │         │         └──────────────────┘                    │
  │         │                  │                              │
  │         ▼                  ▼                              │
  │  ┌─────────────────────────────────────┐                 │
  │  │    Cascaded Unet1D (Denoiser)       │                 │
  │  │    w_sem * c_sem + w_geo * c_geo    │                 │
  │  └─────────────────────────────────────┘                 │
  │         │                                                │
  │         ▼                                                │
  │    Predicted Action                                      │
  └──────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from typing import Dict, Optional, Tuple
from termcolor import cprint
import copy

from sg_dp3_workspace.model.semantic.pi0_wrapper import (
    Pi0SemanticWrapper,
    Pi0SemanticWrapperLight,
)
from sg_dp3_workspace.model.semantic.purification import (
    SemanticPurifier,
    PurificationLoss,
)
from sg_dp3_workspace.model.vision.pointnet import SGDP3VisionEncoder
from sg_dp3_workspace.model.diffusion.cascaded_unet import CascadedUnet1D
from sg_dp3_workspace.model.diffusion.ddim_scheduler import SGDP3DDIMScheduler


class SGDP3Policy(nn.Module):
    """
    SG-DP3: Semantic-Guided 3D Diffusion Policy.

    核心创新:
      1. 语义提纯 (Semantic Purification): VLM 2D 掩码 → 过滤 3D 点云
      2. 级联解耦条件注入 (Cascaded Decoupled Injection):
         w_sem = τ/T_max, w_geo = 1 - w_sem
         c_stage = w_sem * c_sem + w_geo * c_geo

    Args:
        shape_meta: 数据形状元信息
        horizon: 动作序列总长度
        n_action_steps: 实际执行的动作步数
        n_obs_steps: 观测步数
        num_inference_steps: DDIM 推理步数
        num_train_timesteps: 训练扩散步数 (T_max)
        diffusion_step_embed_dim: 时间步嵌入维度
        down_dims: U-Net 下采样维度列表
        kernel_size: 卷积核大小
        encoder_output_dim: 点云编码器输出维度
        semantic_feature_dim: 语义条件维度
        use_pc_color: 是否使用点云颜色信息
        pointnet_type: PointNet 类型
        use_light_vlm: 是否使用轻量版 VLM
        vlm_model_name: PaliGemma 模型名称
        purification_num_points: 提纯后点数 (N_min)
        image_height: 输入图像高度
        image_width: 输入图像宽度
        use_text_condition: 是否使用文本条件
        max_token_len: 最大 token 长度
        purification_loss_weight: 提纯损失权重
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        num_inference_steps: int = 16,
        num_train_timesteps: int = 1000,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        encoder_output_dim: int = 256,
        semantic_feature_dim: int = 256,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        use_light_vlm: bool = False,
        vlm_model_name: str = "google/paligemma-3b-mix-224",
        purification_num_points: int = 1024,
        image_height: int = 224,
        image_width: int = 224,
        use_text_condition: bool = True,
        max_token_len: int = 48,
        purification_loss_weight: float = 0.1,
        pointcloud_encoder_cfg: Optional[dict] = None,
        state_mlp_size: tuple = (64, 64),
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        **kwargs,
    ):
        super().__init__()

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.semantic_feature_dim = semantic_feature_dim
        self.purification_loss_weight = purification_loss_weight

        # ========= 解析 shape_meta =========
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape: {action_shape}")
        self.action_dim = action_dim

        obs_shape_meta = shape_meta["obs"]
        obs_dict = {k: v["shape"] for k, v in obs_shape_meta.items()}

        cprint(f"[SGDP3Policy] action_dim={action_dim}, horizon={horizon}", "yellow")
        cprint(f"[SGDP3Policy] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}", "yellow")

        # ========= 1. 语义理解模块 (Pi0 PyTorch 移植) =========
        if use_light_vlm:
            self.semantic_encoder = Pi0SemanticWrapperLight(
                semantic_feature_dim=semantic_feature_dim,
                image_height=image_height,
                image_width=image_width,
            )
        else:
            self.semantic_encoder = Pi0SemanticWrapper(
                model_name=vlm_model_name,
                freeze_vlm=True,
                semantic_feature_dim=semantic_feature_dim,
                image_height=image_height,
                image_width=image_width,
                use_text_condition=use_text_condition,
                max_token_len=max_token_len,
            )

        # ========= 2. 语义提纯模块 =========
        self.purifier = SemanticPurifier(
            num_points=purification_num_points,
            point_dim=6 if use_pc_color else 3,
            use_learnable_projection=True,
            image_height=image_height,
            image_width=image_width,
        )

        # ========= 3. 几何特征编码 (PointNet) =========
        # 更新 observation_space，点云维度 = 提纯后点数
        purified_obs_dict = copy.deepcopy(obs_dict)
        purified_obs_dict["point_cloud"] = (purification_num_points, 3 if not use_pc_color else 6)

        self.obs_encoder = SGDP3VisionEncoder(
            observation_space=purified_obs_dict,
            out_channel=encoder_output_dim,
            state_mlp_size=state_mlp_size,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        obs_feature_dim = self.obs_encoder.output_shape()

        # ========= 4. 几何条件投影 =========
        # 将 obs_feature_dim 投影到与 semantic_feature_dim 相同维度
        _geo_single_mid_dim = max(512, semantic_feature_dim)
        self.geo_projection = nn.Sequential(
            nn.Linear(obs_feature_dim, _geo_single_mid_dim),
            nn.LayerNorm(_geo_single_mid_dim),
            nn.Mish(),
            nn.Linear(_geo_single_mid_dim, semantic_feature_dim),
        )

        # 多帧观测投影: 对齐 DP3 — 直接使用展平的 obs_features
        # DP3 做法: global_cond = nobs_features.reshape(B, -1)，即直接展平
        # 这里不需要投影，在 forward 中直接使用 obs_features 的展平形式
        self.geo_projection_multi_obs = nn.Identity()

        # 计算 Unet 的 geometric_cond_dim
        geo_cond_dim_for_unet = obs_feature_dim * n_obs_steps  # 直接展平，对齐 DP3

        # ========= 5. 级联解耦 U-Net 去噪器 =========
        self.model = CascadedUnet1D(
            input_dim=action_dim,
            semantic_cond_dim=semantic_feature_dim,
            geometric_cond_dim=geo_cond_dim_for_unet,  # DP3 对齐: obs_feature_dim * n_obs_steps
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=8,
            T_max=num_train_timesteps,
            use_cascaded=True,
        )

        # ========= 6. 噪声调度器 =========
        self.noise_scheduler = SGDP3DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # ========= 7. 辅助损失 =========
        self.purification_criterion = PurificationLoss()

        # ========= 8. 存储 prediction_type =========
        self.prediction_type = prediction_type

        # ========= 9. Normalizer (训练时由外部设置) =========
        self.register_buffer("_normalizer_set", torch.tensor(False))

        # 打印参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cprint(f"[SGDP3Policy] Total params: {total_params:e}", "green")
        cprint(f"[SGDP3Policy] Trainable params: {trainable_params:e}", "green")

        self.normalizer = None

        # 语义可视化保存计数器与目录
        self._semantic_vis_counter = 0
        self._semantic_vis_interval = 500   # 每 500 个 global step 保存一次
        self._semantic_vis_dir = None       # 由训练脚本通过 set_semantic_vis_dir() 设置

    def set_normalizer(self, normalizer):
        """设置归一化器。"""
        self.normalizer = normalizer
        self._normalizer_set = torch.tensor(True)

    def set_semantic_vis_dir(self, vis_dir):
        """设置语义可视化数据保存目录。"""
        import os
        self._semantic_vis_dir = vis_dir
        os.makedirs(os.path.join(vis_dir, "mask"), exist_ok=True)
        os.makedirs(os.path.join(vis_dir, "process"), exist_ok=True)
        os.makedirs(os.path.join(vis_dir, "Original"), exist_ok=True)

    def _save_semantic_vis(self, mask_2d, original_pc, purified_pc, loss_dict):
        """
        定期保存语义掩码和点云对比数据到 numpy 文件,
        供 GUI 的语义引导可视化标签页加载显示。
        """
        import os
        import numpy as np

        if self._semantic_vis_dir is None:
            return

        self._semantic_vis_counter += 1
        if self._semantic_vis_counter % self._semantic_vis_interval != 0:
            return

        step = self._semantic_vis_counter
        save_idx = 0  # 保存 batch 中第一个样本

        try:
            # 1. 保存 2D 语义掩码
            if mask_2d is not None:
                mask_np = mask_2d[save_idx].detach().cpu().numpy()  # (H, W)
                mask_path = os.path.join(self._semantic_vis_dir, "mask", f"step_{step}.npy")
                np.save(mask_path, mask_np)

            # 2. 保存原始点云
            if original_pc is not None:
                orig_np = original_pc[save_idx].detach().cpu().numpy()  # (N, 3|6)
                orig_path = os.path.join(self._semantic_vis_dir, "Original", f"step_{step}.npy")
                np.save(orig_path, orig_np)

            # 3. 保存提纯后点云
            if purified_pc is not None:
                pur_np = purified_pc[save_idx].detach().cpu().numpy()  # (N, 3|6)
                pur_path = os.path.join(self._semantic_vis_dir, "process", f"step_{step}.npy")
                np.save(pur_path, pur_np)

        except Exception as e:
            # 保存失败不影响训练
            pass

    # ========= 训练 =========

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        计算训练损失。

        Args:
            batch: dict with keys:
                'obs': dict with 'point_cloud', 'agent_pos', 'image', (optional) 'instruction'
                'action': (B, T, action_dim)

        Returns:
            loss: scalar
            loss_dict: dict of detailed losses
        """
        # 1. 归一化
        # 提取非归一化字段 (instruction 等)
        _non_norm_fields = {}
        obs_input = batch["obs"]
        if "instruction" in obs_input:
            _non_norm_fields["instruction"] = obs_input["instruction"]
            obs_input = {k: v for k, v in obs_input.items() if k != "instruction"}

        if self.normalizer is not None:
            nobs = self.normalizer.normalize(obs_input)
            nactions = self.normalizer["action"].normalize(batch["action"])
        else:
            nobs = obs_input
            nactions = batch["action"]

        # 恢复非归一化字段
        nobs.update(_non_norm_fields)

        # 截断点云颜色通道 (如果不使用)
        if not self.use_pc_color and "point_cloud" in nobs:
            nobs = dict(nobs)  # 避免修改原始 dict
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = nactions.shape[0]
        device = nactions.device

        # 2. 语义编码 (c_sem) + 分割掩码
        semantic_input = {}
        if "image" in nobs:
            # (B, T, C, H, W) -> 取最后一帧
            images = nobs["image"][:, -1]  # (B, C, H, W)
            semantic_input["pixel_values"] = images

        if "instruction" in nobs:
            semantic_input["input_ids"] = nobs["instruction"][:, -1]  # (B, max_token_len)
            semantic_input["attention_mask"] = torch.ones_like(nobs["instruction"][:, -1])

        # 只有当有 pixel_values 时才调用语义编码器 (Pi0SemanticWrapperLight 要求)
        if "pixel_values" in semantic_input:
            semantic_output = self.semantic_encoder(**semantic_input)
            c_sem = semantic_output["c_sem"]  # (B, semantic_feature_dim)
            mask_2d = semantic_output["mask"]  # (B, H, W)
        else:
            # 无图像输入时，使用零条件
            c_sem = torch.zeros(batch_size, self.semantic_feature_dim, device=device)
            mask_2d = None

        # 3. 语义提纯
        # 获取当前帧的点云 (最后一帧用于语义提纯)
        point_cloud = nobs["point_cloud"]  # (B, T, N, 3) or (B, T, N, 6)
        pc_channels = point_cloud.shape[-1]
        current_pc = point_cloud[:, -1]  # (B, N, 3|6) - 最后一帧用于提纯

        if mask_2d is not None:
            purification_output = self.purifier(current_pc, mask_2d)
            purified_pc = purification_output["purified_points"]  # (B, num_points, 3|6)
        else:
            # 无掩码时直接截取
            purified_pc = current_pc[:, :self.purifier.num_points]

        # 4. 几何编码 (c_geo) - 使用所有 obs_steps 帧 (完全对齐 DP3)
        # DP3 做法: 将 B, n_obs_steps, ... reshape 为 B*n_obs_steps, ... → 编码 → reshape 回 B, -1
        # 关键修复: 每个 obs_step 使用各自时间步的原始点云，而非复制同一帧
        # 这与 DP3 完全一致: dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

        all_agent_pos = nobs["agent_pos"][:, :self.n_obs_steps]  # (B, n_obs_steps, D_state)
        B_obs = all_agent_pos.shape[0] * all_agent_pos.shape[1]  # B * n_obs_steps

        # 对点云: 每个 obs_step 使用各自时间步的原始点云 (对齐 DP3)
        # nobs["point_cloud"]: (B, T, N, 3|6) → 取前 n_obs_steps 帧 → reshape 为 (B*n_obs_steps, N, 3|6)
        num_points = self.purifier.num_points
        all_pc = nobs["point_cloud"][:, :self.n_obs_steps, :num_points, :]  # (B, n_obs_steps, N_min, C)
        pc_flat = all_pc.reshape(B_obs, num_points, pc_channels)  # (B*n_obs_steps, N_min, C)

        # agent_pos 展平
        agent_pos_flat = all_agent_pos.reshape(B_obs, -1)

        obs_for_encoder = {
            "point_cloud": pc_flat,    # (B*n_obs_steps, N_min, 3|6)
            "agent_pos": agent_pos_flat,         # (B*n_obs_steps, D_state)
        }
        geo_feat = self.obs_encoder(obs_for_encoder)  # (B*n_obs_steps, obs_feature_dim)

        # reshape 回 (B, n_obs_steps * obs_feature_dim) - 完全对齐 DP3
        geo_feat = geo_feat.reshape(batch_size, -1)  # (B, n_obs_steps * obs_feature_dim)
        c_geo = self.geo_projection_multi_obs(geo_feat)  # Identity → (B, n_obs_steps * obs_feature_dim)

        # 5. 准备扩散训练
        trajectory = nactions  # (B, T, action_dim)

        # 采样噪声和时间步
        noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        ).long()

        # 前向加噪
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # 6. 预测噪声 (级联解耦条件注入)
        c_sem_expanded = c_sem  # (B, semantic_feature_dim)
        c_geo_expanded = c_geo  # (B, semantic_feature_dim)

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            c_sem=c_sem_expanded,
            c_geo=c_geo_expanded,
        )

        # 7. 计算损失
        pred_type = self.prediction_type if hasattr(self, 'prediction_type') else "epsilon"
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            # V-prediction
            alpha_t = self.noise_scheduler.alphas_cumprod.to(device)[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = (1 - alpha_t).sqrt()
            target = alpha_t.sqrt() * noise - sigma_t * trajectory
        else:
            target = noise

        diffusion_loss = F.mse_loss(pred, target, reduction="none")
        diffusion_loss = reduce(diffusion_loss, "b ... -> b (...)", "mean")
        diffusion_loss = diffusion_loss.mean()

        # 8. 提纯辅助损失
        purify_loss = torch.tensor(0.0, device=device)
        if mask_2d is not None:
            purify_loss = self.purification_criterion(purification_output["point_mask"])

        # 总损失
        total_loss = diffusion_loss + self.purification_loss_weight * purify_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "diffusion_loss": diffusion_loss.item(),
            "purification_loss": purify_loss.item(),
        }

        # 9. 保存语义掩码和点云对比可视化数据 (每 N 步保存一次)
        self._save_semantic_vis(
            mask_2d=mask_2d,
            original_pc=current_pc,
            purified_pc=purified_pc if mask_2d is not None else None,
            loss_dict=loss_dict,
        )

        return total_loss, loss_dict

    # ========= 推理 =========

    def conditional_sample(
        self,
        c_sem: torch.Tensor,
        c_geo: torch.Tensor,
        shape: tuple,
        generator=None,
    ) -> torch.Tensor:
        """
        DDIM 条件采样。

        Args:
            c_sem: (B, semantic_feature_dim) 语义条件
            c_geo: (B, semantic_feature_dim) 几何条件
            shape: (B, T, action_dim) 输出形状

        Returns:
            trajectory: (B, T, action_dim) 去噪后的动作序列
        """
        device = c_sem.device
        dtype = c_sem.dtype

        # 初始化纯噪声
        trajectory = torch.randn(shape, dtype=dtype, device=device)

        # 设置推理时间步
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # 预测噪声
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                c_sem=c_sem,
                c_geo=c_geo,
            )

            # DDIM 采样一步
            trajectory, _ = self.noise_scheduler.step(
                model_output, t, trajectory
            )

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        推理: 预测动作。

        Args:
            obs_dict: dict with 'point_cloud', 'agent_pos', 'image', 'instruction'

        Returns:
            result: dict with 'action', 'action_pred'
        """
        # 1. 归一化
        if self.normalizer is not None:
            nobs = self.normalizer.normalize(obs_dict)
        else:
            nobs = obs_dict

        if not self.use_pc_color and "point_cloud" in nobs:
            nobs = dict(nobs)
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        B = nobs["agent_pos"].shape[0]
        T = self.horizon
        Da = self.action_dim
        device = next(self.parameters()).device

        # 2. 语义编码
        semantic_input = {}
        if "image" in nobs:
            images = nobs["image"][:, -1]  # (B, C, H, W)
            semantic_input["pixel_values"] = images
        if "instruction" in nobs:
            semantic_input["input_ids"] = nobs["instruction"][:, -1]
            semantic_input["attention_mask"] = torch.ones_like(nobs["instruction"][:, -1])

        with torch.no_grad():
            if len(semantic_input) > 0:
                semantic_output = self.semantic_encoder(**semantic_input)
                c_sem = semantic_output["c_sem"]
                mask_2d = semantic_output["mask"]
            else:
                c_sem = torch.zeros(B, self.semantic_feature_dim, device=device)
                mask_2d = None

            # 3. 语义提纯 (仅用于最后一帧, 保留架构兼容性)
            point_cloud_last = nobs["point_cloud"][:, -1]  # (B, N, 3|6)
            pc_channels = point_cloud_last.shape[-1]
            if mask_2d is not None:
                purification_output = self.purifier(point_cloud_last, mask_2d)
                purified_pc = purification_output["purified_points"]
            else:
                purified_pc = point_cloud_last[:, :self.purifier.num_points]

            # 4. 几何编码 - 每个 obs_step 使用各自时间步的点云 (完全对齐 DP3)
            all_agent_pos = nobs["agent_pos"][:, :self.n_obs_steps]  # (B, n_obs_steps, D_state)
            B_obs = B * self.n_obs_steps

            # 关键修复: 每个 obs_step 使用各自时间步的原始点云
            num_points = self.purifier.num_points
            all_pc = nobs["point_cloud"][:, :self.n_obs_steps, :num_points, :]  # (B, n_obs_steps, N_min, C)
            pc_flat = all_pc.reshape(B_obs, num_points, pc_channels)  # (B*n_obs_steps, N_min, C)

            agent_pos_flat = all_agent_pos.reshape(B_obs, -1)

            obs_for_encoder = {
                "point_cloud": pc_flat,
                "agent_pos": agent_pos_flat,
            }
            geo_feat = self.obs_encoder(obs_for_encoder)
            geo_feat = geo_feat.reshape(B, -1)  # (B, n_obs_steps * obs_feature_dim)
            c_geo = self.geo_projection_multi_obs(geo_feat)  # Identity → (B, n_obs_steps * obs_feature_dim)

        # 5. 条件采样
        with torch.no_grad():
            nsample = self.conditional_sample(
                c_sem=c_sem,
                c_geo=c_geo,
                shape=(B, T, Da),
            )

        # 6. 反归一化
        if self.normalizer is not None:
            action_pred = self.normalizer["action"].unnormalize(nsample)
        else:
            action_pred = nsample

        # 7. 截取执行动作
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_trainable_param_groups(self):
        """
        获取可训练参数组 (用于差异化学习率)。

        Returns:
            param_groups: list of dicts for optimizer
        """
        # VLM 相关 (已冻结，不包含)
        # 分割头 + 投影层
        semantic_params = list(self.semantic_encoder.get_trainable_params()) \
            if hasattr(self.semantic_encoder, 'get_trainable_params') \
            else list(self.semantic_encoder.parameters())

        # 提纯模块
        purifier_params = list(self.purifier.parameters())

        # 编码器
        encoder_params = (list(self.obs_encoder.parameters()) 
                         + list(self.geo_projection.parameters())
                         + list(self.geo_projection_multi_obs.parameters()))

        # U-Net
        unet_params = list(self.model.parameters())

        return [
            {"params": semantic_params, "lr": 1e-4, "name": "semantic"},
            {"params": purifier_params, "lr": 1e-4, "name": "purifier"},
            {"params": encoder_params, "lr": 1e-4, "name": "encoder"},
            {"params": unet_params, "lr": 1e-4, "name": "unet"},
        ]
