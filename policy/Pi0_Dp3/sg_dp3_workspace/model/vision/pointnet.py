"""
PointNet++ Geometric Feature Extractor for SG-DP3.

参考来源: policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py

关键变化:
  - 输入从全局点云变为"经过语义提纯后的目标点云"
  - 保留 PointNetEncoderXYZ / PointNetEncoderXYZRGB 的核心结构
  - 增加 output_attention 接口用于可视化/调试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Type
from termcolor import cprint


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    创建多层感知机。

    直接迁移自 DP3 的 pointnet_extractor.py。
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """
    PointNet 编码器 (XYZ + RGB 版)。

    迁移自 DP3 的 PointNetEncoderXYZRGB，核心逻辑不变。
    用于编码带有颜色信息的提纯后点云。
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 256,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint(f"[PointNetEncoderXYZRGB] use_layernorm={use_layernorm}", "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_channels) 输入点云 (提纯后)

        Returns:
            feat: (B, out_channels) 全局点云特征
        """
        x = self.mlp(x)          # (B, N, 512)
        x = torch.max(x, 1)[0]   # (B, 512) - 全局最大池化
        x = self.final_projection(x)  # (B, out_channels)
        return x


class PointNetEncoderXYZ(nn.Module):
    """
    PointNet 编码器 (仅 XYZ 版)。

    迁移自 DP3 的 PointNetEncoderXYZ，核心逻辑不变。
    用于编码仅含坐标信息的提纯后点云。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        super().__init__()
        block_channel = [64, 128, 256]
        cprint(f"[PointNetEncoderXYZ] use_layernorm={use_layernorm}", "cyan")

        assert in_channels == 3, f"PointNetEncoderXYZ only supports 3 channels, got {in_channels}"

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) 输入点云坐标 (提纯后)

        Returns:
            feat: (B, out_channels) 全局点云特征
        """
        x = self.mlp(x)          # (B, N, 256)
        x = torch.max(x, 1)[0]   # (B, 256) - 全局最大池化
        x = self.final_projection(x)  # (B, out_channels)
        return x


class SGDP3VisionEncoder(nn.Module):
    """
    SG-DP3 视觉编码器: 融合提纯后点云的几何特征与机器人本体状态。

    迁移自 DP3 的 DP3Encoder，核心变化:
      - point_cloud_key 对应的输入是"语义提纯后的目标点云"
      - 增加 semantic_condition 接口 (可选)

    结构:
      purified_pointcloud -> PointNet -> geo_feat
      agent_pos -> MLP -> state_feat
      [geo_feat, state_feat] -> concat -> final_feat (c_geo)
    """

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel: int = 256,
        state_mlp_size: tuple = (64, 64),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
        pointcloud_encoder_cfg: Optional[Dict] = None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
    ):
        super().__init__()

        # 键名定义
        self.state_key = "agent_pos"
        self.point_cloud_key = "point_cloud"
        self.n_output_channels = out_channel

        # 获取观测形状
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.use_pc_color = use_pc_color

        cprint(f"[SGDP3VisionEncoder] point_cloud_shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[SGDP3VisionEncoder] state_shape: {self.state_shape}", "yellow")

        # ========= PointNet 编码器 =========
        if pointnet_type == "pointnet":
            if use_pc_color:
                pc_in_channels = 6
                if pointcloud_encoder_cfg is not None:
                    pointcloud_encoder_cfg["in_channels"] = pc_in_channels
                self.extractor = PointNetEncoderXYZRGB(**(pointcloud_encoder_cfg or {}))
            else:
                if pointcloud_encoder_cfg is not None:
                    pointcloud_encoder_cfg["in_channels"] = 3
                self.extractor = PointNetEncoderXYZ(**(pointcloud_encoder_cfg or {}))
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # ========= 状态 MLP =========
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )

        cprint(f"[SGDP3VisionEncoder] output_dim: {self.n_output_channels}", "red")

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        编码提纯后的点云和机器人状态 -> 几何条件 c_geo。

        Args:
            observations: dict with keys:
                'point_cloud': (B, N, 3|6) 提纯后的点云
                'agent_pos': (B, D_state) 机器人本体状态

        Returns:
            final_feat: (B, n_output_channels) = c_geo (几何条件)
        """
        points = observations[self.point_cloud_key]  # (B, N, 3|6)
        assert len(points.shape) == 3, f"point_cloud shape should be 3D, got {points.shape}"

        # PointNet 编码
        pn_feat = self.extractor(points)  # (B, out_channel)

        # 状态编码
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # (B, 64)

        # 拼接
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)  # (B, out_channel + 64)

        return final_feat

    def output_shape(self) -> int:
        return self.n_output_channels
