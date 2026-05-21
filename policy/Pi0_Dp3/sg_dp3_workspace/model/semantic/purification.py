"""
Semantic Purification Module for SG-DP3.

Core Innovation #1: Semantic Purification
利用 VLM 分割出的 2D 掩码过滤 3D 点云，提取目标相关的点云。
当提纯后的点云数量 < N_min 时，使用 torch.randint 进行有放回的随机重采样。

关键约束: 严禁生成破坏表面流形的均匀噪声点！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from termcolor import cprint


class SemanticPurifier(nn.Module):
    """
    语义提纯器：通过 2D 掩码将 3D 点云提纯为仅包含目标物体的子集。

    工作流程:
    1. 接收 VLM 生成的 2D 语义分割掩码 (B, H, W)
    2. 接收对应的 3D 点云 (B, N, 3+)
    3. 利用相机内参将 3D 点投影到 2D 平面
    4. 用 2D 掩码过滤投影后的点
    5. 若过滤后点数 < N_min，使用有放回重采样确保点数达标

    Args:
        num_points: 目标输出点数 (N_min)，默认 1024
        point_dim: 点云特征维度 (3 for XYZ, 6 for XYZ+RGB)
        use_learnable_projection: 是否使用可学习的投影矩阵 (替代固定相机内参)
        image_height: 图像高度
        image_width: 图像宽度
    """

    def __init__(
        self,
        num_points: int = 1024,
        point_dim: int = 3,
        use_learnable_projection: bool = True,
        image_height: int = 224,
        image_width: int = 224,
    ):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.use_learnable_projection = use_learnable_projection
        self.image_height = image_height
        self.image_width = image_width

        if use_learnable_projection:
            # 可学习的投影矩阵，模拟相机内参 + 变换
            # 输入: point_dim (3D 坐标) -> 输出: 2D 坐标 + 深度
            self.projection_head = nn.Sequential(
                nn.Linear(point_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # (u, v) 归一化坐标
            )

        cprint(
            f"[SemanticPurifier] num_points={num_points}, point_dim={point_dim}, "
            f"use_learnable_projection={use_learnable_projection}",
            "cyan",
        )

    def project_points_to_2d(
        self,
        point_cloud: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 3D 点云投影到 2D 空间，获取每个点的 2D 坐标。

        Args:
            point_cloud: (B, N, 3+) 3D 点云
            mask: (B, H, W) 2D 语义分割掩码

        Returns:
            proj_coords: (B, N, 2) 归一化 2D 坐标 [0, 1]
        """
        if self.use_learnable_projection:
            # 可学习投影: 直接从 3D 坐标映射到 2D 归一化坐标
            proj_coords = self.projection_head(point_cloud)  # (B, N, 2)
            proj_coords = torch.sigmoid(proj_coords)  # 归一化到 [0, 1]
        else:
            # 简单正交投影: 取前两个维度并归一化
            xyz = point_cloud[..., :3]
            # 归一化到 [0, 1]
            mins = xyz[..., :2].min(dim=1, keepdim=True)[0]
            maxs = xyz[..., :2].max(dim=1, keepdim=True)[0]
            proj_coords = (xyz[..., :2] - mins) / (maxs - mins + 1e-8)

        return proj_coords

    def filter_points_by_mask(
        self,
        point_cloud: torch.Tensor,
        proj_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 2D 掩码过滤 3D 点云。

        Args:
            point_cloud: (B, N, D) 原始 3D 点云
            proj_coords: (B, N, 2) 2D 投影坐标 (归一化 [0, 1])
            mask: (B, H, W) 2D 语义分割掩码

        Returns:
            filtered_points: (B, N_filtered, D) 过滤后的点云
            point_mask: (B, N) bool, 每个点是否在掩码内
        """
        B, N, D = point_cloud.shape
        H, W = mask.shape[1], mask.shape[2]

        # 将投影坐标映射到掩码网格
        # proj_coords: (B, N, 2) in [0, 1] -> grid: (B, N, 2) in [-1, 1]
        grid_x = proj_coords[..., 0] * 2.0 - 1.0  # (B, N)
        grid_y = proj_coords[..., 1] * 2.0 - 1.0  # (B, N)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, N, 2)

        # 使用 grid_sample 对掩码进行双线性采样
        # mask: (B, H, W) -> (B, 1, H, W) for grid_sample
        mask_4d = mask.unsqueeze(1).float()  # (B, 1, H, W)
        grid_4d = grid.unsqueeze(2)  # (B, N, 1, 2) - grid_sample 需要 (B, H_out, W_out, 2)
        # 实际需要: (B, 1, N, 2)
        grid_4d = grid.unsqueeze(1)  # (B, 1, N, 2)

        # 采样掩码值
        sampled_mask = F.grid_sample(
            mask_4d,
            grid_4d,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, 1, 1, N)
        sampled_mask = sampled_mask.squeeze(1).squeeze(1)  # (B, N)

        # 阈值化
        point_mask = sampled_mask > 0.5  # (B, N) bool

        return point_mask

    def resample_with_replacement(
        self,
        point_cloud: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        对过滤后的点云进行有放回随机重采样，确保输出点数 = self.num_points。

        【核心约束】当提纯后的点云数量 < N_min 时，
        必须使用 torch.randint 进行有放回的随机重采样 (Random Sampling with Replacement)，
        严禁生成破坏表面流形的均匀噪声点。

        Args:
            point_cloud: (B, N, D) 原始点云
            point_mask: (B, N) bool, 每个点是否为目标点

        Returns:
            purified_points: (B, num_points, D) 提纯并重采样后的点云
        """
        B, N, D = point_cloud.shape
        device = point_cloud.device

        purified_list = []
        for b in range(B):
            # 获取该 batch 中目标点的索引
            valid_indices = torch.where(point_mask[b])[0]  # (N_valid,)

            if valid_indices.shape[0] == 0:
                # 极端情况: 没有任何点落入掩码区域
                # 回退策略: 从所有点中均匀随机采样
                cprint(
                    "[SemanticPurifier] WARNING: No points fall within mask, "
                    "falling back to uniform random sampling from all points.",
                    "red",
                )
                fallback_indices = torch.randint(
                    0, N, (self.num_points,), device=device
                )
                purified = point_cloud[b, fallback_indices, :]  # (num_points, D)
            elif valid_indices.shape[0] < self.num_points:
                # 关键分支: 过滤后点数 < N_min，使用有放回重采样
                # 【严禁使用均匀噪声！必须从已有目标点中重采样】
                resample_indices = torch.randint(
                    0,
                    valid_indices.shape[0],
                    (self.num_points,),
                    device=device,
                )
                selected_indices = valid_indices[resample_indices]
                purified = point_cloud[b, selected_indices, :]  # (num_points, D)
            else:
                # 过滤后点数足够，随机无放回采样
                perm = torch.randperm(valid_indices.shape[0], device=device)
                selected_indices = valid_indices[perm[: self.num_points]]
                purified = point_cloud[b, selected_indices, :]  # (num_points, D)

            purified_list.append(purified)

        purified_points = torch.stack(purified_list, dim=0)  # (B, num_points, D)
        return purified_points

    def forward(
        self,
        point_cloud: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播: 语义提纯主函数。

        Args:
            point_cloud: (B, N, D) 原始 3D 点云
            mask: (B, H, W) 2D 语义分割掩码 (来自 pi0_wrapper 的分割头)

        Returns:
            dict with keys:
                'purified_points': (B, num_points, D) 提纯后的点云
                'point_mask': (B, N) 每个原始点是否在掩码内
                'num_valid': (B,) 每个batch中有效点数
        """
        B, N, D = point_cloud.shape

        # Step 1: 将 3D 点云投影到 2D
        proj_coords = self.project_points_to_2d(point_cloud, mask)  # (B, N, 2)

        # Step 2: 使用 2D 掩码过滤 3D 点
        point_mask = self.filter_points_by_mask(point_cloud, proj_coords, mask)  # (B, N)

        # Step 3: 有放回重采样确保点数
        purified_points = self.resample_with_replacement(point_cloud, point_mask)  # (B, num_points, D)

        # 统计信息
        num_valid = point_mask.sum(dim=-1)  # (B,)

        return {
            "purified_points": purified_points,
            "point_mask": point_mask,
            "num_valid": num_valid,
        }


class PurificationLoss(nn.Module):
    """
    提纯模块的辅助损失: 鼓励掩码覆盖合理的点云区域。
    """

    def __init__(self, min_coverage: float = 0.05, max_coverage: float = 0.95):
        super().__init__()
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage

    def forward(self, point_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_mask: (B, N) bool/float mask
        Returns:
            coverage_penalty: scalar loss
        """
        coverage = point_mask.float().mean(dim=-1)  # (B,)

        # 惩罚覆盖过低或过高
        penalty_low = F.relu(self.min_coverage - coverage)
        penalty_high = F.relu(coverage - self.max_coverage)

        loss = (penalty_low + penalty_high).mean()
        return loss
