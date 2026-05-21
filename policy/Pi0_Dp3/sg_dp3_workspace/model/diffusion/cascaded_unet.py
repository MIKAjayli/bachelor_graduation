"""
Cascaded Decoupled Conditional U-Net 1D for SG-DP3.

核心创新 #2: 级联解耦条件注入 (Cascaded Decoupled Injection)

参考来源: policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/diffusion/conditional_unet1d.py

关键改造:
  - 不直接拼接条件，而是修改 forward 和 AdaLN 调制层
  - 通过时间步 τ 动态混合条件: w_sem = τ/T_max, w_geo = 1.0 - w_sem
  - 最终条件 c_stage = w_sem * c_sem + w_geo * c_geo
  - 每个下采样/上采样 block 都使用 "级联解耦" 后的条件

架构:
  - 编码器 (下采样): ConditionalResidualBlock1D x N stages
  - 中间层: ConditionalResidualBlock1D x 2
  - 解码器 (上采样): ConditionalResidualBlock1D x N stages
  - 条件注入: CascadedAdaLN (级联自适应层归一化)
"""

import logging
from typing import Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from termcolor import cprint

logger = logging.getLogger(__name__)


# ========= 辅助组件 (迁移自 DP3) =========

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，用于时间步嵌入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


import math

class Conv1dBlock(nn.Module):
    """
    Conv1d -> GroupNorm -> Mish
    迁移自 DP3 的 conv1d_components.py
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """1D 下采样 (stride=2 卷积)"""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D 上采样 (最近邻插值 + 卷积)"""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ========= 核心创新: 级联 AdaLN =========

class CascadedAdaLN(nn.Module):
    """
    级联自适应层归一化 (Cascaded Adaptive Layer Normalization)。

    核心创新: 不直接拼接语义条件 c_sem 和几何条件 c_geo，
    而是通过时间步 τ 动态混合:

        w_sem = τ / T_max
        w_geo = 1.0 - w_sem
        c_stage = w_sem * c_sem + w_geo * c_geo

    然后用 c_stage 生成 scale 和 bias 进行 AdaLN 调制。

    Args:
        cond_dim: 条件维度 (c_sem 和 c_geo 需要维度一致)
        out_channels: 输出通道数
        use_cascaded: 是否使用级联解耦 (False 时退化为普通 FiLM)
    """

    def __init__(
        self,
        cond_dim: int,
        out_channels: int,
        use_cascaded: bool = True,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_channels = out_channels
        self.use_cascaded = use_cascaded

        # 预测 per-channel scale and bias
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # 可学习的层级缩放
        self.level_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(
        self,
        x: torch.Tensor,
        c_sem: torch.Tensor,
        c_geo: torch.Tensor,
        timestep: torch.Tensor,
        T_max: int = 1000,
    ) -> torch.Tensor:
        """
        级联解耦条件注入。

        Args:
            x: (B, out_channels, L) 输入特征
            c_sem: (B, cond_dim) 语义条件
            c_geo: (B, cond_dim) 几何条件
            timestep: (B,) 当前扩散时间步
            T_max: 最大时间步数

        Returns:
            modulated: (B, out_channels, L) 调制后的特征
        """
        if self.use_cascaded:
            # 检测语义条件是否为零 (无图像输入)
            c_sem_norm = c_sem.abs().sum(dim=-1)  # (B,)
            has_semantic = c_sem_norm.max() > 1e-4  # 只要 batch 中有一个非零就认为有语义

            if not has_semantic:
                # 无语义输入: 直接使用几何条件 (全强度)
                c_stage = c_geo
            else:
                # === 有语义输入: 级联解耦混合 ===
                tau = timestep.float()
                w_sem = (tau / T_max).unsqueeze(-1)  # (B, 1)
                w_geo = 1.0 - w_sem                    # (B, 1)

                level_alpha = torch.sigmoid(self.level_scale)  # (1,)
                w_sem = w_sem * level_alpha
                w_geo = w_geo * (1.0 - level_alpha) + (1.0 - level_alpha)

                c_stage = w_sem * c_sem + w_geo * c_geo  # (B, cond_dim)
        else:
            # 退化模式: 简单拼接
            c_stage = c_sem + c_geo

        # AdaLN 调制
        embed = self.cond_encoder(c_stage)  # (B, out_channels*2, 1)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]  # (B, out_channels, 1)
        bias = embed[:, 1, ...]   # (B, out_channels, 1)

        modulated = scale * x + bias  # (B, out_channels, L)

        return modulated


class CascadedResidualBlock1D(nn.Module):
    """
    级联解耦条件残差块。

    基于 DP3 的 ConditionalResidualBlock1D 改造:
      - 将 FiLM 条件注入替换为 CascadedAdaLN
      - 支持语义+几何条件的级联解耦注入
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        film_input_dim: int = None,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_cascaded: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        self.use_cascaded = use_cascaded
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        # film_input_dim: 无语义时的 FiLM 输入维度 (对齐 DP3: dsed + geo_dim)
        if film_input_dim is None:
            film_input_dim = cond_dim * 2

        if use_cascaded:
            # 级联解耦条件注入
            self.cascaded_adaln = CascadedAdaLN(
                cond_dim=cond_dim,
                out_channels=out_channels,
                use_cascaded=True,
            )
            # 无语义时的 FiLM 路径 (对齐 DP3)
            # DP3: cond_encoder = Mish() → Linear(dsed + global_cond_dim, out_channels*2)
            cond_channels = out_channels * 2
            self.film_cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(film_input_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            # 退化: 标准 FiLM
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )

        # 残差连接
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        c_sem: Optional[torch.Tensor] = None,
        c_geo: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        T_max: int = 1000,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, L)
            c_sem: (B, cond_dim) 语义条件
            c_geo: (B, cond_dim) 几何条件
            timestep: (B,) 时间步
            T_max: 最大时间步
            global_cond: (B, cond_dim) 兼容模式的全局条件

        Returns:
            out: (B, out_channels, L)
        """
        out = self.blocks[0](x)  # (B, out_channels, L)

        # 条件注入
        if self.use_cascaded and c_sem is not None and c_geo is not None:
            # 检查是否有真实语义信息
            c_sem_norm = c_sem.abs().sum(dim=-1)
            has_semantic = c_sem_norm.max() > 1e-4

            if has_semantic:
                # 有语义: 使用级联解耦混合
                out = self.cascaded_adaln(out, c_sem, c_geo, timestep, T_max)
            else:
                # 无语义但 c_sem/c_geo 非零: 用 film_cond_encoder
                film_input = torch.cat([c_sem, c_geo], dim=-1)
                embed = self.film_cond_encoder(film_input)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
        elif self.use_cascaded and global_cond is not None:
            # 无语义模式: global_cond = cat([time_cond, c_geo]), 使用 film_cond_encoder
            embed = self.film_cond_encoder(global_cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        elif global_cond is not None:
            # 退化模式 (use_cascaded=False): 标准 FiLM
            embed = self.cond_encoder(global_cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias

        out = self.blocks[1](out)  # (B, out_channels, L)
        out = out + self.residual_conv(x)

        return out


class CascadedUnet1D(nn.Module):
    """
    级联解耦条件 U-Net 1D。

    基于 DP3 的 ConditionalUnet1D 改造:
      - forward 中不再直接拼接条件
      - 每个 ResidualBlock 使用 CascadedAdaLN 调制
      - 实现时间步动态混合: w_sem = τ/T_max, w_geo = 1 - w_sem

    Args:
        input_dim: 动作维度
        semantic_cond_dim: 语义条件维度 (c_sem)
        geometric_cond_dim: 几何条件维度 (c_geo)
        diffusion_step_embed_dim: 时间步嵌入维度
        down_dims: 下采样各层维度
        kernel_size: 卷积核大小
        n_groups: GroupNorm 分组数
        T_max: 最大扩散时间步
    """

    def __init__(
        self,
        input_dim: int,
        semantic_cond_dim: Optional[int] = None,
        geometric_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 3,
        n_groups: int = 8,
        T_max: int = 1000,
        use_cascaded: bool = True,
    ):
        super().__init__()
        self.T_max = T_max
        self.use_cascaded = use_cascaded
        self.input_dim = input_dim

        # 确保语义和几何条件维度一致
        if semantic_cond_dim is not None and geometric_cond_dim is not None:
            assert semantic_cond_dim == geometric_cond_dim, \
                f"semantic_cond_dim ({semantic_cond_dim}) must equal geometric_cond_dim ({geometric_cond_dim})"
            cond_dim = semantic_cond_dim
        elif semantic_cond_dim is not None:
            cond_dim = semantic_cond_dim
        elif geometric_cond_dim is not None:
            cond_dim = geometric_cond_dim
        else:
            cond_dim = diffusion_step_embed_dim

        self.cond_dim = cond_dim

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),  # 输出维度保持 dsed (对齐 DP3)
        )

        # DP3 对齐: cond_dim_for_film = dsed + geometric_cond_dim
        # 当无语义输入时，global_feature = cat([time_embed(dsed), c_geo(geometric_cond_dim)])
        # FiLM encoder 接收的维度就是这个 cond_dim_for_film
        if geometric_cond_dim is not None:
            self.film_input_dim = dsed + geometric_cond_dim
        else:
            self.film_input_dim = dsed

        # time_cond_proj 保留为 identity 以便兼容旧 checkpoint
        self.time_cond_proj = nn.Identity()

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # 中间层
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            CascadedResidualBlock1D(
                mid_dim, mid_dim,
                cond_dim=cond_dim,
                film_input_dim=self.film_input_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                use_cascaded=use_cascaded,
            ),
            CascadedResidualBlock1D(
                mid_dim, mid_dim,
                cond_dim=cond_dim,
                film_input_dim=self.film_input_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                use_cascaded=use_cascaded,
            ),
        ])

        # 下采样
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                CascadedResidualBlock1D(
                    dim_in, dim_out,
                    cond_dim=cond_dim,
                    film_input_dim=self.film_input_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    use_cascaded=use_cascaded,
                ),
                CascadedResidualBlock1D(
                    dim_out, dim_out,
                    cond_dim=cond_dim,
                    film_input_dim=self.film_input_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    use_cascaded=use_cascaded,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # 上采样
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                CascadedResidualBlock1D(
                    dim_out * 2, dim_in,
                    cond_dim=cond_dim,
                    film_input_dim=self.film_input_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    use_cascaded=use_cascaded,
                ),
                CascadedResidualBlock1D(
                    dim_in, dim_in,
                    cond_dim=cond_dim,
                    film_input_dim=self.film_input_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    use_cascaded=use_cascaded,
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        # 最终卷积
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.down_modules = down_modules
        self.up_modules = up_modules

        num_params = sum(p.numel() for p in self.parameters())
        cprint(f"[CascadedUnet1D] parameters: {num_params:e}", "cyan")
        cprint(f"[CascadedUnet1D] T_max={T_max}, use_cascaded={use_cascaded}", "cyan")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        c_sem: Optional[torch.Tensor] = None,
        c_geo: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向传播: 去噪网络。

        Args:
            sample: (B, T, input_dim) 噪声动作序列
            timestep: (B,) or scalar, 扩散时间步
            c_sem: (B, cond_dim) 语义条件
            c_geo: (B, cond_dim) 几何条件
            global_cond: (B, cond_dim) 兼容模式的全局条件

        Returns:
            output: (B, T, input_dim) 预测噪声/样本
        """
        # (B, T, D) -> (B, D, T)
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. 时间步编码
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        time_embed = self.diffusion_step_encoder(timesteps)  # (B, dsed) - 对齐 DP3，不投影

        # 构建条件 - 完全对齐 DP3
        # 检测是否有真实语义信息
        has_semantic = False
        if c_sem is not None:
            c_sem_norm = c_sem.abs().sum(dim=-1)  # (B,)
            has_semantic = c_sem_norm.max() > 1e-4

        if not has_semantic:
            # 无语义输入: 完全对齐 DP3 的条件路径
            # DP3: global_feature = cat([timestep_embed(dsed), global_cond(obs_feature_dim * n_obs_steps)])
            if c_geo is None:
                if global_cond is not None:
                    c_geo = global_cond
                else:
                    c_geo = time_embed
            global_feature = torch.cat([time_embed, c_geo], dim=-1)  # (B, dsed + geo_dim)
            # 传给 block 的参数: c_sem=None, c_geo=None, global_cond=global_feature
            c_sem_pass = None
            c_geo_pass = None
            global_cond_pass = global_feature
        else:
            # 有语义输入: 使用级联解耦混合
            c_sem_pass = c_sem
            c_geo_pass = c_geo if c_geo is not None else time_embed
            global_cond_pass = None

        # 2. 编码器 (下采样)
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, c_sem=c_sem_pass, c_geo=c_geo_pass, timestep=timesteps, T_max=self.T_max, global_cond=global_cond_pass)
            x = resnet2(x, c_sem=c_sem_pass, c_geo=c_geo_pass, timestep=timesteps, T_max=self.T_max, global_cond=global_cond_pass)
            h.append(x)
            x = downsample(x)

        # 3. 中间层
        for mid_module in self.mid_modules:
            x = mid_module(x, c_sem=c_sem_pass, c_geo=c_geo_pass, timestep=timesteps, T_max=self.T_max, global_cond=global_cond_pass)

        # 4. 解码器 (上采样)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c_sem=c_sem_pass, c_geo=c_geo_pass, timestep=timesteps, T_max=self.T_max, global_cond=global_cond_pass)
            x = resnet2(x, c_sem=c_sem_pass, c_geo=c_geo_pass, timestep=timesteps, T_max=self.T_max, global_cond=global_cond_pass)
            x = upsample(x)

        # 5. 最终卷积
        x = self.final_conv(x)

        # (B, D, T) -> (B, T, D)
        x = einops.rearrange(x, "b t h -> b h t")

        return x
