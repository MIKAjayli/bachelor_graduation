"""
DDIM Scheduler for SG-DP3.

基于 Hugging Face diffusers 的 DDIMScheduler 封装，
支持级联解耦条件注入所需的自定义采样接口。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, Union
from termcolor import cprint

try:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    cprint(
        "[DDIMScheduler] WARNING: diffusers not found, using built-in simplified scheduler",
        "yellow",
    )


class SGDP3DDIMScheduler:
    """
    SG-DP3 的 DDIM 采样调度器。

    功能:
      - 管理 DDIM 采样流程
      - 提供级联解耦采样接口 (支持在每步注入 c_sem 和 c_geo)
      - 兼容 diffusers.DDIMScheduler 或使用内置简化版本
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 16,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        if HAS_DIFFUSERS:
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                prediction_type=prediction_type,
                clip_sample=clip_sample,
                set_alpha_to_one=set_alpha_to_one,
            )
            self._use_diffusers = True
        else:
            self._build_builtin_scheduler(
                num_train_timesteps, beta_start, beta_end, beta_schedule
            )
            self._use_diffusers = False

        cprint(
            f"[SGDP3DDIMScheduler] train_steps={num_train_timesteps}, "
            f"inference_steps={num_inference_steps}, prediction_type={prediction_type}",
            "cyan",
        )

    def _build_builtin_scheduler(
        self,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
    ):
        """内置简化调度器 (当 diffusers 不可用时)。"""
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps)
            betas = 0.5 * (1.0 - np.cos(np.pi * betas / beta_end))
        else:
            betas = np.linspace(beta_start, beta_end, num_train_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.from_numpy(np.cumprod(alphas)).float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def set_timesteps(self, num_inference_steps: int = None):
        """设置推理时间步。"""
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps

        if self._use_diffusers:
            self.scheduler.set_timesteps(self.num_inference_steps)

    @property
    def timesteps(self):
        if self._use_diffusers:
            return self.scheduler.timesteps
        else:
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy()
            return torch.from_numpy(timesteps).long()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """前向加噪过程。"""
        if self._use_diffusers:
            return self.scheduler.add_noise(original_samples, noise, timesteps)

        # 内置实现
        sqrt_alpha = self.sqrt_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[timesteps]

        sqrt_alpha = sqrt_alpha.flatten()
        while len(sqrt_alpha.shape) < len(original_samples.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)

        sqrt_one_minus_alpha = sqrt_one_minus_alpha.flatten()
        while len(sqrt_one_minus_alpha.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_samples = sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """DDIM 采样一步。"""
        if self._use_diffusers:
            result = self.scheduler.step(model_output, timestep, sample)
            return result.prev_sample, None

        # 简化 DDIM 步骤
        device = model_output.device
        t_idx = timestep if isinstance(timestep, int) else timestep.item()
        alpha_t = self.alphas_cumprod.to(device)[t_idx]

        # 计算前一步的 alpha
        if t_idx > 0:
            alpha_t_prev = self.alphas_cumprod.to(device)[t_idx - 1]
        else:
            alpha_t_prev = torch.tensor(1.0, device=device)

        # 预测 x_0
        if self.prediction_type == "epsilon":
            pred_original = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        elif self.prediction_type == "sample":
            pred_original = model_output
        elif self.prediction_type == "v_prediction":
            pred_original = torch.sqrt(alpha_t) * sample - torch.sqrt(1 - alpha_t) * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            pred_original = pred_original.clamp(-1, 1)

        # DDIM 更新
        pred_dir = torch.sqrt(1 - alpha_t_prev) * model_output
        prev_sample = torch.sqrt(alpha_t_prev) * pred_original + pred_dir

        return prev_sample, None

    @property
    def config(self):
        """兼容 diffusers 接口。"""
        if self._use_diffusers:
            return self.scheduler.config

        class _Config:
            num_train_timesteps = self.num_train_timesteps
            prediction_type = self.prediction_type

        return _Config()
