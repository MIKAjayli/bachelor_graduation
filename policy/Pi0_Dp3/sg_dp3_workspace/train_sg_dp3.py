"""
SG-DP3 Training Script.

参考: policy/DP3/3D-Diffusion-Policy/train_dp3.py

训练流程:
  1. 加载配置
  2. 初始化数据集 (MultimodalDataset)
  3. 初始化模型 (SGDP3Policy)
  4. 训练循环:
     - 语义编码 (VLM) → c_sem + mask
     - 语义提纯 → purified_points
     - 几何编码 (PointNet) → c_geo
     - 级联解耦去噪 → loss
  5. EMA 更新
  6. 检查点保存
"""

import os
import sys
import pathlib
import argparse
import copy
import time
import random

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "sg_dp3_workspace"))

import torch
import torch.nn as nn
import numpy as np
import tqdm
from termcolor import cprint
from omegaconf import OmegaConf

from sg_dp3_workspace.policy.sg_dp3_policy import SGDP3Policy
from sg_dp3_workspace.dataset.multimodal_dataset import MultimodalDataset
from torch.utils.data import DataLoader


class EMAModel:
    """
    指数移动平均 (EMA) 模型。
    
    迁移自 DP3 的 EMAModel，使用 power schedule:
      decay = 1 - (1 + step / inv_gamma)^(-power)
    
    DP3 默认参数: power=0.75, inv_gamma=1.0
    训练 1000 epochs, batch=256 时约 22000 steps → decay ≈ 0.9999
    """

    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.model = copy.deepcopy(model)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.optimization_step = 0
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def get_decay(self, optimization_step: int) -> float:
        """Compute the decay factor for the exponential moving average."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    def step(self, model: nn.Module):
        """更新 EMA 参数 (power schedule)。"""
        self.decay = self.get_decay(self.optimization_step)
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.model.parameters(), model.parameters()
            ):
                if model_param.requires_grad:
                    ema_param.data.mul_(self.decay).add_(
                        model_param.data, alpha=1.0 - self.decay
                    )
        self.optimization_step += 1


class TrainSGDP3Workspace:
    """SG-DP3 训练工作空间。"""

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.output_dir = cfg.get("output_dir", "outputs/sg_dp3")
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 自动检测 action 维度 (从 zarr 数据中读取)
        self._auto_detect_action_dim(cfg)

        # 配置模型
        cprint("[TrainSGDP3] Initializing model...", "green")
        self.model = SGDP3Policy(**OmegaConf.to_container(cfg.policy))

        # 配置 EMA (power schedule, 对齐 DP3)
        self.ema_model = None
        if cfg.training.get("use_ema", True):
            self.ema_model = copy.deepcopy(self.model)
            self.ema = EMAModel(
                self.ema_model,
                update_after_step=cfg.training.get("ema_update_after_step", 0),
                inv_gamma=cfg.training.get("ema_inv_gamma", 1.0),
                power=cfg.training.get("ema_power", 0.75),
                min_value=cfg.training.get("ema_min_value", 0.0),
                max_value=cfg.training.get("ema_max_value", 0.9999),
            )

        # 配置优化器 (对齐 DP3: AdamW, betas=[0.95, 0.999])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            betas=(0.95, 0.999),
            weight_decay=cfg.training.get("weight_decay", 1e-6),
        )

        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.num_epochs,
            eta_min=cfg.training.get("lr_min", 1e-6),
        )

        self.global_step = 0
        self.epoch = 0
        self.device = torch.device(cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        cprint(f"[TrainSGDP3] Device: {self.device}", "cyan")

    def _auto_detect_action_dim(self, cfg):
        """从 zarr 数据中自动检测 action 和 state 维度，覆盖配置中的值。"""
        import zarr as _zarr

        zarr_path = cfg.dataset.get("zarr_path", None)
        if zarr_path is None:
            return
        if not os.path.isabs(zarr_path):
            zarr_path = os.path.abspath(zarr_path)
        if not os.path.exists(zarr_path):
            return

        root = _zarr.open(zarr_path, mode="r")
        data_group = root["data"] if "data" in root else root

        # 检测 action 维度
        if "action" in data_group:
            action_data = data_group["action"]
            actual_action_dim = action_data.shape[-1]
            cfg_action_dim = cfg.policy.shape_meta.action.shape[0]
            if actual_action_dim != cfg_action_dim:
                cprint(
                    f"[TrainSGDP3] Auto-detected action_dim={actual_action_dim} "
                    f"(config was {cfg_action_dim}), updating config",
                    "yellow",
                )
                cfg.policy.shape_meta.action.shape = [actual_action_dim]

        # 检测 state 维度
        if "state" in data_group:
            state_data = data_group["state"]
            actual_state_dim = state_data.shape[-1]
            cfg_state_dim = cfg.policy.shape_meta.obs.agent_pos.shape[0]
            if actual_state_dim != cfg_state_dim:
                cprint(
                    f"[TrainSGDP3] Auto-detected state_dim={actual_state_dim} "
                    f"(config was {cfg_state_dim}), updating config",
                    "yellow",
                )
                cfg.policy.shape_meta.obs.agent_pos.shape = [actual_state_dim]

    def run(self):
        """启动训练。"""
        cfg = self.cfg

        # 配置数据集
        cprint("[TrainSGDP3] Loading dataset...", "green")
        dataset = MultimodalDataset(**OmegaConf.to_container(cfg.dataset))
        train_dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # 设置语义可视化保存目录
        vis_dir = os.path.join(self.output_dir, "semantic_vis")
        self.model.set_semantic_vis_dir(vis_dir)
        cprint(f"[TrainSGDP3] Semantic vis dir: {vis_dir}", "cyan")

        # 设备迁移
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
            self.ema.model.to(self.device)

        # 恢复训练
        if cfg.training.get("resume", False):
            self._load_checkpoint()

        # 训练循环
        cprint("[TrainSGDP3] Starting training...", "green")
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        for epoch_idx in range(cfg.training.num_epochs):
            train_losses = []
            epoch_start = time.time()

            with tqdm.tqdm(
                train_dataloader,
                desc=f"Epoch {self.epoch}/{cfg.training.num_epochs}",
                leave=False,
            ) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # 设备迁移
                    batch = self._to_device(batch)

                    # 前向 + 损失
                    loss, loss_dict = self.model.compute_loss(batch)

                    # 反向传播
                    loss = loss / cfg.training.get("gradient_accumulate_every", 1)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.get("max_grad_norm", 10.0),
                    )

                    # 优化器步进
                    if self.global_step % cfg.training.get("gradient_accumulate_every", 1) == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # EMA 更新
                    if self.ema_model is not None:
                        self.ema.step(self.model)

                    # 日志
                    train_losses.append(loss.item())
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        diff=f"{loss_dict.get('diffusion_loss', 0):.4f}",
                    )

                    self.global_step += 1

                    if cfg.training.get("max_train_steps") and batch_idx >= cfg.training.max_train_steps - 1:
                        break

            # Epoch 统计
            epoch_loss = np.mean(train_losses)
            epoch_time = time.time() - epoch_start
            cprint(
                f"Epoch {self.epoch}: loss={epoch_loss:.4f}, time={epoch_time:.1f}s",
                "yellow",
            )

            # 验证
            if self.epoch % cfg.training.get("val_every", 5) == 0:
                val_loss = self._validate(dataset)
                cprint(f"  Validation loss: {val_loss:.4f}", "cyan")

            # 保存检查点
            if (self.epoch + 1) % cfg.training.get("checkpoint_every", 10) == 0:
                self._save_checkpoint()

            # 学习率调度
            self.lr_scheduler.step()
            self.epoch += 1

        # 最终保存
        self._save_checkpoint(name="final")
        cprint("[TrainSGDP3] Training complete!", "green")

    def _to_device(self, batch: dict) -> dict:
        """将 batch 数据移到设备。"""
        result = {}
        for key, val in batch.items():
            if isinstance(val, dict):
                result[key] = {k: v.to(self.device) for k, v in val.items()}
            elif isinstance(val, torch.Tensor):
                result[key] = val.to(self.device)
            else:
                result[key] = val
        return result

    def _validate(self, dataset) -> float:
        """验证。"""
        val_dataset = dataset.get_validation_dataset()
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                loss, _ = self.model.compute_loss(batch)
                val_losses.append(loss.item())
        self.model.train()
        return np.mean(val_losses) if val_losses else 0.0

    def _save_checkpoint(self, name=None):
        """保存检查点。"""
        if name is None:
            name = f"epoch_{self.epoch}"
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{name}.ckpt")

        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        if self.ema_model is not None:
            state["ema_model"] = self.ema.model.state_dict()  # Fix: save actual EMA weights, not the stale reference
            state["ema_optimization_step"] = self.ema.optimization_step
        # 保存 normalizer 状态，以便推理时加载
        if self.model.normalizer is not None:
            try:
                state["normalizer"] = self.model.normalizer.state_dict()
            except Exception:
                pass

        torch.save(state, path)
        cprint(f"[TrainSGDP3] Saved checkpoint: {path}", "green")

    def _load_checkpoint(self):
        """加载检查点。"""
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if ckpts:
                latest = sorted(ckpts)[-1]
                path = os.path.join(ckpt_dir, latest)
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state["model"])
                if self.ema_model is not None and "ema_model" in state:
                    self.ema_model.load_state_dict(state["ema_model"])
                    self.ema.model.load_state_dict(state["ema_model"])  # Fix: also load into actual EMA model
                self.epoch = state.get("epoch", 0)
                self.global_step = state.get("global_step", 0)
                if "ema_optimization_step" in state:
                    self.ema.optimization_step = state["ema_optimization_step"]
                cprint(f"[TrainSGDP3] Resumed from {path}", "magenta")


def main():
    parser = argparse.ArgumentParser(description="Train SG-DP3")
    parser.add_argument("--config", type=str, default="config/sg_dp3.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--zarr_path", type=str, default=None,
                        help="Override zarr data path")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Task name")
    parser.add_argument("--task_config", type=str, default=None,
                        help="Task config (demo_clean / demo_randomized)")
    parser.add_argument("--expert_data_num", type=int, default=None,
                        help="Number of expert episodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(ROOT_DIR, args.config)
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cprint(f"[Train] Config not found at {config_path}, using defaults", "yellow")
        cfg = OmegaConf.create(_get_default_config())

    if args.output_dir:
        cfg.output_dir = args.output_dir

    # 命令行参数覆盖配置
    if args.zarr_path:
        cfg.dataset.zarr_path = args.zarr_path
    if args.seed is not None:
        cfg.training.seed = args.seed
        cfg.dataset.seed = args.seed

    # 启动训练
    workspace = TrainSGDP3Workspace(cfg)
    workspace.run()


def _get_default_config() -> dict:
    """默认配置。"""
    return {
        "output_dir": "outputs/sg_dp3",
        "training": {
            "seed": 42,
            "num_epochs": 1000,
            "batch_size": 256,
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "max_grad_norm": 10.0,
            "use_ema": True,
            "ema_power": 0.75,
            "ema_inv_gamma": 1.0,
            "ema_min_value": 0.0,
            "ema_max_value": 0.9999,
            "ema_update_after_step": 0,
            "device": "cuda",
            "gradient_accumulate_every": 1,
            "checkpoint_every": 100,
            "val_every": 50,
            "num_workers": 4,
        },
        "policy": {
            "shape_meta": {
                "action": {"shape": [7]},
                "obs": {
                    "point_cloud": {"shape": [1024, 3]},
                    "agent_pos": {"shape": [14]},
                },
            },
            "horizon": 8,
            "n_action_steps": 6,
            "n_obs_steps": 3,
            "num_train_timesteps": 100,
            "num_inference_steps": 10,
            "prediction_type": "sample",
            "encoder_output_dim": 128,
            "semantic_feature_dim": 128,
            "use_pc_color": False,
            "use_light_vlm": True,
        },
        "dataset": {
            "zarr_path": "data/demo_task.zarr",
            "horizon": 8,
            "pad_before": 2,
            "pad_after": 5,
            "seed": 42,
            "val_ratio": 0.02,
            "use_image": True,
            "instruction": "pick up the block",
        },
    }


if __name__ == "__main__":
    main()
