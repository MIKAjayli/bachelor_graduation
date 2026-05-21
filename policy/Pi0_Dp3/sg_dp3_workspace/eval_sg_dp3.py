"""
SG-DP3 Evaluation Script.

加载训练好的 SG-DP3 模型，在环境中进行推理评估。
"""

import os
import sys
import pathlib
import argparse
import json
import time

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "sg_dp3_workspace"))

import torch
import numpy as np
from termcolor import cprint
from omegaconf import OmegaConf

from sg_dp3_workspace.policy.sg_dp3_policy import SGDP3Policy


class EvalSGDP3Workspace:
    """SG-DP3 评估工作空间。"""

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    def load_model(self, checkpoint_path: str) -> SGDP3Policy:
        """加载模型检查点。"""
        cprint(f"[Eval] Loading model from {checkpoint_path}", "cyan")

        model = SGDP3Policy(**OmegaConf.to_container(self.cfg.policy))

        state = torch.load(checkpoint_path, map_location=self.device)
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "ema_model" in state:
            model.load_state_dict(state["ema_model"])
        else:
            model.load_state_dict(state)

        model.to(self.device)
        model.eval()
        cprint("[Eval] Model loaded successfully", "green")
        return model

    @torch.no_grad()
    def predict_action(
        self,
        model: SGDP3Policy,
        obs_dict: dict,
    ) -> np.ndarray:
        """
        预测动作。

        Args:
            model: SG-DP3 模型
            obs_dict: 观测字典，包含:
                'point_cloud': (1, n_obs_steps, N, 3)
                'agent_pos': (1, n_obs_steps, D)
                'image': (1, n_obs_steps, C, H, W) [可选]
                'instruction': (1, n_obs_steps, max_token_len) [可选]

        Returns:
            action: (n_action_steps, action_dim) 预测的动作
        """
        # 转为 tensor
        tensor_obs = {}
        for key, val in obs_dict.items():
            if isinstance(val, np.ndarray):
                tensor_obs[key] = torch.from_numpy(val).float().to(self.device)
            elif isinstance(val, torch.Tensor):
                tensor_obs[key] = val.to(self.device)
            else:
                tensor_obs[key] = val

        result = model.predict_action(tensor_obs)
        action = result["action"].cpu().numpy()
        return action[0]  # (n_action_steps, action_dim)

    def run_evaluation(
        self,
        checkpoint_path: str,
        num_episodes: int = 50,
        save_results: bool = True,
    ):
        """
        运行完整评估。

        注意: 实际环境评估需要配合 RoboTwin 环境运行，
        此处提供模型加载和推理接口。
        """
        model = self.load_model(checkpoint_path)

        results = {
            "checkpoint": checkpoint_path,
            "num_episodes": num_episodes,
            "success_rate": 0.0,
            "episode_returns": [],
        }

        cprint(f"[Eval] Starting evaluation for {num_episodes} episodes", "cyan")

        # TODO: 对接 RoboTwin 环境进行实际评估
        # 这里提供推理接口，用户需要根据实际环境实现循环
        cprint(
            "[Eval] Evaluation loop requires RoboTwin environment.\n"
            "       Use predict_action() method for step-by-step inference.",
            "yellow",
        )

        # 示例: 单次推理测试
        obs = self._create_dummy_obs()
        action = self.predict_action(model, obs)
        cprint(f"[Eval] Sample action shape: {action.shape}", "cyan")
        cprint(f"[Eval] Sample action:\n{action}", "cyan")

        if save_results:
            results_path = os.path.join(
                os.path.dirname(checkpoint_path),
                "eval_results.json",
            )
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            cprint(f"[Eval] Results saved to {results_path}", "green")

        return results

    def _create_dummy_obs(self) -> dict:
        """创建测试用的虚拟观测。"""
        n_obs_steps = self.cfg.policy.get("n_obs_steps", 2)
        n_points = self.cfg.policy.get("purification_num_points", 1024)
        action_dim = self.cfg.policy.shape_meta.action.shape[0]
        state_dim = self.cfg.policy.shape_meta.obs.agent_pos.shape[0]

        return {
            "point_cloud": np.random.randn(1, n_obs_steps, n_points, 3).astype(np.float32),
            "agent_pos": np.random.randn(1, n_obs_steps, state_dim).astype(np.float32),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SG-DP3")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/sg_dp3.yaml",
                        help="Path to config file")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(ROOT_DIR, args.config)
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({"policy": TrainSGDP3Workspace._get_default_config()["policy"]})

    cfg.device = args.device

    workspace = EvalSGDP3Workspace(cfg)
    workspace.run_evaluation(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()
