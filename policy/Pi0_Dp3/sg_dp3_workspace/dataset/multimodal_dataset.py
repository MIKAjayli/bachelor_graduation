"""
Multimodal Dataset for SG-DP3.

参考: policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py

扩展 DP3 的 RobotDataset 以同时加载:
  - point_cloud (3D 点云)
  - agent_pos (机器人本体状态)
  - action (动作)
  - image (RGB 图像) [新增]
  - instruction (任务指令文本) [新增]

数据格式: 基于 ReplayBuffer 的 zarr 存储。
"""

import sys
import os
from typing import Dict, Optional
import torch
import numpy as np
import copy
from termcolor import cprint

# 数据工具 (兼容 DP3 框架)
try:
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
    from diffusion_policy_3d.common.sampler import (
        SequenceSampler,
        get_val_mask,
        downsample_mask,
    )
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
    from diffusion_policy_3d.dataset.base_dataset import BaseDataset
    HAS_DP3 = True
except ImportError:
    HAS_DP3 = False
    cprint("[MultimodalDataset] DP3 dependencies not found, using standalone mode", "yellow")


class MultimodalDataset(torch.utils.data.Dataset):
    """
    SG-DP3 多模态数据集。

    扩展 DP3 的 RobotDataset:
      - 新增 image 字段支持
      - 新增 instruction 字段支持 (文本 tokenized)
      - 支持 zarr 格式的 ReplayBuffer 数据存储

    数据格式要求 (zarr):
      - 'state': (N, D_state) 机器人状态
      - 'action': (N, D_action) 动作
      - 'point_cloud': (N, num_points, 3|6) 点云
      - 'image': (N, C, H, W) RGB 图像 [可选]
      - 'instruction': (max_token_len,) 文本 token IDs [可选, 全局共享]
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        task_name: Optional[str] = None,
        # 新增多模态参数
        instruction: Optional[str] = None,
        image_key: str = "image",
        use_image: bool = True,
        max_token_len: int = 48,
    ):
        super().__init__()
        self.task_name = task_name
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_image = use_image
        self.max_token_len = max_token_len
        self.instruction = instruction

        # 路径处理
        if not os.path.isabs(zarr_path):
            # 相对路径基于当前工作目录解析 (而非文件所在目录)
            zarr_path = os.path.abspath(zarr_path)

        self.zarr_path = zarr_path

        # 加载 ReplayBuffer
        if HAS_DP3:
            # 尝试加载含图像的数据 (支持 "image" 或 "images" 键名)
            buffer_keys = ["state", "action", "point_cloud"]
            image_loaded = False
            
            if use_image:
                # 先尝试 "images" 键 (zarr 数据格式)
                for img_key in ["images", "image"]:
                    try:
                        buffer_keys_with_img = buffer_keys + [img_key]
                        self.replay_buffer = ReplayBuffer.copy_from_path(
                            zarr_path, keys=buffer_keys_with_img
                        )
                        image_loaded = True
                        cprint(f"[MultimodalDataset] Loaded image data with key '{img_key}'", "cyan")
                        break
                    except Exception:
                        continue

            if not image_loaded:
                try:
                    self.replay_buffer = ReplayBuffer.copy_from_path(
                        zarr_path, keys=buffer_keys
                    )
                except Exception as e:
                    cprint(
                        f"[MultimodalDataset] Failed to load replay buffer: {e}",
                        "yellow",
                    )
                    raise
            self.use_image = image_loaded
        else:
            # 独立模式: 手动加载 zarr
            self.replay_buffer = self._load_zarr_standalone(zarr_path)
            # 检查是否成功加载图像 (独立模式下跳过嵌套组)
            has_image = "image" in self.replay_buffer.data or "images" in self.replay_buffer.data
            if self.use_image and not has_image:
                cprint("[MultimodalDataset] Image data not available (nested group or missing), disabling image", "yellow")
                self.use_image = False

        # 划分训练/验证集
        if HAS_DP3:
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask, max_n=max_train_episodes, seed=seed
            )
        else:
            n_episodes = self.replay_buffer.n_episodes
            n_val = max(1, int(n_episodes * val_ratio))
            val_mask = np.zeros(n_episodes, dtype=bool)
            val_mask[:n_val] = True
            train_mask = ~val_mask

        self.train_mask = train_mask

        if HAS_DP3:
            self.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
            )
        else:
            self.sampler = self._create_simple_sampler(
                train_mask, horizon, pad_before, pad_after
            )

        # 处理文本指令
        self.tokenized_instruction = None
        if instruction is not None:
            self.tokenized_instruction = self._tokenize_instruction(instruction)

        cprint(
            f"[MultimodalDataset] zarr_path={zarr_path}, "
            f"use_image={self.use_image}, episodes={self.replay_buffer.n_episodes}, "
            f"samples={len(self.sampler)}",
            "cyan",
        )

    def _load_zarr_standalone(self, zarr_path: str):
        """独立加载 zarr 数据 (无 DP3 依赖时)。"""
        import zarr

        root = zarr.open(zarr_path, mode="r")

        class SimpleReplayBuffer:
            def __init__(self, root):
                self.root = root
                self.data = {}

                # 支持两种 zarr 结构:
                # 1. 嵌套结构: root/data/action, root/meta/episode_ends
                # 2. 扁平结构: root/action, root/episode_ends
                if "data" in root and hasattr(root["data"], "keys"):
                    # 嵌套结构: 数据在 data/ 子目录下
                    data_group = root["data"]
                    for key in data_group.keys():
                        item = data_group[key]
                        # 跳过嵌套组 (如 images/ 包含多个相机)
                        if hasattr(item, "shape"):
                            self.data[key] = np.array(item)
                        elif hasattr(item, "keys"):
                            # 这是一个组，跳过 (暂不支持嵌套图像)
                            cprint(f"[SimpleReplayBuffer] Skipping nested group: {key}", "yellow")
                    # episode_ends 在 meta/ 子目录下
                    if "meta" in root and "episode_ends" in root["meta"]:
                        self.episode_ends = np.array(root["meta"]["episode_ends"])
                    else:
                        # 回退: 假设单个 episode
                        n = self._get_data_length()
                        self.episode_ends = np.array([n])
                else:
                    # 扁平结构: 数据直接在 root 下
                    for key in root.keys():
                        item = root[key]
                        if hasattr(item, "shape"):  # 是数组而非 Group
                            self.data[key] = np.array(item)
                    if "episode_ends" in root:
                        self.episode_ends = np.array(root["episode_ends"])
                    else:
                        n = self._get_data_length()
                        self.episode_ends = np.array([n])

                self.n_episodes = len(self.episode_ends)

            def _get_data_length(self):
                """获取数据长度 (从 state 或 action 推断)。"""
                if "state" in self.data:
                    return self.data["state"].shape[0]
                elif "action" in self.data:
                    return self.data["action"].shape[0]
                else:
                    raise ValueError("No 'state' or 'action' found in zarr data")

            def __getitem__(self, key):
                return self.data[key]

        return SimpleReplayBuffer(root)

    def _create_simple_sampler(self, episode_mask, horizon, pad_before, pad_after):
        """创建简单的序列采样器。"""

        class SimpleSampler:
            def __init__(self, replay_buffer, episode_mask, horizon, pad_before, pad_after):
                self.replay_buffer = replay_buffer
                self.horizon = horizon
                self.pad_before = pad_before
                self.pad_after = pad_after

                # 构建采样索引
                episode_ends = replay_buffer.episode_ends
                episode_starts = np.concatenate([[0], episode_ends[:-1]])

                self.sample_indices = []
                for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
                    if episode_mask[i]:
                        for j in range(start, end):
                            self.sample_indices.append(j)

            def __len__(self):
                return len(self.sample_indices)

            def sample_sequence(self, idx):
                center = self.sample_indices[idx]
                data = {}
                for key in self.replay_buffer.data:
                    arr = self.replay_buffer[key]
                    seq = np.zeros((self.horizon,) + arr.shape[1:], dtype=arr.dtype)
                    for t in range(self.horizon):
                        pos = center - self.pad_before + t
                        if 0 <= pos < arr.shape[0]:
                            seq[t] = arr[pos]
                    data[key] = seq
                return data

        return SimpleSampler(self.replay_buffer, episode_mask, horizon, pad_before, pad_after)

    def _tokenize_instruction(self, instruction: str) -> torch.Tensor:
        """
        简单的文本 tokenization (使用空格分词 + 截断/填充)。

        注意: 实际使用时建议使用 transformers tokenizer。
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-mix-224")
            tokens = tokenizer(
                instruction,
                max_length=self.max_token_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(0)  # (max_token_len,)
        except Exception:
            # 降级: 使用简单的字符级 tokenization
            words = instruction.lower().split()
            tokens = [hash(w) % 50000 for w in words]
            tokens = tokens[:self.max_token_len]
            tokens = tokens + [0] * (self.max_token_len - len(tokens))
            return torch.tensor(tokens, dtype=torch.long)

    def get_validation_dataset(self):
        """获取验证集。"""
        val_set = copy.copy(self)
        if HAS_DP3:
            val_set.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_mask,
            )
        else:
            val_set.sampler = self._create_simple_sampler(
                ~self.train_mask, self.horizon, self.pad_before, self.pad_after
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """获取归一化器。"""
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }

        if HAS_DP3:
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        else:
            normalizer = self._create_simple_normalizer(data)

        return normalizer

    def _create_simple_normalizer(self, data: dict, last_n_dims=1, mode="limits", range_eps=1e-4):
        """
        创建与 DP3 LinearNormalizer (mode="limits", last_n_dims=1) 完全一致的归一化器。

        关键：使用全局统计量，对最后一个维度独立归一化。
        例如 point_cloud shape (N, 1024, 6) → dim=6，scale/offset shape=(6,)
        """
        output_min, output_max = -1.0, 1.0

        # 计算每个字段的全局 scale 和 offset
        field_params = {}
        for key, arr in data.items():
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr).astype(np.float32)
            arr = arr.astype(np.float32)

            dim = int(np.prod(arr.shape[-last_n_dims:]))
            flat = arr.reshape(-1, dim)

            input_min = flat.min(axis=0)  # (dim,)
            input_max = flat.max(axis=0)  # (dim,)
            input_range = input_max - input_min

            ignore_dim = input_range < range_eps
            input_range_adj = input_range.copy()
            input_range_adj[ignore_dim] = output_max - output_min

            scale = (output_max - output_min) / input_range_adj  # (dim,)
            offset = output_min - scale * input_min  # (dim,)
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

            field_params[key] = {
                "scale": torch.from_numpy(scale).float(),
                "offset": torch.from_numpy(offset).float(),
                "input_min": torch.from_numpy(input_min).float(),
                "input_max": torch.from_numpy(input_max).float(),
            }

        class SimpleNormalizer:
            def __init__(self, field_params):
                self.params = field_params
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
                        x = x * scale + offset
                        result[key] = x.reshape(src_shape)
                    else:
                        result[key] = val
                return result

            def __getitem__(self, key):
                if key not in self._sub_normalizers:
                    self._sub_normalizers[key] = _SubNormalizer(self, key)
                return self._sub_normalizers[key]

            def state_dict(self):
                return {"params": self.params}

        class _SubNormalizer:
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

        return SimpleNormalizer(field_params)

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: dict) -> dict:
        """将原始采样数据转换为训练格式。"""
        agent_pos = sample["state"].astype(np.float32)
        point_cloud = sample["point_cloud"].astype(np.float32)
        action = sample["action"].astype(np.float32)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # (T, N, 3|6)
                "agent_pos": agent_pos,      # (T, D_state)
            },
            "action": action,  # (T, D_action)
        }

        # 添加图像 (支持 "image" 或 "images" 两种键名)
        if self.use_image:
            if "images" in sample:
                data["obs"]["image"] = sample["images"].astype(np.float32)  # (T, C, H, W)
            elif "image" in sample:
                data["obs"]["image"] = sample["image"].astype(np.float32)  # (T, C, H, W)

        # 添加指令 (广播到所有时间步)
        if self.tokenized_instruction is not None:
            T = action.shape[0]
            data["obs"]["instruction"] = self.tokenized_instruction.unsqueeze(0).expand(T, -1).numpy()

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本。"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        # 转为 torch.Tensor
        torch_data = {}
        for key, val in data.items():
            if isinstance(val, dict):
                torch_data[key] = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in val.items()
                }
            elif isinstance(val, np.ndarray):
                torch_data[key] = torch.from_numpy(val)
            else:
                torch_data[key] = val

        return torch_data
