"""
Pi0 Semantic Wrapper for SG-DP3 (Pure PyTorch Implementation).

使用 Hugging Face transformers.PaliGemmaModel 替代 JAX/Flax 版 Pi0 实现。
冻结 VLM 权重，附加自定义 MLP_seg 分割头。

【红线约束】本文件 100% 基于 PyTorch，绝对禁止引入 jax 或 flax。

参考逻辑来源:
  - policy/pi0/src/openpi/models/pi0.py 中的 embed_prefix (图文前缀编码)
  - policy/pi0/src/openpi/models/pi0_fast.py 中的 embed_inputs

PyTorch 等效:
  - PaliGemmaModel 替代 Flax PaliGemma (Siglip + Gemma)
  - 图像编码 -> 文本编码 -> 前缀融合 的逻辑不变
  - 新增: MLP_seg 分割头用于语义分割掩码生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from termcolor import cprint

# 延迟导入: 只在需要时才导入 transformers，避免不必要的 HuggingFace 连接
_HAS_TRANSFORMERS = None

def _check_transformers():
    """延迟检查 transformers 是否可用。"""
    global _HAS_TRANSFORMERS
    if _HAS_TRANSFORMERS is None:
        try:
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
            _HAS_TRANSFORMERS = True
        except ImportError:
            _HAS_TRANSFORMERS = False
    return _HAS_TRANSFORMERS


class MLPSegHead(nn.Module):
    """
    自定义 MLP 分割头: 从 VLM 的图像特征中预测 2D 语义分割掩码。

    输入: PaliGemma 的 SigLip 图像 patch embeddings (B, num_patches, hidden_dim)
    输出: 2D 分割掩码 (B, H, W)

    Architecture:
        patch_features -> MLP -> reshape -> upsample -> sigmoid -> mask
    """

    def __init__(
        self,
        in_features: int = 1152,  # SigLip So400m/14 hidden dim
        hidden_dim: int = 256,
        num_patches: int = 256,  # 16x16 for 224x224 with patch_size=14
        patch_grid_size: int = 16,  # sqrt(num_patches)
        output_height: int = 224,
        output_width: int = 224,
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_grid_size = patch_grid_size
        self.output_height = output_height
        self.output_width = output_width

        # MLP: patch feature -> 1 (二分类: 目标/背景)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        cprint(
            f"[MLPSegHead] in_features={in_features}, hidden_dim={hidden_dim}, "
            f"patch_grid_size={patch_grid_size}, output=({output_height}, {output_width})",
            "cyan",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (B, num_patches, hidden_dim) SigLip patch embeddings

        Returns:
            mask: (B, H, W) 语义分割掩码，值域 [0, 1]
        """
        B, num_patches, _ = image_features.shape

        # MLP 预测每个 patch 的分割值
        patch_logits = self.mlp(image_features)  # (B, num_patches, 1)
        patch_logits = patch_logits.squeeze(-1)   # (B, num_patches)

        # Reshape 为 2D 网格
        patch_mask = patch_logits.reshape(
            B, 1, self.patch_grid_size, self.patch_grid_size
        )  # (B, 1, grid_h, grid_w)

        # 双线性插值上采样到原始分辨率
        mask = F.interpolate(
            patch_mask,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=True,
        )  # (B, 1, H, W)

        mask = torch.sigmoid(mask).squeeze(1)  # (B, H, W)

        return mask


class Pi0SemanticWrapper(nn.Module):
    """
    Pi0 语义理解模块的 PyTorch 移植。

    基于 PaliGemma (SigLip + Gemma) 实现:
    1. 图像编码 (SigLip) -> 提取视觉特征
    2. 文本编码 (Gemma) -> 获取语义理解
    3. 融合前缀编码 -> 生成全局语义条件 c_sem
    4. MLPSegHead -> 生成 2D 分割掩码

    【关键】VLM 权重全部冻结，仅训练 SegHead 和条件投影层。
    """

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-mix-224",
        freeze_vlm: bool = True,
        seg_hidden_dim: int = 256,
        patch_grid_size: int = 16,
        image_height: int = 224,
        image_width: int = 224,
        semantic_feature_dim: int = 256,
        use_text_condition: bool = True,
        max_token_len: int = 48,
        device: str = "cuda",
    ):
        super().__init__()
        self.freeze_vlm = freeze_vlm
        self.use_text_condition = use_text_condition
        self.max_token_len = max_token_len
        self.semantic_feature_dim = semantic_feature_dim
        self.device_str = device

        # 延迟检查 transformers 是否可用
        if not _check_transformers():
            raise ImportError(
                "transformers library is required for Pi0SemanticWrapper. "
                "Install with: pip install transformers"
            )

        # 延迟导入: 只在需要完整 VLM 时才导入
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        # ========= 加载 PaliGemma 模型 =========
        cprint(f"[Pi0SemanticWrapper] Loading PaliGemma from {model_name}...", "yellow")
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # 保持 float32 以兼容后续计算
        )
        self.processor = PaliGemmaProcessor.from_pretrained(model_name)

        # 获取隐藏维度
        self.vision_hidden_dim = self.paligemma.config.vision_config.hidden_size
        self.text_hidden_dim = self.paligemma.config.text_config.hidden_size

        cprint(
            f"[Pi0SemanticWrapper] vision_hidden_dim={self.vision_hidden_dim}, "
            f"text_hidden_dim={self.text_hidden_dim}",
            "cyan",
        )

        # ========= 冻结 VLM 权重 =========
        if freeze_vlm:
            for param in self.paligemma.parameters():
                param.requires_grad = False
            self.paligemma.eval()
            cprint("[Pi0SemanticWrapper] VLM weights FROZEN", "green")

        # ========= 分割头 =========
        self.seg_head = MLPSegHead(
            in_features=self.vision_hidden_dim,
            hidden_dim=seg_hidden_dim,
            patch_grid_size=patch_grid_size,
            output_height=image_height,
            output_width=image_width,
        )

        # ========= 语义条件投影 =========
        # 将融合后的 VLM 特征投影到统一的语义条件空间
        # vision + text (optional) -> semantic_feature_dim
        cond_input_dim = self.vision_hidden_dim
        if use_text_condition:
            cond_input_dim += self.text_hidden_dim

        self.semantic_projection = nn.Sequential(
            nn.Linear(cond_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, semantic_feature_dim),
        )

        cprint(
            f"[Pi0SemanticWrapper] semantic_projection: {cond_input_dim} -> {semantic_feature_dim}",
            "cyan",
        )

    def encode_image(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 SigLip 编码图像，提取 patch-level 视觉特征。

        等效于 Pi0 JAX 版中的 self.PaliGemma.img(images, train=False)

        Args:
            pixel_values: (B, 3, H, W) 预处理后的图像张量

        Returns:
            image_features: (B, num_patches, vision_hidden_dim) 图像 patch 特征
        """
        # 使用 PaliGemma 的 vision tower (SigLip) 编码
        vision_outputs = self.paligemma.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state  # (B, num_patches, D_vision)

        # 通过 multi-modal projector 映射到语言模型空间
        image_features = self.paligemma.multi_modal_projector(image_features)

        return image_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用 Gemma 编码文本指令，提取语义特征。

        等效于 Pi0 JAX 版中的 self.PaliGemma.llm(tokenized_prompt, embed_only=True)

        Args:
            input_ids: (B, max_token_len) tokenized 文本
            attention_mask: (B, max_token_len) 注意力掩码

        Returns:
            text_feature: (B, text_hidden_dim) 全局文本特征 (pooled)
        """
        text_outputs = self.paligemma.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 取最后一层 hidden states，使用平均池化获取全局特征
        hidden_states = text_outputs.hidden_states[-1]  # (B, seq_len, D_text)

        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            text_feature = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            text_feature = hidden_states.mean(dim=1)

        return text_feature  # (B, D_text)

    def compute_semantic_condition(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算全局语义条件 c_sem 和 2D 分割掩码。

        等效于 Pi0 的 embed_prefix 逻辑，融合视觉和语言信息。

        Args:
            pixel_values: (B, 3, H, W) 输入图像
            input_ids: (B, max_token_len) 可选，tokenized 文本指令
            attention_mask: (B, max_token_len) 可选，注意力掩码

        Returns:
            c_sem: (B, semantic_feature_dim) 全局语义条件向量
            mask: (B, H, W) 2D 语义分割掩码
        """
        # Step 1: 编码图像
        image_features = self.encode_image(pixel_values)  # (B, num_patches, D)

        # Step 2: 生成 2D 分割掩码
        mask = self.seg_head(image_features)  # (B, H, W)

        # Step 3: 全局视觉特征 (平均池化)
        visual_feature = image_features.mean(dim=1)  # (B, D_vision)

        # Step 4: 融合文本条件 (可选)
        if self.use_text_condition and input_ids is not None:
            text_feature = self.encode_text(input_ids, attention_mask)  # (B, D_text)
            combined_feature = torch.cat([visual_feature, text_feature], dim=-1)
        else:
            combined_feature = visual_feature

        # Step 5: 投影到语义条件空间
        c_sem = self.semantic_projection(combined_feature)  # (B, semantic_feature_dim)

        return c_sem, mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播: 输出语义条件和分割掩码。

        Args:
            pixel_values: (B, 3, H, W)
            input_ids: (B, max_token_len) optional
            attention_mask: (B, max_token_len) optional

        Returns:
            dict with keys:
                'c_sem': (B, semantic_feature_dim) 语义条件
                'mask': (B, H, W) 分割掩码
                'image_features': (B, num_patches, D_vision) 图像特征 (用于可视化)
        """
        c_sem, mask = self.compute_semantic_condition(
            pixel_values, input_ids, attention_mask
        )

        # 同时返回图像特征 (用于可视化/调试)
        with torch.no_grad():
            image_features = self.encode_image(pixel_values)

        return {
            "c_sem": c_sem,
            "mask": mask,
            "image_features": image_features,
        }

    def get_trainable_params(self):
        """返回可训练参数 (仅 seg_head + projection，VLM 冻结)。"""
        params = []
        params.extend(self.seg_head.parameters())
        params.extend(self.semantic_projection.parameters())
        return params

    def get_num_trainable_params(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.get_trainable_params())


class Pi0SemanticWrapperLight(nn.Module):
    """
    轻量版语义包装器: 当无法加载完整 PaliGemma 时的降级方案。
    使用简单的 CNN + 文本编码器替代。
    """

    def __init__(
        self,
        semantic_feature_dim: int = 256,
        image_height: int = 224,
        image_width: int = 224,
        seg_hidden_dim: int = 256,
    ):
        super().__init__()
        self.semantic_feature_dim = semantic_feature_dim

        # 简易 CNN 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, semantic_feature_dim),
        )

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        # 保存中间特征图
        self._intermediate_channels = 128
        self._intermediate_size = image_height // 8

        cprint(
            f"[Pi0SemanticWrapperLight] semantic_feature_dim={semantic_feature_dim}",
            "cyan",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids=None,
        attention_mask=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W)

        Returns:
            dict with 'c_sem', 'mask'
        """
        B = pixel_values.shape[0]

        # 提取特征
        x = self.image_encoder[:-2](pixel_values)  # 卷积特征
        feat_map = x  # (B, 128, H/8, W/8)

        # 分割掩码
        mask_logits = self.seg_head(feat_map)  # (B, 1, H/8, W/8)
        mask = F.interpolate(
            torch.sigmoid(mask_logits),
            size=(pixel_values.shape[2], pixel_values.shape[3]),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)  # (B, H, W)

        # 全局特征
        pooled = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)  # (B, 128)
        c_sem = self.image_encoder[-1:](pooled.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) if False else \
            self.image_encoder[0](pixel_values)  # fallback

        # 直接使用线性层
        c_sem = self.image_encoder[-1](  # nn.Linear(128, semantic_feature_dim)
            F.adaptive_avg_pool2d(feat_map, (1, 1)).view(B, -1)
        )

        return {
            "c_sem": c_sem,
            "mask": mask,
            "image_features": feat_map.flatten(2).transpose(1, 2),  # (B, num_patches, 128)
        }
