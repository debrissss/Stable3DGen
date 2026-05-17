# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    """时间步编码器。
    
    将标量时间步（t）映射为向量 Embedding，用于在 Flow Matching 过程中告知模型当前的演化阶段。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """创建正弦曲线时间步 Embedding。

        Args:
            t: 包含 N 个索引的 1-D Tensor，每个批次元素一个。
            dim: 输出向量的维度。
            max_period: 控制 Embedding 的最小频率。

        Returns:
            形状为 (N, D) 的位置 Embedding Tensor。
        """
        # 参考 OpenAI GLIDE 实现: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """执行前向传播计算时间步嵌入。"""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    """稀疏结构流模型。
    
    采用 Transformer 架构（类似 DiT）通过 Flow Matching 生成 3D 空间的占用率潜变量（Occupancy Latent）。
    这是生成管线的第一阶段，负责确定物体的几何骨架。
    """
    def __init__(
        self,
        resolution: int,              # 空间分辨率
        in_channels: int,            # 输入通道数
        model_channels: int,         # Transformer 隐藏层维度
        cond_channels: int,          # 条件特征（如图像）的维度
        out_channels: int,           # 输出通道数
        num_blocks: int,             # Block 堆叠数量
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,         # 3D Patch 的尺寸
        pe_mode: Literal["ape", "rope"] = "ape", # 位置编码模式 (Absolute / RoPE)
        use_fp16: bool = False,
        use_checkpoint: bool = False, # 是否开启梯度检查点以节省显存
        share_mod: bool = False,      # 是否在所有 Block 间共享调制参数
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # 1. 初始化时间步嵌入
        self.t_embedder = TimestepEmbedder(model_channels)
        
        # 2. 初始化自适应层归一化（adaLN）调制层
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        # 3. 初始化绝对位置编码（APE）
        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        # 4. 输入投影层：将 3D Patch 映射为向量
        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
            
        # 5. Transformer 核心 Block 堆叠
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        # 6. 输出投影层：将向量映射回 3D Patch 空间
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """获取模型所在的计算设备。"""
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """将模型主体转换为 float16 以加速推理。"""
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """将模型主体还原为 float32。"""
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        """初始化网络权重，采用 DiT 风格的零初始化策略。"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化时间步嵌入 MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 初始化 adaLN 调制层，初始设为 0 以保证训练初始阶段的稳定性
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 将输出层初始化为零
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """模型前向传播。

        Args:
            x: 输入的 3D 结构张量 [B, C, R, R, R]。
            t: 时间步张量 [B]。
            cond: 条件特征（如 DINOv2 提取的图像特征）。

        Returns:
            预测的向量场输出 [B, C, R, R, R]。
        """
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"输入形状不匹配, 得到 {x.shape}, 期望 {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        # 1. 空间切片 (Patchify)：将 3D 体积转换为序列
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        # 2. 投影与位置编码
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        
        # 3. 调制信号处理
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        
        # 4. 数据类型对齐
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        
        # 5. Transformer 核心计算
        for block in self.blocks:
            h = block(h, t_emb, cond)
            
        # 6. 后处理与输出投影
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        # 7. 还原空间结构 (Unpatchify)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h
