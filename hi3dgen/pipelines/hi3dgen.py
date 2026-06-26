"""Hi3DGen 生成流水线模块。

该模块定义了 Hi3DGenPipeline 类，整合了图像预处理、特征提取、稀疏结构采样、
结构化潜空间采样以及最终的 3D 网格解码逻辑。
"""
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
#
# Original file was released under MIT, with the full license text
# available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
#
# This modified file is released under the same license.
import os
from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class Hi3DGenPipeline(Pipeline):
    """Hi3DGen 流水线类，负责 3D 生成的核心逻辑。

    该类集成了稀疏结构生成、结构化潜空间 (Structured Latent, SLAT) 生成以及最终的网格解码。
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        """初始化 Hi3DGen 流水线。

        Args:
            models: 模型组件字典。
            sparse_structure_sampler: 稀疏结构采样器。
            slat_sampler: 结构化潜空间采样器。
            slat_normalization: SLAT 归一化参数。
            image_cond_model: 图像条件模型名称。
        """
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str, slat_flow_model_path: str = None, ss_flow_model_path: str = None) -> "Hi3DGenPipeline":
        """从预训练路径加载 Hi3DGen 流水线。

        Args:
            path: 本地路径或 Hugging Face 仓库名称。
            slat_flow_model_path: 自定义结构化潜空间流模型的权重路径。
            ss_flow_model_path: 自定义稀疏结构流模型的权重路径。

        Returns:
            加载后的 Hi3DGenPipeline 实例。
        """
        pipeline = super(Hi3DGenPipeline, Hi3DGenPipeline).from_pretrained(path, slat_flow_model_path=slat_flow_model_path, ss_flow_model_path=ss_flow_model_path)
        new_pipeline = Hi3DGenPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # 初始化稀疏结构采样器
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        # 初始化 SLAT 采样器
        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        # 初始化图像条件模型 (如 DINOv2)
        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """初始化图像条件模型（默认为 DINOv2）。

        Args:
            name: DINOv2 模型的名称（例如 'dinov2_vitl14'）。
        """
        try:
            # 尝试从本地加载 DINOv2
            dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), name, source='local',pretrained=True)
        except:
            # 本地加载失败则从 hub 加载
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        # 设置图像预处理的归一化参数
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image, resolution=518) -> Image.Image:
        """使用 BiRefNet 进行背景移除并预处理输入图像。

        该方法包括移除背景、提取前景边界框、保持长宽比的填充以及缩放到目标分辨率。

        Args:
            input: 输入的原始图像（PIL.Image）。
            resolution: 输出图像的目标分辨率（正方形）。默认为 518。

        Returns:
            Image.Image: 预处理后的 RGBA 格式图像。
        """
        # 检查图像是否已有 Alpha 通道（RGBA），如果有且非全白，则直接使用
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, -1]
            if not np.all(alpha == 255):
                has_alpha = True
        
        if has_alpha:
            output = input
        else:
            # 如果没有 Alpha 通道，则使用 BiRefNet 进行背景移除
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            
            # 延迟加载 BiRefNet 模型
            if getattr(self, 'birefnet_model', None) is None:
                self._lazy_load_birefnet()
            
            # 获取 BiRefNet 生成的掩码（Mask）
            mask = self._get_birefnet_mask(input)
            
            # 将输入转换为 RGBA 并应用掩码到 Alpha 通道
            input_rgba = input.convert('RGBA')
            input_array = np.array(input_rgba)
            input_array[:, :, 3] = mask * 255  # 应用掩码
            output = Image.fromarray(input_array)

        # 处理输出图像：提取前景区域
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        
        # 查找非透明像素的边界框（Bounding Box）
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) == 0:  # 如果未检测到前景，则返回原始图像的 RGB 版本
            return input.convert('RGB')
        
        # 计算边界框及其中心点
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)  # 留出 20% 的边距
        
        # 计算裁剪区域
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2)
        )
        
        # 确保边界框在图像范围内
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(output.width, bbox[2]),
            min(output.height, bbox[3])
        )
        
        output = output.crop(bbox)
        
        # 添加填充以保持长宽比为 1:1
        width, height = output.size
        if width > height:
            new_height = width
            padding = (width - height) // 2
            padded_output = Image.new('RGBA', (width, new_height), (0, 0, 0, 0))
            padded_output.paste(output, (0, padding))
        else:
            new_width = height
            padding = (height - width) // 2
            padded_output = Image.new('RGBA', (new_width, height), (0, 0, 0, 0))
            padded_output.paste(output, (padding, 0))
        
        # 将填充后的图像缩放到目标分辨率
        padded_output = padded_output.resize((resolution, resolution), Image.Resampling.LANCZOS)
        
        # 最终处理：Alpha 预乘（Premultiplication）并转换回 PIL 格式
        output = np.array(padded_output).astype(np.float32) / 255
        output = np.dstack((
            output[:, :, :3] * output[:, :, 3:4],  # RGB 通道与 Alpha 通道相乘
            output[:, :, 3]                         # 原始 Alpha 通道
        ))
        output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
        
        return output

    def _lazy_load_birefnet(self):
        """延迟加载 BiRefNet 模型，用于背景移除。"""
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModelForImageSegmentation
        self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'weights/BiRefNet',
            trust_remote_code=True
        ).to(self.device)
        self.birefnet_model.eval()

    def _get_birefnet_mask(self, image: Image.Image) -> np.ndarray:
        """使用 BiRefNet 获取图像的前景掩码。

        Args:
            image: 输入图像。

        Returns:
            np.ndarray: 二值掩码数组。
        """
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images = transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取预测结果并应用 sigmoid
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)

        # 阈值处理生成二值掩码
        return (mask_np > 128).astype(np.uint8)

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """对图像进行特征编码。
        用 DINOv2 提取特征
        Args:
            image: 输入图像，可以是 batch 化的张量或 PIL 图像列表。

        Returns:
            torch.Tensor: 编码后的 Patch Tokens 特征。
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "图像张量应该是 batch 化的 (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "图像列表应包含 PIL 图像"
            # 缩放并转换为张量
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        # 应用归一化变换
        image = self.image_cond_model_transform(image).to(self.device)
        # 获取 DINOv2 特征
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        # 层归一化
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """获取模型的条件信息。

        Args:
            image: 图像提示。

        Returns:
            dict: 包含 'cond' (正向条件) 和 'neg_cond' (负向条件) 的字典。
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """根据给定的条件采样稀疏结构。
        
        Args:
            cond: 条件信息。
            num_samples: 生成样本的数量。
            sampler_params: 采样器的额外参数。

        Returns:
            torch.Tensor: 稀疏结构的坐标。
        """
        # 采样占用率潜变量 (Occupancy Latent)
        # 获取稀疏结构生成所使用的 Flow Model
        flow_model = self.models['sparse_structure_flow_model']
        # 获取模型分辨率
        reso = flow_model.resolution
        # 生成初始噪声
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )["samples"]
        
        # 解码占用率潜变量以获取坐标
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh',],
    ) -> dict:
        """对结构化潜空间 (SLAT) 进行解码。

        Args:
            slat: 结构化潜空间张量。
            formats: 解码的目标格式列表（如 ['mesh']）。

        Returns:
            dict: 包含解码后数据的字典。
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """根据给定的条件和坐标采样结构化潜空间 (SLAT)。
        
        Args:
            cond: 条件信息。
            coords: 稀疏结构的坐标。
            sampler_params: 采样器的额外参数。

        Returns:
            sp.SparseTensor: 采样后的 SLAT。
        """
        # 采样 SLAT
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )["samples"]

        # 反归一化
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh',],
        preprocess_image: bool = True,
    ) -> dict:
        """执行生成流水线。

        Args:
            image: 图像提示。
            num_samples: 生成样本的数量。
            seed: 随机种子。
            sparse_structure_sampler_params: 稀疏结构采样器的参数。
            slat_sampler_params: SLAT 采样器的参数。
            formats: 输出格式。
            preprocess_image: 是否对输入图像进行预处理（如抠图）。

        Returns:
            dict: 解码后的结果。
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """注入支持多图像条件的采样逻辑（上下文管理器）。
        
        Args:
            sampler_name: 采样器名称（如 'sparse_structure_sampler'）。
            num_images: 条件图像的数量。
            num_steps: 采样步数。
            mode: 模式，可以是 'stochastic' (随机选取) 或 'multidiffusion' (多扩散融合)。
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """使用多张图像作为条件执行流水线。

        Args:
            images: 素材的多视图图像列表。
            num_samples: 生成样本的数量。
            seed: 随机种子。
            sparse_structure_sampler_params: 稀疏结构采样器的参数。
            slat_sampler_params: SLAT 采样器的参数。
            formats: 输出格式。
            preprocess_image: 是否预处理图像。
            mode: 多视图融合模式。

        Returns:
            dict: 解码后的结果。
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
