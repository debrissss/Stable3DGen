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
# SPDX-License-Identifier: MIT
from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """流水线的基类。

    该类用于管理多个模型组件，并提供统一的加载和设备切换接口。
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        """初始化流水线。

        Args:
            models: 模型名称到模型的映射字典。
        """
        if models is None:
            return
        self.models = models
        # 将所有模型设置为评估模式
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str, slat_flow_model_path: str = None) -> "Pipeline":
        """从预训练路径加载流水线。

        Args:
            path: 本地路径或 Hugging Face 模型库标识符。
            slat_flow_model_path: 自定义结构化潜空间流模型的权重路径。

        Returns:
            实例化并加载后的流水线对象。
        """
        import os
        import json
        # 检查本地是否存在配置文件
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            # 从 Hugging Face Hub 下载配置文件
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        # 递归加载所有子模型
        _models = {}
        for k, v in args['models'].items():
            if k == 'slat_flow_model' and slat_flow_model_path is not None:
                default_dir_prefix = os.path.dirname(v) # 例如 "ckpts"
                if os.path.isabs(slat_flow_model_path):
                    model_path = slat_flow_model_path
                elif slat_flow_model_path.startswith("weights/") or slat_flow_model_path.startswith("./") or slat_flow_model_path.startswith("../"):
                    model_path = slat_flow_model_path
                else:
                    # 尝试拼接预训练目录和默认子文件夹路径
                    candidate1 = f"{path}/{default_dir_prefix}/{slat_flow_model_path}" if default_dir_prefix else f"{path}/{slat_flow_model_path}"
                    candidate2 = f"{path}/{slat_flow_model_path}"
                    
                    # 优先检测本地文件是否存在
                    if os.path.exists(f"{candidate1}.json") or os.path.exists(f"{candidate1}.safetensors"):
                        model_path = candidate1
                    elif os.path.exists(f"{candidate2}.json") or os.path.exists(f"{candidate2}.safetensors"):
                        model_path = candidate2
                    else:
                        # 本地都不存在时，如果包含 '/' 则可能是 HF repo 路径，否则默认使用 candidate1
                        if '/' in slat_flow_model_path and not slat_flow_model_path.startswith(default_dir_prefix + "/"):
                            model_path = slat_flow_model_path
                        else:
                            model_path = candidate1
                
                print(f"Loading custom slat_flow_model from: {model_path}")
                _models[k] = models.from_pretrained(model_path)
            else:
                _models[k] = models.from_pretrained(f"{path}/{v}")

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        """获取流水线所在的设备。

        尝试从子模型中推断设备。

        Returns:
            torch.device: 检测到的设备。

        Raises:
            RuntimeError: 如果找不到任何设备信息。
        """
        # 优先检查模型是否具有 device 属性
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        # 其次通过检查模型参数来确定设备
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("未找到设备信息。")

    def to(self, device: torch.device) -> None:
        """将流水线中的所有模型移动到指定设备。

        Args:
            device: 目标设备。
        """
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        """将流水线移动到 CUDA 设备。"""
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        """将流水线移动到 CPU 设备。"""
        self.to(torch.device("cpu"))
