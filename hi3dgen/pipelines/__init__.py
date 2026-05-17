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
# 导入采样器模块
from . import samplers
# 导入主要的 Hi3DGen 流水线类
from .hi3dgen import Hi3DGenPipeline

def from_pretrained(path: str):
    """从本地路径或 Hugging Face 模型库加载预训练流水线。

    Args:
        path: 模型的路径。可以是一个本地目录路径，或者是一个 Hugging Face 模型库的标识符。

    Returns:
        根据配置实例化并加载后的流水线对象。
    """
    import os
    import json
    # 检查 pipeline.json 是否存在于本地路径中
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        # 如果不是本地路径，则从 Hugging Face Hub 下载配置文件
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    # 读取配置文件以确定流水线名称
    with open(config_file, 'r') as f:
        config = json.load(f)

    # 动态获取流水线类并调用其 from_pretrained 方法
    return globals()[config['name']].from_pretrained(path)
