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

"""Hi3DGen Gradio 演示应用。

此模块实现了基于 Hi3DGen 管道的 Gradio 界面，用于从单张图像生成高保真 3D 几何模型。
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import gradio as gr
import os
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
from hi3dgen.pipelines import Hi3DGenPipeline
import trimesh
import tempfile

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def cache_weights(weights_dir: str) -> dict:
    """下载并缓存预训练模型权重。

    Args:
        weights_dir: 权重存储的本地目录。

    Returns:
        dict: 包含模型名称和对应本地路径的字典。
    """
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(weights_dir, exist_ok=True)
    model_ids = [
        "Stable-X/trellis-normal-v0-1",
        "Stable-X/yoso-normal-v1-8-1",
        "ZhengPeng7/BiRefNet",
    ]
    cached_paths = {}
    for model_id in model_ids:
        print(f"Caching weights for: {model_id}")
        # Check if the model is already cached
        local_path = os.path.join(weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            print(f"Already cached at: {local_path}")
            cached_paths[model_id] = local_path
            continue
        # Download the model and cache it
        print(f"Downloading and caching model: {model_id}")
        # Use snapshot_download to download the model
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        print(f"Cached at: {local_path}")

    return cached_paths

def preprocess_mesh(mesh_prompt):
    """预处理 Mesh 模型并导出为 GLB 格式。

    Args:
        mesh_prompt: 待处理的 Mesh 文件路径。

    Returns:
        str: 导出的 GLB 文件路径。
    """
    print("Processing mesh")
    trimesh_mesh = trimesh.load_mesh(mesh_prompt)
    trimesh_mesh.export(mesh_prompt+'.glb')
    return mesh_prompt+'.glb'

def preprocess_image(image):
    """预处理输入图像。

    Args:
        image: 输入的原始图像。

    Returns:
        处理后的图像 Tensor 或 PIL 对象。
    """
    if image is None:
        return None
    image = hi3dgen_pipeline.preprocess_image(image, resolution=1024)
    return image

def generate_3d(image, seed=-1,  
                ss_guidance_strength=3, ss_sampling_steps=50,
                slat_guidance_strength=3, slat_sampling_steps=6,
                slat_flow_model_path=None, ss_flow_model_path=None):
    """从图像生成 3D 模型的主函数。

    Args:
        image: 输入图像。
        seed: 随机种子，-1 表示随机生成。
        ss_guidance_strength: Stage 1 (Sparse Structure) 的 CFG Guidance 强度。
        ss_sampling_steps: Stage 1 的采样步数。
        slat_guidance_strength: Stage 2 (Structured Latent) 的 CFG Guidance 强度。
        slat_sampling_steps: Stage 2 的采样步数。
        slat_flow_model_path: 自定义结构化潜空间流模型的权重路径。
        ss_flow_model_path: 自定义稀疏结构流模型的权重路径。

    Returns:
        tuple: (法向图, 模型路径, 下载路径)
    """
    if image is None:
        return None, None, None

    global hi3dgen_pipeline
    current_slat_path = getattr(hi3dgen_pipeline, '_slat_flow_model_path', None)
    current_ss_path = getattr(hi3dgen_pipeline, '_ss_flow_model_path', None)
    
    # 标准化路径比较，空字符串或默认值均视同为 None (即 pipeline.json 里的默认设置)
    norm_slat_flow_model_path = slat_flow_model_path
    if slat_flow_model_path == "slat_flow_normal_dit_L_64l8p2_fp16" or not slat_flow_model_path or slat_flow_model_path.strip() == "":
        norm_slat_flow_model_path = None
        
    norm_current_slat_path = current_slat_path
    if current_slat_path == "slat_flow_normal_dit_L_64l8p2_fp16":
        norm_current_slat_path = None

    norm_ss_flow_model_path = ss_flow_model_path
    if ss_flow_model_path == "ss_flow_normal_dit_L_16l8p2_fp16" or not ss_flow_model_path or ss_flow_model_path.strip() == "":
        norm_ss_flow_model_path = None

    norm_current_ss_path = current_ss_path
    if current_ss_path == "ss_flow_normal_dit_L_16l8p2_fp16":
        norm_current_ss_path = None

    if norm_slat_flow_model_path != norm_current_slat_path or norm_ss_flow_model_path != norm_current_ss_path:
        print(f"Switching models: slat_flow_model from {current_slat_path} to {slat_flow_model_path}, ss_flow_model from {current_ss_path} to {ss_flow_model_path}...")
        if hi3dgen_pipeline is not None:
            hi3dgen_pipeline.cpu()
            del hi3dgen_pipeline
            torch.cuda.empty_cache()
        hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1", slat_flow_model_path=norm_slat_flow_model_path, ss_flow_model_path=norm_ss_flow_model_path)
        hi3dgen_pipeline._slat_flow_model_path = norm_slat_flow_model_path
        hi3dgen_pipeline._ss_flow_model_path = norm_ss_flow_model_path
        hi3dgen_pipeline.cuda()

    # 如果 seed 为 -1，则随机生成一个种子
    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)
    
    # 预处理输入图像
    image = hi3dgen_pipeline.preprocess_image(image, resolution=1024)
    # 使用法向预测器生成法向图（Normal Bridge）
    normal_image = normal_predictor(image, resolution=768, match_input_resolution=True, data_type='object')

    # 运行 Hi3DGen 生成管道
    outputs = hi3dgen_pipeline.run(
        normal_image,
        seed=seed,
        formats=["mesh",],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    generated_mesh = outputs['mesh'][0]
    
    # 保存生成结果
    import datetime
    output_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(os.path.join(TMP_DIR, output_id), exist_ok=True)
    mesh_path = f"{TMP_DIR}/{output_id}/mesh.glb"
    
    # 将生成的模型转换为 trimesh 格式并导出
    trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
    trimesh_mesh.export(mesh_path)

    return normal_image, mesh_path, mesh_path

def convert_mesh(mesh_path, export_format):
    """将 Mesh 转换为选定的导出格式。

    Args:
        mesh_path: 原始 GLB 模型路径。
        export_format: 目标格式（如 obj, ply, stl 等）。

    Returns:
        str: 转换后的临时文件路径。
    """
    if not mesh_path:
        return None
    
    # Create a temporary file to store the mesh data
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{export_format}", delete=False)
    temp_file_path = temp_file.name
    
    new_mesh_path = mesh_path.replace(".glb", f".{export_format}")
    mesh = trimesh.load_mesh(mesh_path)
    mesh.export(temp_file_path)  # Export to the temporary file
    
    return temp_file_path # Return the path to the temporary file

# 创建 Gradio 界面，优化布局
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging</h1>
        <p style='text-align: center;'>
            <strong>V0.1, Introduced By 
            <a href="https://gaplab.cuhk.edu.cn/" target="_blank">GAP Lab</a> from CUHKSZ and 
            <a href="https://www.nvsgames.cn/" target="_blank">Game-AIGC Team</a> from ByteDance</strong>
        </p>
        """
    )
    
    with gr.Row():
        gr.Markdown("""
                    <p align="center">
                    <a title="Website" href="https://stable-x.github.io/Hi3DGen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                    </a>
                    <a title="arXiv" href="https://stable-x.github.io/Hi3DGen/hi3dgen_paper.pdf" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                    </a>
                    <a title="Github" href="https://github.com/Stable-X/Hi3DGen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/Stable-X/Hi3DGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                    </a>
                    <a title="Social" href="https://x.com/ychngji6" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
                    </a>
                    </p>
                    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 左侧面板：输入与设置
            with gr.Tabs():
                
                with gr.Tab("Single Image"):
                    with gr.Row():
                        image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil")
                        normal_output = gr.Image(label="Normal Bridge", image_mode="RGBA", type="pil")
                        
                with gr.Tab("Multiple Images"):
                    gr.Markdown("<div style='text-align: center; padding: 40px; font-size: 24px;'>Multiple Images functionality is coming soon!</div>")
                        
            with gr.Accordion("Advanced Settings", open=False):
                ss_flow_model_path = gr.Dropdown(
                    choices=[
                        "ss_flow_normal_dit_L_16l8p2_fp16",
                        "ss_flow_normal_dit_M_16l8p2_fp16"
                    ],
                    value="ss_flow_normal_dit_L_16l8p2_fp16",
                    label="Sparse Structure Flow Model (ss_flow_model 权重名称/路径)",
                    allow_custom_value=True
                )
                slat_flow_model_path = gr.Dropdown(
                    choices=[
                        "slat_flow_normal_dit_L_64l8p2_fp16",
                        "slat_flow_normal_dit_M_64l8p2_fp16"
                    ],
                    value="slat_flow_normal_dit_L_64l8p2_fp16",
                    label="Structured Latent Flow Model (slat_flow_model 权重名称/路径)",
                    allow_custom_value=True
                )
                seed = gr.Slider(-1, MAX_SEED, label="Seed", value=0, step=1)
                gr.Markdown("#### Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=50, step=1)
                gr.Markdown("#### Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=6, step=1)
                    
            with gr.Group():
                with gr.Row():
                    gen_shape_btn = gr.Button("Generate Shape", size="lg", variant="primary")
                        
        # 右侧面板：输出结果
        with gr.Column(scale=1):
            with gr.Column():
                model_output = gr.Model3D(label="3D Model Preview (Each model is approximately 40MB, may take around 1 minute to load)")
            with gr.Column():
                export_format = gr.Dropdown(
                    choices=["obj", "glb", "ply", "stl"],
                    value="glb",
                    label="File Format"
                )
                download_btn = gr.DownloadButton(label="Export Mesh", interactive=False)

    # 事件处理绑定
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt]
    )
    
    gen_shape_btn.click(
        generate_3d,
        inputs=[
            image_prompt, seed,  
            ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps,
            slat_flow_model_path, ss_flow_model_path
        ],
        outputs=[normal_output, model_output, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    
    def update_download_button(mesh_path, export_format):
        """更新下载按钮的状态和文件路径。"""
        if not mesh_path:
            return gr.File.update(value=None, interactive=False)
        
        download_path = convert_mesh(mesh_path, export_format)
        return download_path
    
    export_format.change(
        update_download_button,
        inputs=[model_output, export_format],
        outputs=[download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    examples = gr.Examples(
        examples=[
            f'assets/example_image/{image}'
            for image in os.listdir("assets/example_image")
        ],
        inputs=image_prompt,
    )

    gr.Markdown(
        """
        **Acknowledgments**: Hi3DGen is built on the shoulders of giants. We would like to express our gratitude to the open-source research community and the developers of these pioneering projects:
        - **3D Modeling:** Our 3D Model is finetuned from the SOTA open-source 3D foundation model [Trellis](https://github.com/microsoft/TRELLIS) and we draw inspiration from the teams behind [Rodin](https://hyperhuman.deemos.com/rodin), [Tripo](https://www.tripo3d.ai/app/home), and [Dora](https://github.com/Seed3D/Dora).
        - **Normal Estimation:** Our Normal Estimation Model builds on the leading normal estimation research such as [StableNormal](https://github.com/hugoycj/StableNormal) and [GenPercept](https://github.com/aim-uofa/GenPercept).
        
        **Your contributions and collaboration push the boundaries of 3D modeling!**
        """
    )

if __name__ == "__main__":
    # 下载并缓存权重
    cache_weights(WEIGHTS_DIR)

    # 初始化 Hi3DGen Pipeline
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    hi3dgen_pipeline._slat_flow_model_path = None
    hi3dgen_pipeline._ss_flow_model_path = None
    hi3dgen_pipeline.cuda()

    # 初始化 Normal Predictor (法向估计模型)
    try:
        normal_predictor = torch.hub.load(os.path.join(torch.hub.get_dir(), 'hugoycj_StableNormal_main'), "StableNormal_turbo", yoso_version='yoso-normal-v1-8-1', source='local', local_cache_dir='./weights', pretrained=True)
    except:
        normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1', local_cache_dir='./weights')    

    # 启动 Gradio 应用
    demo.launch(share=False, server_name="0.0.0.0")

