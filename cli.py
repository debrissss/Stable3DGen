# MIT License
# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.

"""Hi3DGen CLI 应用。

此模块实现了基于 Hi3DGen 管道的单图命令行生成工具，用于从单张图像生成高保真 3D 几何模型。
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import tempfile
import shutil
from typing import *
import torch
import numpy as np
from PIL import Image
from hi3dgen.pipelines import Hi3DGenPipeline
import trimesh

os.environ['SPCONV_ALGO'] = 'native'

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
                slat_guidance_strength=3, slat_sampling_steps=6,):
    """从图像生成 3D 模型的主函数。

    Args:
        image: 输入图像。
        seed: 随机种子，-1 表示随机生成。
        ss_guidance_strength: Stage 1 (Sparse Structure) 的 CFG Guidance 强度。
        ss_sampling_steps: Stage 1 的采样步数。
        slat_guidance_strength: Stage 2 (Structured Latent) 的 CFG Guidance 强度。
        slat_sampling_steps: Stage 2 的采样步数。

    Returns:
        tuple: (法向图, 模型路径, 下载路径)
    """
    if image is None:
        return None, None, None

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hi3DGen 单图 3D 重建命令行工具")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径 (例如: configs/default.yaml)")
    parser.add_argument("--input_image", type=str, default=None, help="输入的单张图像文件路径 (例如: input.png)")
    parser.add_argument("--output_mesh", type=str, default=None, help="输出的 3D 模型文件路径，后缀决定格式 (例如: output.glb, output.obj)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，-1表示随机 (默认: 0)")
    parser.add_argument("--ss_guidance_strength", type=float, default=None, help="Stage 1 Guidance 强度 (默认: 3.0)")
    parser.add_argument("--ss_sampling_steps", type=int, default=None, help="Stage 1 采样步数 (默认: 50)")
    parser.add_argument("--slat_guidance_strength", type=float, default=None, help="Stage 2 Guidance 强度 (默认: 3.0)")
    parser.add_argument("--slat_sampling_steps", type=int, default=None, help="Stage 2 采样步数 (默认: 6)")
    
    args = parser.parse_args()

    # 默认配置字典
    default_config = {
        "input_image": None,
        "output_mesh": None,
        "seed": 0,
        "ss_guidance_strength": 3.0,
        "ss_sampling_steps": 50,
        "slat_guidance_strength": 3.0,
        "slat_sampling_steps": 6
    }

    # 加载 YAML 配置文件
    yaml_config = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: 找不到配置文件: {args.config}")
            exit(1)
        import yaml
        print(f"正在从配置文件加载参数: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}

    # 参数优先级合并：命令行输入 (非 None) > 配置文件参数 > 默认值
    final_config = {}
    for key, def_val in default_config.items():
        cli_val = getattr(args, key)
        if cli_val is not None:
            final_config[key] = cli_val
        elif key in yaml_config:
            final_config[key] = yaml_config[key]
        else:
            final_config[key] = def_val

    # 校验必要的参数
    if not final_config["input_image"]:
        print("Error: 必须提供输入图像路径。请通过命令行参数 --input_image 或配置文件指定。")
        exit(1)
    if not final_config["output_mesh"]:
        print("Error: 必须提供输出模型路径。请通过命令行参数 --output_mesh 或配置文件指定。")
        exit(1)

    # 将合并后的最终参数写回 args 命名空间，确保对原有逻辑完全兼容且无侵入性
    for key, val in final_config.items():
        setattr(args, key, val)

    # 1. 路径与后缀检查
    if not os.path.exists(args.input_image):
        print(f"Error: 找不到输入图像文件: {args.input_image}")
        exit(1)

    _, ext = os.path.splitext(args.output_mesh)
    if not ext:
        print("Error: 输出文件路径必须包含格式后缀 (例如 .glb, .obj, .ply)")
        exit(1)
    target_format = ext[1:].lower() 

    # 2. 模型下载与缓存
    print("检查并缓存模型权重...")
    cache_weights(WEIGHTS_DIR)

    # 3. 初始化全局变量：Pipeline 和 Normal Predictor 
    # (为了兼容原封不动复制的 generate_3d 方法中的全局变量引用)
    print("初始化模型 Pipeline...")
    global hi3dgen_pipeline
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    hi3dgen_pipeline.cuda()

    global normal_predictor
    try:
        normal_predictor = torch.hub.load(os.path.join(torch.hub.get_dir(), 'hugoycj_StableNormal_main'), "StableNormal_turbo", yoso_version='yoso-normal-v1-8-1', source='local', local_cache_dir='./weights', pretrained=True)
    except:
        normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1', local_cache_dir='./weights')    

    # 4. 加载输入图片 (模拟 Gradio 的 RGBA 转换)
    print(f"加载输入图像: {args.input_image}")
    input_img = Image.open(args.input_image)
    if input_img.mode != 'RGBA':
        input_img = input_img.convert('RGBA')

    # 5. 执行 3D 生成
    print("开始执行 3D 生成...")
    normal_img, original_mesh_path, _ = generate_3d(
        image=input_img,
        seed=args.seed,
        ss_guidance_strength=args.ss_guidance_strength,
        ss_sampling_steps=args.ss_sampling_steps,
        slat_guidance_strength=args.slat_guidance_strength,
        slat_sampling_steps=args.slat_sampling_steps
    )

    if not original_mesh_path or not os.path.exists(original_mesh_path):
        print("Error: 3D 模型生成失败。")
        exit(1)

    # 6. 格式转换 (如果输出要求的格式并非 .glb)
    if target_format != 'glb':
        print(f"正在将模型转换为 {target_format} 格式...")
        final_mesh_path = convert_mesh(original_mesh_path, target_format)
    else:
        final_mesh_path = original_mesh_path

    # 7. 复制文件到用户指定的输出路径
    output_dir = os.path.dirname(os.path.abspath(args.output_mesh))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    shutil.copy(final_mesh_path, args.output_mesh)
    print(f"\n成功! 3D 模型已保存至: {args.output_mesh}")
