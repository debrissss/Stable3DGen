# 项目开发文档：Hi3DGen

Hi3DGen 是一个通过“法线桥接”（Normal Bridging）方法从图像生成高保真 3D 几何形状的框架。它将 3D 生成任务分解为两个阶段：稀疏结构生成和结构化潜变量生成，并由表面法线引导生成过程。

## 🏗 高层架构

该项目采用了现代生成式 3D 流线中常见的模块化设计：

1.  **预处理 (Preprocessing)**：对输入图像进行分割（背景移除），并估计表面法线以提供几何引导。
2.  **第一阶段：稀疏结构生成 (Stage 1: Sparse Structure Generation)**：基于流（Flow-based）的模型预测物体所在的稀疏 3D 占据情况（体素/立方体）。
3.  **第二阶段：结构化潜变量生成 (Stage 2: Structured Latent Generation)**：另一个基于流的模型为占据的稀疏坐标生成详细的潜变量特征（结构化潜变量，简称 SLAT）。
4.  **解码 (Decoding)**：生成的潜变量通过一种感知变形的 Marching Cubes 算法解码为三角网格模型（Mesh）。

---

## 📂 目录结构与文件作用

### 根目录
- **`app.py`**：项目主入口。托管基于 Gradio 的 Web 界面，供用户上传图像并生成 3D 模型。
- **`requirements.txt`**：Python 依赖列表。
- **`LICENSE`**：项目许可证 (MIT)。
- **`README.md`**：面向用户的基本文档和安装说明。
- **`assets/`**：包含示例图像和 UI 图标等静态资源。

### `hi3dgen/` (核心包)
生成逻辑的核心，分为几个子包：

#### 🚀 `hi3dgen/pipelines/` (执行流)
- **`base.py`**：流线的基类，处理模型加载和设备管理。
- **`hi3dgen.py`**：**核心流线类**。协调预处理、两阶段生成和解码。支持单图和多图输入。
- **`samplers/`**：包含流模型使用的采样算法（如 Euler 采样器）。

#### 🧠 `hi3dgen/models/` (神经网络)
- **`sparse_structure_flow.py`**：第一阶段模型的实现。一种基于 Transformer (DiT) 的流模型，用于预测稀疏占据结构。
- **`sparse_structure_vae.py`**：用于编码/解码稀疏结构潜变量的 VAE。
- **`structured_latent_flow.py`**：第二阶段模型的实现。为稀疏坐标预测高维特征。
- **`structured_latent_vae/`**：
    - **`encoder.py`**：将 3D 几何编码为结构化潜变量。
    - **`decoder_mesh.py`**：将结构化潜变量解码回 3D 网格表示。

#### 🧩 `hi3dgen/modules/` (构建模块)
- **`attention/`**：自定义注意力机制（如 Flash Attention）。
- **`transformer/`**：模块化的 Transformer 块和位置嵌入。
- **`sparse/`**：处理稀疏张量（体素）的工具。
- **`spatial.py`**：3D 空间数据的操作（如 patchify/unpatchify）。
- **`norm.py`**：归一化层（LayerNorm, RMSNorm）。
- **`utils.py`**：通用的神经网络工具函数。

#### 📐 `hi3dgen/representations/` (3D 数据表示)
- **`mesh/cube2mesh.py`**：**潜变量与几何体之间的桥梁**。利用 Marching Cubes 算法将稀疏立方体特征转换为三角网格。
- **`mesh/utils_cube.py`**：处理基于立方体的 3D 网格的几何工具。

---

## 🛠 关键技术概念

-   **法线桥接 (Normal Bridging)**：利用估计的表面法线作为中间表示来引导 3D 形状生成，确保极高的几何保真度。
-   **流匹配 (Flow Matching)**：项目使用条件流匹配 (CFM) 代替传统的扩散模型 (Diffusion)，生成速度更快且更稳定。
-   **结构化潜变量 (SLAT)**：一种将 3D 几何存储为稀疏体素网格特征的表示方法，能够在保持低内存开销的同时实现高分辨率细节。
-   **变形 Marching Cubes (Deformable Marching Cubes)**：Marching Cubes 的扩展版本，允许顶点从网格位置偏移，从而能够表示锐利的边缘和平滑的曲线。
