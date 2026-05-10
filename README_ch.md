# Stable3DGen

## 安装 (Installation)
克隆 (Clone) 仓库 (repo)：
```bash
git clone --recursive https://github.com/Stable-X/Stable3DGen.git
cd Stable3DGen
```

创建 (Create) 一个 conda 环境 (conda environment)（可选）：
```bash
conda create -n stablex python=3.10
conda activate stablex
```

安装依赖 (Install dependencies)：
```bash
# pytorch (选择正确的 CUDA 版本)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/{your-cuda-version}
pip install spconv-cu{your-cuda-version}==2.3.6 xformers==0.0.27.post2
# 其他依赖
pip install -r requirements.txt
```

## 本地 Demo (Local Demo) 🤗
运行方式：
```bash
python app.py
```

<!-- License -->
## 许可证 (License)
Stable3DGen 的模型和代码改编自 [**Trellis**](https://github.com/microsoft/TRELLIS)，后者遵循 [MIT License](LICENSE)。虽然原始的 Trellis 采用 MIT License，但我们专门移除了其对某些 NVIDIA 库 (kaolin, nvdiffrast, flexicube) 的依赖，以确保此改编版本可以用于商业用途。Stable3DGen 本身在 [MIT License](LICENSE) 下发布。

## 引用 (Citation)
如果你觉得这项工作有所帮助，请考虑引用我们的论文：
```
@article{ye2025hi3dgen,
  title={Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging},
  author={Ye, Chongjie and Wu, Yushuang and Lu, Ziteng and Chang, Jiahao and Guo, Xiaoyang and Zhou, Jiaqing and Zhao, Hao and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2503.22236}, 
  year={2025}
}
```
