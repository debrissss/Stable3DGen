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
import torch

# 定义立方体的 8 个顶点相对于原点的偏移
cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
        1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int)

# 定义立方体的 6 个相邻方向的偏移 (上下左右前后)
cube_neighbor = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

# 定义立方体的 12 条边，每条边由两个顶点的索引对表示
cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, requires_grad=False)
     
def construct_dense_grid(res, device='cuda'):
    """根据分辨率构建一个稠密体素网格 (Dense Grid)。

    Args:
        res (int): 网格的分辨率（边长）。
        device (str): 运行计算的设备。默认为 'cuda'。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 包含 (顶点坐标, 每个立方体对应的 8 个顶点索引) 的元组。
    """
    res_v = res + 1
    # 生成所有顶点的线性索引
    vertsid = torch.arange(res_v ** 3, device=device)
    # 提取每个立方体“起始角点”的线性索引
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    # 计算 8 个角点相对于起始角点的线性偏移
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    # 得到每个立方体对应的 8 个顶点的全局索引 [res^3, 8]
    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))
    # 将线性索引转换回三维坐标 (x, y, z)
    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)
    return verts, cube_fx8


def construct_voxel_grid(coords):
    """根据输入的坐标点构建体素网格 (Voxel Grid)。

    该函数会为每个坐标点周围生成 8 个顶点，并去除重复顶点，最后返回唯一顶点集及立方体与顶点的映射。

    Args:
        coords (torch.Tensor): 输入的体素中心坐标 [N, 3]。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 包含 (唯一顶点坐标, 每个体素对应的 8 个顶点索引) 的元组。
    """
    # 为每个坐标生成 8 个角点坐标
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    # 去重并获取逆向索引（用于重构立方体顶点连接关系）
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    """将立方体上的特征值散射 (Scatter) 并聚合到顶点上。
    根据每一个立方体八个顶点的值计算出模型顶点的值

    Args:
        num_verts (int): 目标顶点的总数。
        cubes (torch.Tensor): 每个立方体对应的 8 个顶点索引 [V, 8]。
        value (torch.Tensor): 要分发的特征值 [V, 8, M]。
        reduce (str): 聚合方式，如 'mean' (平均) 或 'sum' (求和)。默认为 'mean'。

    Returns:
        torch.Tensor: 聚合到顶点后的特征张量 [num_verts, M]。
    """
    M = value.shape[2] # 特征通道数
    reduced = torch.zeros(num_verts, M, device=cubes.device)
    # 使用 scatter_reduce 将立方体属性聚合到顶点上
    return torch.scatter_reduce(reduced, 0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), reduce=reduce, include_self=False)
    
def sparse_cube2verts(coords, feats, training=True):
    """将稀疏立方体特征 (Sparse Cube Features) 转换为顶点特征。

    Args:
        coords (torch.Tensor): 稀疏立方体的坐标。
        feats (torch.Tensor): 每个立方体的特征。
        training (bool): 是否处于训练模式（用于计算一致性损失）。默认为 True。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含 (新顶点坐标, 顶点特征, 一致性正则项损失) 的元组。
    """
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        # 计算原始特征与从顶点重构回来的特征之间的一致性损失 (Consistency Loss)
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
    

def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    """将稀疏属性映射到稠密网格属性中。

    Args:
        coords (torch.Tensor): 稀疏点的坐标 [N, 3]。
        feats (torch.Tensor): 稀疏点的特征 [N, F]。
        res (int): 稠密网格的分辨率。
        sdf_init (bool): 是否初始化 SDF 值为 1（外部）。默认为 True。

    Returns:
        torch.Tensor: 扁平化后的稠密属性张量 [res^3, F]。
    """
    F = feats.shape[-1]
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    if sdf_init:
        # 第 0 个通道通常是 SDF，初始化为 1 表示物体外部
        dense_attrs[..., 0] = 1 
    # 将稀疏特征填充到对应坐标的稠密张量中
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    return dense_attrs.reshape(-1, F)


def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    """计算应用了形变偏移 (Deformation) 后的顶点坐标。

    该函数将归一化的坐标结合预测的形变值（通过 tanh 限制范围）来计算最终位置。

    Args:
        v_pos (torch.Tensor): 原始网格顶点坐标。
        deform (torch.Tensor): 预测的形变偏移量向量。
        res (int): 分辨率。

    Returns:
        torch.Tensor: 形变后的最终顶点坐标。
    """
    # 将坐标缩放到 [-0.5, 0.5] 范围，并应用形变
    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)
        