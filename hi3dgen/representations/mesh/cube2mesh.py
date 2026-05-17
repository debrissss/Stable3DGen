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
# Copyright (c) [2025] [jclarkk] 
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
#
# Original file was released under MIT, with the full license text
# available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
#
# This modified file is released under the same license.
import torch
from ...modules.sparse import SparseTensor
from .utils_cube import *
import numpy as np
import trimesh
import numpy as np
from skimage import measure
from typing import Tuple, Optional

class MeshExtractResult:
    """Mesh 提取结果类。

    该类用于存储从体素网格中提取出的 Mesh 数据，包括顶点、面片、属性以及法线，
    并提供将其转换为 trimesh 格式的方法。

    Attributes:
        vertices (torch.Tensor): Mesh 顶点坐标。
        faces (torch.Tensor): Mesh 面片索引 (LongTensor)。
        vertex_attrs (torch.Tensor, optional): 顶点属性（如颜色）。
        vertex_normal (torch.Tensor): 计算出的顶点法线。
        face_normal (torch.Tensor): 计算出的面片法线。
        res (int): 体素网格的分辨率。
        success (bool): 提取是否成功（是否存在顶点和面片）。
        tsdf_v (torch.Tensor, optional): 训练用的 TSDF 顶点。
        tsdf_s (torch.Tensor, optional): 训练用的 TSDF 标量值。
        reg_loss (torch.Tensor, optional): 训练用的正则化损失。
    """

    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        """初始化 MeshExtractResult。

        Args:
            vertices (torch.Tensor): 顶点张量。
            faces (torch.Tensor): 面片张量。
            vertex_attrs (torch.Tensor, optional): 顶点属性。默认为 None。
            res (int): 分辨率。默认为 64。
        """
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.vertex_normal = self.comput_v_normals(vertices, faces)
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        """计算面片法线 (Face Normals)。

        Args:
            verts (torch.Tensor): 顶点张量。
            faces (torch.Tensor): 面片索引张量。

        Returns:
            torch.Tensor: 标准化后的面片法线。
        """
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        # 使用叉积计算法线
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        """计算顶点法线 (Vertex Normals)。

        通过累加相邻面片的法线并进行标准化来获得。

        Args:
            verts (torch.Tensor): 顶点张量。
            faces (torch.Tensor): 面片索引张量。

        Returns:
            torch.Tensor: 标准化后的顶点法线。
        """
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        # 将面片法线散射累加到对应的三个顶点上
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   
    
    def to_trimesh(self, transform_pose=False):
        """将当前结果转换为 trimesh.Trimesh 对象。

        Args:
            transform_pose (bool): 是否应用坐标系变换（通常用于从 y-up 转换到 z-up）。默认为 False。

        Returns:
            trimesh.Trimesh: 转换后的 Mesh 对象。
        """
        vertices = self.vertices.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        
        if transform_pose:
            # 旋转矩阵，用于调整模型朝向
            transform_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            vertices = vertices @ transform_matrix
            vertex_normals = self.vertex_normal.detach().cpu().numpy() @ transform_matrix
        else:
            vertex_normals = self.vertex_normal.detach().cpu().numpy()
        
        # 创建 trimesh 实例
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_normals=self.face_normal.detach().cpu().numpy(),
            vertex_normals=vertex_normals
        )
        
        return mesh

class EnhancedMarchingCubes:
    """增强型 Marching Cubes 实现。

    该类封装了基础的 Marching Cubes 算法，并增加了对顶点形变 (Deformation) 
    和颜色插值的支持，适用于动态生成的 3D 几何体。

    Attributes:
        device (str): 运行计算的设备 (如 "cuda" 或 "cpu")。
    """

    def __init__(self, device="cuda"):
        """初始化 EnhancedMarchingCubes。

        Args:
            device (str): 设备名称。默认为 "cuda"。
        """
        self.device = device

    def __call__(self,
                 voxelgrid_vertices: torch.Tensor,
                 scalar_field: torch.Tensor,
                 voxelgrid_colors: Optional[torch.Tensor] = None,
                 training: bool = False
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                 
        """执行增强型 Marching Cubes 提取。

        Args:
            voxelgrid_vertices (torch.Tensor): 用于形变的体素网格的顶点坐标。
            scalar_field (torch.Tensor): 标量场 (SDF)，用于提取等值面。
            voxelgrid_colors (torch.Tensor, optional): 体素颜色网格。默认为 None。
            training (bool): 是否处于训练模式。默认为 False。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                包含 (形变后的顶点, 面片索引, 偏差损失, 颜色属性) 的元组。
        """
        # 确保标量场 (scalar_field) 是三维形状 [res, res, res]
        if scalar_field.dim() == 1:
            # 如果是一维张量（扁平化的），通过开三次方根（还原立方体边长）计算网格分辨率并重塑形状
            grid_size = int(round(scalar_field.shape[0] ** (1 / 3)))
            scalar_field = scalar_field.reshape(grid_size, grid_size, grid_size)
        elif scalar_field.dim() > 3:
            # 如果维度大于 3（例如带有 batch 维度），则压缩多余维度
            scalar_field = scalar_field.squeeze()

        # 转换为 numpy 数组进行算法处理
        scalar_np = scalar_field.cpu().numpy() # sdf

        if scalar_np.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {scalar_np.shape}")

        # 运行基础 Marching Cubes 算法提取等值面，从 SDF 得到 Mesh
        vertices, faces, normals, _ = measure.marching_cubes(
            scalar_np,
            level=0.0,
            gradient_direction='ascent'
        )

        vertices = torch.from_numpy(np.ascontiguousarray(vertices)).float().to(self.device)
        faces = torch.from_numpy(np.ascontiguousarray(faces)).long().to(self.device)

        # 应用形变 (Deformations)
        if voxelgrid_vertices is not None:
            # 如果需要，重构并归一化体素顶点
            if voxelgrid_vertices.dim() == 2:
                voxelgrid_vertices = voxelgrid_vertices.reshape(grid_size, grid_size, grid_size, 3)
            deformed_vertices = self._apply_deformations(vertices, voxelgrid_vertices)
        else:
            deformed_vertices = vertices

        # 处理颜色插值
        colors = None
        if voxelgrid_colors is not None:
            if voxelgrid_colors.dim() == 2:
                voxelgrid_colors = voxelgrid_colors.reshape(grid_size, grid_size, grid_size, -1)
            colors = self._interpolate_colors(vertices, voxelgrid_colors)
            # 使用 sigmoid 确保颜色值在 [0, 1] 范围内
            colors = torch.sigmoid(colors)

        # 在训练模式下计算偏差损失 (Deviation Loss)
        deviation_loss = torch.tensor(0.0, device=self.device)
        if training:
            deviation_loss = self._compute_deviation_loss(vertices, deformed_vertices)

        # 翻转面片索引顺序（可能是由于某种坐标系差异）
        faces = faces.flip(dims=[1])

        return deformed_vertices, faces, deviation_loss, colors

    def _apply_deformations(self, vertices: torch.Tensor,
                            voxelgrid_vertices: torch.Tensor) -> torch.Tensor:
        """通过三线性插值 (Trilinear Interpolation) 对顶点应用形变。

        Args:
            vertices (torch.Tensor): 原始网格顶点。
            voxelgrid_vertices (torch.Tensor): 存储在体素网格中的偏移量。

        Returns:
            torch.Tensor: 形变后的顶点。
        """
        grid_positions = vertices.clone()

        # 获取整数网格索引
        grid_coords = grid_positions.long()

        # 获取网格内局部相对位置（小数）
        local_coords = grid_positions - grid_coords.float()

        if voxelgrid_vertices.dim() == 2:
            grid_size = int(round(voxelgrid_vertices.shape[0] ** (1 / 3)))
            voxelgrid_vertices = voxelgrid_vertices.reshape(grid_size, grid_size, grid_size, 3)

        # 限制坐标范围，防止溢出
        grid_coords = torch.clamp(grid_coords, 0, voxelgrid_vertices.shape[0] - 1)

        # 执行三线性插值
        deformed_vertices = self._trilinear_interpolate(
            grid_coords, local_coords, voxelgrid_vertices
        )

        return deformed_vertices

    def _interpolate_colors(self, vertices: torch.Tensor,
                            voxelgrid_colors: torch.Tensor) -> torch.Tensor:
        """为网格顶点插值颜色。

        Args:
            vertices (torch.Tensor): 网格顶点。
            voxelgrid_colors (torch.Tensor): 体素颜色网格。

        Returns:
            torch.Tensor: 插值后的顶点颜色。
        """
        grid_positions = vertices.clone()

        grid_coords = grid_positions.long()
        local_coords = grid_positions - grid_coords.float()

        if voxelgrid_colors.dim() == 2:
            grid_size = int(round(voxelgrid_colors.shape[0] ** (1 / 3)))
            color_channels = voxelgrid_colors.shape[1]
            voxelgrid_colors = voxelgrid_colors.reshape(grid_size, grid_size, grid_size, color_channels)

        grid_coords = torch.clamp(grid_coords, 0, voxelgrid_colors.shape[0] - 1)

        return self._trilinear_interpolate(
            grid_coords, local_coords, voxelgrid_colors, is_color=True
        )

    def _trilinear_interpolate(self, grid_coords: torch.Tensor,
                               local_coords: torch.Tensor,
                               values: torch.Tensor,
                               is_color: bool = False) -> torch.Tensor:
        """执行三线性插值。

        Args:
            grid_coords (torch.Tensor): 整数网格坐标。
            local_coords (torch.Tensor): 网格内的局部坐标 [0, 1]。
            values (torch.Tensor): 体素网格值。
            is_color (bool): 是否正在处理颜色。默认为 False。

        Returns:
            torch.Tensor: 插值后的值。
        """
        x, y, z = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]

        if is_color and values.dim() == 2:
            grid_size = int(round(values.shape[0] ** (1 / 3)))
            color_channels = values.shape[1]
            values = values.reshape(grid_size, grid_size, grid_size, color_channels)

        # 根据维度获取 8 个角点的值
        if values.dim() == 4:  # (grid x grid x grid x channels)
            c000 = values[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], :]
            c001 = values[grid_coords[:, 0], grid_coords[:, 1],
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
            c010 = values[grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1),
                   grid_coords[:, 2], :]
            c011 = values[grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1),
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
            c100 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1],
                   grid_coords[:, 2], :]
            c101 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1],
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
            c110 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1),
                   torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1), grid_coords[:, 2], :]
            c111 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1),
                   torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1),
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
        else:
            c000 = values[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
            c001 = values[
                grid_coords[:, 0], grid_coords[:, 1], torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]
            c010 = values[
                grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1), grid_coords[:, 2]]
            c011 = values[grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1), torch.clamp(
                grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]
            c100 = values[
                torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1], grid_coords[:, 2]]
            c101 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1], torch.clamp(
                grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]
            c110 = values[
                torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), torch.clamp(grid_coords[:, 1] + 1, 0,
                                                                                        values.shape[
                                                                                            1] - 1), grid_coords[:, 2]]
            c111 = values[
                torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), torch.clamp(grid_coords[:, 1] + 1, 0,
                                                                                        values.shape[
                                                                                            1] - 1), torch.clamp(
                    grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]

        # 如果需要，增加通道维度
        if values.dim() == 3:
            c000, c001, c010, c011 = [c[..., None] if c.dim() == 1 else c for c in [c000, c001, c010, c011]]
            c100, c101, c110, c111 = [c[..., None] if c.dim() == 1 else c for c in [c100, c101, c110, c111]]

        # 沿 x 轴插值
        c00 = c000 * (1 - x)[:, None] + c100 * x[:, None]
        c01 = c001 * (1 - x)[:, None] + c101 * x[:, None]
        c10 = c010 * (1 - x)[:, None] + c110 * x[:, None]
        c11 = c011 * (1 - x)[:, None] + c111 * x[:, None]

        # 沿 y 轴插值
        c0 = c00 * (1 - y)[:, None] + c10 * y[:, None]
        c1 = c01 * (1 - y)[:, None] + c11 * y[:, None]

        # 沿 z 轴插值并返回结果
        return c0 * (1 - z)[:, None] + c1 * z[:, None]

    def _compute_deviation_loss(self, original_vertices: torch.Tensor,
                                deformed_vertices: torch.Tensor) -> torch.Tensor:
        """计算形变偏差损失 (L2 Loss)。

        Args:
            original_vertices (torch.Tensor): 原始提取顶点。
            deformed_vertices (torch.Tensor): 形变后的顶点。

        Returns:
            torch.Tensor: 平均平方误差。
        """
        return torch.mean((deformed_vertices - original_vertices) ** 2)

class SparseFeatures2Mesh:
    """稀疏特征转 Mesh 工具类。

    该类负责将输入的稀疏张量 (SparseTensor) 转换为稠密表示，并利用 
    EnhancedMarchingCubes 提取最终的 3D 网格。

    Attributes:
        device (str): 运行设备。
        res (int): 目标网格分辨率。
        mesh_extractor (EnhancedMarchingCubes): Mesh 提取器实例。
        sdf_bias (float): SDF 的初始偏移量。
        reg_c (torch.Tensor): 用于稠密网格化的正则化 Cube。
        reg_v (torch.Tensor): 用于稠密网格化的正则化顶点。
        use_color (bool): 是否生成颜色。
        layouts (dict): 存储特征通道布局定义的字典。
    """

    def __init__(self, device="cuda", res=128, use_color=True):
        """初始化 SparseFeatures2Mesh。

        Args:
            device (str): 设备名称。默认为 "cuda"。
            res (int): 分辨率。默认为 128。
            use_color (bool): 是否使用颜色。默认为 True。
        """
        super().__init__()
        self.device = device
        self.res = res
        self.mesh_extractor = EnhancedMarchingCubes(device=device)
        self.sdf_bias = -1.0 / res
        # 构建基础稠密网格
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()

    def _calc_layout(self):
        """定义稀疏特征张量的布局 (Channel Layout)。"""
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            包含法向图在内的 6 通道颜色特征。
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = LAYOUTS
        start = 0
        # 计算每个特征的起始和终止位置
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        """根据名称从特征张量中切片并重构对应布局。

        Args:
            feats (torch.Tensor): 原始特征张量。
            name (str): 布局名称 (如 'sdf', 'color')。

        Returns:
            torch.Tensor: 切片后的特征张量。
        """
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name][
            'shape'])

    def __call__(self, cubefeats: SparseTensor, training=False):
        """执行从稀疏特征到 Mesh 的全流程转换。

        Args:
            cubefeats (SparseTensor): 输入的立方体稀疏特征。
            training (bool): 是否处于训练阶段。默认为 False。

        Returns:
            MeshExtractResult: 包含生成 Mesh 及其元数据的结果对象。
        """
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats

        # 从原始特征中解析各个组件
        sdf, deform, color, weights = [self.get_layout(feats, name)
                                       for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        
        # 将稀疏立方体特征映射到顶点
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1),
                                                     training=training)

        # 获取稠密属性
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res + 1, sdf_init=True)

        if self.use_color:
            sdf_d, deform_d, colors_d = (v_attrs_d[..., 0], v_attrs_d[..., 1:4],
                                         v_attrs_d[..., 4:])
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        # 获取形变后的网格顶点坐标
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)

        # 运行增强型 Marching Cubes 提取 Mesh
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            voxelgrid_colors=colors_d,
            training=training
        )

        # 封装结果
        mesh = MeshExtractResult(vertices=vertices, faces=faces,
                                 vertex_attrs=colors, res=self.res)

        # 如果在训练中，计算额外的正则化损失
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]

        return mesh