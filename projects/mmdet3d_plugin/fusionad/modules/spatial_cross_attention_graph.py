
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
# Modifications:
# - Modified by FusionAD on 2023.5
# - Added extended support from FusionAD (https://arxiv.org/abs/2308.01006)

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

# for graph tree
from scipy.spatial import cKDTree
import numpy as np
##################################
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttentionGraph(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 k_graph=0,
                 **kwargs
                 ):
        super(SpatialCrossAttentionGraph, self).__init__(init_cfg)
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first

        self.k_graph = k_graph
        if self.k_graph >= 1:
            self.dtransform = nn.Linear(embed_dims*2, embed_dims)
            self.dtransform_conv1 = nn.Linear(embed_dims, embed_dims)
            self.dtransform_conv2 = nn.Linear(embed_dims*self.k_graph, embed_dims)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def cKDTree_neighbor(self, query_points):
        """"
        input:
            query_points: [num_cam, batch_size, num_query, num_points, 2(x, y)]
        return:
            neighbor_values_torch: [num_cam, batch_size, num_query, num_points, self.k_graph, 2(x, y)]
        """
        num_cam, batch_size, num_query, num_points, _ = query_points.shape
        query_points = query_points.view(-1, 2)
        query_points_numpy = query_points.cpu().numpy()
        # 使用 np.unique 去除重复元素，同时返回唯一坐标及其在原数组中的索引
        unique_coords, unique_indices = np.unique(query_points_numpy, axis=0, return_index=True)

        # 构建 KD 树
        kdtree = cKDTree(unique_coords)

        # 查询示例：获取最近的8个邻居 +1是本身
        distances, neighbor_indices = kdtree.query(query_points_numpy, k=self.k_graph+1)

        # 假设unique_indices的大小至少为19
        if all(np.any(neighbor_index >= len(unique_indices)) for neighbor_index in neighbor_indices):
            print("neighbor_indices中存在无效索引")
        else:
            neighbor_indices = unique_indices[neighbor_indices]

        neighbor_indices = neighbor_indices[:, 1:]  # 去掉本身
        neighbor_values = query_points_numpy[neighbor_indices]


        # 将NumPy数组转换为PyTorch张量
        neighbor_values_torch = torch.from_numpy(neighbor_values).view(num_cam, batch_size, num_query, num_points, self.k_graph, -1).cuda()
  
        return neighbor_values_torch

    def dual_transform(self, queries, queries_neighbor):

        x1 = self.dtransform_conv1(queries)
        x2 = self.dtransform_conv2(queries_neighbor)
        
        x = torch.cat([x1, x2], dim=-1)
        x = self.dtransform(x)

        return x


    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # reference_points_cam [num_cam, batch_size, num_query, num_points, 2(x, y)]
        if key is None: # [6, 30825, 1, 256]
            key = query
        if value is None: # [6, 30825, 1, 256]
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3) # [6, 1, 40000, 4, 2]
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        if self.k_graph >= 1:
            # 生成K个邻域reference_points
            reference_points_rebatch_graph = self.cKDTree_neighbor(reference_points_rebatch) # [1, 6, 9463, 4, 2] --> [1, 6, 9463, 4, 4, 2]
            # [batch_size, num_cam, max_num_query, num_points, (x,y)] --> [batch_size, num_cam, max_num_query, num_points, self.k_graph, (x,y)]

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        
        if self.k_graph >= 1:
            # 用K个邻域reference_points执行deformable_attention
            queries_group = []
            for i in range(self.k_graph):
                query_item = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                                    reference_points=reference_points_rebatch_graph[..., i, :].view(bs*self.num_cams, max_len, D, 2),
                                                    spatial_shapes=spatial_shapes,
                                                    level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
                queries_group.append(query_item)
            queries_neighbor = torch.concat(queries_group, -1)
            queries = self.dual_transform(queries, queries_neighbor)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual