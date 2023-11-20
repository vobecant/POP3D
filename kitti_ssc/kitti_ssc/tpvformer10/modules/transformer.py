# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TPVTemporalSelfAttention
from .spatial_cross_attention import TPVMSDeformableAttention3D
from .decoder import TPVCustomMSDeformableAttention


@TRANSFORMER.register_module()
class TPVPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        # self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(m, TPVTemporalSelfAttention) \
                    or isinstance(m, TPVCustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries, # list
            bev_h,
            bev_w,
            bev_z,
            grid_length=[0.512, 0.512],
            bev_pos=None, # list
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0) # bs, num_cam, C, h, w
        bev_queries_hw = bev_queries[0].unsqueeze(1).repeat(1, bs, 1)
        bev_queries_zh = bev_queries[1].unsqueeze(1).repeat(1, bs, 1)
        bev_queries_wz = bev_queries[2].unsqueeze(1).repeat(1, bs, 1)
        bev_pos_hw = bev_pos[0].flatten(2).permute(2, 0, 1)
        if bev_pos[1] is not None:
            bev_pos_zh = bev_pos[1].flatten(2).permute(2, 0, 1)
            bev_pos_wz = bev_pos[2].flatten(2).permute(2, 0, 1)
        else:
            bev_pos_zh = None
            bev_pos_wz = None

        # obtain rotation angle and shift with ego motion
        # delta_x = kwargs['img_metas'][0]['can_bus'][0]
        # delta_y = kwargs['img_metas'][0]['can_bus'][1]
        # ego_angle = kwargs['img_metas'][0]['can_bus'][-2] / np.pi * 180
        # rotation_angle = kwargs['img_metas'][0]['can_bus'][-1]
        # grid_length_y = grid_length[0]
        # grid_length_x = grid_length[1]
        # translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        # translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        # if translation_angle < 0:
        #     translation_angle += 360
        # bev_angle = ego_angle - translation_angle
        # shift_y = translation_length * \
        #     np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        # shift_x = translation_length * \
        #     np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        # shift_y = shift_y * self.use_shift
        # shift_x = shift_x * self.use_shift
        # shift = bev_queries_hw.new_tensor([shift_x, shift_y])

        # if prev_bev is not None:
        #     if prev_bev.shape[1] == bev_h * bev_w:
        #         prev_bev = prev_bev.permute(1, 0, 2)
        #     if self.rotate_prev_bev:
        #         num_prev_bev = prev_bev.size(1)
        #         prev_bev = prev_bev.reshape(bev_h, bev_w, -1).permute(2, 0, 1)
        #         prev_bev = rotate(prev_bev, rotation_angle,
        #                           center=self.rotate_center)
        #         prev_bev = prev_bev.permute(1, 2, 0).reshape(
        #             bev_h * bev_w, num_prev_bev, -1)

        # # add can bus signals
        # can_bus = bev_queries_hw.new_tensor(kwargs['img_metas'][0]['can_bus'])[
        #     None, None, :]
        # can_bus = self.can_bus_mlp(can_bus)
        # bev_queries_hw = bev_queries_hw + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos_hw.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        bev_embed = self.encoder(
            [bev_queries_hw, bev_queries_zh, bev_queries_wz],
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_z=bev_z,
            bev_pos=[bev_pos_hw, bev_pos_zh, bev_pos_wz],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=0,
            **kwargs
        )

        return bev_embed

    # def forward(self,
    #             mlvl_feats,
    #             bev_queries,
    #             object_query_embed,
    #             bev_h,
    #             bev_w,
    #             bev_z,
    #             grid_length=[0.512, 0.512],
    #             bev_pos=None,
    #             reg_branches=None,
    #             cls_branches=None,
    #             prev_bev=None,
    #             **kwargs):
    #     """Forward function for `Detr3DTransformer`.
    #     Args:
    #         mlvl_feats (list(Tensor)): Input queries from
    #             different level. Each element has shape
    #             [bs, num_cams, embed_dims, h, w].
    #         bev_queries (Tensor): (bev_h*bev_w, c)
    #         bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
    #         object_query_embed (Tensor): The query embedding for decoder,
    #             with shape [num_query, c].
    #         reg_branches (obj:`nn.ModuleList`): Regression heads for
    #             feature maps from each decoder layer. Only would
    #             be passed when `with_box_refine` is True. Default to None.
    #     Returns:
    #         tuple[Tensor]: results of decoder containing the following tensor.
    #             - bev_embed: BEV features
    #             - inter_states: Outputs from decoder. If
    #                 return_intermediate_dec is True output has shape \
    #                   (num_dec_layers, bs, num_query, embed_dims), else has \
    #                   shape (1, bs, num_query, embed_dims).
    #             - init_reference_out: The initial value of reference \
    #                 points, has shape (bs, num_queries, 4).
    #             - inter_references_out: The internal value of reference \
    #                 points in decoder, has shape \
    #                 (num_dec_layers, bs,num_query, embed_dims)
    #             - enc_outputs_class: The classification score of \
    #                 proposals generated from \
    #                 encoder's feature maps, has shape \
    #                 (batch, h*w, num_classes). \
    #                 Only would be returned when `as_two_stage` is True, \
    #                 otherwise None.
    #             - enc_outputs_coord_unact: The regression results \
    #                 generated from encoder's feature maps., has shape \
    #                 (batch, h*w, 4). Only would \
    #                 be returned when `as_two_stage` is True, \
    #                 otherwise None.
    #     """
    #     # import pdb; pdb.set_trace()
    #     bev_embed = self.get_bev_features(
    #         mlvl_feats,
    #         bev_queries,
    #         bev_h,
    #         bev_w,
    #         bev_z,
    #         grid_length=grid_length,
    #         bev_pos=bev_pos,
    #         prev_bev=prev_bev,
    #         **kwargs) # bev_embed shape: bs, bev_h*bev_w, embed_dims
        
    #     # import pdb; pdb.set_trace()
    #     bs = mlvl_feats[0].size(0)
    #     query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
    #     query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
    #     query = query.unsqueeze(0).expand(bs, -1, -1)
    #     reference_points = self.reference_points(query_pos)
    #     reference_points = reference_points.sigmoid()
    #     init_reference_out = reference_points

    #     query = query.permute(1, 0, 2)
    #     query_pos = query_pos.permute(1, 0, 2)
    #     bev_embed = [bev.permute(1, 0, 2) for bev in bev_embed]

    #     inter_states, inter_references = self.decoder(
    #         query=query,
    #         key=None,
    #         value=bev_embed,
    #         query_pos=query_pos,
    #         reference_points=reference_points,
    #         reg_branches=reg_branches,
    #         cls_branches=cls_branches,
    #         spatial_shapes=torch.tensor([[bev_w, bev_h, bev_z]], device=query.device),
    #         level_start_index=torch.tensor([0], device=query.device),
    #         **kwargs)

    #     inter_references_out = inter_references

    #     return bev_embed, inter_states, init_reference_out, inter_references_out


if __name__ == "__main__":
    pass