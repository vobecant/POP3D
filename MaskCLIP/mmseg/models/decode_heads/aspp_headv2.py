# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ASPPModuleV2(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModuleV2, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHeadV2(BaseDecodeHead):
    """
    This head is the implementation of DeepLabV2

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (6, 12, 18, 24).
    """

    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        super(ASPPHeadV2, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModuleV2(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        output = sum(aspp_outs)
        output = self.cls_seg(output)
        return output

    def forward_module(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        output = sum(aspp_outs)
        return output