import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS


def build_decoder(num_layers, in_dims, hidden_dims, out_dims):
    def build_block(in_dims, out_dims):
        return [nn.Softplus(), nn.Linear(in_dims, out_dims)]

    if num_layers == 1:
        layers = [nn.Linear(in_dims, out_dims)]
    else:
        layers = [nn.Linear(in_dims, hidden_dims)]
        for _i in range(1, num_layers - 1):
            layers.extend(build_block(hidden_dims, hidden_dims))
        layers.extend(build_block(hidden_dims, out_dims))
    decoder = nn.Sequential(*layers)
    return decoder


@HEADS.register_module()
class TPVAggregator(BaseModule):
    def __init__(
            self, tpv_h, tpv_w, tpv_z, nbr_classes=20,
            in_dims=64, hidden_dims=128, out_dims=None,
            scale_h=2, scale_w=2, scale_z=2, use_checkpoint=True,
            feature_dim=None, hidden_dims_ft=512, out_dims_ft=None,
            dec_layers_occupancy=2, dec_layers_features=2,
            feature_dim_dino=None, hidden_dims_ft_dino=512, out_dims_ft_dino=None,
            dec_layers_features_dino=2,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.feature_dim = feature_dim
        self.feature_dim_dino = feature_dim_dino

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = build_decoder(dec_layers_occupancy, in_dims, hidden_dims, out_dims)
        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        # feature learning branch
        self.decoder_ft = None
        if self.feature_dim is not None:
            out_dims_ft = in_dims if out_dims_ft is None else out_dims_ft
            self.decoder_ft = build_decoder(dec_layers_features, in_dims, hidden_dims_ft, out_dims_ft)
            self.classifier_ft = nn.Linear(out_dims_ft, self.feature_dim)

        self.decoder_ft_dino = None
        if self.feature_dim_dino is not None:
            out_dims_ft_dino = in_dims if out_dims_ft_dino is None else out_dims_ft_dino
            self.decoder_ft_dino = build_decoder(dec_layers_features_dino, in_dims, hidden_dims_ft_dino,
                                                 out_dims_ft_dino)
            self.classifier_ft_dino = nn.Linear(out_dims_ft_dino, self.feature_dim_dino)

    def forward(self, tpv_list, points_input=None, features=None, features_only=False, voxel_features=True):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """

        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear'
            )

        logits_vox_fts, logits_pts_fts = None, None
        logits_pts_fts_dino, logits_vox_fts_dino = None, None
        if features is not None:
            # features: bs, n, 3
            _, n, _ = features.shape
            features = features.reshape(bs, 1, n, 3).float()
            features[..., 0] = features[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
            features[..., 1] = features[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
            features[..., 2] = features[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1
            sample_loc = features[:, :, :, [0, 1]]
            tpv_hw_fts = F.grid_sample(tpv_hw, sample_loc).squeeze(2)  # bs, c, n
            sample_loc = features[:, :, :, [1, 2]]
            tpv_zh_fts = F.grid_sample(tpv_zh, sample_loc).squeeze(2)
            sample_loc = features[:, :, :, [2, 0]]
            tpv_wz_fts = F.grid_sample(tpv_wz, sample_loc).squeeze(2)

            tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)

            fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
            fused_pts = tpv_hw_fts + tpv_zh_fts + tpv_wz_fts
            if voxel_features:
                fused = torch.cat([fused_vox, fused_pts], dim=-1)  # bs, c, whz+n
                fused = fused.permute(0, 2, 1)
                logits_pts_fts, logits_vox_fts = self.forward_decoder_classifier(bs, fused, n, self.decoder_ft,
                                                                                 self.classifier_ft, self.feature_dim)
            else:
                fused = fused_pts
                fused = fused.permute(0, 2, 1)
                logits_pts_fts = self.forward_decoder_classifier_points_only(bs, fused, n,self.decoder_ft,
                                                                             self.classifier_ft, self.feature_dim)
                logits_vox_fts = None

            if self.decoder_ft_dino is not None:
                logits_pts_fts_dino = self.forward_decoder_classifier_points_only(bs, fused_pts.permute(0, 2, 1), n,
                                                                                  self.decoder_ft_dino,
                                                                                  self.classifier_ft_dino,
                                                                                  self.feature_dim_dino)

            if features_only:
                return logits_pts_fts, logits_pts_fts_dino

        if points_input is not None:
            points = points_input.detach().clone()
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3)
            points[..., 0] = points[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1
            sample_loc = points[:, :, :, [0, 1]]
            tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc).squeeze(2)  # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc).squeeze(2)

            tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)

            fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
            fused = torch.cat([fused_vox, fused_pts], dim=-1)  # bs, c, whz+n

            fused = fused.permute(0, 2, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)
            logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, self.scale_w * self.tpv_w,
                                                     self.scale_h * self.tpv_h, self.scale_z * self.tpv_z)
            logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)
            return logits_vox, logits_pts, logits_vox_fts, logits_pts_fts, logits_vox_fts_dino, logits_pts_fts_dino

        else:
            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)

            fused = tpv_hw + tpv_zh + tpv_wz
            fused = fused.permute(0, 2, 3, 4, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)

            return logits

    def forward_decoder_classifier(self, bs, fused, n, decoder, classifier, feature_dim):
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(classifier, fused)
        else:
            fused = decoder(fused)
            logits = classifier(fused)
        logits = logits.permute(0, 2, 1)
        logits_vox_fts = logits[:, :, :(-n)].reshape(bs, feature_dim, self.scale_w * self.tpv_w,
                                                     self.scale_h * self.tpv_h, self.scale_z * self.tpv_z)
        logits_pts_fts = logits[:, :, (-n):].reshape(bs, feature_dim, n).permute(0, 2, 1)
        return logits_pts_fts, logits_vox_fts

    def forward_decoder_classifier_points_only(self, bs, fused, n, decoder, classifier, feature_dim):
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(classifier, fused)
        else:
            fused = decoder(fused)
            logits = classifier(fused)
        logits = logits.permute(0, 2, 1)
        logits_pts_fts = logits.reshape(bs, feature_dim, n).permute(0, 2, 1)
        return logits_pts_fts
