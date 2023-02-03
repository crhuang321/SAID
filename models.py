import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import (MLP, Bicubic, Lanczos, MeanShift, ResidualGroup,
                         default_conv, make_coord)


class SAID_Bicubic(nn.Module):
    def __init__(self, SAID_spec, Bicubic_spec=None, rgb_range=255):
        super().__init__()
        self.dsn = SAID(**SAID_spec, rgb_range=rgb_range)
        self.usn = Bicubic()
        self.quantization = Quantization(rgb_range)

    def forward(self, x, lr_size, gt_size):
        lr = self.dsn(x, lr_size)
        lr_quant = self.quantization(lr)
        sr = self.usn(lr_quant, gt_size)
        return lr, sr


class SAID_Lanczos(nn.Module):
    def __init__(self, SAID_spec, Lanczos_spec=None, rgb_range=255):
        super().__init__()
        self.dsn = SAID(**SAID_spec, rgb_range=rgb_range)
        self.usn = Lanczos()
        self.quantization = Quantization(rgb_range)

    def forward(self, x, lr_size, gt_size):
        lr = self.dsn(x, lr_size)
        lr_quant = self.quantization(lr)
        sr = self.usn(lr_quant, gt_size)
        return lr, sr


class Quantization(nn.Module):
    def __init__(self, rgb_range):
        super().__init__()
        self.rgb_range = rgb_range

    def forward(self, x):
        if self.training:
            x = x - (x - torch.clamp(x, 0, self.rgb_range)).detach()
            x = x * (255 / self.rgb_range)
            x = x - (x - torch.round(x)).detach()
        else:
            x = torch.clamp(x, 0, self.rgb_range)
            x = torch.round(x * (255 / self.rgb_range))

        x /= (255 / self.rgb_range)
        return x


class SAID(nn.Module):
    def __init__(self, n_feats, n_resgroups, n_resblocks, feat_enh, downsampler, rgb_range):
        super().__init__()
        self.sub_mean = MeanShift(rgb_range, (0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0))
        self.add_mean = MeanShift(rgb_range, (0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0), 1)
        self.ds_encoder = RCANplug(n_feats, n_resgroups, n_resblocks, feat_enh)
        self.downsampler = AFResampler(n_feats, **downsampler)

    def forward(self, x, size):
        x = self.sub_mean(x)
        x = self.ds_encoder(x, size)
        x = self.downsampler(x, size)
        x = self.add_mean(x)
        return x


class RCANplug(nn.Module):
    def __init__(
        self, n_feats, n_resgroups, n_resblocks, feat_enh=False,
        kernel_size=3, reduction=16, n_colors=3, res_scale=1, conv=default_conv):
        super().__init__()
        self.n_resgroups = n_resgroups
        if feat_enh:
            enh_blocks = [FeatEnhance(n_feats) for _ in range(self.n_resgroups)]
            self.feat_enh = nn.Sequential(*enh_blocks)
        else:
            self.feat_enh = None

        self.head = nn.Sequential(conv(n_colors, n_feats, kernel_size))

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=nn.ReLU(True), res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(conv(n_feats, n_feats, kernel_size))

    def forward(self, x, size):
        x = self.head(x)

        res = x
        for i in range(self.n_resgroups):
            res = self.body[i](res)
            if self.feat_enh is not None:
                res = self.feat_enh[i](res, size)
        res = self.body[-1](res)
        res += x

        x = self.tail(res)
        return x


class FeatEnhance(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.fc = nn.Linear(2, n_feats, bias=False)

    def forward(self, feat, size):
        rels = relcoordscale(feat.shape, size)
        rel_coord = rels[0].to(feat.device)
        rel_scale = rels[1].to(feat.device)

        res = self.conv1(feat)
        x = res
        x = torch.stack(torch.split(x, 2, dim=1), dim=1)
        x = torch.mul(x.permute(1, 0, 2, 3, 4), rel_coord.permute(0, 3, 1, 2)).permute(1, 0, 2, 3, 4).view(res.shape)
        x += self.fc(rel_scale.view(-1, 2)).view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
        x = self.conv2(x)
        res = res + x
        return res


def relcoordscale(feat_size, tar_size):
    """
    feat_size: [b, c, h, w]
    tar_size:  [h, w]
    rel_coord: [b, h, w, 2]
    rel_scale: [b, h, w, 2]
    """
    feat_coord = make_coord(feat_size[-2:], flatten=False)
    feat_coord = torch.stack([feat_coord] * feat_size[0], 0)

    tar_coord = make_coord(tar_size, flatten=False)
    tar_coord = torch.stack([tar_coord] * feat_size[0], 0)

    sample_tar_coord = F.grid_sample(
        tar_coord.permute(0, 3, 1, 2), feat_coord.flip(-1), mode='nearest', align_corners=False)
    sample_tar_coord = sample_tar_coord.permute(0, 2, 3, 1)

    rel_coord = feat_coord - sample_tar_coord
    rel_coord[:, :, :, 0] *= feat_size[-2]
    rel_coord[:, :, :, 1] *= feat_size[-1]

    rel_scale = torch.ones_like(rel_coord)
    rel_scale[:, :, :, 0] *= 2 / tar_size[0] * feat_size[-2]
    rel_scale[:, :, :, 1] *= 2 / tar_size[1] * feat_size[-1]

    return rel_coord, rel_scale


def normalize(x, range=[-1, 1]):
    # normalize to [0, 1]
    x = (x - x.min()) / (x.max() - x.min())
    if range==[-1, 1]:
        # normalize to [-1, 1]
        x = x * 2 - 1
    return x


class AFResampler(nn.Module):
    def __init__(self, n_feats, imnet_spec):
        super().__init__()
        self.imF = MLP(**imnet_spec)
        self.feat_fusion = nn.Sequential(
            nn.Conv2d(n_feats, 64, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1))
    
    def forward(self, feat, size):
        # (1) Scale Factor and Implicit Weight Generation Function
        B, C, in_H, in_W = feat.shape
        out_H, out_W = size
        scale_H, scale_W = in_H / out_H, in_W / out_W
        imF = self.imF

        # (2) Center Coord Mapping
        # Coordinates in Output Space
        coord_out = [torch.arange(0, out_H, 1).float(), torch.arange(0, out_W, 1).float()]
        # Mapping from Output to Input
        coord_in = [(coord_out[0] + 0.5) * scale_H - 0.5, (coord_out[1] + 0.5) * scale_W - 0.5]
        # Convert to 2D
        coord_in = torch.stack([
            torch.cat([coord_in[1].unsqueeze(0)] * out_H, dim=0),
            torch.cat([coord_in[0].unsqueeze(-1)] * out_W, dim=-1),
        ], dim=-1).unsqueeze(0)  # 1, out_H, out_W, 2
        coord_in = torch.cat([coord_in] * B, dim=0)  # B, out_H, out_W, 2

        # (3) Neighborhood Range Determinztion
        neighbor_H, neighbor_W = math.ceil(scale_H), math.ceil(scale_W)
        neighbor_H = neighbor_H + 1 if neighbor_H % 2 == 0 else neighbor_H
        neighbor_W = neighbor_W + 1 if neighbor_W % 2 == 0 else neighbor_W
        neighbor_H, neighbor_W = neighbor_H // 2, neighbor_W // 2

        # (4) Neighbors and Weights
        value = 0
        w_sum = 0
        for offset_H in range(-neighbor_H, neighbor_H + 1):
            for offset_W in range(-neighbor_W, neighbor_W + 1):
                # offset
                curr_offset = torch.cat([
                    torch.ones([B, out_H, out_W, 1]) * offset_W,
                    torch.ones([B, out_H, out_W, 1]) * offset_H,
                ], dim=-1)  # B, out_H, out_W, 2

                # calculate coord for sampling
                curr_coord = coord_in + curr_offset  # B, out_H, out_W, 2
                # normalize to [-1, 1]
                curr_coord[..., 0] = normalize(curr_coord[..., 0])
                curr_coord[..., 1] = normalize(curr_coord[..., 1])

                # sample a neighbor
                curr_F = F.grid_sample(
                    feat, curr_coord.to(feat.device), padding_mode='zeros', align_corners=True)

                # calculate weight
                w_param = torch.cat([
                    curr_offset,
                    torch.ones([B, out_H, out_W, 1]) * scale_H,
                    torch.ones([B, out_H, out_W, 1]) * scale_W
                ], dim=-1)
                weight = imF(w_param.to(feat.device)).view(B, 1, out_H, out_W)
                w_sum += weight

                # Weighting
                value += weight * curr_F

        # (5) Output Value
        out = value / w_sum

        # (6) Channel Feature Fusion
        out = self.feat_fusion(out)
        return out