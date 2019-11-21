import torch
import torch.nn as nn
import torch.nn.functional as F


class FastNormalizedFusion(nn.Module):

    def __init__(self, in_nodes):
        super().__init__()
        self.in_nodes = in_nodes
        self.weight = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(0.0001))

    def forward(self, *x):
        if len(x) != self.in_nodes:
            raise RuntimeError(
                "Expected to have {} input nodes, but have {}.".format(self.in_nodes, len(x))
            )

        # where wi ≥ 0 is ensured by applying a relu after each wi (paper)
        weight = F.relu(self.weight)
        weighted_xs = [xi * wi for xi, wi in zip(x, weight)]
        normalized_weighted_x = sum(weighted_xs) / (weight.sum() + self.eps)
        return normalized_weighted_x


class DepthwiseSeparableConv2d(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class Conv3x3BnReLU(nn.Sequential):

    def __init__(self, in_channels, stride=1):
        conv = DepthwiseSeparableConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            bias=False,
            padding=1,
            stride=stride,
        )
        bn = nn.BatchNorm2d(in_channels, momentum=0.03)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class BiFPN(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.p4_tr = Conv3x3BnReLU(in_channels)
        self.p3_tr = Conv3x3BnReLU(in_channels)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.fuse_p4_tr = FastNormalizedFusion(in_nodes=2)
        self.fuse_p3_tr = FastNormalizedFusion(in_nodes=2)

        self.down_p2 = Conv3x3BnReLU(in_channels, stride=2)
        self.down_p3 = Conv3x3BnReLU(in_channels, stride=2)
        self.down_p4 = Conv3x3BnReLU(in_channels, stride=2)

        self.fuse_p5_out = FastNormalizedFusion(in_nodes=2)
        self.fuse_p4_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p3_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p2_out = FastNormalizedFusion(in_nodes=2)

        self.p5_out = Conv3x3BnReLU(in_channels)
        self.p4_out = Conv3x3BnReLU(in_channels)
        self.p3_out = Conv3x3BnReLU(in_channels)
        self.p2_out = Conv3x3BnReLU(in_channels)

    def forward(self, *ps):
        p2, p3, p4, p5 = ps

        p4_tr = self.p4_tr(self.fuse_p4_tr(p4, self.up(p5)))
        p3_tr = self.p3_tr(self.fuse_p3_tr(p3, self.up(p4_tr)))

        p2_out = self.p2_out(self.fuse_p2_out(p2, self.up(p3_tr)))
        p3_out = self.p3_out(self.fuse_p3_out(p3, p3_tr, self.down_p2(p2_out)))
        p4_out = self.p4_out(self.fuse_p4_out(p4, p4_tr, self.down_p3(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5, self.down_p4(p4_out)))

        return p2_out, p3_out, p4_out, p5_out


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class BiFPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[encoder_depth - 3:encoder_depth + 1]

        self.p_convs = nn.ModuleList([nn.Conv2d(in_ch, pyramid_channels, 1) for in_ch in encoder_channels])
        self.bifpn = BiFPN(pyramid_channels)

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in range(4)
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        cs = features[-4:]
        ps = [p_conv(c) for p_conv, c in zip(self.p_convs, cs)]
        ps = self.bifpn(*ps)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, ps)]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x
