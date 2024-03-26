import torch
from torch import nn

from projects.common.modules.ops.conv_blocks import Conv2dReLU


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        upsample_mode: str,
        use_batchnorm=True,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

        self.conv1 = Conv2dReLU(
            in_ch + skip_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        center_ch: int,
        skip_chs: list[int],
        decoder_chs: list[int],
        upsample_mode: str,
        use_batchnorm=True,
        use_center_block=False,
    ):
        super().__init__()

        if use_center_block:
            self.center = CenterBlock(center_ch, center_ch, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        in_chs = [center_ch] + decoder_chs[:-1]
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, upsample_mode, use_batchnorm=use_batchnorm)
            for in_ch, skip_ch, out_ch in zip(in_chs, skip_chs, decoder_chs, strict=True)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, center_feat, skip_feats):

        x = self.center(center_feat)
        xs = []
        for decoder_block, skip_feat in zip(self.blocks, skip_feats, strict=True):
            x = decoder_block(x, skip_feat)
            xs.append(x)

        return xs
