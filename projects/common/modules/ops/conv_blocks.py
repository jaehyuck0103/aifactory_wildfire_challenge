from torch import nn


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super().__init__(conv, bn, relu)


class Conv2dUpsample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, upsampling: int):
        conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = (
            nn.Upsample(scale_factor=upsampling, mode="bilinear")
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)
