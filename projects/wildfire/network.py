import torch
from mmengine.model import BaseModel as MMBaseModel
from mmengine.registry import MODELS
from torch import nn

from projects.common.losses.lovasz_losses import lovasz_hinge
from projects.common.modules.ops.conv_blocks import Conv2dUpsample


class UNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = MODELS.build(encoder)
        encoder_chs = self.encoder.out_channels

        decoder["center_ch"] = encoder_chs[-1]
        decoder["skip_chs"] = encoder_chs[::-1][1:]
        self.decoder = MODELS.build(decoder)

        # head
        decoder_chs = decoder["decoder_chs"]
        # self.head4 = Conv2dUpsample(decoder_chs[-5], 1, kernel_size=3, upsampling=16)
        # self.head3 = Conv2dUpsample(decoder_chs[-4], 1, kernel_size=3, upsampling=8)
        # self.head2 = Conv2dUpsample(decoder_chs[-3], 1, kernel_size=3, upsampling=4)
        # self.head1 = Conv2dUpsample(decoder_chs[-2], 1, kernel_size=3, upsampling=2)
        self.head0 = Conv2dUpsample(decoder_chs[-1], 1, kernel_size=3, upsampling=1)

    def forward(self, inputs):
        x = inputs["img"]  # (B, C, H, W)

        # Normalize
        # x = (x - 0.5) / 0.25

        # Forward Pass
        encoded = self.encoder(x)
        decoder_out = self.decoder(encoded[-1], encoded[::-1][1:])

        # output_h4 = self.head4(decoder_out[-5])
        # output_h3 = self.head3(decoder_out[-4])
        # output_h2 = self.head2(decoder_out[-3])
        # output_h1 = self.head1(decoder_out[-2])
        output_h0 = self.head0(decoder_out[-1])

        return output_h0


class OurBaseModel(MMBaseModel):
    def __init__(self, unet):
        super().__init__()

        self.unet = MODELS.build(unet)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, mode, **inputs):
        y_pred = self.unet(inputs)  # (B, 1, H, W)
        y_gt = inputs["mask"]  # (B, 1, H, W)

        if mode == "loss":
            lovasz_loss = lovasz_hinge(y_pred, y_gt, use_elu=True)
            return {"lovasz_loss": lovasz_loss}
        elif mode == "predict":
            return torch.sigmoid(y_pred), y_gt


if __name__ == "__main__":
    from projects.common.modules.unet.decoder import UNetDecoder
    from projects.common.modules.unet.encoders.regnet import RegNetEncoder

    net = UNet(
        encoder=dict(type=RegNetEncoder, name="regnetx_002", in_ch=10, empty_out_depths=[0]),
        decoder=dict(
            type=UNetDecoder,
            decoder_chs=[128, 64, 48, 32, 24],
            upsample_mode="nearest",
            use_batchnorm=True,
        ),
    )

    x = torch.rand([4, 10, 224, 224], dtype=torch.float32)
    y = net({"img": x})
    print(y)
    print(y.shape)
