import timm
import torch
from torch import nn


class RegNetEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        in_ch: int,
        empty_out_depths: list[int],
    ):

        super().__init__()

        if name == "regnetx_002":
            self.model = timm.create_model("regnetx_002", pretrained=True)
        elif name == "regnetx_004":
            self.model = timm.create_model("regnetx_004", pretrained=True)
        elif name == "regnetx_006":
            self.model = timm.create_model("regnetx_006", pretrained=True)
        elif name == "regnetx_008":
            self.model = timm.create_model("regnetx_008", pretrained=True)
        else:
            raise ValueError(name)

        # Remove original fc layer
        del self.model.final_conv
        del self.model.head

        # conv0
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_ch * 2, 32, kernel_size=1, padding=0, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        # Patch first layer
        patch_ch = 32
        with torch.no_grad():
            orig_weight = self.model.stem.conv.weight.detach()

            new_conv = nn.Conv2d(patch_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)
            new_conv.weight[:] = (
                orig_weight.repeat(1, (patch_ch + 2) // 3, 1, 1)[:, :patch_ch] * 3 / patch_ch
            )
            self.model.stem.conv = new_conv

        # ETC
        self.empty_out_depths = empty_out_depths

    def _get_stages(self) -> list[nn.Module]:
        return [
            # nn.Identity(),
            nn.Sequential(self.conv0, self.model.stem),
            self.model.s1,
            self.model.s2,
            self.model.s3,
            self.model.s4,
        ]

    @property
    def out_channels(self) -> list[int]:
        channels = [
            # self.model.stem.conv.in_channels,
            self.model.stem.conv.out_channels,
            self.model.s1.b1.conv1.conv.out_channels,
            self.model.s2.b1.conv1.conv.out_channels,
            self.model.s3.b1.conv1.conv.out_channels,
            self.model.s4.b1.conv1.conv.out_channels,
        ]
        return [0 if d in self.empty_out_depths else ch for d, ch in enumerate(channels)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages = self._get_stages()

        features = []
        for depth, stage in enumerate(stages):
            x = stage(x)

            if depth in self.empty_out_depths:
                B, _, H, W = x.shape
                empty_tensor = torch.zeros((B, 0, H, W), dtype=x.dtype, device=x.device)
                features.append(empty_tensor)
            else:
                features.append(x)

        return features
