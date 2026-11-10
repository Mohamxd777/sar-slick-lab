"""Scratch U-Net and DeepLabV3+ with a torchvision ResNet-18 encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def conv(cin, cout, kernel=3, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=(kernel // 2) * dilation,
                  dilation=dilation, groups=groups, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )


def double_conv(cin, cout):
    return nn.Sequential(conv(cin, cout), conv(cout, cout))


def upsample(tensor, shape):
    return F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (24, 48, 96, 192)
        self.down = nn.ModuleList(double_conv(a, b) for a, b in zip((3, *widths), widths))
        self.pool = nn.MaxPool2d(2)
        self.middle = double_conv(192, 384)
        self.up = nn.ModuleList(double_conv(a + b, b) for a, b in zip((384, 192, 96, 48), reversed(widths)))
        self.head = nn.Conv2d(24, 5, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.middle(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = block(torch.cat([upsample(x, skip.shape[-2:]), skip], dim=1))
        return self.head(x)


def separable(cin, cout, dilation=1):
    return nn.Sequential(conv(cin, cin, dilation=dilation, groups=cin), conv(cin, cout, 1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        del self.encoder.fc, self.encoder.avgpool
        # Keep layer4 on the layer3 grid and enlarge its receptive field.
        layer4 = self.encoder.layer4
        layer4[0].conv1.stride = (1, 1)
        layer4[0].downsample[0].stride = (1, 1)
        for layer in (layer4[0].conv2, layer4[1].conv1, layer4[1].conv2):
            layer.dilation = layer.padding = (2, 2)
        self.aspp = nn.ModuleList([conv(512, 96, 1), *(separable(512, 96, rate) for rate in (3, 6, 9))])
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv(512, 96, 1))
        self.project = nn.Sequential(conv(480, 96, 1), nn.Dropout(.1))
        self.low = conv(64, 32, 1)
        self.decoder = nn.Sequential(separable(128, 96), separable(96, 96), nn.Conv2d(96, 5, 1))

    def forward(self, x):
        encoder = self.encoder
        stem = encoder.relu(encoder.bn1(encoder.conv1(x)))
        low = encoder.layer1(encoder.maxpool(stem))
        high = encoder.layer4(encoder.layer3(encoder.layer2(low)))
        branches = [branch(high) for branch in self.aspp]
        branches.append(upsample(self.global_context(high), high.shape[-2:]))
        context = upsample(self.project(torch.cat(branches, dim=1)), low.shape[-2:])
        return upsample(self.decoder(torch.cat([context, self.low(low)], dim=1)), x.shape[-2:])
