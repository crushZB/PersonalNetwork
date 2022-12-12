import torch.nn as nn


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self._body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self._body(x)
        return x * y


class PALayer(nn.Module):
    def __init__(self, in_channels, reduction=16, bias=False) -> None:
        super(PALayer, self).__init__()
        self._body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1,
                      stride=1, padding=0, bias=bias),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1, stride=1,
                      padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self._body(x)
        return x * y
