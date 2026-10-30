import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.ds1 = DownsamplingBlock(in_channels=1, out_channels=16)
        self.ds2 = DownsamplingBlock(in_channels=16, out_channels=32)
        self.bottleneck = BottleneckBlock(in_channels=32, out_channels=64)
        self.us1 = UpsamplingBlock(in_channels=64, out_channels=32)
        self.us2 = UpsamplingBlock(in_channels=32, out_channels=16)
        self.output_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out, skip1 = self.ds1(x)
        out, skip2 = self.ds2(out)
        out = self.bottleneck(out)
        out = self.us1(out, skip2)
        out = self.us2(out, skip1)
        out = self.output_conv(out)
        out = self.sigmoid(out)

        return out


class DownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        skip = self.relu(self.conv(x))
        out = self.pool(skip)

        return out, skip


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.conv(x)
        out = self.relu(out)

        return out


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        # Takes the upconv output concatenated with an equally sized skip connection
        self.conv = nn.Conv2d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()

    def forward(self, x, skip):

        out = self.upconv(x)
        out = torch.concat((out, skip), dim=1)
        out = self.conv(out)
        out = self.relu(out)

        return out
