"""Baseline PyTorch models for deep beamforming."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ResizeConvDBF(nn.Module):
    """Simple resize-then-convolve baseline for DAS-IQ prediction."""

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        if len(output_size) != 2:
            raise ValueError("output_size must be a tuple/list of (H, W)")

        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, 2, C, T].
        """
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Expected input x with shape [B, 2, C, T]")

        x_resized = F.interpolate(
            x,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.net(x_resized)


class MemorizeImageDBF(nn.Module):
    """Directly learn one output image, ignoring the input."""

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        if len(output_size) != 2:
            raise ValueError("output_size must be a tuple/list of (H, W)")

        h = int(output_size[0])
        w = int(output_size[1])
        self.output_size = (h, w)
        self.y_param = nn.Parameter(torch.zeros(1, 2, h, w))

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input x with shape [B, 2, C, T]")
        return self.y_param.expand(x.shape[0], -1, -1, -1)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with LeakyReLU activations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReflectionConv2d3x3(nn.Module):
    """3x3 convolution using reflection padding instead of zero padding."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReflectionConvBlock(nn.Module):
    """Two reflection-padded 3x3 convolutions with LeakyReLU activations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ReflectionConv2d3x3(in_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ReflectionConv2d3x3(out_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordResizeUNetDBF(nn.Module):
    """Resize input to image grid, append coordinates, and apply a small U-Net."""

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        if len(output_size) != 2:
            raise ValueError("output_size must be a tuple/list of (H, W)")

        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.enc1 = ConvBlock(4, 32)
        self.enc2 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        self.out_conv = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def _coord_maps(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h, w = self.output_size
        z = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        lateral = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        z_coord = z.expand(b, 1, h, w)
        x_coord = lateral.expand(b, 1, h, w)
        return torch.cat([z_coord, x_coord], dim=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Expected input x with shape [B, 2, C, T]")

        x_resized = F.interpolate(
            x,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        if coords is None:
            coord_maps = self._coord_maps(x_resized)
        else:
            if coords.ndim != 4 or coords.shape[1] != 2:
                raise ValueError("Expected coords with shape [B, 2, H, W]")
            if tuple(coords.shape[-2:]) != self.output_size:
                raise ValueError(f"Expected coords spatial size {self.output_size}, got {tuple(coords.shape[-2:])}")
            if coords.shape[0] != x.shape[0]:
                raise ValueError("coords batch size must match x batch size")
            coord_maps = coords.to(device=x_resized.device, dtype=x_resized.dtype)
        input_img = torch.cat([x_resized, coord_maps], dim=1)

        skip1 = self.enc1(input_img)
        down1 = F.avg_pool2d(skip1, kernel_size=2, stride=2)
        skip2 = self.enc2(down1)
        down2 = F.avg_pool2d(skip2, kernel_size=2, stride=2)
        bottleneck = self.bottleneck(down2)

        up2 = F.interpolate(bottleneck, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = F.interpolate(dec2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))
        return self.out_conv(dec1)

class CoordResizeUNetDBFReflectionPad(CoordResizeUNetDBF):
    """CoordResizeUNetDBF variant with reflection padding in all 3x3 convolutions."""

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__(output_size=output_size)
        self.enc1 = ReflectionConvBlock(4, 32)
        self.enc2 = ReflectionConvBlock(32, 64)
        self.bottleneck = ReflectionConvBlock(64, 128)
        self.dec2 = ReflectionConvBlock(128 + 64, 64)
        self.dec1 = ReflectionConvBlock(64 + 32, 32)
        self.out_conv = ReflectionConv2d3x3(32, 2)

class CoordLocalGlobalUNetDBFReflectionPad(CoordResizeUNetDBFReflectionPad):
    """Reflection-padded U-Net using local and global physical coordinates.

    Expects coords with shape [B, 4, H, W] in the order
    [z_local, x_local, z_global, x_global]. The resized RF/IQ input has 2
    channels, so the first encoder block receives 6 channels total.
    """

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__(output_size=output_size)
        self.enc1 = ReflectionConvBlock(6, 32)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Expected input x with shape [B, 2, C, T]")
        if coords is None:
            raise ValueError("CoordLocalGlobalUNetDBFReflectionPad requires coords with shape [B, 4, H, W]")

        x_resized = F.interpolate(
            x,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        if coords.ndim != 4 or coords.shape[1] != 4:
            raise ValueError("Expected coords with shape [B, 4, H, W]")
        if tuple(coords.shape[-2:]) != self.output_size:
            raise ValueError(f"Expected coords spatial size {self.output_size}, got {tuple(coords.shape[-2:])}")
        if coords.shape[0] != x.shape[0]:
            raise ValueError("coords batch size must match x batch size")
        coord_maps = coords.to(device=x_resized.device, dtype=x_resized.dtype)
        input_img = torch.cat([x_resized, coord_maps], dim=1)

        skip1 = self.enc1(input_img)
        down1 = F.avg_pool2d(skip1, kernel_size=2, stride=2)
        skip2 = self.enc2(down1)
        down2 = F.avg_pool2d(skip2, kernel_size=2, stride=2)
        bottleneck = self.bottleneck(down2)

        up2 = F.interpolate(bottleneck, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = F.interpolate(dec2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))
        return self.out_conv(dec1)

