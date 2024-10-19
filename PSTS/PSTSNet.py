from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv3 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(self.conv3(self.relu(x1)))
        x_img = x_img * x2
        return x_img


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class Encoder(nn.Module):
    def __init__(self, in_channels, base_c, factor, block_dims):
        super(Encoder, self).__init__()
        self.down1 = Down(block_dims[0], block_dims[1])
        self.down2 = Down(block_dims[1], block_dims[2])
        self.down3 = Down(block_dims[2], block_dims[2])

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, base_c, factor, bilinear, block_dims):
        super(Decoder, self).__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True))
        self.up1 = Up(block_dims[2] * 2, block_dims[2] // 2, bilinear)
        self.up2 = Up(block_dims[2], block_dims[1] // 2, bilinear)
        self.up3 = Up(block_dims[0] + base_c, base_c, bilinear)

    def forward(self, x1, x2, x3, x4):
        out1 = self.conv1x1(x4)
        out2 = self.up1(out1, x3)
        out3 = self.up2(out2, x2)
        out4 = self.up3(out3, x1)
        return out1, out2, out3, out4


class PSTS_Pure(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(PSTS_Pure, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        block_dims1 = [64, 128, 256]

        # stage1
        self.encoder1 = Encoder(in_channels, base_c, factor, block_dims1)
        self.decoder1 = Decoder(base_c, factor, bilinear, block_dims1)
        self.out_conv1 = OutConv(base_c, 11)

        self.sam = SAM(base_c, 3, False)

        # stage 2
        self.encoder2 = Encoder(in_channels, base_c, factor, block_dims1)
        self.decoder2 = Decoder(base_c, factor, bilinear, block_dims1)
        self.out_conv2 = OutConv(base_c, out_channels)

        self.fuature_fusion1 = nn.Conv2d(256, 256, kernel_size=1)
        self.fuature_fusion2 = nn.Conv2d(128, 256, kernel_size=1)
        self.fuature_fusion3 = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.in_conv(img)
        e11, e12, e13 = self.encoder1(x)
        d11, d12, d13, d14 = self.decoder1(x, e11, e12, e13)
        sam1 = self.sam(d14, x)
        out1 = self.out_conv1(d14)

        e21, e22, e23 = self.encoder2(sam1)
        x1 = e21 + self.fuature_fusion3(d13)
        x2 = e22 + self.fuature_fusion2(d12)
        x3 = e23 + self.fuature_fusion1(d11)
        d21, d22, d23, d24 = self.decoder2(sam1, x1, x2, x3)
        out2 = self.out_conv2(d24)
        return out1, out2



class PSTS_Blended(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(PSTS_Blended, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        block_dims1 = [64, 128, 256]

        # stage1
        self.encoder1 = Encoder(in_channels, base_c, factor, block_dims1)
        self.decoder1 = Decoder(base_c, factor, bilinear, block_dims1)
        self.out_conv1 = OutConv(base_c, 7)

        self.sam = SAM(base_c, 3, False)

        # stage 2
        self.encoder2 = Encoder(in_channels, base_c, factor, block_dims1)
        self.decoder2 = Decoder(base_c, factor, bilinear, block_dims1)
        self.out_conv2 = OutConv(base_c, out_channels)

        self.fuature_fusion1 = nn.Conv2d(256, 256, kernel_size=1)
        self.fuature_fusion2 = nn.Conv2d(128, 256, kernel_size=1)
        self.fuature_fusion3 = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.in_conv(img)
        e11, e12, e13 = self.encoder1(x)
        d11, d12, d13, d14 = self.decoder1(x, e11, e12, e13)
        sam1 = self.sam(d14, x)
        out1 = self.out_conv1(d14)

        e21, e22, e23 = self.encoder2(sam1)
        x1 = e21 + self.fuature_fusion3(d13)
        x2 = e22 + self.fuature_fusion2(d12)
        x3 = e23 + self.fuature_fusion1(d11)
        d21, d22, d23, d24 = self.decoder2(sam1, x1, x2, x3)
        out2 = self.out_conv2(d24)
        return out1, out2