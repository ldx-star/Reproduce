from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FENet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 bilinear: bool = True):
        super(FENet, self).__init__()
        base = 64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, base * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base * 4)
        self.conv4 = nn.Conv2d(base * 4, base * 8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(base * 8)
        self.conv5 = nn.Conv2d(base * 8, base * 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base * 16)

        self.conv6 = nn.Conv2d(base * 24, base * 8, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(base * 8)
        self.conv7 = nn.Conv2d(base * 12, base * 4, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(base * 4)
        self.conv8 = nn.Conv2d(base * 6, base * 2, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(base * 2)
        self.conv9 = nn.Conv2d(base * 3, base, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(base)

        self.last_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(self.maxpool(x1))))
        x3 = self.relu(self.bn3(self.conv3(self.maxpool(x2))))
        x4 = self.relu(self.bn4(self.conv4(self.maxpool(x3))))
        x5 = self.relu(self.bn5(self.conv5(self.maxpool(x4))))

        d4 = self.relu(self.bn6(self.conv6(torch.cat([x4, self.upsample(x5)], dim=1))))
        d3 = self.relu(self.bn7(self.conv7(torch.cat([x3, self.upsample(d4)], dim=1))))
        d2 = self.relu(self.bn8(self.conv8(torch.cat([x2, self.upsample(d3)], dim=1))))
        d1 = self.relu(self.bn9(self.conv9(torch.cat([x1, self.upsample(d2)], dim=1))))

        y = self.last_conv(d1)

        return y
