from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from debugpy.launcher import channel


class FAB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(FAB, self).__init__()
        self.RReLu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(self.RReLu(self.bn1(x1)))
        x2 = self.conv3(self.RReLu(self.bn2(x1)))
        return x1 + x2


class PEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PEB, self).__init__()
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.upSample(x)

        x2 = self.conv1(x1)
        x2 = self.conv2(x1 + self.relu(self.BN(x2)))
        return x2


class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.Wd = nn.Linear(channels, channels // r)
        self.Wu = nn.Linear(channels // r, channels)
        self.f = nn.Sigmoid()
        self.Hgp = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.Hgp(x).view(x.size(0), -1)
        x1 = self.Wd(x1)
        x1 = self.Wu(x1)
        x1 = self.f(x1)
        return x1.view(x.size(0),x.size(1), 1,1) * x


class HPIU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HPIU, self).__init__()
        self.fab1 = FAB(in_channels, in_channels)
        self.fab2 = FAB(in_channels, out_channels)
        self.fab3 = FAB(2*out_channels + 2 * in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        f1 = self.fab1(x)
        f2 = self.fab2(f1)
        hlf1 = self.relu(torch.cat([f1, x], dim=1))  # [in_channel,in_channel]
        hlf2 = self.relu(torch.cat([f2, hlf1], dim=1))  # [out_channel,2*in_channel]
        f3 = self.fab3(torch.cat([f2, hlf2], dim=1))  # [2*out_channel+2*in_channel,out_channel]
        return f3 + f2


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class SAM(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(SAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class EAU(nn.Module):
    def __init__(self, channels):
        super(EAU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,padding=1)
        self.cam = CAM(channels)
        self.sam = SAM(channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        cam = self.cam(x1)
        sam = self.sam(x2)
        return cam + sam


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, channels):
        super(Encoder, self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hpiu1 = HPIU(in_channels, channels[0])
        self.hpiu2 = HPIU(channels[0], channels[1])
        self.hpiu3 = HPIU(channels[1], channels[2])
        self.hpiu4 = HPIU(channels[2], channels[3])
        self.hpiu5 = HPIU(channels[3], out_channels)

    def forward(self, x):
        x1 = self.hpiu1(x)
        x2 = self.hpiu2(self.maxPool(x1))
        x3 = self.hpiu3(self.maxPool(x2))
        x4 = self.hpiu4(self.maxPool(x3))
        x5 = self.hpiu5(self.maxPool(x4))
        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels,channels):
        super(Decoder, self).__init__()
        self.hpiu1 = HPIU(in_channels, channels[3])
        self.peb1 = PEB(in_channels, channels[3])
        self.eau1 = EAU(channels[3])

        self.hpiu2 = HPIU(channels[3], channels[2])
        self.peb2 = PEB(channels[3], channels[2])
        self.eau2 = EAU(channels[2])

        self.hpiu3 = HPIU(channels[2], channels[1])
        self.peb3 = PEB(channels[2], channels[1])
        self.eau3 = EAU(channels[1])

        self.hpiu4 = HPIU(channels[1], channels[0])
        self.peb4 = PEB(channels[1], channels[0])
        self.eau4 = EAU(channels[0])

        self.hpiu5 = HPIU(channels[0], out_channels)

    def forward(self, x1, x2, x3, x4, x5):
        d = self.hpiu1(torch.cat([x4, self.eau1(self.peb1(x5))], dim=1))
        d = self.hpiu2(torch.cat([x3, self.eau2(self.peb2(d))], dim=1))
        d = self.hpiu3(torch.cat([x2, self.eau3(self.peb3(d))], dim=1))
        d = self.hpiu4(torch.cat([x1, self.eau4(self.peb4(d))], dim=1))
        return d


class Pin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pin, self).__init__()
        midChannels = 1024
        channels = [64, 128, 256, 512]
        self.encoder = Encoder(in_channels, midChannels, channels)
        self.decoder = Decoder(midChannels, out_channels, channels)
        self.last_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        d5 = self.decoder(x1, x2, x3, x4, x5)
        return self.last_conv(d5)


if __name__ == '__main__':
    model = Pin(1, 2)
    inputs = torch.randn(3, 1, 224, 224)
    output = model(inputs)
