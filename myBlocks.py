"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from resnet_blocks import SEBottle2neck, SELayer

# sinc layer


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class My_Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False, conv1=[2, 3, 1, 1, 1, 1], conv2=[2, 3, 0, 1, 1, 3], conv3=[1, 3, 0, 1, 1, 3], pool=(1, 3)):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(conv1[0], conv1[1]),
                               padding=(conv1[2], conv1[3]),
                               stride=(conv1[4], conv1[5]))
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(conv2[0], conv2[1]),
                               padding=(conv2[2], conv2[3]),
                               stride=(conv2[4], conv2[5]))

        self.downsample = True
        self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                         out_channels=nb_filts[1],
                                         kernel_size=(conv3[0], conv3[1]),
                                         padding=(conv3[2], conv3[3]),
                                         stride=(conv3[4], conv3[5]))

        # self.mp = nn.MaxPool2d((1,4))
        self.mp = nn.MaxPool2d((pool[0], pool[1]))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x) 
            out = self.selu(out)  
        else:
            out = x
        out = self.conv1(x)

        out = self.bn2(out)  
        out = self.selu(out)  
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class My_SERes2Net_block(nn.Module):
    def __init__(self, nb_filts, first=False, conv1=[2, 3, 1, 1, 1, 1], conv2=[3, 3, 1, 1, 1, 3], conv3=[1, 3, 0, 1, 1, 3], pool=(1, 3), radix=2, groups=2):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.conv1 = SEBottle2neck(inplanes=nb_filts[0],
                                   planes=nb_filts[1], kernel_size=(conv1[0], conv1[1]))
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(conv2[0], conv2[1]),
                               padding=(conv2[2], conv2[3]),
                               stride=(conv2[4], conv2[5]))

        self.downsample = True
        self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                         out_channels=nb_filts[1],
                                         kernel_size=(conv3[0], conv3[1]),
                                         padding=(conv3[2], conv3[3]),
                                         stride=(conv3[4], conv3[5]))

        self.mp = nn.MaxPool2d((pool[0], pool[1]))

    def forward(self, x): 
        identity = x
        if not self.first:
            out = self.bn1(x)  
            out = self.selu(out)  
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out
