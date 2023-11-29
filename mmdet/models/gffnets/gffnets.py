"""
    Pytorch Implementation of our GFFnet models based on Deeplabv3+ with WiderResNet-38 as backbone.
    Author: Xiangtai Li (lxtpku@pku.edu.cn)
"""

import logging

import torch
from torch import nn

from .nn.mynn import initialize_weights, Norm2d, Upsample
from .nn.operators import conv_bn_relu, conv_sigmoid, DenseBlock


class WiderResNetGFFDFPHead(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(WiderResNetGFFDFPHead, self).__init__()

        self.d_in1 = conv_bn_relu(256, 128, 1, norm_layer=norm_layer)
        self.d_in2 = conv_bn_relu(256, 128, 1, norm_layer=norm_layer)
        self.d_in3 = conv_bn_relu(256, 128, 1, norm_layer=norm_layer)

        self.gate1 = conv_sigmoid(256, 128)
        self.gate2 = conv_sigmoid(256, 128)
        self.gate3 = conv_sigmoid(256, 128)

        in_channel = 128
        self.dense_3 = DenseBlock(in_channel, 128, 128, 3, drop_out=0, norm_layer=norm_layer)
        self.dense_6 = DenseBlock(in_channel + 128, 128, 128, 6, drop_out=0, norm_layer=norm_layer)
        self.dense_9 = DenseBlock(in_channel + 128 * 2, 128, 128, 9, drop_out=0, norm_layer=norm_layer)

        self.downsample = conv_bn_relu(512, 256, 1, norm_layer=norm_layer)

        # self.cls = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
        #     norm_layer(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     norm_layer(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x):
        m2, m5, aspp = x
        m2_size = m2.size()[2:]
        m5_size = m5.size()[2:]
        aspp_size = aspp.size()[2:]

        g_m2 = self.gate1(m2)
        g_m5 = self.gate2(m5)
        g_aspp = self.gate3(aspp)

        m2 = self.d_in1(m2)
        m5 = self.d_in2(m5)
        aspp = self.d_in3(aspp)
        # GFF fusion
        m2 = m2 + g_m2 * m2 + (1 - g_m2) * ( Upsample(g_m5 * m5, size=m2_size) +   Upsample(g_aspp * aspp, size=m2_size) )
        m5 = m5 + g_m5 * m5 + (1 - g_m5) * (
                    Upsample(g_m2 *  m2, size=m5_size) +  Upsample(g_aspp *aspp, size=m5_size))
        aspp_f = aspp + aspp * g_aspp + (1 - g_aspp) * (
                     Upsample(g_m5 *m5, size=aspp_size) +  Upsample(g_m2 * m2, size=aspp_size))

        aspp_f = Upsample(aspp_f, size=m2_size)
        aspp = Upsample(aspp, size=m2_size)
        m5 = Upsample(m5, size=m2_size)

        # DFP fusion
        out = aspp_f
        aspp_f = self.dense_3(out)
        out = torch.cat([aspp_f,m5], dim=1)
        m5 = self.dense_6(out)
        out = torch.cat([aspp_f,m5,m2],dim=1)
        m2 = self.dense_9(out)

        f = torch.cat([aspp_f, aspp, m5, m2], dim=1)
        down_sampled_f = self.downsample(f)
        # out = self.cls(f)

        return down_sampled_f