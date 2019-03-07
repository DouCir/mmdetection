"""
Author:Yuan Yuan
Date: 2019/03/07
Description: This file defines a deep auto-encoder which
             uses 2d-convolution to encode and uses
             2d-transpose convolution to decode.
             Work flows:
             RGB image    --->VGG16--->\                \--->transpose convolution--->reconstructed RGB image
                                       \ concatenate ---
             Thermal image--->VGG16--->\                \--->transpose convolution--->reconstructed Thermal image
"""

from ..models import VGG
import torch.nn as nn
import torch
import numpy as np
from mmcv.cnn import kaiming_init


class MultiAutoEncoder(nn.Module):
    def __init__(self):
        super(MultiAutoEncoder).__init__()
        # encoder
        self.encoder_rgb = VGG(depth=16, with_last_pool=False)
        self.encoder_thermal = VGG(depth=16, with_last_pool=False)
        self.conv = nn.Conv2d(1024, 512, 1)
        # decoder
        self.de_conv1_rgb = nn.ConvTranspose2d(512, 256, kernel=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_rgb = nn.ConvTranspose2d(256, 128, kernel=(4, 4), stride=(2, 2), padding=(1, 3))
        self.de_conv3_rgb = nn.ConvTranspose2d(128, 64, kernel=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_rgb = nn.ConvTranspose2d(64, 3, kernel=(4, 4), stride=(2, 2), padding=(3, 1))
        self.de_conv1_thermal = nn.ConvTranspose2d(512, 256, kernel=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_thermal = nn.ConvTranspose2d(256, 128, kernel=(4, 4), stride=(2, 2), padding=(1, 3))
        self.de_conv3_thermal = nn.ConvTranspose2d(128, 64, kernel=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_thermal = nn.ConvTranspose2d(64, 3, kernel=(4, 4), stride=(2, 2), padding=(3, 1))

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

    def init_weights(self, pretrained=None):
        if not isinstance(pretrained, str):
            raise TypeError('Input must be a string')
        # encoder
        self.encoder_rgb.init_weights(pretrained)
        self.encoder_thermal.init_weights(pretrained)
        kaiming_init(self.conv)
        # decoder
        self.de_conv1_rgb.weight.data = self.bilinear_kernel(512, 256, 4)
        self.de_conv2_rgb.weight.data = self.bilinear_kernel(256, 128, 4)
        self.de_conv3_rgb.weight.data = self.bilinear_kernel(128, 64, 3)
        self.de_conv4_rgb.weight.data = self.bilinear_kernel(64, 3, 4)
        self.de_conv1_thermal.weight.data = self.bilinear_kernel(512, 256, 4)
        self.de_conv2_thermal.weight.data = self.bilinear_kernel(256, 128, 4)
        self.de_conv3_thermal.weight.data = self.bilinear_kernel(128, 64, 3)
        self.de_conv4_thermal.weight.data = self.bilinear_kernel(64, 3, 4)

    def forward(self, img_rgb, img_thermal):
        # encode
        feats_rgb = self.encoder_rgb(img_rgb)
        feats_thermal = self.encoder_thermal(img_thermal)
        code = self.conv(torch.cat((feats_rgb, feats_thermal), 1))
        # decode
        rgb_up_2x = nn.Relu(self.de_conv1_rgb(code), True)
        rgb_up_4x = nn.Relu(self.de_conv2_rgb(rgb_up_2x), True)
        rgb_up_8x = nn.Relu(self.de_conv3_rgb(rgb_up_4x), True)
        rgb_up_16x = nn.Relu(self.de_conv4_rgb(rgb_up_8x), True)
        th_up_2x = nn.Relu(self.de_conv1_thermal(code), True)
        th_up_4x = nn.Relu(self.de_conv2_thermal(th_up_2x), True)
        th_up_8x = nn.Relu(self.de_conv3_thermal(th_up_4x), True)
        th_up_16x = nn.Relu(self.de_conv4_thermal(th_up_8x), True)
        return code, rgb_up_16x, th_up_16x

    def loss(self, img_rgb, img_thermal, decode_rgb, decode_thermal):
        batch_size = img_rgb.shape[0]
        return (nn.MSELoss(img_rgb, decode_rgb, size_average=False) +
                nn.MSELoss(img_thermal, decode_thermal, size_average=False)) / batch_size
