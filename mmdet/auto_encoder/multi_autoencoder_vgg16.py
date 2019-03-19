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
import torch.nn.functional as F


class MultiAutoEncoder(nn.Module):
    def __init__(self):
        super(MultiAutoEncoder, self).__init__()
        # encoder
        self.encoder_rgb = VGG(depth=16, out_indices=(4,), frozen_stages=1, with_last_pool=False)
        self.encoder_thermal = VGG(depth=16, out_indices=(4,), frozen_stages=1, with_last_pool=False)
        self.conv = nn.Conv2d(1024, 128, 1)
        # decoder
        self.de_conv1_rgb = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_rgb = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv3_rgb = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_rgb = nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv1_thermal = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_thermal = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv3_thermal = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_thermal = nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.criterion = nn.MSELoss(size_average=False, reduce=False)

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
        # weight[range(in_channels), range(out_channels), :, :] = filt
        for i in range(in_channels):
            for j in range(out_channels):
                weight[i, j, :, :] = filt
        return torch.from_numpy(weight)

    def init_weights(self, pretrained=None):
        if not isinstance(pretrained, str):
            raise TypeError('Input must be a string')
        # encoder
        self.encoder_rgb.init_weights(pretrained)
        self.encoder_thermal.init_weights(pretrained)
        kaiming_init(self.conv)
        # decoder
        self.de_conv1_rgb.weight.data = self.bilinear_kernel(self.de_conv1_rgb.in_channels,
                                                             self.de_conv1_rgb.out_channels, 4)
        self.de_conv2_rgb.weight.data = self.bilinear_kernel(self.de_conv2_rgb.in_channels,
                                                             self.de_conv2_rgb.out_channels, 4)
        self.de_conv3_rgb.weight.data = self.bilinear_kernel(self.de_conv3_rgb.in_channels,
                                                             self.de_conv3_rgb.out_channels, 4)
        self.de_conv4_rgb.weight.data = self.bilinear_kernel(self.de_conv4_rgb.in_channels,
                                                             self.de_conv4_rgb.out_channels, 4)
        self.de_conv1_thermal.weight.data = self.bilinear_kernel(self.de_conv1_thermal.in_channels,
                                                                 self.de_conv1_thermal.out_channels, 4)
        self.de_conv2_thermal.weight.data = self.bilinear_kernel(self.de_conv2_thermal.in_channels,
                                                                 self.de_conv2_thermal.out_channels, 4)
        self.de_conv3_thermal.weight.data = self.bilinear_kernel(self.de_conv3_thermal.in_channels,
                                                                 self.de_conv3_thermal.out_channels, 4)
        self.de_conv4_thermal.weight.data = self.bilinear_kernel(self.de_conv4_thermal.in_channels,
                                                                 self.de_conv4_thermal.out_channels, 4)
        # kaiming_init(self.de_conv1_rgb)
        # kaiming_init(self.de_conv2_rgb)
        # kaiming_init(self.de_conv3_rgb)
        # kaiming_init(self.de_conv4_rgb)
        # kaiming_init(self.de_conv1_thermal)
        # kaiming_init(self.de_conv2_thermal)
        # kaiming_init(self.de_conv3_thermal)
        # kaiming_init(self.de_conv4_thermal)

    def forward(self, img_rgb, img_thermal):
        # encode
        feats_rgb = self.encoder_rgb(img_rgb)
        feats_thermal = self.encoder_thermal(img_thermal)
        code = self.conv(torch.cat((feats_rgb[0], feats_thermal[0]), 1))
        # decode
        rgb_up_2x = F.relu(self.de_conv1_rgb(code), True)
        rgb_up_4x = F.relu(self.de_conv2_rgb(rgb_up_2x), True)
        rgb_up_8x = F.relu(self.de_conv3_rgb(rgb_up_4x), True)
        rgb_up_16x = torch.tanh(self.de_conv4_rgb(rgb_up_8x))
        th_up_2x = F.relu(self.de_conv1_thermal(code), True)
        th_up_4x = F.relu(self.de_conv2_thermal(th_up_2x), True)
        th_up_8x = F.relu(self.de_conv3_thermal(th_up_4x), True)
        th_up_16x = torch.tanh(self.de_conv4_thermal(th_up_8x))
        return code, rgb_up_16x, th_up_16x

    def loss(self, img_rgb, img_thermal, decode_rgb, decode_thermal, batch_size):
        loss_rgb = self.criterion(img_rgb, decode_rgb)
        loss_thermal = self.criterion(img_thermal, decode_thermal)
        return (loss_rgb + loss_thermal) / batch_size
