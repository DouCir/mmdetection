"""
Author:Yuan Yuan
Date: 2019/03/07
Description: This file defines a deep auto-encoder which
             uses 2d-convolution to encode and uses
             2d-transpose convolution to decode.
             Work flows:
             RGB image    --->4x[conv]--->--->transpose convolution--->reconstructed RGB image

             Thermal image--->4x[conv]--->--->transpose convolution--->reconstructed Thermal image
"""

from ..models import VGG
import torch.nn as nn
import torch
import numpy as np
from mmcv.cnn import kaiming_init
import torch.nn.functional as F


class SingleKaistAutoEncoder(nn.Module):
    def __init__(self):
        super(SingleKaistAutoEncoder, self).__init__()
        # encoder
        self.conv1_rgb = nn.Conv2d(3, 32, 5, 4, 2)
        self.conv2_rgb = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_rgb = nn.Conv2d(64, 256, 3, 2, 1)
        # self.conv4_rgb = nn.Conv2d(64, 128, 3, 2, 1)
        # decoder
        self.de_conv1_rgb = nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_rgb = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv3_rgb = nn.ConvTranspose2d(32, 3, kernel_size=(6, 6), stride=(4, 4), padding=(1, 1))
        # self.de_conv4_rgb = nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

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

    def init_weights(self):
        # encoder
        kaiming_init(self.conv1_rgb)
        kaiming_init(self.conv2_rgb)
        kaiming_init(self.conv3_rgb)
        # kaiming_init(self.conv4_rgb)
        # decoder
        self.de_conv1_rgb.weight.data = self.bilinear_kernel(self.de_conv1_rgb.in_channels,
                                                             self.de_conv1_rgb.out_channels, 4)
        self.de_conv2_rgb.weight.data = self.bilinear_kernel(self.de_conv2_rgb.in_channels,
                                                             self.de_conv2_rgb.out_channels, 4)
        self.de_conv3_rgb.weight.data = self.bilinear_kernel(self.de_conv3_rgb.in_channels,
                                                             self.de_conv3_rgb.out_channels, 6)
        # self.de_conv4_rgb.weight.data = self.bilinear_kernel(self.de_conv4_rgb.in_channels,
        #                                                      self.de_conv4_rgb.out_channels, 4)

    def forward(self, img_rgb):
        # encode
        rgb_down_2x = F.relu(self.conv1_rgb(img_rgb), True)
        rgb_down_4x = F.relu(self.conv2_rgb(rgb_down_2x), True)
        # rgb_down_8x = F.relu(self.conv3_rgb(rgb_down_4x), True)
        code = F.relu(self.conv3_rgb(rgb_down_4x), True)
        # decode
        rgb_up_2x = F.relu(self.de_conv1_rgb(code), True)
        rgb_up_4x = F.relu(self.de_conv2_rgb(rgb_up_2x), True)
        # rgb_up_8x = F.relu(self.de_conv3_rgb(rgb_up_4x), True)
        decode = torch.tanh(self.de_conv3_rgb(rgb_up_4x))
        return code, decode
