"""
Author:Yuan Yuan
Date: 2019/03/07
Description: This file defines a deep auto-encoder which
             uses 2d-convolution to encode and uses
             2d-transpose convolution to decode.
             Work flows:
             RGB image    --->4x[conv,pool]--->\                \--->transpose convolution--->reconstructed RGB image
                                               \ concatenate ---
             Thermal image--->4x[conv,pool]--->\                \--->transpose convolution--->reconstructed Thermal image
"""

import torch.nn as nn
import torch
import numpy as np
from mmcv.cnn import kaiming_init
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.conv1_rgb = nn.Conv2d(3, 32, 5, 4, 2)
        self.bn1_rgb = nn.BatchNorm2d(32)
        self.conv2_rgb = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2_rgb = nn.BatchNorm2d(64)
        self.conv3_rgb = nn.Conv2d(64, 256, 3, 2, 1)
        self.bn3_rgb = nn.BatchNorm2d(256)
        self.conv1_thermal = nn.Conv2d(3, 32, 5, 4, 2)
        self.bn1_thermal = nn.BatchNorm2d(32)
        self.conv2_thermal = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2_thermal = nn.BatchNorm2d(64)
        self.conv3_thermal = nn.Conv2d(64, 256, 3, 2, 1)
        self.bn3_thermal = nn.BatchNorm2d(256)
        # decoder
        self.de_conv1_rgb = nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn4_rgb = nn.BatchNorm2d(64)
        self.de_conv2_rgb = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn5_rgb = nn.BatchNorm2d(32)
        self.de_conv3_rgb = nn.ConvTranspose2d(32, 3, kernel_size=(6, 6), stride=(4, 4), padding=(1, 1))
        self.de_conv1_thermal = nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn4_thermal = nn.BatchNorm2d(64)
        self.de_conv2_thermal = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn5_thermal = nn.BatchNorm2d(32)
        self.de_conv3_thermal = nn.ConvTranspose2d(32, 3, kernel_size=(6, 6), stride=(4, 4), padding=(1, 1))

        # concatenate feature maps
        self.conv_cat = nn.Conv2d(512, 256, 1)
        self.bn_cat = nn.BatchNorm2d(256)

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
        kaiming_init(self.conv1_thermal)
        kaiming_init(self.conv2_thermal)
        kaiming_init(self.conv3_thermal)
        kaiming_init(self.conv_cat)
        # decoder
        self.de_conv1_rgb.weight.data = self.bilinear_kernel(self.de_conv1_rgb.in_channels,
                                                             self.de_conv1_rgb.out_channels, 4)
        self.de_conv2_rgb.weight.data = self.bilinear_kernel(self.de_conv2_rgb.in_channels,
                                                             self.de_conv2_rgb.out_channels, 4)
        self.de_conv3_rgb.weight.data = self.bilinear_kernel(self.de_conv3_rgb.in_channels,
                                                             self.de_conv3_rgb.out_channels, 6)
        self.de_conv1_thermal.weight.data = self.bilinear_kernel(self.de_conv1_thermal.in_channels,
                                                                 self.de_conv1_thermal.out_channels, 4)
        self.de_conv2_thermal.weight.data = self.bilinear_kernel(self.de_conv2_thermal.in_channels,
                                                                 self.de_conv2_thermal.out_channels, 4)
        self.de_conv3_thermal.weight.data = self.bilinear_kernel(self.de_conv3_thermal.in_channels,
                                                                 self.de_conv3_thermal.out_channels, 6)

    def forward(self, img_rgb, img_thermal):
        # encode
        rgb_down_2x = F.relu(self.bn1_rgb(self.conv1_rgb(img_rgb)), True)
        rgb_down_4x = F.relu(self.bn2_rgb(self.conv2_rgb(rgb_down_2x)), True)
        rgb_down_16x = F.relu(self.bn3_rgb(self.conv3_rgb(rgb_down_4x)), True)
        thermal_down_2x = F.relu(self.bn1_thermal(self.conv1_thermal(img_thermal)), True)
        thermal_down_4x = F.relu(self.bn2_thermal(self.conv2_thermal(thermal_down_2x)), True)
        thermal_down_16x = F.relu(self.bn3_thermal(self.conv3_thermal(thermal_down_4x)), True)
        code = self.bn_cat(self.conv_cat(torch.cat((rgb_down_16x, thermal_down_16x), 1)))
        # decode
        rgb_up_2x = F.relu(self.bn4_rgb(self.de_conv1_rgb(code)), True)
        rgb_up_4x = F.relu(self.bn5_rgb(self.de_conv2_rgb(rgb_up_2x)), True)
        rgb_up_16x = torch.sigmoid(self.de_conv3_rgb(rgb_up_4x))
        th_up_2x = F.relu(self.bn4_thermal(self.de_conv1_thermal(code)), True)
        th_up_4x = F.relu(self.bn5_thermal(self.de_conv2_thermal(th_up_2x)), True)
        th_up_16x = torch.sigmoid(self.de_conv3_thermal(th_up_4x))
        return code, rgb_up_16x, th_up_16x

    def loss(self, img_rgb, img_thermal, decode_rgb, decode_thermal, batch_size):
        loss_rgb = self.criterion(img_rgb, decode_rgb)
        loss_thermal = self.criterion(img_thermal, decode_thermal)
        return (loss_rgb + loss_thermal) / batch_size
