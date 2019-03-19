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

from ..models import VGG
import torch.nn as nn
import torch
import numpy as np
from mmcv.cnn import kaiming_init
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.conv1_rgb = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool1_rgb = nn.MaxPool2d(2, 2)
        self.conv2_rgb = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2_rgb = nn.MaxPool2d(2, 2)
        self.conv3_rgb = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3_rgb = nn.MaxPool2d(2, 2)
        self.conv4_rgb = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool4_rgb = nn.MaxPool2d(2, 2)
        self.conv1_thermal = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool1_thermal = nn.MaxPool2d(2, 2)
        self.conv2_thermal = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2_thermal = nn.MaxPool2d(2, 2)
        self.conv3_thermal = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3_thermal = nn.MaxPool2d(2, 2)
        self.conv4_thermal = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool4_thermal = nn.MaxPool2d(2, 2)

        self.conv_cat = nn.Conv2d(256, 128, 1)
        # decoder
        self.de_conv1_rgb = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_rgb = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv3_rgb = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_rgb = nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv1_thermal = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_thermal = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv3_thermal = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_thermal = nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.criterion = nn.MSELoss(size_average=True, reduce=True)

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
        kaiming_init(self.conv4_rgb)
        kaiming_init(self.conv1_thermal)
        kaiming_init(self.conv2_thermal)
        kaiming_init(self.conv3_thermal)
        kaiming_init(self.conv4_thermal)
        kaiming_init(self.conv_cat)
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

    def forward(self, img_rgb, img_thermal):
        # encode
        rgb_down_2x = self.pool1_rgb(F.relu(self.conv1_rgb(img_rgb), True))
        rgb_down_4x = self.pool2_rgb(F.relu(self.conv2_rgb(rgb_down_2x), True))
        rgb_down_8x = self.pool3_rgb(F.relu(self.conv3_rgb(rgb_down_4x), True))
        rgb_down_16x = self.pool4_rgb(F.relu(self.conv4_rgb(rgb_down_8x), True))
        thermal_down_2x = self.pool1_thermal(F.relu(self.conv1_thermal(img_thermal), True))
        thermal_down_4x = self.pool2_thermal(F.relu(self.conv2_thermal(thermal_down_2x), True))
        thermal_down_8x = self.pool3_thermal(F.relu(self.conv3_thermal(thermal_down_4x), True))
        thermal_down_16x = self.pool4_thermal(F.relu(self.conv4_thermal(thermal_down_8x), True))
        code = self.conv_cat(torch.cat((rgb_down_16x, thermal_down_16x), 1))
        # decode
        rgb_up_2x = torch.tanh(F.relu(self.de_conv1_rgb(code), True))
        rgb_up_4x = F.relu(self.de_conv2_rgb(rgb_up_2x), True)
        rgb_up_8x = F.relu(self.de_conv3_rgb(rgb_up_4x), True)
        rgb_up_16x = torch.tanh(self.de_conv4_rgb(rgb_up_8x))
        th_up_2x = torch.tanh(F.relu(self.de_conv1_thermal(code), True))
        th_up_4x = F.relu(self.de_conv2_thermal(th_up_2x), True)
        th_up_8x = F.relu(self.de_conv3_thermal(th_up_4x), True)
        th_up_16x = torch.tanh(self.de_conv4_thermal(th_up_8x))
        # th_up_16x = F.relu(self.de_conv4_thermal(th_up_8x))
        return code, rgb_up_16x, th_up_16x

    def loss(self, img_rgb, img_thermal, decode_rgb, decode_thermal, batch_size):
        loss_rgb = self.criterion(img_rgb, decode_rgb)
        loss_thermal = self.criterion(img_thermal, decode_thermal)
        return (loss_rgb + loss_thermal) / batch_size
