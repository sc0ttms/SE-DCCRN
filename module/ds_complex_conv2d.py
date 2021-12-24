# -*- coding: utf-8 -*-

import sys
import os
import torch
from torchinfo import summary

sys.path.append(os.getcwd())
from module.complex_conv2d import ComplexConv2d
from module.ds_conv2d import DSConv2d


class DSComplexConv2d(ComplexConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv2d_real = DSConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.padding[0], 0],
        )
        self.conv2d_imag = DSConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.padding[0], 0],
        )


if __name__ == "__main__":
    print(f"Test DSComplexConv2d Module Start...")

    # get model
    model = DSComplexConv2d(2 // 2, 16 // 2)
    # get inputs [B, C, F, T]
    X = torch.randn([2, 2, 256, 401])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    Y_hat = model(X)

    print(f"Test DSComplexConv2d Module End...")
