# -*- coding: utf-8 -*-

import sys
import os
import torch
from torchinfo import summary

sys.path.append(os.getcwd())
from module.complex_conv_transpose2d import ComplexConvTranspose2d
from module.ds_conv_transpose2d import DSConvTranspose2d


class DSComplexConvTranspose2d(ComplexConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 2),
        stride=(2, 1),
        padding=(2, 0),
        output_padding=(1, 0),
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        self.conv_transpose2d_real = DSConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        self.conv_transpose2d_imag = DSConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )


if __name__ == "__main__":
    print(f"Test DSComplexConvTranspose2d Module Start...")

    # get model
    model = DSComplexConvTranspose2d(2 // 2, 16 // 2)
    # get inputs [B, C, F, T]
    X = torch.randn([2, 2, 256, 401])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    Y_hat = model(X)

    print(f"Test DSComplexConvTranspose2d Module End...")
