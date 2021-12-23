# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 2),
        stride=(2, 1),
        padding=(2, 1),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv2d_real = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.padding[0], 0],
        )
        self.conv2d_imag = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.padding[0], 0],
        )

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.05)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input):
        # input [B, C, F, T]
        # pad [left, right, top, bottom]
        input = F.pad(input, [self.padding[1], 0, 0, 0])

        # get real, imag
        # [B, C // 2, F, T]
        real, imag = torch.chunk(input, 2, axis=1)

        # (Xr*Wr-Xi*Wi) + j(Xr*Wi+Xi*Wr)
        rr = self.conv2d_real(real)
        ii = self.conv2d_imag(imag)
        ri = self.conv2d_imag(real)
        ir = self.conv2d_real(imag)
        real = rr + (-ii)
        imag = ri + ir

        # -> [B, C, F, T]
        return torch.cat([real, imag], axis=1)


if __name__ == "__main__":
    print(f"Test ComplexConv2d Module Start...")

    # get model
    model = ComplexConv2d(2 // 2, 16 // 2)
    # get inputs [B, C, F, T]
    X = torch.randn([2, 2, 256, 401])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    Y_hat = model(X)

    print(f"Test ComplexConv2d Module End...")
