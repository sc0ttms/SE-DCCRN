# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchinfo import summary


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 2),
        stride=(2, 1),
        padding=(2, 0),
        output_padding=(1, 0),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.conv_transpose2d_real = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        self.conv_transpose2d_imag = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=0.05)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input):
        # input [B, C, F, T]
        # get real, imag
        # [B, C // 2, F, T]
        real, imag = torch.chunk(input, 2, axis=1)

        # (Xr*Wr-Xi*Wi) + j(Xr*Wi+Xi*Wr)
        rr = self.conv_transpose2d_real(real)
        ii = self.conv_transpose2d_imag(imag)
        ri = self.conv_transpose2d_imag(real)
        ir = self.conv_transpose2d_real(imag)
        real = rr + (-ii)
        imag = ri + ir

        # -> [B, C, F, T]
        return torch.cat([real, imag], axis=1)


if __name__ == "__main__":
    print(f"Test ComplexConvTranspose2d Module Start...")

    # get model
    model = ComplexConvTranspose2d(2 // 2, 16 // 2)
    # get inputs [B, C, F, T]
    X = torch.randn([2, 2, 256, 401])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    Y_hat = model(X)

    print(f"Test ComplexConvTranspose2d Module End...")
