# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn
from torchinfo import summary

sys.path.append(os.getcwd())
from module.ds_complex_conv2d import DSComplexConv2d
from module.ds_complex_conv_transpose2d import DSComplexConvTranspose2d
from module.dc_crn import DCCRN


class DSDCCRN(DCCRN):
    def __init__(
        self,
        n_fft=512,
        rnn_layers=2,
        rnn_units=256,
        kernel_num=[2, 32, 64, 128, 256, 256, 256],
        kernel_size=5,
    ):
        super().__init__(
            n_fft=n_fft,
            rnn_layers=rnn_layers,
            rnn_units=rnn_units,
            kernel_num=kernel_num,
            kernel_size=kernel_size,
        )

        # encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    DSComplexConv2d(
                        self.kernel_num[i] // 2,
                        self.kernel_num[i + 1] // 2,
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                    ),
                    nn.BatchNorm2d(self.kernel_num[i + 1]),
                    nn.PReLU(),
                )
            )

        # decoder
        self.decoder = nn.ModuleList()
        for i in range(len(self.kernel_num) - 1, 0, -1):
            if i != 1:
                self.decoder.append(
                    nn.Sequential(
                        DSComplexConvTranspose2d(
                            self.kernel_num[i] * 2 // 2,
                            self.kernel_num[i - 1] // 2,
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        ),
                        nn.BatchNorm2d(self.kernel_num[i - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        DSComplexConvTranspose2d(
                            self.kernel_num[i] * 2 // 2,
                            self.kernel_num[i - 1] // 2,
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        )
                    )
                )


if __name__ == "__main__":
    print(f"Test DSDCCRN Module Start...")

    # get model
    model = DSDCCRN(
        n_fft=512,
        rnn_layers=2,
        rnn_units=256,
        kernel_num=[2, 32, 64, 128, 256, 256, 256],
        kernel_size=5,
    )
    # get inputs [B, F, T, 2]
    X = torch.randn([2, 257, 401, 2])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    mask = model(X)

    print(f"Test DSDCCRN Module End...")

    pass
