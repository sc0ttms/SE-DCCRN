# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

sys.path.append(os.getcwd())
from module.complex_conv2d import ComplexConv2d
from module.complex_lstm import ComplexLSTM
from module.complex_conv_transpose2d import ComplexConvTranspose2d
from audio.utils import flatten_parameters


class DCCRN(nn.Module):
    def __init__(
        self,
        n_fft=512,
        rnn_layers=2,
        rnn_units=256,
        kernel_num=[2, 32, 64, 128, 256, 256, 256],
        kernel_size=5,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        # encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
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

        # rnn
        rnn_input_size = self.n_fft // (2 ** (len(self.kernel_num))) * self.kernel_num[-1]
        rnns = []
        for i in range(self.rnn_layers):
            rnns.append(
                ComplexLSTM(
                    rnn_input_size // 2 if i == 0 else self.rnn_units // 2,
                    self.rnn_units // 2,
                    projection_size=rnn_input_size // 2 if i == self.rnn_layers - 1 else None,
                )
            )
        self.rnn = nn.Sequential(*rnns)

        # decoder
        self.decoder = nn.ModuleList()
        for i in range(len(self.kernel_num) - 1, 0, -1):
            if i != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
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
                        ComplexConvTranspose2d(
                            self.kernel_num[i] * 2 // 2,
                            self.kernel_num[i - 1] // 2,
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        )
                    )
                )

        # flatten_parameters
        self.apply(flatten_parameters)

    @staticmethod
    def skip_connect(decoder_in, encoder_out):
        decoder_in_real, decoder_in_imag = torch.chunk(decoder_in, 2, axis=1)
        encoder_out_real, encoder_out_imag = torch.chunk(encoder_out, 2, axis=1)
        out_real = torch.cat([decoder_in_real, encoder_out_real], axis=1)
        out_imag = torch.cat([decoder_in_imag, encoder_out_imag], axis=1)
        return torch.cat([out_real, out_imag], axis=1)

    def forward(self, noisy_spec):
        # input [B, F, T, 2]

        # [B, F, T, 2] -> [B, 2, F, T]
        noisy_spec = noisy_spec.permute(0, 3, 1, 2)

        # dropout first bin
        out = noisy_spec[:, :, 1:, :]

        # encoder
        encoder_out = []
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)

        # rnn
        out = self.rnn(out)

        # decoder
        for i, layer in enumerate(self.decoder):
            out = self.skip_connect(out, encoder_out[-1 - i])
            out = layer(out)[:, :, :, 1:]

        # mask [B, 2, F, T]
        mask = F.pad(out, [0, 0, 1, 0])

        return mask


if __name__ == "__main__":
    print(f"Test DCCRN Module Start...")

    # get model
    model = DCCRN(
        n_fft=512,
        rnn_layers=2,
        rnn_units=256,
        kernel_num=[2, 32, 64, 128, 256, 256, 256],
        kernel_size=5,
    )
    # get inputs [B, F, T, 2]
    X = torch.randn([1, 257, 401, 2])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    mask = model(X)

    print(f"Test DCCRN Module End...")

    pass
