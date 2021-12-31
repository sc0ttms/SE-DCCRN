# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchinfo import summary


class ComplexLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        projection_size=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_real = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.lstm_imag = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        if projection_size is not None:
            self.projection_size = projection_size
            self.fc_real = nn.Linear(self.hidden_size, self.projection_size)
            self.fc_imag = nn.Linear(self.hidden_size, self.projection_size)
        else:
            self.projection_size = None

    def forward(self, input):
        # [B, C, F, T]
        [batch_size, num_channels, num_freqs, num_frames] = input.shape

        # get real, imag
        # [B, C // 2, F, T]
        real, imag = torch.chunk(input, 2, axis=1)
        # [B, C // 2, F, T] -> [B, T, C // 2, F] -> [B, T, (C // 2) * F]
        real = real.permute(0, 3, 1, 2).reshape(batch_size, num_frames, (num_channels // 2) * num_freqs)
        imag = imag.permute(0, 3, 1, 2).reshape(batch_size, num_frames, (num_channels // 2) * num_freqs)

        # (Xr*Wr-Xi*Wi) + j(Xr*Wi+Xi*Wr)
        rr, _ = self.lstm_real(real)
        ii, _ = self.lstm_imag(imag)
        ri, _ = self.lstm_imag(real)
        ir, _ = self.lstm_real(imag)
        real = rr + (-ii)
        imag = ri + ir
        if self.projection_size is not None:
            real = self.fc_real(real)
            imag = self.fc_imag(imag)

        # [B, T, C // 2 * F] -> [B, T, C // 2, F] -> [B, C // 2, F, T]
        real = real.reshape(batch_size, num_frames, num_channels // 2, -1).permute(0, 2, 3, 1)
        imag = imag.reshape(batch_size, num_frames, num_channels // 2, -1).permute(0, 2, 3, 1)

        # -> [B, C, F, T]
        return torch.cat([real, imag], axis=1)


if __name__ == "__main__":
    print(f"Test ComplexLSTM Module Start...")

    # get model
    model = ComplexLSTM(256, 128)
    model.flatten_parameters()
    # get inputs [B, C, F, T]
    X = torch.randn([2, 2, 256, 401])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    Y_hat = model(X)

    # get model
    model = ComplexLSTM(256, 128, projection_size=128)
    model.flatten_parameters()
    # get inputs [B, C, F, T]
    X = torch.randn([2, 2, 256, 401])
    # print network
    summary(model, input_size=tuple(X.shape))
    # forward
    Y_hat = model(X)

    print(f"Test ComplexLSTM Module End...")
