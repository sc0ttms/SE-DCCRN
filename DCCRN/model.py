# -*- coding: utf-8 -*-

import sys
import os
import toml
from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


sys.path.append(os.getcwd())
from audio.metrics import SI_SDR


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
        real = rr - ii
        imag = ri + ir

        # -> [B, C, F, T]
        return torch.cat([real, imag], axis=1)


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

    def flatten_parameters(self):
        self.lstm_real.flatten_parameters()
        self.lstm_imag.flatten_parameters()

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
        real = rr - ii
        imag = ri + ir
        if self.projection_size is not None:
            real = self.fc_real(real)
            imag = self.fc_imag(imag)

        # [B, T, C // 2 * F] -> [B, T, C // 2, F] -> [B, C // 2, F, T]
        real = real.reshape(batch_size, num_frames, num_channels // 2, -1).permute(0, 2, 3, 1)
        imag = imag.reshape(batch_size, num_frames, num_channels // 2, -1).permute(0, 2, 3, 1)

        # -> [B, C, F, T]
        return torch.cat([real, imag], axis=1)


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
        real = rr - ii
        imag = ri + ir

        # -> [B, C, F, T]
        return torch.cat([real, imag], axis=1)


class DCCRN(nn.Module):
    def __init__(self, config, mode="train", device="cpu"):
        super().__init__()

        # set mode
        self.mode = mode

        # set device
        self.device = device

        # get dataset args
        self.n_fft = config["dataset"]["n_fft"]
        self.win_len = config["dataset"]["win_len"]
        self.hop_len = config["dataset"]["hop_len"]
        self.audio_len = config["dataset"]["audio_len"]
        self.window = torch.hann_window(self.win_len, periodic=False, device=self.device)

        # get model args
        self.encoder_decoder_num_channels = config["model"]["encoder_decoder_num_channels"]
        self.encoder_decoder_kernel_size = config["model"]["encoder_decoder_kernel_size"]
        self.rnn_hidden_size = config["model"]["rnn_hidden_size"]
        self.rnn_num_layers = config["model"]["rnn_num_layers"]
        self.look_ahead = config["model"]["look_ahead"]

        # encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.encoder_decoder_num_channels) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.encoder_decoder_num_channels[i] // 2,
                        self.encoder_decoder_num_channels[i + 1] // 2,
                        kernel_size=(self.encoder_decoder_kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                    ),
                    nn.BatchNorm2d(self.encoder_decoder_num_channels[i + 1]),
                    nn.PReLU(),
                )
            )

        # rnn
        rnn_input_size = (
            self.n_fft // (2 ** (len(self.encoder_decoder_num_channels))) * self.encoder_decoder_num_channels[-1]
        )
        rnns = []
        for i in range(self.rnn_num_layers):
            rnns.append(
                ComplexLSTM(
                    rnn_input_size // 2 if i == 0 else self.rnn_hidden_size // 2,
                    self.rnn_hidden_size // 2,
                    projection_size=rnn_input_size // 2 if i == self.rnn_num_layers - 1 else None,
                )
            )
        self.rnn = nn.Sequential(*rnns)

        # decoder
        self.decoder = nn.ModuleList()
        for i in range(len(self.encoder_decoder_num_channels) - 1, 0, -1):
            if i != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.encoder_decoder_num_channels[i] * 2 // 2,
                            self.encoder_decoder_num_channels[i - 1] // 2,
                            kernel_size=(self.encoder_decoder_kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        ),
                        nn.BatchNorm2d(self.encoder_decoder_num_channels[i - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.encoder_decoder_num_channels[i] * 2 // 2,
                            self.encoder_decoder_num_channels[i - 1] // 2,
                            kernel_size=(self.encoder_decoder_kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        )
                    )
                )

        # set LSTM params
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.rnn, nn.LSTM):
            self.rnn.flatten_parameters()

    @staticmethod
    def skip_connect(decoder_in, encoder_out):
        decoder_in_real, decoder_in_imag = torch.chunk(decoder_in, 2, axis=1)
        encoder_out_real, encoder_out_imag = torch.chunk(encoder_out, 2, axis=1)
        out_real = torch.cat([decoder_in_real, encoder_out_real], axis=1)
        out_imag = torch.cat([decoder_in_imag, encoder_out_imag], axis=1)
        return torch.cat([out_real, out_imag], axis=1)

    @staticmethod
    def loss(enh, clean):
        return -(torch.mean(SI_SDR(enh, clean)))

    def print_networks(self, input_size=(2, 480000)):
        summary(self, input_size=input_size)

    def forward(self, noisy):
        # noisy [B, S]

        # [B, F, T, 2]
        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
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

        # mask
        mask = F.pad(out, [0, 0, 1, 0])
        enh_spec_real = mask[:, 0, :, :] * noisy_spec[:, 0, :, :] - mask[:, 1, :, :] * noisy_spec[:, 1, :, :]
        enh_spec_imag = mask[:, 1, :, :] * noisy_spec[:, 0, :, :] + mask[:, 0, :, :] * noisy_spec[:, 1, :, :]
        # [B, F, T] -> [B, F, T, 1]
        enh_spec_real = enh_spec_real.unsqueeze(dim=3)
        enh_spec_imag = enh_spec_imag.unsqueeze(dim=3)
        # [B, F, T, 1] -> [B, F, T, 2]
        enh_spec = torch.cat([enh_spec_real, enh_spec_imag], dim=3)

        # [B, S]
        enh = torch.istft(
            enh_spec,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
        enh = torch.clamp(enh, min=-1.0, max=1.0)

        return enh


if __name__ == "__main__":
    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)
    # get train args
    use_amp = False if device == "cpu" else config["train"]["use_amp"]
    clip_grad_norm_value = config["train"]["clip_grad_norm_value"]

    # config model
    model = DCCRN(config, mode="train", device=device)
    model = model.to(device)
    model.print_networks()

    # config optimizer
    optimizer = getattr(torch.optim, config["train"]["optimizer"])(
        params=model.parameters(),
        lr=config["train"]["lr"],
    )

    # scaler
    scaler = GradScaler(enabled=use_amp)

    # gen test data
    noisy = torch.randn([3, 48000])  # [B, S]
    clean = torch.randn([3, 48000])  # [B, S]

    # to device
    noisy = noisy.to(device)
    clean = clean.to(device)

    # test model and optimizer
    optimizer.zero_grad()
    with autocast(enabled=use_amp):
        enh = model(noisy)
        loss = model.loss(enh, clean)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)
    scaler.step(optimizer)
    scaler.update()

    pass
