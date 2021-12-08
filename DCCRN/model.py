# -*- coding: utf-8 -*-

import sys
import os
import toml
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.signal import stft, istft
from paddle.framework import ParamAttr
from paddle.nn.initializer import Constant, Normal


sys.path.append(os.getcwd())
from audio.feature import offline_laplace_norm, cumulative_laplace_norm
from audio.metrics import SI_SDR


class ComplexConv2D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[5, 2],
        stride=[2, 1],
        padding=[2, 1],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv2d_real = nn.Conv2D(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.padding[0], 0],
            weight_attr=ParamAttr(initializer=Normal(mean=0, std=0.05)),
            bias_attr=ParamAttr(initializer=Constant(value=0.0)),
        )
        self.conv2d_imag = nn.Conv2D(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.padding[0], 0],
            weight_attr=ParamAttr(initializer=Normal(mean=0, std=0.05)),
            bias_attr=ParamAttr(initializer=Constant(value=0.0)),
        )

    def forward(self, input):
        # [B, C, F, T]
        [_, num_channels, _, _] = input.shape
        assert num_channels // 2 == self.in_channels

        # pad [left, right, top, bottom]
        input = F.pad(input, [self.padding[1], 0, 0, 0])

        # get real, imag
        # [B, C // 2, F, T]
        real, imag = paddle.chunk(input, 2, axis=1)

        # (Xr*Wr-Xi*Wi) + j(Xr*Wi+Xi*Wr)
        rr = self.conv2d_real(real)
        ii = self.conv2d_imag(imag)
        ri = self.conv2d_imag(real)
        ir = self.conv2d_real(imag)
        real = rr - ii
        imag = ri + ir

        # -> [B, C, F, T]
        return paddle.concat([real, imag], axis=1)


class ComplexLSTM(nn.Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        projection_size=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_real = nn.LSTM(self.input_size, self.hidden_size)
        self.lstm_imag = nn.LSTM(self.input_size, self.hidden_size)
        if projection_size is not None:
            self.projection_size = projection_size
            self.fc_real = nn.Linear(self.hidden_size, self.projection_size)
            self.fc_imag = nn.Linear(self.hidden_size, self.projection_size)
        else:
            self.projection_size = None

    def forward(self, input):
        # [B, C, F, T]
        [batch_size, num_channels, num_freqs, num_frames] = input.shape
        assert (num_channels // 2) * num_freqs == self.input_size

        # get real, imag
        # [B, C // 2, F, T]
        real, imag = paddle.chunk(input, 2, axis=1)
        # [B, C // 2, F, T] -> [B, T, C // 2, F] -> [B, T, (C // 2) * F]
        real = real.transpose([0, 3, 1, 2]).reshape([batch_size, num_frames, (num_channels // 2) * num_freqs])
        imag = imag.transpose([0, 3, 1, 2]).reshape([batch_size, num_frames, (num_channels // 2) * num_freqs])

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
        real = real.reshape([batch_size, num_frames, num_channels // 2, -1]).transpose([0, 2, 3, 1])
        imag = imag.reshape([batch_size, num_frames, num_channels // 2, -1]).transpose([0, 2, 3, 1])

        # -> [B, C, F, T]
        return paddle.concat([real, imag], axis=1)


class ComplexConv2DTranspose(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[5, 2],
        stride=[2, 1],
        padding=[2, 0],
        output_padding=[1, 0],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.conv2d_transpose_real = nn.Conv2DTranspose(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            weight_attr=ParamAttr(initializer=Normal(mean=0, std=0.05)),
            bias_attr=ParamAttr(initializer=Constant(value=0.0)),
        )
        self.conv2d_transpose_imag = nn.Conv2DTranspose(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            weight_attr=ParamAttr(initializer=Normal(mean=0, std=0.05)),
            bias_attr=ParamAttr(initializer=Constant(value=0.0)),
        )

    def forward(self, input):
        # [B, C, F, T]
        [_, num_channels, _, _] = input.shape
        assert num_channels // 2 == self.in_channels

        # get real, imag
        # [B, C // 2, F, T]
        real, imag = paddle.chunk(input, 2, axis=1)

        # (Xr*Wr-Xi*Wi) + j(Xr*Wi+Xi*Wr)
        rr = self.conv2d_transpose_real(real)
        ii = self.conv2d_transpose_imag(imag)
        ri = self.conv2d_transpose_imag(real)
        ir = self.conv2d_transpose_real(imag)
        real = rr - ii
        imag = ri + ir

        # -> [B, C, F, T]
        return paddle.concat([real, imag], axis=1)


class DCCRN(nn.Layer):
    def __init__(self, config, mode="train"):
        super().__init__()

        # set mode
        self.mode = mode

        # get dataset args
        self.n_fft = config["dataset"]["n_fft"]
        self.win_len = config["dataset"]["win_len"]
        self.hop_len = config["dataset"]["hop_len"]
        self.audio_len = config["dataset"]["audio_len"]
        self.window = paddle.to_tensor(np.hanning(self.win_len), dtype=paddle.float32)

        # get model args
        self.encoder_decoder_num_channels = config["model"]["encoder_decoder_num_channels"]
        self.encoder_decoder_kernel_size = config["model"]["encoder_decoder_kernel_size"]
        self.rnn_hidden_size = config["model"]["rnn_hidden_size"]
        self.rnn_num_layers = config["model"]["rnn_num_layers"]
        self.look_ahead = config["model"]["look_ahead"]

        # encoder
        self.encoder = nn.LayerList()
        for i in range(len(self.encoder_decoder_num_channels) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2D(
                        self.encoder_decoder_num_channels[i] // 2,
                        self.encoder_decoder_num_channels[i + 1] // 2,
                        kernel_size=self.encoder_decoder_kernel_size,
                        stride=[2, 1],
                        padding=[2, 1],
                    ),
                    nn.BatchNorm2D(self.encoder_decoder_num_channels[i + 1]),
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
        self.decoder = nn.LayerList()
        for i in range(len(self.encoder_decoder_num_channels) - 1, 0, -1):
            if i != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConv2DTranspose(
                            self.encoder_decoder_num_channels[i] * 2 // 2,
                            self.encoder_decoder_num_channels[i - 1] // 2,
                            kernel_size=self.encoder_decoder_kernel_size,
                            stride=[2, 1],
                            padding=[2, 0],
                            output_padding=[1, 0],
                        ),
                        nn.BatchNorm2D(self.encoder_decoder_num_channels[i - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConv2DTranspose(
                            self.encoder_decoder_num_channels[i] * 2 // 2,
                            self.encoder_decoder_num_channels[i - 1] // 2,
                            kernel_size=self.encoder_decoder_kernel_size,
                            stride=[2, 1],
                            padding=[2, 0],
                            output_padding=[1, 0],
                        )
                    )
                )

    @staticmethod
    def skip_connect(decoder_in, encoder_out):
        decoder_in_real, decoder_in_imag = paddle.chunk(decoder_in, 2, axis=1)
        encoder_out_real, encoder_out_imag = paddle.chunk(encoder_out, 2, axis=1)
        out_real = paddle.concat([decoder_in_real, encoder_out_real], axis=1)
        out_imag = paddle.concat([decoder_in_imag, encoder_out_imag], axis=1)
        return paddle.concat([out_real, out_imag], axis=1)

    @staticmethod
    def loss(enh, clean):
        return -(paddle.mean(SI_SDR(enh, clean)))

    def forward(self, noisy):
        # noisy [B, S]

        # stft
        # [B, F, T]
        noisy_spec = stft(noisy, self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window)

        # get noisy_spec_real, noisy_spec_imag
        # [B, F, T]
        noisy_spec_real = paddle.real(noisy_spec)
        noisy_spec_imag = paddle.imag(noisy_spec)
        # -> [B, 2, F, T]
        noisy_spec = paddle.stack([noisy_spec_real, noisy_spec_imag], axis=1)
        # dropout first bin
        out = noisy_spec[:, :, 1:, :]

        # check num_channels
        [_, num_channels, _, _] = out.shape
        assert num_channels == 2

        # norm
        # if self.mode in ["train", "valid"]:
        #     # [B, 2, F, T]
        #     out = offline_laplace_norm(out)
        # else:
        #     # [B, 2, F, T]
        #     out = cumulative_laplace_norm(out)

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
        cRM = F.pad(out, [0, 0, 1, 0])
        enh_spec_real = cRM[:, 0, :, :] * noisy_spec_real - cRM[:, 1, :, :] * noisy_spec_imag
        enh_spec_imag = cRM[:, 1, :, :] * noisy_spec_real + cRM[:, 0, :, :] * noisy_spec_imag
        enh_spec = paddle.squeeze(enh_spec_real + 1j * enh_spec_imag, axis=1)

        # istft
        # [B, S]
        enh = istft(enh_spec, self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window)

        return enh


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)
    # get train args
    use_amp = False if device == "cpu" else config["train"]["use_amp"]
    clip_grad_norm_value = config["train"]["clip_grad_norm_value"]

    # config model
    model = DCCRN(config, mode="train")
    print(
        paddle.summary(
            model,
            input_size=(
                config["dataloader"]["batch_size"],
                int(config["dataset"]["sr"] * config["dataset"]["audio_len"]),
            ),
        )
    )

    # config optimizer
    optimizer = getattr(paddle.optimizer, config["train"]["optimizer"])(
        parameters=model.parameters(),
        learning_rate=config["train"]["lr"],
        grad_clip=nn.ClipGradByNorm(clip_norm=clip_grad_norm_value),
    )

    # scaler
    scaler = paddle.amp.GradScaler()

    # gen test data
    noisy = paddle.randn([3, 48000]).astype(paddle.float32)  # [B, S]
    clean = paddle.randn([3, 48000]).astype(paddle.float32)  # [B, S]

    # test model and optimizer
    with paddle.amp.auto_cast(enable=use_amp):
        enh = model(noisy)
        loss = model.loss(enh, clean)
    scaled = scaler.scale(loss)
    scaled.backward()
    scaler.minimize(optimizer, scaled)
    optimizer.clear_grad()

    pass
