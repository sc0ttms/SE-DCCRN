# -*- coding: utf-8 -*-

import numpy as np
import paddle

EPS = np.finfo(np.float32).eps


def compress_cIRM(mask, K=10, C=0.1):
    if paddle.is_tensor(mask):
        mask = -100 * (mask <= -100).astype(paddle.float32) + mask * (mask > -100).astype(paddle.float32)
        mask = K * (1 - paddle.exp(-C * mask)) / (1 + paddle.exp(-C * mask))
        return mask.astype(paddle.float32)
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
        return mask.astype(np.float32)


def decompress_cIRM(mask, K=10, limit=9.999):
    if paddle.is_tensor(mask):
        mask = (
            limit * (mask >= limit).astype(paddle.float32)
            - limit * (mask <= -limit).astype(paddle.float32)
            + mask * (paddle.abs(mask) < limit).astype(paddle.float32)
        )
        mask = -K * paddle.log((K - mask) / (K + mask))
        return mask.astype(paddle.float32)
    else:
        mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (np.abs(mask) < limit)
        mask = -K * np.log((K - mask) / (K + mask))
        return mask.astype(np.float32)


def get_cIRM(noisy_spec, clean_spec):
    if paddle.is_tensor(noisy_spec) and paddle.is_tensor(clean_spec):
        noisy_real = paddle.real(noisy_spec)
        noisy_imag = paddle.imag(noisy_spec)
        clean_real = paddle.real(clean_spec)
        clean_imag = paddle.imag(clean_spec)

        denominator = (noisy_real ** 2 + clean_imag ** 2) + EPS

        cIRM_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
        cIRM_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator
        cIRM = paddle.stack([cIRM_real, cIRM_imag], axis=-1)
    else:
        noisy_real = np.real(noisy_spec)
        noisy_imag = np.imag(noisy_spec)
        clean_real = np.real(clean_spec)
        clean_imag = np.imag(clean_spec)

        denominator = (noisy_real ** 2 + clean_imag ** 2) + EPS

        cIRM_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
        cIRM_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator
        cIRM = np.stack((cIRM_real, cIRM_imag), axis=-1)

    return compress_cIRM(cIRM)  # [B, F, T, 2]
