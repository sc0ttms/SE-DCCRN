# -*- coding: utf-8 -*-

import numpy as np
import paddle

EPS = np.finfo(np.float32).eps


def is_clipped(data, clipping_threshold=0.99):
    return any(abs(data) > clipping_threshold)


def sub_sample(noisy, clean, samples):
    """random select fixed-length data from noisy and clean

    Args:
        noisy (float): noisy data
        clean (float): clean data
        samples (int): fixed length

    Returns:
        noisy, clean: fixed-length noisy and clean
    """
    length = len(noisy)

    if length > samples:
        start_idx = np.random.randint(length - samples)
        end_idx = start_idx + samples
        noisy = noisy[start_idx:end_idx]
        clean = clean[start_idx:end_idx]
    elif length < samples:
        noisy = np.append(noisy, np.zeros(samples - length))
        clean = np.append(clean, np.zeros(samples - length))
    else:
        pass

    assert len(noisy) == len(clean) == samples

    return noisy, clean


def offline_laplace_norm(input):
    """offline laplace norm

    Args:
        input (float): [B, C, F, T]

    Returns:
        normed (float): [B, C, F, T]
    """
    # utterance-level mu
    mu = paddle.mean(input, axis=list(range(1, input.dim())), keepdim=True)

    normed = input / (mu + EPS)

    return normed


def cumulative_laplace_norm(input):
    """cumulative laplace norm

    Args:
        input (float): [B, C, F, T]

    Returns:
        normed (float): [B, C, F, T]
    """
    [batch_size, num_channels, num_freqs, num_frames] = input.shape
    input = input.reshape([batch_size * num_channels, num_freqs, num_frames])

    step_sum = paddle.sum(input, axis=1)  # [B * C, F, T] => [B, T]
    cumulative_sum = paddle.cumsum(step_sum, axis=-1)  # [B, T]

    entry_count = paddle.arange(
        num_freqs,
        num_freqs * num_frames + 1,
        num_freqs,
        dtype=input.dtype,
    )
    entry_count = entry_count.reshape([1, num_frames])  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cumulative_mean = cumulative_sum / entry_count  # B, T
    cumulative_mean = cumulative_mean.reshape([batch_size * num_channels, 1, num_frames])

    normed = input / (cumulative_mean + EPS)

    return normed.reshape([batch_size, num_channels, num_freqs, num_frames])


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    [batch_size, _, num_freqs, _] = input.shape
    assert (
        batch_size > num_groups
    ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = paddle.arange(group_idx, batch_size, num_groups)
        freqs_indices = paddle.arange(group_idx, num_freqs, num_groups)

        selected_samples = paddle.index_select(input, samples_indices, axis=0)
        selected = paddle.index_select(selected_samples, freqs_indices, axis=2)

        output.append(selected)  # [B // num_groups, C, F // num_groups, T]

    return paddle.concat(output, axis=0)
