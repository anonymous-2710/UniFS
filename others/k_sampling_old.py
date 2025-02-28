import pickle

import torch
import numpy as np
from scipy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt


def degrade_random(image, center_fraction=0.08, acceleration=4):
    """
    Applies random subsampling in k-space.

    Args:
        image (ndarray): Input 2D image.
        center_fraction (float): Fraction of low-frequency data to retain in the center.
        acceleration (int): Acceleration factor for subsampling.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        ndarray: Reconstructed image after random subsampling.
        ndarray: k-space mask applied.
    """
    assert len(image.shape) == 2
    # Step 1: 2D Fourier transform to k-space
    k_space = fft2(image)
    k_space_shifted = fftshift(k_space)

    # Step 2: Generate random mask
    num_cols = k_space.shape[1]
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.uniform(size=num_cols) < prob  # (1, num_cols)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = 1

    # Step 3: Apply mask in k-space
    mask = np.tile(mask, (k_space.shape[0], 1))  # (1, num_cols) ==> (num_rows, num_cols)
    k_space_cropped = k_space_shifted * mask

    # Step 4: Inverse Fourier transform back to image space
    k_space_cropped_shifted_back = ifftshift(k_space_cropped)
    image_reconstructed = np.abs(ifft2(k_space_cropped_shifted_back))

    return image_reconstructed, k_space_cropped, mask


def degrade_equispace(image, center_fraction=0.08, acceleration=4):
    """
    Applies equispace subsampling in k-space.

    Args:
        image (ndarray): Input 2D image.
        center_fraction (float): Fraction of low-frequency data to retain in the center.
        acceleration (int): Acceleration factor for subsampling.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        ndarray: Reconstructed image after equispaced subsampling.
        ndarray: k-space mask applied.
    """
    assert len(image.shape) == 2
    # Step 1: 2D Fourier transform to k-space
    k_space = fft2(image)
    k_space_shifted = fftshift(k_space)

    # Step 2: Generate equispaced mask
    num_cols = k_space.shape[1]
    num_low_freqs = int(round(num_cols * center_fraction))
    mask = np.zeros(num_cols, dtype=np.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = 1

    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
    offset = np.random.randint(0, round(adjusted_accel))  # random choose start point
    accel_samples = np.arange(offset, num_cols, adjusted_accel)  # equispace arrs from offset to num_cols, step=accel
    accel_samples = np.around(accel_samples).astype(np.uint)  # to nearest integer
    mask[accel_samples] = 1

    # Step 3: Apply mask in k-space
    mask = np.tile(mask, (k_space.shape[0], 1))  # (1, num_cols) ==> (num_rows, num_cols)
    k_space_cropped = k_space_shifted * mask

    # Step 4: Inverse Fourier transform back to image space
    k_space_cropped_shifted_back = ifftshift(k_space_cropped)
    image_reconstructed = np.abs(ifft2(k_space_cropped_shifted_back))

    return image_reconstructed, k_space_cropped, mask


def degrade_center(image, crop_factor=0.25):
    assert len(image.shape) == 2
    # Step 1: 2D傅里叶变换到k-space
    k_space = fft2(image)

    # Step 2: 将零频率部分移到图像的中心
    k_space_shifted = fftshift(k_space)

    # Step 3: 裁剪k-space的中心区域
    center_x, center_y = k_space.shape[0] // 2, k_space.shape[1] // 2
    crop_size_x = int(k_space.shape[0] * crop_factor)
    crop_size_y = int(k_space.shape[1] * crop_factor)

    mask = np.zeros_like(image)
    mask[
    center_x - crop_size_x // 2: center_x + crop_size_x // 2,
    center_y - crop_size_y // 2: center_y + crop_size_y // 2
    ] = 1
    k_space_cropped = k_space_shifted * mask

    # Step 4: 逆傅里叶变换回图像空间
    k_space_cropped_shifted_back = ifftshift(k_space_cropped)  # 将低频部分移动回去
    image_reconstructed = np.abs(ifft2(k_space_cropped_shifted_back))

    return image_reconstructed, k_space_cropped, mask

