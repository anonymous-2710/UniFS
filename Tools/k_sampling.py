import pickle
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fftshift, ifftshift, fft2, ifft2
from skimage.draw import line
from sigpy.mri import poisson


def _get_shifted_kspace(input_data):
    k_space = fft2(input_data)
    return fftshift(k_space)


def _apply_mask_and_reconstruct(k_space_shifted, mask):
    """公共函数：应用掩码并重建图像"""
    k_space_cropped = k_space_shifted * mask
    k_space_cropped_shifted_back = ifftshift(k_space_cropped)
    image_reconstructed = np.abs(ifft2(k_space_cropped_shifted_back))
    return image_reconstructed, k_space_cropped, mask


def _generate_random_mask(num_cols, center_fraction, acceleration):
    """生成随机采样掩码 (1D列掩码)"""
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = 1
    return mask


def _generate_equispace_mask(num_cols, center_fraction, acceleration):
    """生成等间距采样掩码 (1D列掩码)"""
    num_low_freqs = int(round(num_cols * center_fraction))
    mask = np.zeros(num_cols, dtype=np.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = 1

    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
    offset = np.random.randint(0, round(adjusted_accel))
    accel_samples = np.arange(offset, num_cols, adjusted_accel)
    # FixBug at 0216: index out of range on mask for some num_cols
    # accel_samples = np.around(accel_samples).astype(np.uint)
    accel_samples = np.floor(accel_samples).astype(np.uint)
    mask[accel_samples] = 1
    return mask


def _generate_center_mask(shape, crop_factor):
    """生成中心裁剪掩码 (2D掩码)"""
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    crop_size_x = int(rows * crop_factor)
    crop_size_y = int(cols * crop_factor)
    mask = np.zeros(shape)
    mask[
    center_x - crop_size_x // 2: center_x + crop_size_x // 2,
    center_y - crop_size_y // 2: center_y + crop_size_y // 2
    ] = 1
    return mask


def _generate_poisson_mask(shape, calib_size, acceleration):
    """生成2D泊松圆盘采样掩码"""
    # 生成泊松圆盘采样模板
    mask = poisson(shape, calib=(calib_size, calib_size), accel=acceleration)  # complex dataytype mask?
    # 强制中心校准区域全采样
    # ctr_start = (shape[0] // 2 - calib_size // 2, shape[1] // 2 - calib_size // 2)
    # ctr_end = (ctr_start[0] + calib_size, ctr_start[1] + calib_size)
    # mask[ctr_start[0]:ctr_end[0], ctr_start[1]:ctr_end[1]] = 1
    return mask.astype(np.float32)


def degrade_poisson(image, calib_size=64, acceleration=4):
    """
    使用泊松圆盘采样的k空间退化函数
    :param image: 输入图像 (H, W)
    :param input_kspace: 是否直接输入k空间数据
    :param calib_size: 校准区域大小 (默认32x32)
    :param acceleration: 加速倍数
    :return: 重建图像, 欠采样k空间, 采样掩码
    """
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image)
    mask = _generate_poisson_mask(k_space_shifted.shape, calib_size, acceleration)
    return _apply_mask_and_reconstruct(k_space_shifted, mask)


# center_fraction=0.08, acceleration=4 for x4
# center_fraction=0.04, acceleration=8 for x8
def degrade_random(image, center_fraction=0.08, acceleration=4):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image)
    mask_1d = _generate_random_mask(k_space_shifted.shape[1], center_fraction, acceleration)
    mask = np.tile(mask_1d, (k_space_shifted.shape[0], 1))
    return _apply_mask_and_reconstruct(k_space_shifted, mask)


def degrade_equispace(image, center_fraction=0.08, acceleration=4):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image)
    mask_1d = _generate_equispace_mask(k_space_shifted.shape[1], center_fraction, acceleration)
    mask = np.tile(mask_1d, (k_space_shifted.shape[0], 1))
    return _apply_mask_and_reconstruct(k_space_shifted, mask)


# crop_factor=0.25 for x4
# crop_factor=0.125 for x8
def degrade_center(image, crop_factor=0.25):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image)
    mask = _generate_center_mask(k_space_shifted.shape, crop_factor)
    return _apply_mask_and_reconstruct(k_space_shifted, mask)

