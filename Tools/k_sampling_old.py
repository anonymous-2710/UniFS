import pickle
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fftshift, ifftshift, fft2, ifft2
from skimage.draw import line


def process_fastMRI_knee(image, target_shape=(320, 320)):
    # This method for fastMRI Knee dataset only
    assert len(image.shape) == 2

    shifted = fftshift(image)  # special operation
    h, w = image.shape
    new_h, new_w = target_shape
    start_h = (h - new_h) // 2
    end_h = start_h + new_h
    start_w = (w - new_w) // 2
    end_w = start_w + new_w
    cropped = shifted[start_h:end_h, start_w:end_w]

    return cropped


def _get_shifted_kspace(input_data, input_kspace=False):
    if input_kspace:
        # return fftshift(input_data)
        return input_data  # if shifted
    else:
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


def generate_radial_mask(shape, num_angles):
    """
    生成径向k-space采样掩模
    Parameters:
        shape (tuple): 图像尺寸 (N, N)
        num_angles (int): 角度采样数量
    Returns:
        mask (ndarray): 生成的二值掩模
    """
    N = shape[0]
    mask = np.zeros(shape, dtype=np.uint8)
    center = N // 2, N // 2

    # 生成均匀分布的角度（0到π之间）
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    for theta in angles:
        # 计算方向向量
        direction = np.array([np.cos(theta), np.sin(theta)])

        # 找到直线与图像边界的交点
        t_values = []
        for sign in [-1, 1]:
            t = 0
            x, y = center
            while True:
                x_new = x + sign * direction[0] * t
                y_new = y + sign * direction[1] * t
                if x_new < 0 or x_new >= N or y_new < 0 or y_new >= N:
                    break
                t += 1
            t_values.append(t - 1)

        # 计算起点和终点坐标
        start = (int(center[0] + direction[0] * t_values[0]),
                 int(center[1] + direction[1] * t_values[0]))
        end = (int(center[0] - direction[0] * t_values[1]),
               int(center[1] - direction[1] * t_values[1]))

        # 生成直线上的采样点
        rr, cc = line(start[0], start[1], end[0], end[1])

        # 过滤超出图像范围的坐标
        valid = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
        mask[rr[valid], cc[valid]] = 1

    return mask


def degrade_radial(image, input_kspace=False, num_angles=32):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image, input_kspace)
    mask = generate_radial_mask(k_space_shifted.shape, num_angles)
    return _apply_mask_and_reconstruct(k_space_shifted, mask)


# center_fraction=0.08, acceleration=4 for x4
# center_fraction=0.04, acceleration=8 for x8
def degrade_random(image, input_kspace=False, center_fraction=0.08, acceleration=4):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image, input_kspace)
    mask_1d = _generate_random_mask(k_space_shifted.shape[1], center_fraction, acceleration)
    mask = np.tile(mask_1d, (k_space_shifted.shape[0], 1))
    return _apply_mask_and_reconstruct(k_space_shifted, mask)


def degrade_equispace(image, input_kspace=False, center_fraction=0.08, acceleration=4):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image, input_kspace)
    mask_1d = _generate_equispace_mask(k_space_shifted.shape[1], center_fraction, acceleration)
    mask = np.tile(mask_1d, (k_space_shifted.shape[0], 1))
    return _apply_mask_and_reconstruct(k_space_shifted, mask)


# crop_factor=0.25 for x4
# crop_factor=0.125 for x8
def degrade_center(image, input_kspace=False, crop_factor=0.25):
    assert len(image.shape) == 2
    k_space_shifted = _get_shifted_kspace(image, input_kspace)
    mask = _generate_center_mask(k_space_shifted.shape, crop_factor)
    return _apply_mask_and_reconstruct(k_space_shifted, mask)

