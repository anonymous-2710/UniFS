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


if __name__ == '__main__':
    # Step 1: Generate random 200x200 matrix and save as grayscale image
    # image = np.random.rand(200, 200)
    original_image = "/data1/lijl/Restormer/Dataset/BraTs/test/BraTS2021_00254/BraTS2021_00254_XZ75.pkl"
    with open(original_image, "rb") as f:
        data = pickle.load(f)
        image = data['t2_hr']
        image = image[0]  # 1,200,200 ==> 200,200

    # Step 2: Apply degrade_random function
    # reconstructed_image, sampled_k_space = degrade_random(image)
    # reconstructed_image, sampled_k_space = degrade_equispace(image)
    reconstructed_image, sampled_k_space = degrade_center(image)

    # Step 3: Combine images into a single figure using subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    # Reconstructed Image
    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Reconstructed Image')

    # Sampled k-Space Image
    axes[2].imshow(np.log1p(np.abs(sampled_k_space)), cmap='gray')  # Log transform for better visualization
    axes[2].axis('off')
    axes[2].set_title('Sampled k-Space')

    # Adjust layout and save the combined figure
    plt.tight_layout()
    plt.title('Center crop degrade')
    plt.savefig('Center crop degrade.png', bbox_inches='tight')
    plt.close()
