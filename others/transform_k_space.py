import pickle
import torch
import numpy as np
from scipy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from Tools.k_sampling import *

# To test if transform can be applied on images and corresponding k-space images at the same time

if __name__ == '__main__':
    # Step 1: Generate random 200x200 matrix and save as grayscale image
    # image = np.random.rand(200, 200)
    original_image = "/mnt/lijl/Restormer/Dataset/Multix4/test/random/BraTS2021_00048/BraTS2021_00048_RND_XZ60.pkl"
    with open(original_image, "rb") as f:
        data = pickle.load(f)
        image = data['t2_hr']
        image = image[0]  # 1,200,200 ==> 200,200

    image_reconstructed, k_space_cropped = degrade_random(image)

    # Step 2: 对退化图像和裁剪后的频域图像做变换
    rotated_image = rotate(image_reconstructed, 15, reshape=False)
    flipped_image = np.fliplr(image_reconstructed)

    rotated_k_space = rotate(np.abs(k_space_cropped), 15, reshape=False)
    flipped_k_space = np.fliplr(k_space_cropped)

    # Step 3: 对变换后的频域图像做逆傅里叶变换
    k_space_cropped_shifted_back_rotate = ifftshift(rotated_k_space)
    reconstructed_from_k_space_rotate = np.abs(ifft2(k_space_cropped_shifted_back_rotate))

    k_space_cropped_shifted_back_flipped = ifftshift(flipped_k_space)
    reconstructed_from_k_space_flipped = np.abs(ifft2(k_space_cropped_shifted_back_flipped))

    # 可视化结果
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # 原始MRI图像
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title("Original MRI Image")
    axs[0, 0].axis('off')

    # 退化图像
    axs[0, 1].imshow(image_reconstructed, cmap='gray')
    axs[0, 1].set_title("Degraded Image")
    axs[0, 1].axis('off')

    # 裁剪的频域图像
    axs[0, 2].imshow(np.log(np.abs(k_space_cropped) + 1), cmap='gray')
    axs[0, 2].set_title("Cropped K-Space")
    axs[0, 2].axis('off')

    # 变换后的退化图像
    axs[1, 0].imshow(rotated_image, cmap='gray')
    axs[1, 0].set_title("Rotated Degraded Image")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(flipped_image, cmap='gray')
    axs[1, 1].set_title("Flipped Degraded Image")
    axs[1, 1].axis('off')

    # 变换后的频域图像
    axs[1, 2].imshow(np.log(np.abs(flipped_k_space) + 1), cmap='gray')
    axs[1, 2].set_title("Flipped K-Space")
    axs[1, 2].axis('off')

    axs[2, 2].imshow(np.log(np.abs(rotated_k_space) + 1), cmap='gray')
    axs[2, 2].set_title("Rotated K-Space")
    axs[2, 2].axis('off')

    # 从频域恢复的图像
    axs[2, 0].imshow(reconstructed_from_k_space_rotate, cmap='gray')
    axs[2, 0].set_title("Reconstructed from Rotated K-Space")
    axs[2, 0].axis('off')

    axs[2, 1].imshow(reconstructed_from_k_space_flipped, cmap='gray')
    axs[2, 1].set_title("Reconstructed from Flipped K-Space")
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()