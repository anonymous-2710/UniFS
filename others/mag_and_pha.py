from Tools.utils import *
from Tools.k_sampling import *

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

"""
This experiment test cropped_k_space ==> t2_lr ==> k_space, and cropped_k_space != k_space, not accurate enough.

In practical calculations, due to the limited precision of computers (floating-point accuracy), 
small numerical errors may occur when converting between frequency domain and image space.

These errors are usually very small (such as in the range of 1e-7 or smaller) and can be ignored for most applications.

However, if cropping, filtering, phase modification, or other operations are performed on the frequency domain image, 
the information cannot be fully restored when the inverse transformation is followed by the forward transformation.
"""


def visualize_degradation(image, center_fraction=0.08, acceleration=4):
    """
    Visualize the process of k-space degradation and reconstruction.

    Args:
        image (ndarray): Input 2D image.
        center_fraction (float): Fraction of low-frequency data to retain in the center.
        acceleration (int): Acceleration factor for subsampling.
    """
    # Perform degradation
    image_reconstructed, k_space_cropped, mask = degrade_equispace(image, center_fraction=center_fraction,
                                                                   acceleration=acceleration)

    # Calculate k-space from reconstructed image
    k_space_from_reconstructed = fftshift(fft2(image_reconstructed))

    # Plot original image, mask, k-space, and results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Original k-space (log scale)")
    plt.imshow(np.log(np.abs(fftshift(fft2(image)) + 1)), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("k-space Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Degraded k-space (log scale)")
    plt.imshow(np.log(np.abs(k_space_cropped) + 1), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Reconstructed Image")
    plt.imshow(image_reconstructed, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Reconstructed k-space (log scale)")
    plt.imshow(np.log(np.abs(k_space_from_reconstructed) + 1), cmap='gray')
    plt.axis('off')

    # plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create a sample 200x200 image
    image = np.random.rand(200, 200)

    # Visualize the degradation process
    visualize_degradation(image)
from Tools.utils import *
from Tools.k_sampling import *

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

"""
This experiment test cropped_k_space ==> t2_lr ==> k_space, and cropped_k_space != k_space, not accurate enough.

In practical calculations, due to the limited precision of computers (floating-point accuracy), 
small numerical errors may occur when converting between frequency domain and image space.

These errors are usually very small (such as in the range of 1e-7 or smaller) and can be ignored for most applications.

However, if cropping, filtering, phase modification, or other operations are performed on the frequency domain image, 
the information cannot be fully restored when the inverse transformation is followed by the forward transformation.
"""


def visualize_degradation(image, center_fraction=0.08, acceleration=4):
    """
    Visualize the process of k-space degradation and reconstruction.

    Args:
        image (ndarray): Input 2D image.
        center_fraction (float): Fraction of low-frequency data to retain in the center.
        acceleration (int): Acceleration factor for subsampling.
    """
    # Perform degradation
    image_reconstructed, k_space_cropped, mask = degrade_equispace(image, center_fraction=center_fraction,
                                                                   acceleration=acceleration)

    # Calculate k-space from reconstructed image
    k_space_from_reconstructed = fftshift(fft2(image_reconstructed))

    # Plot original image, mask, k-space, and results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Degraded k-space (log scale)")
    plt.imshow(np.log(np.abs(k_space_cropped) + 1), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Degraded k-space magnitude")
    plt.imshow(np.log(np.abs(k_space_cropped) + 1), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Degraded k-space phase")
    plt.imshow(np.angle(k_space_cropped), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Reconstructed Image")
    plt.imshow(image_reconstructed, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Reconstructed k-space (log scale)")
    plt.imshow(np.log(np.abs(k_space_from_reconstructed) + 1), cmap='gray')
    plt.axis('off')

    # plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create a sample 200x200 image
    image = np.random.rand(200, 200)

    # Visualize the degradation process
    visualize_degradation(image)
