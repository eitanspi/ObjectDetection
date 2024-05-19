import cv2
import numpy as np
import math
import os


def histogram_equalization(img):
    """
    Applies histogram equalization to an input image to improve its contrast.

    Args:
    img (numpy.ndarray): Input image in BGR format.

    Returns:
    numpy.ndarray: Image after applying histogram equalization.
    """
    # Convert the image from BGR to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization on the Y channel (brightness)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # Convert the YUV image back to BGR format
    output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return output


def contrast_stretching(img):
    """
    Applies contrast stretching to an input image to enhance its contrast.

    Args:
    img (numpy.ndarray): Input image in BGR format.

    Returns:
    numpy.ndarray: Image after applying contrast stretching.
    """
    # Define input and output intensity ranges
    xp = [0, 64, 128, 192, 255]
    fp = [0, 8, 128, 250, 255]

    # Create a lookup table to map input intensities to output intensities
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')

    # Apply the lookup table to the image
    output = cv2.LUT(img, table)

    return output


def add_haze(image_path, output_path, beta, A):
    """
    Adds haze to an input image and saves the output image.

    Args:
    image_path (str): Path to the input image.
    output_path (str): Path where the output image will be saved.
    beta (float): Haze density coefficient.
    A (float): Atmospheric light intensity.

    Returns:
    None
    """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")

    # Normalize the image to the range [0, 1]
    img_f = image / 255.0

    # Apply haze effect to the image
    img_f = apply_haze_numba(img_f, beta, A)

    # Convert the image back to the range [0, 255] and to uint8 type
    img_f = np.clip(img_f * 255, 0, 255).astype(np.uint8)

    # Save the hazy image
    cv2.imwrite(output_path, img_f)


def apply_haze_numba(img_f, beta, A):
    """
    Applies haze effect to a normalized image.

    Args:
    img_f (numpy.ndarray): Normalized input image (values in the range [0, 1]).
    beta (float): Haze density coefficient.
    A (float): Atmospheric light intensity.

    Returns:
    numpy.ndarray: Image after applying haze effect.
    """
    row, col, _ = img_f.shape
    size = math.sqrt(max(row, col))
    center = (row // 2, col // 2)

    # Apply haze effect to each pixel
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j, l, :] = img_f[j, l, :] * td + A * (1 - td)

    return img_f

# Example usage


# if __name__ == "__main__":
#     # Example usage
#     image_path = 'lexus.jpeg'  # Path to the input image
#     output_haze_path = 'lexus_haze.jpeg'  # Path to save the hazy image
#     output_histogram_path = 'lexus_histogram_equalized.jpeg'  # Path to save the histogram equalized image
#     output_contrast_path = 'lexus_contrast_stretched.jpeg'  # Path to save the contrast stretched image
#
#     # Apply haze effect
#     beta = 0.1  # Haze density coefficient
#     A = 0.5  # Atmospheric light intensity
#     add_haze(image_path, output_haze_path, beta, A)
#     print(f"Hazy image saved as {output_haze_path}")
#
#     # Read the hazy image
#     hazy_img = cv2.imread(output_haze_path)
#
#     # Apply histogram equalization to the hazy image
#     if hazy_img is not None:
#         hist_eq_img = histogram_equalization(hazy_img)
#         cv2.imwrite(output_histogram_path, hist_eq_img)
#         print(f"Histogram equalized image saved as {output_histogram_path}")
#
#     # Apply contrast stretching to the hazy image
#     if hazy_img is not None:
#         cont_str_img = contrast_stretching(hazy_img)
#         cv2.imwrite(output_contrast_path, cont_str_img)
#         print(f"Contrast stretched image saved as {output_contrast_path}")
