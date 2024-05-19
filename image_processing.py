import cv2
import numpy as np
import os
from pathlib import Path
from core_functions import histogram_equalization, contrast_stretching, add_haze


def process_images(input_folder):
    """
    Processes images in a folder by applying haze, histogram equalization, and contrast stretching, and saves the results.

    Args:
    input_folder (str): Path to the input folder containing images.

    Returns:
    None
    """
    # Define output folder names
    output_haze_folder = os.path.join(input_folder, 'hazy_images')
    output_hist_folder = os.path.join(input_folder, 'histogram_equalized')
    output_hist_comparison_folder = os.path.join(input_folder, 'hist_comparisons')
    output_contrast_folder = os.path.join(input_folder, 'contrast_stretched')
    output_contrast_comparison_folder = os.path.join(input_folder, 'contrast_comparisons')

    # Ensure output folders exist
    Path(output_haze_folder).mkdir(parents=True, exist_ok=True)
    Path(output_hist_folder).mkdir(parents=True, exist_ok=True)
    Path(output_hist_comparison_folder).mkdir(parents=True, exist_ok=True)
    Path(output_contrast_folder).mkdir(parents=True, exist_ok=True)
    Path(output_contrast_comparison_folder).mkdir(parents=True, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image from {image_path}")
                continue

            # Apply haze effect
            hazy_img_path = os.path.join(output_haze_folder, filename)
            add_haze(image_path, hazy_img_path, beta=0.15, A=0.5)git reset

            hazy_img = cv2.imread(hazy_img_path)

            # Create comparison image of original vs hazy image
            haze_comparison_image = np.hstack((img, hazy_img))
            haze_comparison_img_path = os.path.join(output_hist_comparison_folder,
                                                    f"{os.path.splitext(filename)[0]}_haze_comparison.jpeg")
            cv2.imwrite(haze_comparison_img_path, haze_comparison_image)

            # Apply histogram equalization to hazy image
            hist_eq_img = histogram_equalization(hazy_img)
            hist_eq_img_path = os.path.join(output_hist_folder, filename)
            cv2.imwrite(hist_eq_img_path, hist_eq_img)

            # Create comparison image for histogram equalization
            hist_comparison_image = np.hstack((img, hist_eq_img))
            hist_comparison_img_path = os.path.join(output_hist_comparison_folder,
                                                    f"{os.path.splitext(filename)[0]}_hist_comparison.jpeg")
            cv2.imwrite(hist_comparison_img_path, hist_comparison_image)

            # Apply contrast stretching to hazy image
            cont_str_img = contrast_stretching(hazy_img)
            cont_str_img_path = os.path.join(output_contrast_folder, filename)
            cv2.imwrite(cont_str_img_path, cont_str_img)

            # Create comparison image for contrast stretching
            contrast_comparison_image = np.hstack((img, cont_str_img))
            contrast_comparison_img_path = os.path.join(output_contrast_comparison_folder,
                                                        f"{os.path.splitext(filename)[0]}_contrast_comparison.jpeg")
            cv2.imwrite(contrast_comparison_img_path, contrast_comparison_image)

            print(f"Processed {filename} and saved all versions.")


if __name__ == "__main__":
    # Example usage
    input_folder = '/Users/ytnspybq/PycharmProjects/project_code/clear'  # Path to the input folder containing images
    process_images(input_folder)
