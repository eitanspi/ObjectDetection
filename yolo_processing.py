from ultralytics import YOLO
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from math import sqrt


def process_images_with_yolo(input_folder):
    """
    Processes images in a folder using a YOLO model and saves the results.

    Args:
    input_folder (str): Path to the input folder containing images and subfolders.

    Returns:
    None
    """
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')

    # Define subfolder names
    subfolders = ['hazy_images', 'histogram_equalized', 'contrast_stretched']

    # Define output folder names
    output_folders = {
        'clear_images': 'clear_yolo',
        'hazy_images': 'hazy_yolo',
        'histogram_equalized': 'hist_equal_yolo',
        'contrast_stretched': 'contrast_yolo'
    }

    # Create output folders if they don't exist
    for key in output_folders:
        Path(os.path.join(input_folder, output_folders[key])).mkdir(parents=True, exist_ok=True)

    # Data structure to keep track of metrics
    metrics = []

    # Process clear images first to get the original object counts and positions
    clear_images_folder = input_folder
    original_detections = {}
    for filename in os.listdir(clear_images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(clear_images_folder, filename)
            if os.path.isfile(image_path):
                output_path = os.path.join(input_folder, output_folders['clear_images'], filename)
                tp, fn, fp, centers = process_image_with_yolo(model, image_path, output_path)
                original_detections[filename] = centers
                metrics.append({
                    'Image': filename,
                    'Original_Objects': len(centers),
                    'Hazy_TP': 0, 'Hazy_FP': 0, 'Hazy_FN': 0,
                    'HistEqual_TP': 0, 'HistEqual_FP': 0, 'HistEqual_FN': 0,
                    'Contrast_TP': 0, 'Contrast_FP': 0, 'Contrast_FN': 0
                })

    # Process images in each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.exists(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                    image_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(image_path):
                        output_subfolder = output_folders[subfolder]
                        output_path = os.path.join(input_folder, output_subfolder, filename)
                        tp, fn, fp, centers = process_image_with_yolo(model, image_path, output_path)

                        # Compare detections with the original image
                        original_centers = original_detections.get(filename, [])
                        tp, fn, fp = compare_detections(original_centers, centers, image_path)

                        metric = next((m for m in metrics if m['Image'] == filename), None)
                        if metric:
                            metric[f'{subfolder.capitalize()}_TP'] = tp
                            metric[f'{subfolder.capitalize()}_FN'] = fn
                            metric[f'{subfolder.capitalize()}_FP'] = fp

    # Save metrics to Excel file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_excel(os.path.join(input_folder, 'yolo_metrics.xlsx'), index=False)

    # Adjust column widths in the Excel file
    with pd.ExcelWriter(os.path.join(input_folder, 'yolo_metrics.xlsx'), engine='openpyxl') as writer:
        metrics_df.to_excel(writer, index=False)
        worksheet = writer.sheets['Sheet1']
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width


def process_image_with_yolo(model, image_path, output_path):
    """
    Processes a single image using a YOLO model and saves the result.

    Args:
    model (YOLO): YOLO model.
    image_path (str): Path to the input image.
    output_path (str): Path where the output image will be saved.

    Returns:
    (int, int, int, list): Tuple containing TP, FN, FP counts and centers of detected objects.
    """
    # Run inference on the image
    results = model(image_path)

    # Save the annotated image
    results[0].save(filename=output_path)

    # Extract detection results
    detections = results[0].boxes
    centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in detections.xyxy]

    return len(centers), 0, 0, centers  # Placeholder for TP, FN, FP


def compare_detections(original_centers, processed_centers, image_path):
    """
    Compares detected objects in the original and processed images.

    Args:
    original_centers (list): List of centers of objects in the original image.
    processed_centers (list): List of centers of objects in the processed image.
    image_path (str): Path to the processed image.

    Returns:
    (int, int, int): Tuple containing TP, FN, and FP counts.
    """
    image = Image.open(image_path)
    width, height = image.size
    threshold = sqrt(width ** 2 + height ** 2) * 0.03  # 3% of the image diagonal

    tp = 0
    original_remaining = set(original_centers)
    processed_remaining = set(processed_centers)

    for oc in original_centers:
        for pc in processed_centers:
            if euclidean_distance(oc, pc) <= threshold:
                tp += 1
                original_remaining.discard(oc)
                processed_remaining.discard(pc)
                break

    fn = len(original_remaining)
    fp = len(processed_remaining)

    return tp, fn, fp


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
    point1 (tuple): First point (x1, y1).
    point2 (tuple): Second point (x2, y2).

    Returns:
    float: Euclidean distance between the points.
    """
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


if __name__ == "__main__":
    # Example usage
    input_folder = '/Users/ytnspybq/PycharmProjects/project_code/clear'  # Path to the input folder containing images
    # and subfolders
    process_images_with_yolo(input_folder)
