# ObjectDetection

This project involves processing and enhancing images using the YOLO (You Only Look Once) object detection model. The code includes functionalities for:

- Processing images to detect objects using a pretrained YOLO model.
- Enhancing images using classical dehazing techniques like histogram equalization and contrast stretching.
- Adding synthetic haze to images for simulation and training purposes.
- Generating metrics and visual results to evaluate the performance of the YOLO model on various types of processed images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
   ```sh
   git clone git@github.com:eitanspi/ObjectDetection.git
   cd ObjectDetection
   ```

2. **Install the required dependencies**:
   Make sure you have Python 3.6 or higher installed. Then, install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the YOLO model**:
   Place the YOLO model (`yolov8n.pt`) in the project directory. You can download it from [YOLO's official website](https://github.com/ultralytics/yolov5).

## Usage

1. **Process images with YOLO**:
   ```sh
   python main.py
   ```

2. **Script Descriptions**:
   - `main.py`: The entry point for processing images with the YOLO model.
   - `core_functions.py`: Core functions for image processing and object detection.
   - `image_processing.py`: Functions for applying image enhancements like histogram equalization and contrast stretching.
   - `yolo_processing.py`: Functions for processing images using the YOLO model and generating metrics.
   - `plot.py`: Utility functions for plotting and visualizing results.

3. **Input Folder**:
   - The input folder should contain images and subfolders for different processing types. For example:
     ```
     input_folder/
     ├── clear/
     ├── hazy_images/
     ├── histogram_equalized/
     ├── contrast_stretched/
     ```

## Features

- **Object Detection**: Detects objects in images using a pretrained YOLO model.
- **Image Enhancement**: Applies histogram equalization and contrast stretching to improve image quality.
- **Synthetic Haze Addition**: Adds varying levels of synthetic haze to images for simulation purposes.
- **Performance Metrics**: Generates and saves metrics to an Excel file to evaluate the performance of the YOLO model on different types of processed images.

## Project Structure

```
ObjectDetection/
├── .idea/
├── __pycache__/
├── clear/
├── core_functions.py
├── image_processing.py
├── main.py
├── plot.py
├── yolo_processing.py
├── yolov8n.pt
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
