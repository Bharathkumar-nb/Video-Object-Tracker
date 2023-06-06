# Object Detection

This repository contains the code for object detection using the Faster R-CNN algorithm. The code is organized into three files:

1. `object_detection.py`: This script performs object detection on a video using a trained Faster R-CNN model.

2. `object_detection_dataset.py`: This file defines a custom dataset class for loading the preprocessed images and annotations.

3. `object_detection_utils.py`: This file contains utility functions for preprocessing images and visualizing the detection results.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV

## Usage

1. Prepare the Dataset:
   - Preprocess the images and annotations and save them as a pickle file (`object_detection_data_preprocessed.pkl`).
   - Ensure that the pickle file is located in the same directory as the script.

2. Training the Model:
   - Modify the script to set the desired number of classes and adjust other parameters.
   - Run the script `object_detection.py` to train the Faster R-CNN model.

3. Object Detection:
   - Modify the script `object_detection.py` to set the path to the trained model file (`faster_rcnn_model.pth`).
   - Run the script `object_detection.py` to perform object detection on a video file.
Note:

1. Place the video file (`video.mp4`) in the project directory.
2. Run the `object_detection.py` script: `python object_detection.py`.
3. The script will process the video frames, perform object detection, and display the results.
4. Press 'q' to exit the script.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to explore and modify the code according to your needs!
