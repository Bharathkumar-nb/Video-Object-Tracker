# Object Tracking in videos

This repository contains scripts for object tracking tasks. The scripts are written in Python and utilize the OpenCV and PyTorch libraries.

## Files

- `object_labeling.py`: This script allows you to label objects in a video file by selecting regions of interest (ROIs) in the frames. The labeled objects' coordinates are stored, and the frames with bounding boxes are saved as images. The labeled objects are also saved in a JSON file.

- `preprocessing_pipeline.py`: This script processes the video frames and the labeled objects generated by `object_labeling.py`. It resizes the frames and encodes the annotations to prepare them for training an object detection model. The preprocessed frames and annotations are saved in a pickle file.

- `object_detection_training.py`: This script trains an object detection model using the preprocessed data from `preprocessing_pipeline.py`. It utilizes the Faster R-CNN model architecture and the PyTorch library. The trained model is saved as a `.pth` file.

- `object_detection.py`: This script performs object detection on a video file using the trained model. It loads the model and processes each frame of the video to detect and draw bounding boxes around the objects of interest.

## Usage

Please follow the instructions below to use each script:

### `object_labeling.py`

1. Set the `video_file_name` variable to the path of the video file you want to label.
2. Set the `output_folder` variable to the path of the folder where you want to save the labeled frames.
3. Set the `output_file` variable to the path of the JSON file where you want to save the labeled objects.
4. Run the script.
5. Follow the instructions displayed in the console to label the objects in the video frames.
6. Press the ESC button to finish the selection process.
7. The labeled frames with bounding boxes will be saved in the specified output folder, and the labeled objects will be saved in the specified JSON file.

### `preprocessing_pipeline.py`

1. Set the `video_file_name` variable to the path of the video file you want to preprocess.
2. Set the `labeled_objects_file` variable to the path of the JSON file containing the labeled objects generated by `object_labeling.py`.
3. Set the `preprocessed_data_file` variable to the path where you want to save the preprocessed data as a pickle file.
4. Run the script.
5. The video frames will be processed, resized, and encoded along with the labeled objects.
6. The preprocessed data will be saved in the specified pickle file.

### `object_detection_training.py`

1. Set the `preprocessed_data_file` variable to the path of the pickle file containing the preprocessed data generated by `preprocessing_pipeline.py`.
2. Set the `trained_model_file` variable to the desired path for saving the trained model as a `.pth` file.
3. Set the `num_classes` variable to the number of classes in your dataset.
4. Run the script.
5. The script will load the preprocessed data and the Faster R-CNN model.
6. The model will be trained using the preprocessed data, and the trained model will be saved as a `.pth` file.

### `object_detection.py`

1. Set the `model_file` variable to the path of the trained model file generated by `object_detection_training.py`.
2. Set the `video_file` variable to the path of the video file you want to perform object detection on.
3. Set the `class_labels` variable to a list of class labels corresponding to the trained model.
4. Run the script.
5. The script will load

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to explore and modify the code according to your needs!
