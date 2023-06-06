# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:39:12 2023

@author: Bharath
@credits: https://chat.openai.com/
"""

import cv2
import json
import pickle


class VideoProcessor:
    def __init__(self, video_file_name, labeled_objects_file, preprocessed_data_file):
        self.video_file_name = video_file_name
        self.labeled_objects_file = labeled_objects_file
        self.preprocessed_data_file = preprocessed_data_file
        self.labeled_objects = []
        self.images = []
        self.annotations = []

    def process_video(self):
        try:
            # Open the video file
            video = cv2.VideoCapture(self.video_file_name)
            frame_count = 0
            obj_count = 0
            processed_frame_count = 0
            no_obj_count = 0

            while True:
                success = video.grab()
                if not success:
                    break

                # Read every 10th frame in the video
                if frame_count % 10 == 0:
                    success, frame = video.retrieve()
                    if not success:
                        break

                    processed_frame_count += 1
                    current_object = None

                    if obj_count < len(self.labeled_objects) and self.labeled_objects[obj_count]["frame_id"] == frame_count:
                        # Object found
                        while obj_count < len(self.labeled_objects) and self.labeled_objects[obj_count]["frame_id"] == frame_count:
                            self.images.append(frame)
                            current_object = self.labeled_objects[obj_count]
                            current_object["class"] = 1
                            self.annotations.append(current_object)
                            current_object = None
                            obj_count += 1
                    else:
                        # Object not found
                        current_object = {"x": 0, "y": 0, "w": 0, "h": 0, "frame_id": frame_count, "class": 0}
                        self.annotations.append(current_object)
                        self.images.append(frame)
                        no_obj_count += 1

                frame_count += 1

            # Release the video and close windows
            video.release()
            cv2.destroyAllWindows()

            print(f"frame shape = {frame.shape}")
            print(f"obj_count {obj_count}")
            print(f"no_obj_count {no_obj_count}")
            print(f"processed_frame_count {processed_frame_count}")
            print(f"frame_count {frame_count}")

        except Exception as e:
            print(e)

    def preprocess_images(self):
        SCALED_WIDTH = 400
        SCALED_HEIGHT = 400
        preprocessed_images = []

        def preprocess_image(image):
            # Resize the image to a fixed size
            resized_image = cv2.resize(image, (SCALED_WIDTH, SCALED_HEIGHT))
            return resized_image

        def encode_annotation(annotation, image_width, image_height):
            # Extract the bounding box coordinates
            x, y, w, h = annotation["x"], annotation["y"], annotation["w"], annotation["h"]
            # Calculate the relative coordinates
            x_min = (x / image_width) * SCALED_WIDTH
            y_min = (y / image_height) * SCALED_HEIGHT
            x_max = ((x + w) / image_width) * SCALED_WIDTH
            y_max = ((y + h) / image_height) * SCALED_HEIGHT
            # Encode the annotation as a dictionary
            encoded_annotation = {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "frame_id": annotation["frame_id"],
                "class": annotation["class"]
            }
            return encoded_annotation

        idx = 0
        for annotation, image in zip(self.annotations, self.images):
            image_height, image_width, _ = image.shape
            encoded_annotation = encode_annotation(annotation, image_width, image_height)
            self.annotations[idx] = encoded_annotation  # Update the annotation with encoded version
            preprocessed_image = preprocess_image(image)
            preprocessed_images.append(preprocessed_image)
            idx += 1

        return preprocessed_images

    def save_preprocessed_data(self, preprocessed_images):
        dataset = {
            "images": preprocessed_images,
            "annotations": self.annotations
        }

        # Save the preprocessed data to a file
        with open(self.preprocessed_data_file, 'wb') as f:
            pickle.dump(dataset, f)

    def load_labeled_objects(self):
        # Read the contents of the JSON file
        with open(self.labeled_objects_file, 'r') as file:
            labeled_objects_json = file.read()

        # Convert the JSON string to a Python list
        self.labeled_objects = json.loads(labeled_objects_json)

    def process_video_and_save_data(self):
        # Load the labeled objects
        self.load_labeled_objects()

        # Process the video frames
        self.process_video()

        # Preprocess the images
        preprocessed_images = self.preprocess_images()

        # Save the preprocessed data
        self.save_preprocessed_data(preprocessed_images)


# Example usage
video_file_name = "D:/GamingVideos/OBS_Recordings/talus.mp4"
labeled_objects_file = 'labeled_objects.json'
preprocessed_data_file = 'object_detection_data_preprocessed.pkl'

# Create an instance of the VideoProcessor class
processor = VideoProcessor(video_file_name, labeled_objects_file, preprocessed_data_file)

# Process the video frames and save the preprocessed data
processor.process_video_and_save_data()

# Credits to ChatGPT for assistance in transforming the code into an object-oriented structure.
