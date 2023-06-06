# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:00:54 2023

@author: Bharath
@credits: https://chat.openai.com/
"""

import time
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

SCALED_WIDTH = 400
SCALED_HEIGHT = 400

class ObjectDetection:
    def __init__(self, model_file, video_file, class_labels):
        self.model_file = model_file
        self.video_file = video_file
        self.class_labels = class_labels
        self.model = None
        self.device = None

    def load_model(self):
        # Load the trained model
        self.model = fasterrcnn_resnet50_fpn()
        # Modify the model to match the number of classes
        num_classes = len(self.class_labels)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()

    def preprocess_image(self, image):
        # Resize the image to a fixed size
        resized_image = cv2.resize(image, (SCALED_WIDTH, SCALED_HEIGHT))
        return resized_image

    def process_video(self):
        # Set device for inference (e.g., GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Open the video file
        video = cv2.VideoCapture(self.video_file)

        frame_id = 0
        elapsed_times = []
        # Loop through the frames of the video
        while True:
            # Read the next frame
            success, frame = video.read()
            if not success:
                break
            frame_id += 1
            start_time = time.time()  # Start the timer for the current epoch
            # Preprocess the frame
            preprocessed_frame = self.preprocess_image(frame)
            image_tensor = F.to_tensor(preprocessed_frame)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add an extra dimension

            # Perform inference
            with torch.no_grad():
                predictions = self.model(image_tensor)

            # Process the predictions
            boxes = predictions[0]["boxes"].cpu().numpy()
            labels = predictions[0]["labels"].cpu().numpy()
            scores = predictions[0]["scores"].cpu().numpy()

            # Visualize the predictions
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5:  # Set a threshold for confidence score
                    x, y, xx, yy = box.astype(int)
                    image_height, image_width, _ = frame.shape
                    x *= image_width
                    x //= SCALED_WIDTH
                    xx *= image_width
                    xx //= SCALED_WIDTH
                    y *= image_height
                    y //= SCALED_HEIGHT
                    yy *= image_height
                    yy //= SCALED_HEIGHT
                    class_label = self.class_labels[label]
                    print(f"{label}: {score:.2f}, {x}, {y}, {xx}, {yy}")
                    cv2.rectangle(frame, (x, y), (xx, yy), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_label}: {score:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            end_time = time.time()  # Stop the timer for the current epoch
            epoch_time = end_time - start_time
            elapsed_times.append(epoch_time)
            # Display the frame with predictions
            cv2.imshow("Video", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        total_time = sum(elapsed_times)
        average_time = total_time / len(elapsed_times)
        fps = 1/average_time
        print(f"Total time to process {frame_id} frames: {total_time:.2f} seconds")
        print(f"Average time: {average_time} seconds")
        print(f"FPS: {fps}")

        # Release the video and close windows
        video.release()
        cv2.destroyAllWindows()


# Define the class labels (modify as per your dataset)
class_labels = ["background", "object"]

# Create an object detection instance
object_detection = ObjectDetection(
    model_file="faster_rcnn_model.pth",
    video_file="D:/GamingVideos/OBS_Recordings/tyra.mp4",
    class_labels=class_labels
)

# Load the model
object_detection.load_model()

# Process the video
object_detection.process_video()
