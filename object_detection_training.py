# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:50:13 2023

@author: Bharath
@credits: https://chat.openai.com/
"""

# Object Detection Training

import time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import pickle
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class ObjectDetectionDataset(Dataset):
    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        annotation = self.annotations[index]

        image_tensor = F.to_tensor(image)

        # Extract the bounding box coordinates from the annotation
        x, y, xx, yy = annotation["x_min"], annotation["y_min"], annotation["x_max"], annotation["y_max"]

        # Create the target tensor containing the bounding box coordinates
        # Check if the bounding box is valid
        if xx > x and yy > y:
            # Create the target tensor containing the bounding box coordinates
            target = {
                "boxes": torch.tensor([[x, y, xx, yy]], dtype=torch.float32),
                "labels": torch.tensor([annotation["class"]], dtype=torch.int64),
            }
        else:
            # Invalid bounding box, return None as the target
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64)
            }

        # Convert the image tensor to the desired data type
        image_tensor = image_tensor.to(dtype=torch.float32)
        return image_tensor, target


class ObjectDetectionTrainer:
    def __init__(self, dataset, model, device='cuda'):
        self.dataset = dataset
        self.model = model
        self.device = torch.device(device)

        # Define the optimizer and learning rate scheduler
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def train(self, batch_size=4, num_epochs=10):
        # Create a data loader for the dataset
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        for epoch in range(num_epochs):
            start_time = time.time()  # Start the timer for the current epoch
            self.model.train()

            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                outputs = self.model(images, targets)

                # Compute the loss
                loss_classifier = outputs["loss_classifier"]
                loss_box_reg = outputs["loss_box_reg"]
                loss_objectness = outputs["loss_objectness"]
                loss_rpn_box_reg = outputs["loss_rpn_box_reg"]

                # Compute the total loss
                losses = sum(loss for loss in outputs.values())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            # Update the learning rate
            self.lr_scheduler.step()

            end_time = time.time()  # Stop the timer for the current epoch
            epoch_time = end_time - start_time

            # Print the training loss for this epoch
            print(f"Epoch {epoch+1}/{num_epochs}, loss_classifier: {loss_classifier.item()}, Time: {epoch_time:.2f} seconds")
            print(f"Epoch {epoch+1}/{num_epochs}, loss_box_reg: {loss_box_reg}")
            print(f"Epoch {epoch+1}/{num_epochs}, loss_objectness: {loss_objectness}")
            print(f"Epoch {epoch+1}/{num_epochs}, loss_rpn_box_reg: {loss_rpn_box_reg}")

            # Save the trained model
            torch.save(self.model.state_dict(), f"faster_rcnn_model3_{epoch}.pth")

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# Preprocessed data file
preprocessed_data_file = 'object_detection_data_preprocessed.pkl'
# Trained model file
trained_model_file = "faster_rcnn_model.pth"

# Load dataset object containing preprocessed images and annotations
with open(preprocessed_data_file, 'rb') as f:
    data = pickle.load(f)

dataset = ObjectDetectionDataset(data["images"], data["annotations"])

# Define the model architecture
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Modify the model to match the number of classes in your dataset
num_classes = 2  # Number of classes in your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Set device for training (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create an object detection trainer
trainer = ObjectDetectionTrainer(dataset, model, device)

# Train the model
trainer.train(batch_size=4, num_epochs=10)
