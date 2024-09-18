import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T

class FoodSegmentation:
    def __init__(self, num_classes:int, device:str="cuda"):
        self.num_classes = num_classes
        self.device = device
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((520, 520)),  # Resize to the required input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        model = deeplabv3_resnet50(pretrained=True, num_classes=self.num_classes)
        #model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1,1), stride=(1,1))
        return model

    def set_optimizer(self, learning_rate:float=1e-3):
        self.otimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss()

    def train(self, x, y, epochs=10, batch_size=32):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(zip(x, y)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(x)}")

    def plot_results(self, original_image, predicted_mask):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='jet', alpha=0.7)
        plt.title("Predicted Mask")
        plt.axis('off')
        plt.show()

    def visualize_prediction(self, image):
        original_image = image.copy()
        tensor_image = self.transform(image).unsqueeze(0)
        tensor_image = tensor_image.to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_image)["out"]
            predicted_mask = outputs.argmax(dim=1).cpu().numpy()
        self.plot_results(original_image, predicted_mask[0])
        return original_image, predicted_mask
