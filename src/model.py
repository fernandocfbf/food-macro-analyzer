from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T
from PIL import Image

class FoodSegmentation:
    def __init__(self, num_classes:int, device:str="cuda"):
        self.num_classes = num_classes
        self.device = device
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _image_transform(self, image):
        return T.Compose([
            T.ToPILImage(),
            T.Resize((520, 520)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)
    
    def _mask_transform(self, mask):
        return T.Compose([
            T.Resize((520, 520), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])(mask)

    def _build_model(self):
        model = deeplabv3_resnet50(pretrained=True, num_classes=self.num_classes)
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1,1))
        return model

    def set_optimizer(self, learning_rate:float=1e-3):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                for images, masks in train_loader:

                    images = images.to(self.device)
                    masks = masks.to(self.device).long()
                    outputs = self.model(images)["out"]
                    loss = self.criterion(outputs, masks.squeeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
        print('Training complete')

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
        tensor_image = self._image_transform(image).unsqueeze(0)
        tensor_image = tensor_image.to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_image)["out"]
            predicted_mask = outputs.argmax(dim=1).cpu().numpy()
        self.plot_results(original_image, predicted_mask[0])
        return original_image, predicted_mask
