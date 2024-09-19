import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class FoodSegDataset(Dataset):
    def __init__(self, image_dir:str, mask_dir:str, mode:str="train"):
        self.image_dir = os.path.join(image_dir, mode)
        self.mask_dir = os.path.join(mask_dir, mode)
        self.images = os.listdir(self.image_dir)
        self.annotations = os.listdir(self.mask_dir)
        self.mode = mode
        self.image_size = (520,520)
        self.transform_image = T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size, interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.annotations[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        return image, mask