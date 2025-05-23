import torch
import albumentations as A
import numpy as np
from transformers import SegformerForSemanticSegmentation

from src.base_model import BaseModel

class SegFormerModel(BaseModel):
    def __init__(self, model_path:str="src/model/segformer-b3-finetuned-foodseg103"):
        """
        Initialize the SegFormerModel with the given model path.

        Parameters
        ----------
            model_path : str
                Path to the SegFormer model weights file.
        """
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _preprocess_image(self, image:np.ndarray) -> torch.Tensor:
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.pytorch.ToTensorV2()
        ])
        return transform(image=image)["image"].float().unsqueeze(0)

    def _load_model(self):
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_segmentation_mask(self, image:np.ndarray) -> np.ndarray:
        image_shape = image.shape[:-1]
        processed_image = self._preprocess_image(image.copy()).to(self.device)
        with torch.no_grad():
            outputs = self.model(processed_image)
            logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=image_shape, 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        return predicted