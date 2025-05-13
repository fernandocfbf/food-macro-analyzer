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
        self.transform = self._get_preprocess_tranformation()

    def _get_preprocess_tranformation(self):
        """
        Define the preprocessing transformations for the input images.

        Returns
        -------
            A.Compose
                A composition of transformations to be applied to the input images.
        """
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.pytorch.ToTensorV2()
        ])

    def _preprocess_image(self, image:np.ndarray) -> torch.Tensor:
        """
        Preprocess the input image using the defined transformations.

        Parameters
        ----------
            image : np.ndarray
                The input image to be preprocessed.

        Returns
        -------
            torch.Tensor
                The preprocessed image tensor.
        """
        return self.transform(image=image)["image"].float().unsqueeze(0)

    def _load_model(self):
        """
        Load the SegFormer model from the specified path and move it
        to the appropriate device.
        """
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_segmentation_mask(self, image:np.ndarray) -> np.ndarray:
        """
        Predict the segmentation mask for the given image using the SegFormer model.

        Parameters
        ----------
            image : np.ndarray
                The input image for which the segmentation mask is to be predicted.
        
        Returns
        -------
            np.ndarray
                The predicted segmentation mask.
        """
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