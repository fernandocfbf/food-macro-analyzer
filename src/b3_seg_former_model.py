import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import albumentations as A
import numpy as np
from transformers import SegformerForSemanticSegmentation

from src.constants.category_id import PALLETE, CATEGORY_ID

class SegFormerModel:
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
    
    def generate_pallete_for_mask(self, mask):
            color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for label, color in PALLETE.items():
                color_seg[mask == label, :] = color
            return color_seg

    def apply_mask_on_image(self, image, mask, alpha=0.5):
        print(image.shape, mask.shape)
        imagem_original_float = image.astype(np.float32) / 255.0
        imagem_mascara_float = mask.astype(np.float32) / 255.0
        imagem_overlay = (1 - alpha) * imagem_original_float + alpha * imagem_mascara_float
        imagem_overlay = (imagem_overlay * 255).astype(np.uint8)
        return imagem_overlay

    def preview_image_segmentation(self, image:np.ndarray) -> None:
        """
        Preview the segmentation of the given image using the SegFormer model.

        Parameters
        ----------
            image : np.ndarray 
                Input image in the form of a NumPy array.
        """
        
        
        original_image = image.copy()
        mask = self.predict_segmentation_mask(image)
        color_seg = self.generate_pallete_for_mask(mask)
        blend = self.apply_mask_on_image(image, color_seg)
        fig, axs = plt.subplots(1, 2, figsize=(16, 12))

        axs[0].set_title('Original Image')
        axs[1].set_title('Annotation Mask')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[0].imshow(original_image)
        axs[1].imshow(blend)
        unique_labels = np.unique(mask)
        patches = [
            mpatches.Patch(
                color=np.array(PALLETE[label])/255,
                label=CATEGORY_ID.get(label, f'Class {label}')
            )
            for label in unique_labels if label in CATEGORY_ID]
        axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    