import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abc import ABC, abstractmethod
from typing import Union

from src.utils.visualization import generate_pallete_for_mask, apply_mask_on_image
from src.constants.category_id import PALLETE, CATEGORY_ID

class BaseModel(ABC):

    def generate_annotated_image(self, image:np.ndarray, mask:np.ndarray) -> np.ndarray:
        """
        Generate an annotated image by applying the segmentation mask to the input image.
        
        Parameters
        ----------
            image : np.ndarray
                The input image to be annotated.
            mask : np.ndarray
                The segmentation mask to be applied to the image.

        Returns 
        -------
            np.ndarray
                The annotated image with the segmentation mask applied.
        """
        color_seg = generate_pallete_for_mask(mask)
        return apply_mask_on_image(image, color_seg)
    
    def preview_image_segmentation(self, image:np.ndarray, mask:np.ndarray, legend:bool=True) -> None:
        """
        Preview the segmentation of the given image using the model.
        
        Parameters
        ----------
            image : np.ndarray
                The input image to be segmented.
            mask : np.ndarray
                The predicted segmentation mask.
        """
        annotated_image = self.generate_annotated_image(image.copy(), mask)
        fig, axs = plt.subplots(1, 2, figsize=(16, 12))
        axs[0].set_title('Original Image')
        axs[1].set_title('Annotation Mask')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[0].imshow(image)
        axs[1].imshow(annotated_image)

        if legend:
            unique_labels = np.unique(mask)
            patches = [
                mpatches.Patch(
                    color=np.array(PALLETE[label])/255,
                    label=CATEGORY_ID.get(label, f'Class {label}')
                )
                for label in unique_labels if label in CATEGORY_ID]
            axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the segmentation model and move it to the appropriate device."""
        raise NotImplementedError("Method not implemented")
    
    @abstractmethod
    def _preprocess_image(self, image:np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess the input image using the defined transformations.
        
        Parameters
        ----------
            image : np.ndarray
                The input image to be preprocessed.

        Returns
        -------
            Union[np.ndarray, torch.Tensor]
                The preprocessed image tensor.
        """
        raise NotImplementedError("Method not implemented")
    
    @abstractmethod
    def predict_segmentation_mask(self, image:np.ndarray) -> np.ndarray:
        """
        Predict the segmentation mask for the given image using the model.
        
        Parameters
        ----------
            image : np.ndarray
                The input image for which the segmentation mask is to be predicted.

        Returns
        -------
            np.ndarray
                The predicted segmentation mask.
        """
        raise NotImplementedError("Method not implemented")
