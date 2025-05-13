import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abc import ABC, abstractmethod

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
        """
        color_seg = generate_pallete_for_mask(mask)
        return apply_mask_on_image(image, color_seg)
    
    def preview_image_segmentation(self, image:np.ndarray, mask:np.ndarray) -> None:
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
        raise NotImplementedError("")
    
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
        raise NotImplementedError("")
