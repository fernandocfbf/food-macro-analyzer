import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.segment_anything_model import SAMModel
from src.b3_seg_former_model import SegFormerModel

from src.utils.visualization import generate_pallete_for_mask, apply_mask_on_image
from src.constants.category_id import PALLETE, CATEGORY_ID

class EnhancedModel:
    """
    Combines SegFormer and SAM outputs to generate enhanced segmentation masks.
    """
    def __init__(self, segformer_model_path:str="src/model/segformer-b3-finetuned-foodseg103", sam_model_type:str="vit_h") -> None:
        self.segformer_model = self._load_segformer_model(segformer_model_path)
        self.sam_model = self._load_sam_model(sam_model_type)

    def _load_segformer_model(self, segformer_model_path) -> SegFormerModel:
        """
        Load the SegFormer model.

        Parameters
        ----------
        segformer_model_path : str
            Path to the SegFormer model.

        Returns
        -------
        SegFormerModel
        """
        return SegFormerModel(segformer_model_path)

    def _load_sam_model(self, sam_model_type) -> SAMModel:
        """
        Load the SAM model.

        Parameters
        ----------
        sam_model_type : str
            Identifier for the SAM model type.

        Returns
        -------
        SAMModel
        """
        return SAMModel(sam_model_type)
    
    def _sort_sam_masks(self, sam_masks_list:list[dict]) -> list[np.ndarray]:
        """
        Sort SAM masks by area in descending order.

        Parameters
        ----------
        sam_masks_list : list of dict
            List of SAM masks with segmentation and area.

        Returns
        -------
        list of np.ndarray
            Sorted list of binary segmentation masks.
        """
        return [
            mask['segmentation']
            for mask
            in sorted(sam_masks_list, key=lambda x: x['area'], reverse=True)
        ]
    
    def _enhance_segformer_mask(self, segformer_mask:np.ndarray, sam_masks_list:list[np.ndarray]) -> np.ndarray:
        """
        Enhance SegFormer segmentation using SAM masks.

        Parameters
        ----------
        segformer_mask : np.ndarray
            Raw SegFormer prediction with class IDs.
        sam_masks_list : list of np.ndarray
            Binary masks from SAM sorted by area.

        Returns
        -------
        np.ndarray
            Enhanced segmentation mask.
        """
        base_mask = np.zeros_like(segformer_mask)
        for sam_segmentation_mask in sam_masks_list:
            sam_segmentation_mask = np.copy(sam_segmentation_mask).astype(bool)
            image_target_region = segformer_mask[sam_segmentation_mask]
            class_ids, counts = np.unique(image_target_region, return_counts=True)
            dominant_class = class_ids[np.argmax(counts)]
            base_mask[sam_segmentation_mask] = dominant_class
        return base_mask
    
    def predict_segmentation_mask(self, image:np.ndarray) -> np.ndarray:
        """
        Predict enhanced segmentation mask from input image.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image.

        Returns
        -------
        np.ndarray
            Final enhanced segmentation mask.
        """
        segformer_mask = self.segformer_model.predict_segmentation_mask(image.copy())
        sam_masks_list = self.sam_model.predict_segmentation_mask(image.copy())
        sam_masks_list_sorted = self._sort_sam_masks(sam_masks_list)
        return self._enhance_segformer_mask(segformer_mask, sam_masks_list_sorted)
    
    def generate_annotated_image(self, image:np.ndarray, mask:np.ndarray) -> np.ndarray:
        color_seg = generate_pallete_for_mask(mask)
        return apply_mask_on_image(image, color_seg)
    
    def preview_full_pipeline(self, image:np.ndarray) -> None:
        original_image = image.copy()
        segformer_mask = self.segformer_model.predict_segmentation_mask(image.copy())
        segformer_annotation = self.segformer_model.generate_annotated_image(image.copy(), segformer_mask)

        sam_masks_list = self.sam_model.predict_segmentation_mask(image.copy())
        sam_annotation = self.sam_model.generate_annotated_image(image.copy(), sam_masks_list)

        sam_masks_list_sorted = self._sort_sam_masks(sam_masks_list)
        enhanced_mask = self._enhance_segformer_mask(segformer_mask, sam_masks_list_sorted)
        enhanced_annotation = self.generate_annotated_image(image.copy(), enhanced_mask)

        fig, axs = plt.subplots(1,4, figsize=(20, 10), layout="constrained")

        axs[0].set_title('Original Image')
        axs[1].set_title('Segformer Annotation Mask')
        axs[2].set_title('SAM Annotation Mask')
        axs[3].set_title('Enhanced Annotation Mask')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')
        axs[0].imshow(original_image)
        axs[1].imshow(segformer_annotation)
        axs[2].imshow(sam_annotation)
        axs[3].imshow(enhanced_annotation)
        segformer_unique_labels = np.unique(segformer_mask)
        patches = [
            mpatches.Patch(
                color=np.array(PALLETE[label])/255,
                label=CATEGORY_ID.get(label, f'Class {label}')
            )
            for label in segformer_unique_labels if label in CATEGORY_ID]
        axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        enhanced_unique_labels = np.unique(enhanced_mask)
        patches = [
            mpatches.Patch(
                color=np.array(PALLETE[label])/255,
                label=CATEGORY_ID.get(label, f'Class {label}')
            )
            for label in enhanced_unique_labels if label in CATEGORY_ID]
        axs[3].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')


    def preview_image_segmentation(self, image:np.ndarray, mask:np.ndarray) -> None:
        """
        Display the input image alongside its predicted segmentation.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        mask : np.ndarray
            Segmentation mask with class IDs.
        """
        original_image = image.copy()
        blend = self.generate_annotated_image(image, mask)
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
