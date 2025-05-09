import torch
import numpy as np
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

from src.utils.segmentation import compute_iou

class SAMModel:
    def __init__(self, model_path:str="src/model/sam_vit_b_01ec64.pth", model_type:str="vit_b"):
        """
        Initialize the SAMModel with the given model path and type.

        Parameters
        ----------
            model_path : str 
                Path to the SAM model weights file.
            model_type : str 
                Type of the SAM model (e.g., "vit_b", "vit_l", "vit_h").
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    def _load_model(self):
        """
        Load the SAM model from the specified path and move it
        to the appropriate device.
        """
        self.model = sam_model_registry[self.model_type](checkpoint=self.model_path)
        self.model.to(device=self.device)

    def _merge_similar_masks(self, masks:list[dict], iou_threshold:float=0.9):
        """
        Merge similar masks based on the Intersection over Union (IoU) threshold.

        Parameters
        ----------
            masks : list[dict] 
                List of masks to be merged.
            iou_threshold : float 
                IoU threshold for merging masks.

        Returns
        -------
            list[dict] 
                List of merged masks.
        """
        filtered_sam_results = []
        for mask_info in masks:
            is_duplicated = False
            for unique_mask in filtered_sam_results:
                iou = compute_iou(mask_info["segmentation"], unique_mask["segmentation"])
                if iou > iou_threshold:
                    is_duplicated = True
                    break
            if not is_duplicated:
                filtered_sam_results.append(mask_info)
        return filtered_sam_results

    def generate_masks(self, image:np.ndarray, merge_similar:bool=False) -> list[dict]:
        """
        Generate masks for the given image using the SAM model.

        Parameters
        ----------
            image : np.ndarray 
                Input image in the form of a NumPy array.
            merge_similar : bool
                Whether to merge similar masks based on IoU.

        Returns
        -------
            list[dict] 
                List of dictionaries containing mask information.
        """
        masks = self.mask_generator.generate(image)
        if merge_similar:
            return self._merge_similar_masks(masks) 
        return masks
    
    def preview_image_segmentation(self, image:np.ndarray, merge_similar:bool=False) -> None:
        """
        Preview the segmentation of the given image using the SAM model.

        Parameters
        ----------
            image : np.ndarray 
                Input image in the form of a NumPy array.
            merge_similar : bool
                Whether to merge similar masks based on IoU.
        """
        segmentation_masks = self.generate_masks(image.copy(), merge_similar)
        detections = sv.Detections.from_sam(segmentation_masks)
        annotated_image = self.mask_annotator.annotate(image.copy(), detections)
        sv.plot_images_grid(
            images=[image, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )