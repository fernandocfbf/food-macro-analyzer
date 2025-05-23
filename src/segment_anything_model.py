import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

from src.base_model import BaseModel
from src.utils.segmentation import compute_iou

class SAMModel(BaseModel):
    def __init__(self, model_type:str="vit_b", process_mask_size:tuple=(512, 512)):
        """
        Initialize the SAMModel with the given model path and type.

        Parameters
        ----------
            model_type : str 
                Type of the SAM model (e.g., "vit_b", "vit_h").
            process_mask_size : tuple
                Size of the mask to be used on processing (default is (512, 512)).
        """
        self.model_type = model_type
        self.model_path = self._get_model_path(model_type)
        self.process_mask_size = process_mask_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    def _get_model_path(self, model_type:str) -> str:
        """
        Get the model path based on the model type.

        Parameters
        ----------
            model_type : str 
                Type of the SAM model (e.g., "vit_b", "vit_h").
    
        Returns 
        -------
            str 
                Path to the SAM model weights file.
        """
        available_models = {
            "vit_b": "src/model/sam_vit_b_01ec64.pth",
            "vit_h": "src/model/sam_vit_h_4b8939.pth"
        }
        return available_models[model_type]

    def _preprocess_image(self, image:np.ndarray) -> np.ndarray:
        processed_image = cv2.resize(image, self.process_mask_size)
        return processed_image

    def _load_model(self):
        self.model = sam_model_registry[self.model_type](checkpoint=self.model_path).to(device=self.device)

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

    def predict_segmentation_mask(self, image:np.ndarray, merge_similar:bool=False) -> list[dict]:
        image = self._preprocess_image(image)
        masks = self.mask_generator.generate(image)
        if merge_similar:
            return self._merge_similar_masks(masks) 
        return masks
    
    def generate_annotated_image(self, image:np.ndarray, masks_list:list[dict]):
        detections = sv.Detections.from_sam(masks_list)
        return self.mask_annotator.annotate(image, detections)