import numpy as np
import random
import cv2
from src.segment_enhanced_model import SegmentEnhancedModel
from src.volume_estimation import VolumeEstimator

class MacroNutrientsEstimator:
    def __init__(self, sam_model_type:str="vit_b", depth_layers:int=20, voxel_volume:int=1):
        self.segmentation_model = SegmentEnhancedModel(sam_model_type=sam_model_type)
        self.volume_estimator = VolumeEstimator(depth_layers=depth_layers, voxel_volume=voxel_volume)

    def _get_food_nutrition_info(self, food_id:int) -> dict:
        """
        Placeholder function to get food nutrition information.
        In a real implementation, this would query a database or API.
        """
        nutrition_info = {
            "calories": random.uniform(0.5, 2.5),
            "protein": random.uniform(0.01, 0.2),
            "carbs": random.uniform(0.01, 0.3),
            "fat": random.uniform(0.005, 0.2),
            "density": 0.001
        }
        return nutrition_info

    def estimate_macro_nutrients(self, image:np.ndarray) -> dict:
        mask = self.segmentation_model.predict_segmentation_mask(image)
        self.segmentation_model.preview_image_segmentation(image, mask)
        unique_food_ids = np.unique(mask)
        unique_food_ids = [id for id in unique_food_ids if id != 0]
        total_nutrition = {
            "weight": 0,
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0
        }
        for food_id in unique_food_ids:
            food_mask = np.where(mask == food_id, 1, 0).astype(np.uint8)
            food_mask = cv2.resize(food_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            filtered_image = cv2.bitwise_and(image, image, mask=food_mask)
            volume = self.volume_estimator.estimate_volume(filtered_image)
            nutrition_info = self._get_food_nutrition_info(food_id)
            if nutrition_info:
                weight = volume * nutrition_info["density"]
                print(f"Food ID: {food_id}, Volume: {volume}, Weight: {weight}")
                total_nutrition["weight"] += weight
                total_nutrition["calories"] += nutrition_info["calories"] * weight
                total_nutrition["protein"] += nutrition_info["protein"] * weight
                total_nutrition["carbs"] += nutrition_info["carbs"] * weight
                total_nutrition["fat"] += nutrition_info["fat"] * weight
        return total_nutrition