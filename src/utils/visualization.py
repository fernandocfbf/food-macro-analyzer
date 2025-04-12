import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random
import numpy as np
import torch

from src.utils.dataset import load_foodseg103, decode_image_from_bytes
from src.constants.category_id import CATEGORY_ID

def show_image_preview(dataset: pd.DataFrame, image_id:int) -> None:
    """
    Displays the original image, its annotation mask, and a labeled version with class names.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing images and their corresponding annotation masks.
        Must include the columns "image_decoded" (original image) and "annotation_mask" (segmentation mask).
    
    image_id : int
        Index of the image in the DataFrame to be displayed.
    
    Returns
    -------
    None
        The function displays the images using Matplotlib but does not return any values.
    
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].set_title('Original Image')
    axs[1].set_title('Annotation Mask')
    axs[2].set_title('Original Image with Labels')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    unique_labels = np.unique(dataset.loc[image_id, 'annotation_mask'])
    example_info = dataset.loc[image_id]
    original_image = example_info["image_decoded"]
    annotation_mask = example_info["annotation_mask"]
    label_array = np.array(annotation_mask)
    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[1].imshow(annotation_mask, cmap="tab20")
    axs[1].axis("off")
    axs[2].imshow(annotation_mask, cmap="tab20")
    axs[2].axis("off")
    for lbl in unique_labels:
        if lbl != 0: #ignore background
            mask = (label_array == lbl).astype(np.uint8)
            M = cv2.moments(mask)
            if M["m00"] != 0:  # avoid division by zero
                cx = int(M["m10"] / M["m00"])  # centroid x
                cy = int(M["m01"] / M["m00"])  # centroid y
                class_name = f"{CATEGORY_ID.get(lbl, 'Unknown')}"  # Get class name
                axs[2].text(cx, cy, class_name, color="white", fontsize=12, ha="center", va="center", 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def predict_random_images(model, image_processor, validation_dataset, num_images=5, mask_strengh=0.5) -> None:
    for _ in range(num_images):
        image_idx = random.randint(0, len(validation_dataset)-1)
        test_image = decode_image_from_bytes(validation_dataset.image_dataset.loc[image_idx]["image"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = image_processor(test_image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[test_image.shape[0:2]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
        
        color_seg = np.zeros((predicted_segmentation_map.shape[0],
                        predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[predicted_segmentation_map == label, :] = color
        color_seg = color_seg[..., ::-1]

        combined_image = np.array(test_image) * 0.5 + color_seg * mask_strengh
        combined_image = combined_image.astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(12, 8))
        axs[0].set_title('Original Image')
        axs[1].set_title('Annotation Mask')
        axs[2].set_title('Combined Image')

        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')

        axs[0].imshow(test_image)
        axs[1].imshow(predicted_segmentation_map)
        axs[2].imshow(combined_image)