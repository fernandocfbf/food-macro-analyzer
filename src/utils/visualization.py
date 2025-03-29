import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

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