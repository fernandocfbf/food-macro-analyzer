import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cv2
import random
import numpy as np
import torch

from src.utils.dataset import decode_image_from_bytes
from src.constants.category_id import CATEGORY_ID, PALLETE

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
                
def predict_mask(image, model, transform, device:str="cuda"):
    original_size = image.shape[:2]
    image_transformed = transform(image=image)["image"]
    input_tensor = image_transformed.float().unsqueeze(0)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size,
        mode="bilinear",
        align_corners=False
    )
    predicted_mask = upsampled_logits.argmax(dim=1)[0] 
    predicted_mask = predicted_mask.cpu().numpy()
    return predicted_mask

def _generate_pallete_for_mask(mask):
    color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in PALLETE.items():
        color_seg[mask == label, :] = color
    return color_seg

def apply_mask_on_image(image, mask, alpha=0.5):
    imagem_original_float = image.astype(np.float32) / 255.0
    imagem_mascara_float = mask.astype(np.float32) / 255.0
    imagem_overlay = (1 - alpha) * imagem_original_float + alpha * imagem_mascara_float
    imagem_overlay = (imagem_overlay * 255).astype(np.uint8)
    return imagem_overlay

def predict_random_images(model, transform, validation_dataset, num_images=5, mask_alpha=0.5, device="cuda") -> None:
    random_images_idx_list = random.sample(range(0, validation_dataset.__len__()), num_images) 
    for idx in random_images_idx_list:
        image = decode_image_from_bytes(validation_dataset.image_dataset.loc[idx]["image"])
        mask = predict_mask(image, model, transform, device)
        color_seg = _generate_pallete_for_mask(mask)
        blend = apply_mask_on_image(image, color_seg)
        fig, axs = plt.subplots(1, 3, figsize=(16, 12))

        axs[0].set_title('Original Image')
        axs[1].set_title('Annotation Mask')
        axs[2].set_title('Combined Image')

        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')

        axs[0].imshow(image)
        axs[1].imshow(color_seg)
        axs[2].imshow(blend)

        unique_labels = np.unique(mask)
        patches = [
            mpatches.Patch(
                color=np.array(PALLETE[label])/255,
                label=CATEGORY_ID.get(label, f'Class {label}')
            )
            for label in unique_labels if label in CATEGORY_ID]
        axs[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')