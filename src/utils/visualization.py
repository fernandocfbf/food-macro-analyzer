import matplotlib.pyplot as plt
import pandas as pd

def show_image_preview(dataset: pd.DataFrame, image_id:int) -> None:
    """
    Displays the original image and its corresponding annotation mask side by side.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing images and their annotation masks. 
        It must include the columns "original_image" and "annotation_mask".
    
    image_id : int
        Index of the image in the DataFrame to be displayed.
    
    Returns
    -------
    None
        The function displays the images using Matplotlib but does not return any values.
    """
    image_information = dataset.loc[image_id]
    original_image = image_information["original_image"]
    annotation_mask = image_information["annotation_mask"]
    
    _, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(annotation_mask, cmap="tab20")
    axs[1].set_title("Annotation Mask")
    axs[1].axis("off")