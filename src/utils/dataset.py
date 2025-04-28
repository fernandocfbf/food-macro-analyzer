import pandas as pd
import numpy as np
from PIL import Image
import io
from datasets import load_dataset

def load_foodseg103(type:str="all", sample_size:int=None, random_state=42) -> pd.DataFrame:
    """
    Loads the FoodSeg103 dataset as a Pandas DataFrame.

    Parameters
    ----------
    type : str, optional
        Specifies which subset of the dataset to load. Options:
        - "train": Loads only the training set.
        - "validation": Loads only the validation set.
        - "all" (default): Loads both training and validation sets, concatenated into a single DataFrame.
    
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the requested subset of the FoodSeg103 dataset.
    
    Raises
    ------
    ValueError
        If an invalid `type` is provided.
    """
    valid_types = {"train", "validation", "all"}
    if type not in valid_types:
        raise ValueError(f"Invalid type '{type}'. Expected one of: {valid_types}")
    food_seg_103 = load_dataset("EduardoPacheco/FoodSeg103")
    if type == "all":
        train_df = food_seg_103["train"].to_pandas()
        validation_df = food_seg_103["validation"].to_pandas()
        image_dataset = pd.concat([train_df, validation_df])
    else:
        image_dataset = food_seg_103[type].to_pandas() 
    
    return image_dataset if sample_size is None else image_dataset.sample(sample_size, random_state=random_state).reset_index(drop=True)

def decode_image_from_bytes(byte_data: dict) -> np.ndarray:
    """
    Decodes a NumPy array from a byte-encoded image.

    Parameters
    ----------
    byte_data : dict
        A dictionary containing the key 'bytes' with the image data in byte format.

    Returns
    -------
    np.ndarray
        A NumPy array representing the decoded image.
    """
    image_bytes = byte_data["bytes"]
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)