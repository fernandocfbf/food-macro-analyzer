import pandas as pd
from typing import Optional
import numpy as np
from PIL import Image
import io
from datasets import load_dataset, concatenate_datasets, DatasetDict


def load_foodseg103_splits(sample_size:int=None, random_state:Optional[int]=42) -> pd.DataFrame:
    """
    Loads the FoodSeg103 dataset as a Pandas DataFrame.

    Parameters
    ----------
    sample_size : int, optional
        The number of samples to return. If None, all samples are returned.
    random_state : int, optional
        Random seed for reproducibility when sampling.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the requested subset of the FoodSeg103 dataset.
    """
    food_seg_103 = load_dataset("EduardoPacheco/FoodSeg103")
    splits = [ds for ds in food_seg_103.values()]
    merged_dataset = concatenate_datasets(splits)
    merged_dataset = merged_dataset.shuffle(seed=random_state)

    train_test = merged_dataset.train_test_split(test_size=0.3, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)

    train_dataset = train_test['train']        
    val_dataset = val_test['train']           
    test_dataset = val_test['test']           
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

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