import numpy as np
from PIL import Image
import io

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
