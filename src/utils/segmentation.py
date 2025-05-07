import numpy as np

def compute_iou(mask1, mask2) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
        mask1 : np.ndarray
            First binary mask.
        mask2 : np.ndarray
            Second binary mask.

    Returns
    -------
        float
            IoU value between the two masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union