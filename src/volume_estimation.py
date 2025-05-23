import open3d as o3d
import numpy as np
import cv2

class VolumeEstimator:
    def __init__(self, depth_layers:int=20, voxel_volume:int=1):
        self.depth_layers = depth_layers
        self.voxel_volume = voxel_volume

    def _process_image(self, image:np.ndarray) -> np.ndarray:
        """
        Process the input image to convert it to grayscale and normalize it.
        
        Parameters
        ----------
        image : np.ndarray
            The input image in BGR format.
            
        Returns
        -------
        np.ndarray
            The processed grayscale image.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def estimate_volume(self, image:np.ndarray) -> float:
        """
        Estimate the volume of an object in the image using a depth-based approach.

        Parameters
        ----------
        image : np.ndarray
            The input image in BGR format.

        Returns 
        -------
        float
            The estimated volume of the object in cubic units.
        """
        processed_image = self._process_image(image)
        normalized = processed_image.astype(float) / 255.0
        volume = np.zeros((processed_image.shape[0], processed_image.shape[1], self.depth_layers), dtype=float)
        for z in range(self.depth_layers):
            depth_factor = 1.0 - (z / self.depth_layers)
            volume[:, :, z] = normalized * depth_factor
        return round(volume.sum() * self.voxel_volume, 2)
    

    



