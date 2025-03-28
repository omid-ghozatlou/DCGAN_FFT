"""
Data loading utilities for the DCGAN FFT project.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.transform import resize
from typing import List, Tuple, Optional

class ImageDataset(Dataset):
    """
    Custom dataset for loading and preprocessing images.
    """
    
    def __init__(self, data_path: str, categories: List[str], img_size: int = 64):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            categories: List of category names
            img_size: Target size for images
        """
        self.img_size = img_size
        self.images = []
        self.labels = []
        
        for category in categories:
            print(f'Loading category: {category}')
            path = os.path.join(data_path, category)
            
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img_array = imread(img_path)
                img_resized = resize(img_array, (img_size, img_size))
                
                # Convert to grayscale if needed
                if len(img_resized.shape) == 3:
                    img_resized = img_resized[:, :, 0]  # Take first channel
                
                self.images.append(img_resized)
                self.labels.append(categories.index(category))
            
            print(f'Loaded category: {category} successfully')
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f'Dataset shape: {self.images.shape}')
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Preprocessed image tensor
        """
        img = self.images[idx]
        img = torch.FloatTensor(img)
        img = (img - 0.5) * 2  # Normalize to [-1, 1]
        return img

def create_dataloader(
    data_path: str,
    categories: List[str],
    batch_size: int = 128,
    img_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        data_path: Path to the data directory
        categories: List of category names
        batch_size: Batch size for the DataLoader
        img_size: Target size for images
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster data transfer to GPU
        
    Returns:
        DataLoader instance
    """
    dataset = ImageDataset(data_path, categories, img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader 