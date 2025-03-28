"""
Utility functions for FFT operations and transformations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

def fft_transform(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform image to frequency domain using FFT.
    
    Args:
        img: Input image tensor
        
    Returns:
        Tuple of (magnitude, phase) tensors
    """
    fft = torch.fft.fftn(img, dim=(-2, -1))
    magnitude = torch.abs(fft)
    magnitude = (magnitude / magnitude.max() - 0.5) * 2  # Normalize to [-1, 1]
    
    phase = torch.angle(fft)
    phase = phase / torch.pi  # Normalize to [-1, 1]
    
    return magnitude, phase

def inverse_fft(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Transform frequency domain back to spatial domain using inverse FFT.
    
    Args:
        magnitude: Magnitude spectrum tensor
        phase: Phase spectrum tensor
        
    Returns:
        Reconstructed image tensor
    """
    # Denormalize phase and magnitude
    phase = phase * torch.pi
    magnitude = (magnitude + 1) / 2
    
    # Convert polar to rectangular coordinates
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    
    # Create complex tensor and apply inverse FFT
    fourier = torch.view_as_complex(torch.stack((real, imag), dim=-1))
    spatial = torch.fft.ifft2(fourier).real
    
    return spatial

def batch_fft_transform(imgs: torch.Tensor) -> torch.Tensor:
    """
    Transform a batch of images to frequency domain.
    
    Args:
        imgs: Batch of input images [batch_size, channels, height, width]
        
    Returns:
        Tensor containing magnitude and phase for each image [batch_size, 2, height, width]
    """
    batch_size = imgs.shape[0]
    fft_pairs = torch.zeros((batch_size, 2, imgs.shape[2], imgs.shape[3]))
    
    for i in range(batch_size):
        mag, phase = fft_transform(imgs[i])
        fft_pairs[i] = torch.stack((mag, phase), dim=0)
    
    return fft_pairs

def batch_inverse_fft(fft_pairs: torch.Tensor) -> torch.Tensor:
    """
    Transform a batch of frequency domain representations back to spatial domain.
    
    Args:
        fft_pairs: Batch of FFT pairs [batch_size, 2, height, width]
        
    Returns:
        Batch of reconstructed images [batch_size, height, width]
    """
    batch_size = fft_pairs.shape[0]
    spatial = torch.zeros((batch_size, fft_pairs.shape[2], fft_pairs.shape[3]))
    
    for i in range(batch_size):
        spatial[i] = inverse_fft(fft_pairs[i, 0], fft_pairs[i, 1])
    
    return spatial 