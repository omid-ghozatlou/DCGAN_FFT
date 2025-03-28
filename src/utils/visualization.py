"""
Visualization utilities for training metrics and generated images.
"""

import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from typing import List, Dict
import numpy as np

def plot_metrics(
    metrics: Dict[str, List[float]],
    save_dir: str,
    title: str = "Training Metrics"
) -> None:
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary containing metric values
        save_dir: Directory to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        if metric_name not in ['epoch', 'timestamp']:
            plt.plot(metrics['epoch'], values, label=metric_name)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

def visualize_generated_images(
    generator: torch.nn.Module,
    num_images: int,
    device: torch.device,
    save_path: str,
    nrow: int = 5
) -> None:
    """
    Generate and visualize images from the generator.
    
    Args:
        generator: Generator model
        num_images: Number of images to generate
        device: Device to run generation on
        save_path: Path to save the generated images
        nrow: Number of images per row in the grid
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_images, generator.latent_dim, device=device)
        # Generate images
        gen_imgs = generator(z)
        # Convert to spatial domain
        spatial = batch_inverse_fft(gen_imgs.cpu())
        
        # Create grid of images
        grid = make_grid(spatial, nrow=nrow, normalize=True)
        
        # Save grid
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_fft_components(
    img: torch.Tensor,
    save_path: str
) -> None:
    """
    Plot original image and its FFT components.
    
    Args:
        img: Input image tensor
        save_path: Path to save the plot
    """
    # Transform to frequency domain
    magnitude, phase = fft_transform(img)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(img.cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot magnitude spectrum
    axes[1].imshow(magnitude.cpu().numpy(), cmap='gray')
    axes[1].set_title('Magnitude Spectrum')
    axes[1].axis('off')
    
    # Plot phase spectrum
    axes[2].imshow(phase.cpu().numpy(), cmap='gray')
    axes[2].set_title('Phase Spectrum')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 