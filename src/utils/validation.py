"""
Validation utilities for model evaluation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
from tqdm import tqdm

def validate(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    adversarial_loss: nn.Module
) -> Dict[str, float]:
    """
    Validate the model on the validation dataset.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        val_dataloader: Validation dataloader
        device: Device to run validation on
        adversarial_loss: Loss function
        
    Returns:
        Dictionary containing validation metrics
    """
    generator.eval()
    discriminator.eval()
    
    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for imgs in tqdm(val_dataloader, desc="Validating"):
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
            fake = torch.zeros(batch_size, 1, device=device, requires_grad=False)
            
            # Transform images to frequency domain
            fft_pairs = batch_fft_transform(imgs)
            fft_pairs = fft_pairs.to(device)
            
            # Generate images
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            gen_imgs = generator(z)
            
            # Calculate losses
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            real_loss = adversarial_loss(discriminator(fft_pairs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1
    
    # Calculate average losses
    avg_d_loss = total_d_loss / num_batches
    avg_g_loss = total_g_loss / num_batches
    
    return {
        "d_loss": avg_d_loss,
        "g_loss": avg_g_loss
    }

def early_stopping(
    val_losses: list,
    patience: int
) -> bool:
    """
    Check if training should be stopped early.
    
    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait before stopping
        
    Returns:
        True if training should be stopped, False otherwise
    """
    if len(val_losses) < patience:
        return False
    
    # Check if the last 'patience' losses are worse than the best loss
    best_loss = min(val_losses[:-patience])
    return all(loss > best_loss for loss in val_losses[-patience:]) 