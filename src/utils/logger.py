"""
Logging utility for training metrics and model checkpoints.
"""

import os
import json
import torch
from datetime import datetime
from typing import Dict, Any

class TrainingLogger:
    """Logger for tracking training metrics and saving checkpoints."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs and checkpoints
        """
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize metrics dictionary
        self.metrics = {
            "epoch": [],
            "d_loss": [],
            "g_loss": [],
            "timestamp": []
        }
    
    def log_metrics(self, epoch: int, d_loss: float, g_loss: float) -> None:
        """
        Log training metrics for the current epoch.
        
        Args:
            epoch: Current epoch number
            d_loss: Discriminator loss
            g_loss: Generator loss
        """
        self.metrics["epoch"].append(epoch)
        self.metrics["d_loss"].append(d_loss)
        self.metrics["g_loss"].append(g_loss)
        self.metrics["timestamp"].append(datetime.now().isoformat())
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def save_checkpoint(
        self,
        epoch: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        loss: float
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            generator: Generator model
            discriminator: Discriminator model
            optimizer_G: Generator optimizer
            optimizer_D: Discriminator optimizer
            loss: Current loss value
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            generator: Generator model
            discriminator: Discriminator model
            optimizer_G: Generator optimizer
            optimizer_D: Discriminator optimizer
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(checkpoint_path)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        return checkpoint 