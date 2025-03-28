"""
DCGAN model implementation for FFT image generation.
Contains Generator and Discriminator architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def weights_init_normal(m: nn.Module) -> None:
    """
    Initialize weights using normal distribution.
    
    Args:
        m: Module to initialize weights for
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    """
    Generator network for DCGAN.
    Generates FFT representations of images from random noise.
    """
    
    def __init__(self, config):
        """
        Initialize the Generator.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super(Generator, self).__init__()
        
        self.init_size = config.model.init_size
        self.l1 = nn.Sequential(
            nn.Linear(config.model.latent_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, config.model.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Input noise tensor
            
        Returns:
            Generated FFT image tensor
        """
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN.
    Discriminates between real and generated FFT images.
    """
    
    def __init__(self, config):
        """
        Initialize the Discriminator.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters: int, out_filters: int, bn: bool = True) -> list:
            """
            Create a discriminator block.
            
            Args:
                in_filters: Number of input filters
                out_filters: Number of output filters
                bn: Whether to use batch normalization
                
            Returns:
                List of layers for the discriminator block
            """
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(config.model.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = config.model.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            img: Input image tensor
            
        Returns:
            Validity score tensor
        """
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def create_models(config) -> Tuple[Generator, Discriminator]:
    """
    Create and initialize generator and discriminator models.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (generator, discriminator)
    """
    generator = Generator(config)
    discriminator = Discriminator(config)
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    return generator, discriminator 