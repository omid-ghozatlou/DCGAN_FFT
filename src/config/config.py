"""
Configuration file for DCGAN with FFT.
Contains all hyperparameters and settings for the model and training.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    n_epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    n_cpu: int = 8
    sample_interval: int = 50
    validation_interval: int = 10
    checkpoint_interval: int = 20
    early_stopping_patience: int = 10

@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""
    latent_dim: int = 100
    img_size: int = 64
    channels: int = 2
    init_size: Optional[int] = None

    def __post_init__(self):
        if self.init_size is None:
            self.init_size = self.img_size // 4

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_path: str = "./data"
    img_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    validation_split: float = 0.1
    categories: list = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = ['a']  # Default category

@dataclass
class LoggingConfig:
    """Logging and visualization configuration."""
    log_dir: str = "logs"
    tensorboard: bool = True
    save_images: bool = True
    save_checkpoints: bool = True
    plot_metrics: bool = True

@dataclass
class Config:
    """Main configuration class combining all settings."""
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    use_cuda: bool = True  # Will be set based on availability
    output_dir: str = "imgs/FFT"

# Create a default configuration instance
config = Config() 