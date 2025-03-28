# DCGAN with FFT Image Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) that generates images in the frequency domain using Fast Fourier Transform (FFT). The generator creates FFT representations of images, which are then converted back to the spatial domain for visualization.

## Features

- DCGAN implementation for FFT image generation
- Support for both magnitude and phase components of FFT
- Automatic conversion between frequency and spatial domains
- Configurable training parameters
- Progress visualization during training
- Model checkpointing and early stopping
- Comprehensive logging and metrics tracking
- Validation during training
- Example notebooks for visualization

## Project Structure

```
.
├── src/                    # Source code
│   ├── config/            # Configuration files
│   ├── models/            # Model architectures
│   └── utils/             # Utility functions
├── examples/              # Example notebooks
├── data/                  # Dataset directory
├── logs/                  # Training logs and checkpoints
├── imgs/                  # Generated images
├── setup.py              # Package installation
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DCGAN_FFT.git
cd DCGAN_FFT
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

## Usage

### Training

1. Prepare your dataset in the `data` directory
2. Configure training parameters in `src/config/config.py` or use command line arguments
3. Run the training script:

```bash
python src/train.py
```

### Training Parameters

- `--n_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.0002)
- `--latent_dim`: Dimension of latent space (default: 100)
- `--img_size`: Size of generated images (default: 64)
- `--channels`: Number of image channels (default: 2)

### Example Notebooks

Check the `examples` directory for Jupyter notebooks demonstrating:

- Basic usage and training
- Visualization of FFT components
- Model evaluation and inference

## Results

Generated images are saved in the `imgs/FFT` directory during training. The images are generated in the frequency domain and automatically converted to the spatial domain for visualization.

Training metrics and checkpoints are saved in the `logs` directory:

- `metrics.json`: Training and validation metrics
- `checkpoints/`: Model checkpoints
- `metrics.png`: Visualization of training metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dcgan_fft2024,
  author = {Your Name},
  title = {DCGAN with FFT Image Generation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/DCGAN_FFT}
}
```
