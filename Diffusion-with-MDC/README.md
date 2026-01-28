# Diffusion-with-MDC

A diffusion-based super-resolution framework with Multi-Domain Constraints (MDC) for high-field MRI image super-resolution.

## Features

- **Latent Diffusion Model**: Efficient diffusion process in continuous latent space
- **VQGAN Encoder/Decoder**: High-quality image reconstruction using discrete codebook
- **Multi-Domain Constraints**: 
  - Frequency-domain loss based on FFT
  - QR decomposition-based structural loss
- **Fast Inference**: Only 15 diffusion steps required

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (recommended)
- PyTorch 1.12+

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Inference

1. **Download Pre-trained Models**

   - Main model: Trained model weights should be placed in `weights/` directory
   - VQGAN autoencoder: Will be automatically downloaded 

2. **Run Inference**

   ```bash
   python inference_folder.py -i <input_image_or_folder> -o <output_folder> --scale 4 --steps 15
   ```

   **Arguments:**
   - `-i, --in_path`: Input image or folder path
   - `-o, --out_path`: Output folder path (default: `./results`)
   - `-s, --steps`: Number of diffusion steps (default: 15)
   - `--scale`: Super-resolution scale factor (default: 4)
   - `--seed`: Random seed (default: 12345)
   - `--chop_size`: Chop size for processing large images, choices: [512, 256] (default: 512)

   **Example:**
   ```bash
   python inference_folder.py -i ./testdata -o ./results --scale 4 --steps 15
   ```

### Training

1. **Prepare Data**

   - Modify `configs/Diffusion-with-MDC.yaml`:
     - Update `data.train.params.txt_file_path` with your training data paths
     - Update `data.val.params.dir_path` with your validation data path

2. **Run Training**

   ```bash
   python main.py --cfg_path configs/Diffusion-with-MDC.yaml --save_dir ./logs
   ```

   **Arguments:**
   - `--cfg_path`: Path to config file (default: `./configs/Diffusion-with-MDC.yaml`)
   - `--save_dir`: Directory to save checkpoints and logs (default: `./saved_logs`)
   - `--resume`: Resume from checkpoint or save_dir
   - `--steps`: Number of diffusion steps (default: 15)

## Configuration

Edit `configs/Diffusion-with-MDC.yaml` to customize:

- **Data paths**: Modify `data.train.params.txt_file_path` and `data.val.params.dir_path`
- **Model parameters**: Adjust model architecture in `model.params`
- **Training settings**: Modify learning rate, batch size, etc. in `train` section
- **Diffusion parameters**: Change diffusion steps, schedule, etc. in `diffusion.params`

## Project Structure

```
Diffusion-with-MDC/
├── configs/              # Configuration files
├── models/               # Model definitions
├── ldm/                  # Latent diffusion modules
├── utils/                # Utility functions
├── basicsr/              # BasicSR utilities
├── datapipe/             # Data pipeline
├── weights/              # Model weights (not included)
├── inference_folder.py   # Inference script
├── main.py               # Training script
└── requirements.txt      # Dependencies
```

## Acknowledgments

This project is developed based on [ResShift](https://github.com/zsyOAOA/ResShift). Thanks for their excellent work.

