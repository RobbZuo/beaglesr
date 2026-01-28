#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
import argparse
from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input image or folder path")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output folder path")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Number of diffusion steps")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--output_folder_name", type=str, default="", 
                       help="Output folder name, default uses timestamp")
    parser.add_argument("--img_extensions", type=str, default=".png,.jpg,.jpeg,.tif,.tiff,.bmp", 
                       help="Supported image formats, separated by commas")
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256],
            help="Chop size for processing large images",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="dental",
            choices=['dental'],
            help="Task type",
            )
    return parser.parse_args()

def get_image_files(input_path, extensions):
    """Get all image files with supported formats from the specified path"""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    
    ext_list = extensions.lower().split(',')
    image_files = []
    
    for ext in ext_list:
        if not ext.startswith('.'):
            ext = '.' + ext
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    return sorted(image_files)

def get_configs(args):
    # Load config file based on task type
    if args.task == 'dental':
        configs = OmegaConf.load('./configs/Diffusion-with-MDC.yaml')

    # Prepare model weight files
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    
    ckpt_path = ckpt_dir / f'ema_model_500000.pth'
    
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
        print("Downloading VQGAN autoencoder weights...")
        load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v1.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )

    # Update config
    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # Calculate chop parameters
    if args.chop_size == 512:
        chop_stride = (512 - 64) * (4 // args.scale)
    elif args.chop_size == 256:
        chop_stride = (256 - 32) * (4 // args.scale)
    else:
        raise ValueError("Chop size must be in [512, 256]")
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    # Calculate autoencoder minimum size
    autoencoder_scale = 2 ** (len(configs.autoencoder.params.ddconfig.ch_mult) - 1)
    desired_min_size = 64 * (autoencoder_scale // args.scale)

    return configs, chop_stride, desired_min_size

def main():
    args = get_parser()
    configs, chop_stride, desired_min_size = get_configs(args)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dirname = args.output_folder_name if args.output_folder_name else f"inference_results_{timestamp}"
    out_path = Path(args.out_path) / output_dirname
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    resshift_sampler = ResShiftSampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        desired_min_size=desired_min_size,
    )

    # Get all images to process
    image_files = get_image_files(args.in_path, args.img_extensions)
    
    if len(image_files) == 0:
        print(f"No supported image files found in path: {args.in_path}")
        return

    # Process images
    total_images = len(image_files)
    print(f"Found {total_images} image files")
    print(f"Output directory: {out_path}")

    for idx, img_path in enumerate(image_files, 1):
        print(f"Processing {idx}/{total_images}: {img_path.name}")
        
        # Create corresponding output path
        if Path(args.in_path).is_dir():
            relative_path = img_path.relative_to(Path(args.in_path))
            current_out_path = out_path / relative_path.parent
        else:
            current_out_path = out_path
        current_out_path.mkdir(parents=True, exist_ok=True)

        try:
            # Process image
            resshift_sampler.inference(
                str(img_path), 
                str(current_out_path), 
                bs=1, 
                noise_repeat=False
            )
            print(f"✓ Completed: {img_path.name}")
        except Exception as e:
            print(f"✗ Failed: {img_path.name}, Error: {str(e)}")

    print(f"\nProcessing completed! All results saved to: {out_path}")

if __name__ == '__main__':
    main()
