"""
Simple inference script for Mars image enhancement.
Usage: 
    1. Put test images in 'input/' folder
    2. Run: python inference.py
    3. Enhanced images will be saved in 'results/' folder

Similar to RealSRGAN: https://github.com/noor-ahmad-haral/RealSRGAN
"""

import os
import sys
import torch
from PIL import Image
import argparse
from pathlib import Path
import numpy as np

# Add src to path for imports
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from models.generator import GeneratorRGB_Fixed


def load_model(weights_path, device):
    """Load the trained RGB generator model."""
    print(f"Loading model from {weights_path}...")
    model = GeneratorRGB_Fixed(in_channels=3, out_channels=3)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def enhance_image(model, image_path, device):
    """Enhance a single image."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    
    # Prepare for model - need dimensions divisible by 16 for U-Net
    # Calculate size that's divisible by 16
    def make_divisible(size, divisor=16):
        """Make dimensions divisible by divisor."""
        w, h = size
        new_w = (w // divisor) * divisor
        new_h = (h // divisor) * divisor
        # Ensure minimum size
        new_w = max(new_w, divisor)
        new_h = max(new_h, divisor)
        return (new_w, new_h)
    
    # Resize to appropriate size
    max_size = 512
    if max(img.size) > max_size:
        # Maintain aspect ratio
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Make divisible by 16
    target_size = make_divisible(img.size, 16)
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Enhance
    with torch.no_grad():
        enhanced = model(img_tensor)
    
    # Convert back to image
    enhanced = enhanced.squeeze(0).cpu().clamp(0, 1)
    enhanced = (enhanced.permute(1, 2, 0).numpy() * 255).astype('uint8')
    enhanced_img = Image.fromarray(enhanced)
    
    # Resize back to original size
    if enhanced_img.size != original_size:
        enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)
    
    return enhanced_img


def main():
    parser = argparse.ArgumentParser(description='Enhance Mars images')
    parser.add_argument('--input_dir', type=str, default='input', 
                        help='Directory with input images (default: input/)')
    parser.add_argument('--output_dir', type=str, default='outputs/enhanced', 
                        help='Directory for enhanced images (default: outputs/enhanced/)')
    parser.add_argument('--weights', type=str, default='weights/generator_epoch_5.pth',
                        help='Path to model weights')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    parser.add_argument('--create_comparisons', action='store_true',
                        help='Also create side-by-side comparison images')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        print("Please train the model first or specify correct weights path.")
        return
    
    model = load_model(args.weights, device)
    
    # Get input images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' not found!")
        print(f"Please create it and add some test images.")
        return
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in '{args.input_dir}'!")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    # Apply max_images limit
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Results will be saved to '{args.output_dir}/'\n")
    
    # Process images
    import numpy as np
    for idx, img_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing {img_path.name}...", end=' ')
            
            # Enhance
            enhanced_img = enhance_image(model, img_path, device)
            
            # Save result
            output_path = Path(args.output_dir) / f"enhanced_{img_path.name}"
            enhanced_img.save(output_path)
            
            print(f"✓ Saved to {output_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Done! Enhanced images saved to '{args.output_dir}/'")
    print(f"{'='*60}")
    
    # Create comparisons if requested
    if args.create_comparisons:
        print("\nCreating comparison images...")
        comparison_dir = 'comparisons'
        os.makedirs(comparison_dir, exist_ok=True)
        
        comparison_count = 0
        for img_path in image_files[:len(image_files) if not args.max_images else args.max_images]:
            try:
                input_img = Image.open(img_path).convert('RGB')
                enhanced_path = Path(args.output_dir) / f"enhanced_{img_path.name}"
                
                if enhanced_path.exists():
                    enhanced_img = Image.open(enhanced_path).convert('RGB')
                    
                    # Create side-by-side
                    if input_img.size != enhanced_img.size:
                        enhanced_img = enhanced_img.resize(input_img.size, Image.LANCZOS)
                    
                    width, height = input_img.size
                    comparison = Image.new('RGB', (width * 2, height))
                    comparison.paste(input_img, (0, 0))
                    comparison.paste(enhanced_img, (width, 0))
                    
                    # Add separator
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(comparison)
                    draw.line([(width, 0), (width, height)], fill='white', width=2)
                    
                    comparison_file = Path(comparison_dir) / f"comparison_{img_path.name}"
                    comparison.save(comparison_file, quality=95)
                    comparison_count += 1
            except Exception as e:
                pass
        
        print(f"✓ Created {comparison_count} comparison images in '{comparison_dir}/'")


if __name__ == '__main__':
    main()
