"""
Visualize progressive denoising from blurred to clear image.

Shows the image enhancement in multiple steps to see the gradual improvement.
This simulates what happens during the enhancement process.
"""

import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
import numpy as np
import argparse
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models.generator import GeneratorRGB_Fixed


def progressive_denoise(image_path, weights_path, output_dir='outputs/visualizations', num_steps=8, device='cuda'):
    """Show progressive enhancement from blurred to clear."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {weights_path}...")
    model = GeneratorRGB_Fixed(in_channels=3, out_channels=3)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load and prepare image
    print(f"Processing {image_path}...")
    img = Image.open(image_path).convert('RGB')
    
    # Resize to be divisible by 16
    def make_divisible(size, divisor=16):
        w, h = size
        new_w = (w // divisor) * divisor
        new_h = (h // divisor) * divisor
        return max(new_w, divisor), max(new_h, divisor)
    
    max_size = 512
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    target_size = make_divisible(img.size, 16)
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Get enhanced output
    with torch.no_grad():
        enhanced = model(img_tensor)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating {num_steps} progressive steps from blurred to clear...")
    
    # Create progressive blend from input to output
    steps = []
    for i in range(num_steps + 1):
        alpha = i / num_steps
        # Blend between input (blurred) and output (clear)
        blended = (1 - alpha) * img_tensor + alpha * enhanced
        steps.append((f"Step_{i:02d}_({int(alpha*100)}%)", blended.clamp(0, 1)))
    
    # Save individual steps
    for name, tensor in steps:
        save_image(tensor, f'{output_dir}/{name}.png')
    
    # Create side-by-side comparison
    print("\nCreating comparison images...")
    
    # Grid showing progression
    create_progression_grid(steps, output_dir)
    
    # Create animated-style strip
    create_progression_strip(steps, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Progressive denoising visualization complete!")
    print(f"  Individual steps: {output_dir}/Step_XX_*.png")
    print(f"  Grid view: {output_dir}/progression_grid.png")
    print(f"  Strip view: {output_dir}/progression_strip.png")
    print(f"{'='*60}")
    print(f"\nThe progression shows:")
    print(f"  Step 00 (0%)   → Original blurred input")
    print(f"  Step XX (XX%)  → Gradual enhancement")
    print(f"  Step {num_steps:02d} (100%) → Final clear output")


def create_progression_grid(steps, output_dir):
    """Create a grid showing the progression."""
    
    num_steps = len(steps)
    cols = 3
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('Progressive Enhancement: Blurred → Clear', 
                 fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, tensor) in enumerate(steps):
        row = idx // cols
        col = idx % cols
        
        img = tensor[0].cpu().permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(name.replace('_', ' '), fontsize=12)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(steps), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/progression_grid.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_progression_strip(steps, output_dir):
    """Create a horizontal strip showing the progression."""
    
    fig, axes = plt.subplots(1, len(steps), figsize=(3*len(steps), 4))
    fig.suptitle('Progressive Denoising', fontsize=14, fontweight='bold')
    
    for idx, (name, tensor) in enumerate(steps):
        img = tensor[0].cpu().permute(1, 2, 0).numpy()
        
        axes[idx].imshow(img)
        # Extract percentage from name
        percentage = name.split('(')[1].split('%')[0] if '(' in name else str(idx)
        axes[idx].set_title(f'{percentage}%', fontsize=11)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/progression_strip.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize progressive denoising')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--weights', type=str, default='weights/generator_epoch_2.pth',
                       help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                       help='Directory to save visualization')
    parser.add_argument('--steps', type=int, default=8,
                       help='Number of intermediate steps to show (default: 8)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights not found at {args.weights}")
        return
    
    progressive_denoise(args.image, args.weights, args.output_dir, args.steps)


if __name__ == '__main__':
    main()
