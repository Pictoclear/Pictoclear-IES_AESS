"""
Create a grid comparison showing all input and enhanced images at once.

This creates a single image with:
Row 1: All input images
Row 2: All enhanced images
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def create_grid_comparison(input_dir='input', output_dir='outputs/enhanced', grid_file='outputs/comparisons/comparison_grid.jpg'):
    """Create a grid comparison with all images."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all input images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    input_images = sorted([f for f in input_path.iterdir() 
                          if f.suffix.lower() in image_extensions])
    
    if not input_images:
        print(f"No images found in '{input_dir}'!")
        return
    
    print(f"Creating grid comparison for {len(input_images)} images...")
    
    # Load all image pairs
    pairs = []
    for img_file in input_images:
        enhanced_file = output_path / f"enhanced_{img_file.name}"
        
        if not enhanced_file.exists():
            print(f"⚠️  Skipping {img_file.name} - no enhanced version found")
            continue
        
        try:
            input_img = Image.open(img_file).convert('RGB')
            enhanced_img = Image.open(enhanced_file).convert('RGB')
            
            # Resize enhanced to match input
            if input_img.size != enhanced_img.size:
                enhanced_img = enhanced_img.resize(input_img.size, Image.LANCZOS)
            
            pairs.append((input_img, enhanced_img, img_file.name))
        except Exception as e:
            print(f"✗ Error loading {img_file.name}: {e}")
    
    if not pairs:
        print("No valid image pairs found!")
        return
    
    # Resize all images to same width for grid (keep aspect ratio)
    target_width = 400
    resized_pairs = []
    
    for input_img, enhanced_img, name in pairs:
        ratio = target_width / input_img.width
        new_height = int(input_img.height * ratio)
        
        input_resized = input_img.resize((target_width, new_height), Image.LANCZOS)
        enhanced_resized = enhanced_img.resize((target_width, new_height), Image.LANCZOS)
        
        resized_pairs.append((input_resized, enhanced_resized, name, new_height))
    
    # Calculate grid dimensions
    num_images = len(resized_pairs)
    max_height = max(h for _, _, _, h in resized_pairs)
    
    # Create grid: 2 rows (input, enhanced) x N columns
    grid_width = target_width * num_images
    grid_height = max_height * 2 + 60  # 60px for labels
    
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    # Add images to grid
    for i, (input_img, enhanced_img, name, height) in enumerate(resized_pairs):
        x_offset = i * target_width
        
        # Top row: Input images
        grid.paste(input_img, (x_offset, 30))
        
        # Bottom row: Enhanced images
        grid.paste(enhanced_img, (x_offset, max_height + 30))
    
    # Add labels
    try:
        # Try to use a font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 5), "INPUT IMAGES ↓", fill='black', font=font)
    draw.text((10, max_height + 35), "ENHANCED IMAGES ↓", fill='black', font=font)
    
    # Add separator line
    draw.line([(0, max_height + 25), (grid_width, max_height + 25)], 
             fill='gray', width=2)
    
    # Save grid
    grid.save(grid_file, quality=95)
    print(f"\n{'='*60}")
    print(f"✓ Created grid comparison: {grid_file}")
    print(f"  Size: {grid.size}")
    print(f"  Images: {num_images}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create grid comparison image')
    parser.add_argument('--input_dir', type=str, default='input',
                       help='Directory with input images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory with enhanced images')
    parser.add_argument('--grid_file', type=str, default='comparison_grid.jpg',
                       help='Output grid file name')
    
    args = parser.parse_args()
    create_grid_comparison(args.input_dir, args.output_dir, args.grid_file)
