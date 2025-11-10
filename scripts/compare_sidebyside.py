"""
Create side-by-side comparison images: Input | Enhanced Output

This helps visualize the enhancement results easily.
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def create_comparison(input_dir='input', output_dir='outputs/enhanced', comparison_dir='outputs/comparisons'):
    """Create side-by-side comparison images."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    comparison_path = Path(comparison_dir)
    
    # Create comparison directory
    comparison_path.mkdir(exist_ok=True)
    
    # Get all input images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    input_images = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not input_images:
        print(f"No images found in '{input_dir}'!")
        return
    
    print(f"Creating comparisons for {len(input_images)} images...")
    print(f"Comparisons will be saved to '{comparison_dir}/'\n")
    
    created = 0
    for img_file in input_images:
        # Find corresponding enhanced image
        enhanced_file = output_path / f"enhanced_{img_file.name}"
        
        if not enhanced_file.exists():
            print(f"⚠️  Skipping {img_file.name} - no enhanced version found")
            continue
        
        try:
            # Load images
            input_img = Image.open(img_file).convert('RGB')
            enhanced_img = Image.open(enhanced_file).convert('RGB')
            
            # Resize if they don't match (shouldn't happen, but just in case)
            if input_img.size != enhanced_img.size:
                enhanced_img = enhanced_img.resize(input_img.size, Image.LANCZOS)
            
            # Create side-by-side comparison
            width, height = input_img.size
            comparison = Image.new('RGB', (width * 2, height))
            
            # Paste input on left, enhanced on right
            comparison.paste(input_img, (0, 0))
            comparison.paste(enhanced_img, (width, 0))
            
            # Add a white separator line
            from PIL import ImageDraw
            draw = ImageDraw.Draw(comparison)
            draw.line([(width, 0), (width, height)], fill='white', width=2)
            
            # Save comparison
            comparison_file = comparison_path / f"comparison_{img_file.name}"
            comparison.save(comparison_file, quality=95)
            
            print(f"✓ Created: {comparison_file.name}")
            created += 1
            
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Created {created} comparison images in '{comparison_dir}/'")
    print(f"{'='*60}")
    print("\nEach image shows: Input (left) | Enhanced (right)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create side-by-side comparison images')
    parser.add_argument('--input_dir', type=str, default='input',
                       help='Directory with input images')
    parser.add_argument('--output_dir', type=str, default='outputs/enhanced',
                       help='Directory with enhanced images')
    parser.add_argument('--comparison_dir', type=str, default='outputs/comparisons',
                       help='Directory to save comparisons')
    
    args = parser.parse_args()
    create_comparison(args.input_dir, args.output_dir, args.comparison_dir)
