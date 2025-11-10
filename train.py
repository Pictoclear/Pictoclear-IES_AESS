"""
Quick retraining script with the FIXED generator (no green artifacts).

This will retrain the model from scratch using the corrected architecture.
"""

import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from models.generator import GeneratorRGB_Fixed
from datasets.mars_dataset import MarsBlurDataset


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create fixed model
    model = GeneratorRGB_Fixed().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    
    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    dataset = MarsBlurDataset(args.data_root, target_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=2)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Training for {args.epochs} epochs...")
    
    # Create output directories
    os.makedirs('weights', exist_ok=True)
    os.makedirs('training_samples', exist_ok=True)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (degraded, target) in enumerate(pbar):
            degraded = degraded.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(degraded)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save sample images every 1000 batches
            if batch_idx % 1000 == 0:
                with torch.no_grad():
                    from torchvision.utils import save_image
                    comparison = torch.cat([
                        degraded[:4], 
                        output[:4].clamp(0, 1), 
                        target[:4]
                    ], dim=0)
                    save_image(comparison, 
                              f'training_samples/epoch_{epoch}_batch_{batch_idx}.png',
                              nrow=4, normalize=True)
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = f'weights/generator_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
    
    print('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                       default='data/mars/raw',
                       help='Path to Mars dataset')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    train(args)
