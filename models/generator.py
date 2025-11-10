import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorRGB_Fixed(nn.Module):
    """Fixed RGB generator without problematic attention mechanism.
    
    Simplified U-Net architecture that works properly with RGB images.
    No color artifacts, clean skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 512)
        
        # Decoder (upsampling path)
        self.dec4 = self._conv_block(512 + 512, 256)  # 512 from bottleneck + 512 from enc4
        self.dec3 = self._conv_block(256 + 256, 128)  # 256 from dec4 + 256 from enc3
        self.dec2 = self._conv_block(128 + 128, 64)   # 128 from dec3 + 128 from enc2
        self.dec1 = self._conv_block(64 + 64, 64)     # 64 from dec2 + 64 from enc1
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Convolutional block with BatchNorm and LeakyReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final layer + residual connection
        out = self.final(d1)
        
        # Clean residual connection - just add input back
        return torch.tanh(out) + x  # tanh keeps output bounded


if __name__ == '__main__':
    # Test the model
    model = GeneratorRGB_Fixed()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print('Input shape:', dummy.shape)
    print('Output shape:', out.shape)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
