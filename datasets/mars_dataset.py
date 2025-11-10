from pathlib import Path
from typing import Tuple, Optional

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class MarsBlurDataset(Dataset):
    """Dataset that treats the original Mars RGB image as the target
    and creates a synthetic blurred/noisy input for supervised training.
    """

    def __init__(self, root: str | Path, size: Tuple[int, int] = (320, 256),
                 augment: bool = True, seed: Optional[int] = 42):
        self.root = Path(root)
        # Recursively gather image files (handles nested Kaggle extraction folder)
        self.paths = [p for p in self.root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        self.size = size
        self.augment = augment
        if seed is not None:
            random.seed(seed)

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.size)

    def __len__(self) -> int:
        return len(self.paths)

    def _degrade(self, img: Image.Image) -> Image.Image:
        # Randomize blur strength and add resizing artifacts
        radius = random.uniform(1.0, 2.5)
        degraded = img.filter(ImageFilter.GaussianBlur(radius=radius))
        # Down-up sample to add aliasing
        w, h = degraded.size
        small = degraded.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
        degraded = small.resize((w, h), Image.BILINEAR)
        return degraded

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.resize(img)

        target = self.to_tensor(img)  # [3,H,W] in [0,1]

        degraded = self._degrade(img)
        degraded_t = self.to_tensor(degraded)

        # Additive Gaussian noise
        noise = torch.randn_like(degraded_t) * 0.02
        inp = torch.clamp(degraded_t + noise, 0.0, 1.0)
        return inp, target
