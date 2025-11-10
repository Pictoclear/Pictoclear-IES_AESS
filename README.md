# Image Enhancement


Deblur Mars rover images using deep learning (Attention U-Net).



## Install## Setup



```bash

pip install -r requirements.txt

``````bash Deep learning deblurring for Mars rover images.



## Usagepip install -r requirements.txt



### Enhance Images```

```bash

# Put images in input/ folder

python enhance.py

# Results saved to outputs/enhanced/## Enhance Images## UsageDeep learning-based deblurring for Mars rover images using Attention U-Net.

```



### Train Model

```bash```bash

# Download dataset: https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images

# Extract to data/mars/raw/python enhance.py

python train.py --epochs 10

`````````bash



### Visualizations

```bash

# Side-by-side comparisons## Trainpip install -r requirements.txt

python scripts/compare_sidebyside.py



# Grid view

python scripts/compare_grid.py```bash## Quick StartDeep learning-based image enhancement for Mars rover images using Attention U-Net architecture.



# Progressive denoisingpython train.py --epochs 10

python scripts/visualize_denoising.py --image input/your_image.jpg

``````# Enhance (put images in input/)



## Files



- `enhance.py` - Main enhancement script**Dataset**: [Kaggle Mars Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)python enhance.py

- `train.py` - Training script

- `models/` - U-Net architecture

- `datasets/` - Data loader

- `scripts/` - Comparison and visualization tools

- `weights/` - Model checkpoints# Train (download dataset to data/mars/raw/)```bash

- `input/` - Place input images here

- `outputs/` - Enhanced resultspython train.py --epochs 10



## Options```# Install dependencies



```bash

# Custom weights

python enhance.py --weights weights/model.pth## Datasetpip install -r requirements.txt## Quick StartA deep learning-based image enhancement system for Mars rover images using an Attention U-Net architecture. This project enhances blurry or degraded Mars surface images captured by rovers.



# Training options

python train.py --epochs 5 --batch_size 16 --lr 0.0001

```[Kaggle - Mars Surface Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images) (39k images, 2.5GB)





### 1. Install

# Train model (download dataset from link below)

python train.py --epochs 10```bash

```

pip install -r requirements.txt![Python](https://img.shields.io/badge/Python-3.8%2B-blue)A deep learning-based image enhancement system for Mars rover images using an Attention U-Net architecture. This project enhances blurry or degraded Mars surface images captured by rovers.


# Enhance with custom weights![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

python enhance.py --weights weights/model.pth

### 2. Enhance Images

# Compare results

python scripts/compare_sidebyside.py```bash![License](https://img.shields.io/badge/License-MIT-green)



# Visualize denoising process# Put images in input/ folder, then:

python scripts/visualize_denoising.py --image input/mars.jpg

```python enhance.py




### 3. Train Model

## Requirements

```bash

- Python 3.8+

- PyTorch 1.12+# Download dataset from: https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images

- CUDA GPU (recommended)




**Training details:**Place your Mars images in the `input/` folder, then run:

- Checkpoints automatically saved to `weights/` after each epoch

- Sample images saved to `training_samples/` every 1000 batches## Quick Start

- Loss values displayed in real-time with progress bar

- Best model typically achieved after 5-10 epochs```bash

- Estimated time: ~2 hours for 10 epochs on full dataset (39k images)

python enhance.pypython create_comparisons.py2. **Run inference**:

## Model Architecture

```

**Attention U-Net with RGB Support**

- 4-level encoder-decoder architecture### 1. Setup

- Spatial attention mechanisms for focus on important features

- Skip connections for detail preservationEnhanced images will be saved to `outputs/enhanced/`

- Input/Output: 3-channel RGB images (256x256)

   ```bash

**Key Components**:

- **Encoder**: 4 downsampling levels with attention**Example**:

- **Bottleneck**: Feature extraction at lowest resolution

- **Decoder**: 4 upsampling levels with skip connections```bash```bash

- **Attention**: Spatial attention at each level

# Enhance all images in input folder

**Training Details**:

- Loss: L1 (MAE) loss for sharper resultspython enhance.py# Install dependencies# Grid view of all images   python inference.py

- Optimizer: Adam (lr=1e-4)

- Synthetic degradation: Gaussian blur + downsampling + noise

- Data augmentation: Random crops and flips




## Results



The model effectively:

- ✅ Removes blur from atmospheric effects

- ✅ Enhances fine surface details (rocks, terrain features)

- ✅ Preserves natural Mars color information

- ✅ Reduces compression artifacts

- ✅ Maintains realistic appearance





## Credits

- Mars Dataset: NASA/JPL-Caltech via Kaggle
- Architecture inspired by attention U-Net and RealSRGAN workflows
- Original thermal GAN project (legacy code)

---

## License

See original project license for details.
