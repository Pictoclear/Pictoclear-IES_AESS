# Mars Image Enhancement# Mars Image Enhancement


Deblur Mars rover images using deep learning (Attention U-Net).



## Install## Setup



```bash

pip install -r requirements.txt

``````bashDeep learning deblurring for Mars rover images.



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



## License



MIT## License# Enhance images (put images in input/ folder first)




MITpython enhance.py


### 1. Install

# Train model (download dataset from link below)

python train.py --epochs 10```bash

```

pip install -r requirements.txt![Python](https://img.shields.io/badge/Python-3.8%2B-blue)A deep learning-based image enhancement system for Mars rover images using an Attention U-Net architecture. This project enhances blurry or degraded Mars surface images captured by rovers.

## Usage

```

```bash

# Enhance with custom weights![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

python enhance.py --weights weights/model.pth

### 2. Enhance Images

# Compare results

python scripts/compare_sidebyside.py```bash![License](https://img.shields.io/badge/License-MIT-green)



# Visualize denoising process# Put images in input/ folder, then:

python scripts/visualize_denoising.py --image input/mars.jpg

```python enhance.py



## Dataset```



Download from [Kaggle - Mars Surface Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)  ## Features![Python](https://img.shields.io/badge/Python-3.8%2B-blue)A deep learning-based image enhancement system for Mars rover images using an Attention U-Net architecture. This project enhances blurry or degraded Mars surface images captured by rovers.

Extract to `data/mars/raw/` (39k images, ~2.5GB)

### 3. Train Model

## Requirements

```bash

- Python 3.8+

- PyTorch 1.12+# Download dataset from: https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images

- CUDA GPU (recommended)

# Extract to data/mars/raw/- 🚀 **Image Enhancement**: Deblur and enhance Mars rover images using a trained deep learning model![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

## License



MIT License - NASA/JPL-Caltech images

python train.py --epochs 10- 📊 **Visualization Tools**: Side-by-side comparisons and progressive denoising visualization

```

- 🎮 **GPU Accelerated**: CUDA support for fast inference and training![License](https://img.shields.io/badge/License-MIT-green)

## Features

- 🔧 **Simple Interface**: Easy-to-use command-line scripts

- ✅ Image enhancement (deblurring, detail restoration)

- ✅ Side-by-side comparisons

- ✅ Progressive denoising visualization

- ✅ GPU accelerated## Project Structure



## Usage## Features## FeaturesDeep learning model to enhance Mars surface images using an attention U-Net architecture. Removes blur, noise, and improves clarity while preserving natural Mars colors.Deep learning-based image enhancement for Mars surface images using RGB generator model with attention mechanism.



**Enhance images:**```

```bash

python enhance.py                                    # Basicmars-image-enhancement/

python enhance.py --weights weights/model.pth        # Custom weights

```├── enhance.py                      # Main enhancement script



**Train model:**├── train.py                        # Training script- 🚀 **Image Enhancement**: Deblur and enhance Mars rover images using a trained deep learning model

```bash

python train.py --epochs 10 --batch_size 8├── scripts/

```

│   ├── compare_sidebyside.py      # Create side-by-side comparisons- ⚡ **Progressive Training**: Multiple training scripts for different time/quality tradeoffs

**Compare results:**

```bash│   ├── compare_grid.py            # Create grid comparison view

python scripts/compare_sidebyside.py      # Side-by-side view

python scripts/compare_grid.py            # Grid view│   └── visualize_denoising.py     # Visualize progressive denoising- 📊 **Visualization Tools**: Side-by-side comparisons and progressive denoising visualization- **Image Enhancement**: Deblur and enhance Mars rover images using a trained deep learning model

```

├── src/

**Visualize process:**

```bash│   ├── models/generator.py        # Attention U-Net architecture- 🎮 **GPU Accelerated**: CUDA support for fast inference and training

python scripts/visualize_denoising.py --image input/mars.jpg

```│   ├── datasets/mars_dataset.py   # Mars image dataset loader



## Structure│   └── scripts/train.py           # Alternative training location- **Progressive Training**: Multiple training scripts for different time/quality tradeoffs



```├── weights/                        # Trained model checkpoints

mars-image-enhancement/

├── enhance.py              # Main enhancement script├── input/                          # Place images here for enhancement## Project Structure

├── train.py                # Training script

├── models/                 # U-Net architecture└── outputs/                        # Enhanced results

├── datasets/               # Data loader

├── scripts/                # Utilities (comparison, visualization)    ├── enhanced/                   # Enhanced images- **Visualization Tools**: Side-by-side comparisons and progressive denoising visualization## Quick StartInspired by [RealSRGAN](https://github.com/noor-ahmad-haral/RealSRGAN) workflow for simple testing.

├── weights/                # Model checkpoints

├── input/                  # Input images    ├── comparisons/                # Side-by-side comparisons

└── outputs/                # Results

```    └── visualizations/             # Progressive denoising views```



## Requirements```



- Python 3.8+mars-image-enhancement/- **GPU Accelerated**: CUDA support for fast inference and training

- PyTorch 1.12+

- CUDA GPU (recommended)## Quick Start



## Dataset├── enhance.py                      # Main enhancement script



**Mars Surface Images** - [Kaggle](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)### 1. Installation

- 39,105 images (~2.5GB)

- Curiosity, Opportunity, Spirit rovers├── scripts/



## License```bash



MIT License# Clone the repository│   ├── train_quick.py             # Quick training (1k images, ~5 min)



---git clone https://github.com/yourusername/mars-image-enhancement.git



**NASA/JPL-Caltech** | Inspired by [RealSRGAN](https://github.com/noor-ahmad-haral/RealSRGAN)cd mars-image-enhancement│   ├── train_medium.py            # Medium training (5k images, ~15 min)## Project Structure




# Install dependencies│   ├── compare_sidebyside.py      # Create side-by-side comparisons

pip install -r requirements.txt

```│   ├── compare_grid.py            # Create grid comparison view### 1. Enhance Images (Inference)## Quick Start



### 2. Download Dataset (For Training Only)│   └── visualize_denoising.py     # Visualize progressive denoising



Download the Mars Surface Images dataset from Kaggle:├── src/```



**Dataset**: [Mars Surface Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)│   ├── models/generator.py        # Attention U-Net architecture



```bash│   ├── datasets/mars_dataset.py   # Mars image dataset loadermars-image-enhancement/

# Create data directory

mkdir -p data/mars/raw│   └── scripts/train.py           # Full training script



# Download and extract the dataset to data/mars/raw/├── weights/                        # Trained model checkpoints├── enhance.py                      # Main enhancement script (run this!)

# The dataset contains 39,000+ Mars rover images (~2.5GB)

```├── input/                          # Place images here for enhancement



**Note**: The dataset is only needed for training. Pre-trained weights are included for inference.└── outputs/                        # Enhanced results├── scripts/```bash### 🚀 Simple Testing (Recommended)



### 3. Enhance Images    ├── enhanced/                   # Enhanced images



Place your Mars images in the `input/` folder, then run:    ├── comparisons/                # Side-by-side comparisons│   ├── train_quick.py             # Quick training (1k images, ~5 min)



```bash    └── visualizations/             # Progressive denoising views

python enhance.py

``````│   ├── train_medium.py            # Medium training (5k images, ~15 min)# Put test images in input/ folder, then:



Enhanced images will be saved to `outputs/enhanced/`



**Example**:## Quick Start│   ├── compare_sidebyside.py      # Create side-by-side comparisons

```bash

# Enhance all images in input folder

python enhance.py

### 1. Installation│   ├── compare_grid.py            # Create grid comparison viewpython inference.py**Just want to enhance some Mars images? Use this simple approach:**

# Custom input/output directories

python enhance.py --input_dir my_images --output_dir my_results



# Use specific model weights```bash│   └── visualize_denoising.py     # Visualize progressive denoising

python enhance.py --weights weights/generator_epoch_10.pth

```# Clone the repository



### 4. View Comparisonsgit clone https://github.com/yourusername/mars-image-enhancement.git├── src/```



Create side-by-side before/after comparisons:cd mars-image-enhancement



```bash│   ├── models/

python scripts/compare_sidebyside.py

```# Install dependencies



Or create a grid view of all comparisons:pip install -r requirements.txt│   │   └── generator.py           # Attention U-Net architecture1. **Put test images in the `input/` folder**:



```bash```

python scripts/compare_grid.py

```│   ├── datasets/



### 5. Visualize Progressive Denoising### 2. Download Dataset (For Training)



See how the model gradually enhances an image from blurred to clear:│   │   └── mars_dataset.py        # Mars image dataset with synthetic degradationEnhanced images will be saved to `results/` folder.   ```bash



```bashDownload the Mars Surface Images dataset from Kaggle:

python scripts/visualize_denoising.py --image input/your_image.jpg

```│   └── scripts/



This creates:**Dataset**: [Mars Surface Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)

- Individual frames showing 0%, 12%, 25%, ..., 100% enhancement

- Grid view showing all steps│       └── train.py               # Full training script (all images)   # Add your Mars images (JPG, PNG, etc.) to the input/ directory

- Horizontal strip showing the progression

```bash

## Training

# Create data directory├── weights/                        # Trained model checkpoints

Train the model on the full Mars dataset:

mkdir -p data/mars/raw

```bash

python train.py --epochs 10 --batch_size 8├── input/                          # Place images here for enhancement**Create comparisons**:   # Some sample images are already included!

```

# Download and extract the dataset to data/mars/raw/

**Training options:**

```bash# The dataset contains 39,000+ Mars rover images├── outputs/                        # Enhanced results and visualizations

# Custom number of epochs

python train.py --epochs 5```



# Larger batch size (if you have enough GPU memory)└── data/                          # Training dataset (Mars images)```bash   ```

python train.py --batch_size 16

**Note**: The dataset is ~2.5GB and only needed if you want to train the model. Pre-trained weights are included for inference.

# Custom learning rate

python train.py --lr 0.0001```



# Custom data path### 3. Enhance Images

python train.py --data_root path/to/mars/images

```# Side-by-side comparisons



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

# Custom input/output directoriespip install -r requirements.txt

## Advanced Usage

python enhance.py --input_dir my_images --output_dir my_results

### Custom Enhancement Options

python create_grid.py   ```

```bash

# Process specific directory with custom output# Use specific model weights

python enhance.py \

  --input_dir path/to/mars/images \python enhance.py --weights weights/generator_epoch_5.pth# Download Mars dataset (if training)

  --output_dir path/to/results \

  --weights weights/generator_epoch_10.pth```



# Limit number of images to process# Dataset from Kaggle: Mars Surface Images

python enhance.py --max_images 50

```### 4. View Comparisons



### Training with Custom Parameters```



```bashCreate side-by-side before/after comparisons:

# Full training with all options

python train.py \# Or do it automatically3. **Check results**:

  --epochs 15 \

  --batch_size 16 \```bash

  --lr 0.0001 \

  --data_root data/mars/rawpython scripts/compare_sidebyside.py### 2. Enhance Images

```

```

### Visualization Options

python inference.py --create_comparisons   Enhanced images will be saved in the `results/` folder! ✨

```bash

# More detailed progressive steps (16 instead of 8)Or create a grid view of all comparisons:

python scripts/visualize_denoising.py \

  --image input/mars.jpg \Place your Mars images in the `input/` folder, then run:

  --steps 16

```bash

# Custom output location

python scripts/visualize_denoising.py \python scripts/compare_grid.py```

  --image input/mars.jpg \

  --output_dir custom_visualization```

```

```bash

## Requirements

### 5. Visualize Progressive Denoising

- Python 3.8+

- PyTorch 1.12+python enhance.py4. **Create comparisons** (optional):

- torchvision

- Pillow (PIL)See how the model gradually enhances an image from blurred to clear:

- numpy

- tqdm```

- matplotlib

- CUDA-capable GPU (recommended, but CPU works too)```bash



See `requirements.txt` for exact versions.python scripts/visualize_denoising.py --image input/your_image.jpg### 2. Train Model (Optional)   ```bash



## Dataset Information```



The model is trained on the **Mars Surface Images** dataset from Kaggle:Enhanced images will be saved to `outputs/enhanced/`

- **Source**: [Kaggle - Mars Surface and Curiosity Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)

- **Size**: 39,105 images (~2.5GB)This creates:

- **Rovers**: Curiosity, Opportunity, Spirit

- **Format**: RGB JPG images- Individual frames showing 0%, 12%, 25%, ..., 100% enhancement   # Side-by-side comparisons for each image

- **Resolution**: Various (automatically resized during training)

- Grid view showing all steps

**Citation**: If you use this dataset, please cite the original Kaggle source.

- Horizontal strip showing the progression### 3. View Comparisons

## Results



The model effectively:

- ✅ Removes blur from atmospheric effects## Training**Quick training** (2-3 minutes on 100 images):   python create_comparisons.py

- ✅ Enhances fine surface details (rocks, terrain features)

- ✅ Preserves natural Mars color information

- ✅ Reduces compression artifacts

- ✅ Maintains realistic appearance### Quick Training (~5 minutes)Create side-by-side before/after comparisons:



**Performance Metrics**:For fast testing with 1,000 images:

- Typical PSNR improvement: 2-5 dB on degraded images

- Training time: ~2 hours for full dataset (10 epochs)```bash   

- Inference: ~0.1 seconds per image (GPU)

```bash

## Examples

python scripts/train_quick.py```bash

Check the `outputs/` folder after running the scripts to see:

- `outputs/enhanced/` - Enhanced versions of your input images```

- `outputs/comparisons/` - Side-by-side before/after comparisons

- `outputs/visualizations/` - Progressive denoising demonstrationspython scripts/compare_sidebyside.pypython quick_train.py   # OR create a grid view of all images at once



## Troubleshooting### Medium Training (~15-20 minutes)



**Out of Memory Error**:Balanced quality with 5,000 images:```

```bash

# Reduce batch size

python train.py --batch_size 4

``````bash```   python create_grid.py



**CUDA Not Available**:python scripts/train_medium.py

```bash

# The model will automatically fall back to CPU```Or create a grid view of all comparisons:

# Training will be slower but still works

```



**Import Errors**:### Full Training (~2 hours)   

```bash

# Make sure you're in the project root directoryBest quality with all 39,000+ images:

cd mars-image-enhancement

python enhance.py```bash

```

```bash

**Dataset Not Found**:

```bashpython src/scripts/train.py --epochs 10 --batch_size 8python scripts/compare_grid.py**Full training** (on entire dataset):   # OR do it all in one command

# Make sure dataset is in the correct location

# Should be: data/mars/raw/*.jpg```

```

```

## License

Training checkpoints are automatically saved to `weights/` after each epoch.

MIT License - See [LICENSE](LICENSE) file for details.

```bash   python inference.py --create_comparisons

## Acknowledgments

**Training Progress**:

- 🌌 Mars images courtesy of NASA/JPL-Caltech

- 📊 Dataset from [Kaggle - Mars Surface Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)- Loss values displayed in real-time### 4. Visualize Progressive Denoising

- 🏗️ Inspired by [RealSRGAN](https://github.com/noor-ahmad-haral/RealSRGAN) architecture patterns

- 🎓 U-Net architecture based on Ronneberger et al. (2015)- Sample images saved to `training_samples/` every 200 batches



## Contributing- Best model typically achieved after 5-10 epochspython src/scripts/train.py --epochs 10 --batch_size 4   ```



Contributions are welcome! Please feel free to submit a Pull Request.



## Citation## Model ArchitectureSee how the model gradually enhances an image:



If you use this code in your research, please cite:



```bibtex**Attention U-Net with RGB Support**```

@software{mars_image_enhancement,

  title = {Mars Image Enhancement with Attention U-Net},- 4-level encoder-decoder architecture

  year = {2025},

  author = {Your Name},- Spatial attention mechanisms for focus on important features```bash

  url = {https://github.com/yourusername/mars-image-enhancement}

}- Skip connections for detail preservation

```

- Input/Output: 3-channel RGB images (256x256)python scripts/visualize_denoising.py --image input/your_image.jpg**Advanced options**:

## Contact



For questions or issues, please open an issue on GitHub.

**Key Components**:```

---

- **Encoder**: 4 downsampling levels with attention

**Made with ❤️ for Mars exploration**

- **Bottleneck**: Feature extraction at lowest resolution### 3. Download Dataset (Optional)```bash

- **Decoder**: 4 upsampling levels with skip connections

- **Attention**: Spatial attention at each level## Training



**Training Details**:# Process specific number of images

- Loss: L1 (MAE) loss for sharper results

- Optimizer: Adam (lr=1e-4)### Quick Training (~5 minutes)

- Synthetic degradation: Gaussian blur + downsampling + noise

- Data augmentation: Random crops and flipsFor fast testing with 1,000 images:```bashpython inference.py --max_images 10



## Advanced Usage



### Custom Enhancement Options```bashpython src/scripts/download_mars_dataset.py



```bashpython scripts/train_quick.py

# Process specific directory with custom output

python enhance.py \``````# Use different weights (after training more epochs)

  --input_dir path/to/mars/images \

  --output_dir path/to/results \

  --weights weights/generator_epoch_10.pth

### Medium Training (~15-20 minutes)python inference.py --weights weights_mars/generator_rgb_epoch_10.pth

# Limit number of images to process

python enhance.py --max_images 50Balanced quality with 5,000 images:

```

Downloads ~39k Mars Curiosity rover images from Kaggle.

### Training with Custom Parameters

```bash

```bash

# Quick training with custom settingspython scripts/train_medium.py# Custom input/output directories

python scripts/train_quick.py

```

# Full training with custom configuration

python src/scripts/train.py \---python inference.py --input_dir my_images --output_dir my_results

  --epochs 15 \

  --batch_size 16 \### Full Training (~2 hours)

  --lr 0.0001 \

  --data_root data/mars/rawBest quality with all 39,000+ images:```

```



### Visualization Options

```bash## Project Structure

```bash

# More detailed progressive steps (16 instead of 8)python src/scripts/train.py --epochs 10 --batch_size 8

python scripts/visualize_denoising.py \

  --image input/mars.jpg \```---

  --steps 16



# Custom output location

python scripts/visualize_denoising.py \Training checkpoints are saved to `weights/` after each epoch.```

  --image input/mars.jpg \

  --output_dir custom_visualization

```

## Model Architecture├── input/              # Put test images here### 🔬 Full Training Pipeline (Advanced)

## Requirements



- Python 3.8+

- PyTorch 1.12+**Attention U-Net with RGB Support**├── results/            # Enhanced images appear here

- torchvision

- Pillow (PIL)- 4-level encoder-decoder architecture

- numpy

- tqdm- Spatial attention mechanisms├── inference.py        # Run this to enhance images**Want to train the model yourself?**

- matplotlib

- CUDA-capable GPU (recommended, but CPU works too)- Skip connections for detail preservation



See `requirements.txt` for exact versions.- Input/Output: 3-channel RGB images (256x256)├── quick_train.py      # Fast training on 100 images



## Dataset Information



The model is trained on the **Mars Surface Images** dataset from Kaggle:**Training Details:**├── create_comparisons.py  # Side-by-side comparisons1. **Download Mars Dataset** (optional if training from scratch):

- **Source**: [Kaggle - Mars Surface and Curiosity Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)

- **Size**: 39,105 images (~2.5GB)- Loss: L1 (MAE) loss

- **Rovers**: Curiosity, Opportunity, Spirit

- **Format**: RGB JPG images- Optimizer: Adam (lr=1e-4)├── create_grid.py      # Grid comparison view   ```bash

- **Resolution**: Various (automatically resized during training)

- Synthetic degradation: Gaussian blur + downsampling + noise

**Citation**: If you use this dataset, please cite the original Kaggle source.

- Data augmentation: Random crops and flips│   python src/scripts/download_mars_dataset.py

## Results



The model effectively:

- ✅ Removes blur from atmospheric effects## Advanced Usage├── src/   ```

- ✅ Enhances fine surface details (rocks, terrain features)

- ✅ Preserves natural Mars color information

- ✅ Reduces compression artifacts

- ✅ Maintains realistic appearance### Custom Enhancement│   ├── models/   This downloads ~39k Mars surface images from Kaggle.



**Performance Metrics**:

- Typical PSNR improvement: 2-5 dB on degraded images

- Training time: 15-20 minutes (medium), 2 hours (full)```bash│   │   └── generator.py       # RGB U-Net architecture

- Inference: ~0.1 seconds per image (GPU)

python enhance.py --input_dir path/to/images --output_dir path/to/results --weights weights/generator_epoch_5.pth

## Examples

```│   ├── datasets/2. **Train the model**:

Check the `outputs/` folder after running the scripts to see:

- `outputs/enhanced/` - Enhanced versions of your input images

- `outputs/comparisons/` - Side-by-side before/after comparisons

- `outputs/visualizations/` - Progressive denoising demonstrations### Training Options│   │   └── mars_dataset.py    # Data loader with synthetic degradation   ```bash



## Troubleshooting



**Out of Memory Error**:```bash│   └── scripts/   python src/scripts/train_mars.py --data_root data/mars/raw --epochs 10 --batch_size 4

```bash

# Reduce batch size in training scripts# Medium training with custom settings

python src/scripts/train.py --batch_size 4

```python scripts/train_medium.py│       ├── train.py           # Full training script   ```



**CUDA Not Available**:

```bash

# The model will automatically fall back to CPU# Full training with custom epochs/batch size│       └── download_mars_dataset.py   

# Training will be slower but still works

```python src/scripts/train.py --epochs 5 --batch_size 16 --lr 0.0001



**Import Errors**:```│   Training will:

```bash

# Make sure you're in the project root directory

cd mars-image-enhancement

python enhance.py### Visualization Options├── weights/            # Model weights   - Create synthetic degraded inputs (blur + downsample + noise)

```



## License

```bash├── training_samples/   # Training visualization   - Train RGB generator with L1 content loss

MIT License - See [LICENSE](LICENSE) file for details.

# More detailed progressive steps (16 instead of 8)

## Acknowledgments

python scripts/visualize_denoising.py --image input/mars.jpg --steps 16└── legacy/            # Original thermal GAN code   - Save comparison images per epoch

- 🌌 Mars images courtesy of NASA/JPL-Caltech

- 📊 Dataset from [Kaggle - Mars Surface Images](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-images)

- 🏗️ Inspired by [RealSRGAN](https://github.com/noor-ahmad-haral/RealSRGAN) architecture patterns

- 🎓 U-Net architecture based on Ronneberger et al. (2015)# Custom output location```   - Save model weights to `weights_mars/`



## Contributingpython scripts/visualize_denoising.py --image input/mars.jpg --output_dir my_vis



Contributions are welcome! Please feel free to submit a Pull Request.```



## Citation



If you use this code in your research, please cite:## Requirements---3. **Run batch inference on full dataset**:



```bibtex

@software{mars_image_enhancement,

  title = {Mars Image Enhancement with Attention U-Net},- Python 3.8+   ```bash

  year = {2025},

  author = {Your Name},- PyTorch 1.12+

  url = {https://github.com/yourusername/mars-image-enhancement}

}- CUDA-capable GPU (recommended)## Model Details   python src/scripts/inference_mars_rgb.py --images_dir data/mars/raw --weights weights_mars/generator_rgb_epoch_10.pth --out_dir outputs

```

- See `requirements.txt` for full dependencies

## Contact

   ```

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

## Dataset

---

- **Architecture**: Attention U-Net (RGB, 3→3 channels)

**Made with ❤️ for Mars exploration**

The model is trained on the **Mars Surface Images** dataset from Kaggle, containing 39,000+ images from various Mars rovers (Curiosity, Opportunity, Spirit).

- **Training**: Synthetic degradation (blur + downsample + noise)---

## Results

- **Loss**: L1 content loss

The model effectively:

- Removes blur from atmospheric effects- **Dataset**: Mars Surface images (NASA/JPL-Caltech)## Project Structure

- Enhances fine surface details

- Preserves natural color information

- Reduces compression artifacts

---```

Typical PSNR improvement: 2-5 dB on degraded images.

Ganthermal-main/

## License

## Usage Examples├── input/                          # 📥 Put test images here

MIT License - See LICENSE file for details

├── results/                        # 📤 Enhanced images appear here

## Acknowledgments

```bash├── inference.py                    # 🚀 Simple inference script (RealSRGAN-style)

- Mars images courtesy of NASA/JPL-Caltech

- Dataset from Kaggle Mars Surface Images# Basic enhancement│

- Inspired by RealSRGAN architecture patterns

python inference.py├── src/

## Citation

│   ├── models/

If you use this code in your research, please cite:

# Limit number of images│   │   └── generator_rgb.py       # RGB generator architecture (attention U-Net)

```bibtex

@software{mars_image_enhancement,python inference.py --max_images 10│   ├── datasets/

  title = {Mars Image Enhancement with Attention U-Net},

  year = {2025},│   │   └── mars_dataset.py        # Synthetic degradation dataset

  author = {Your Name},

  url = {https://github.com/yourusername/mars-enhancement}# Use specific weights│   └── scripts/

}

```python inference.py --weights weights/generator_epoch_5.pth│       ├── download_mars_dataset.py    # Download Mars dataset from Kaggle


│       ├── train_mars.py               # Training pipeline

# Create comparisons automatically│       └── inference_mars_rgb.py       # Batch inference script

python inference.py --create_comparisons│

├── weights_mars/                   # Model weights after training

# Custom directories├── samples_mars/                   # Training comparison images

python inference.py --input_dir my_images --output_dir my_results├── data/                          # Mars dataset (downloaded)

```│

├── legacy/                        # Original thermal GAN components (1-channel)

---├── weights/legacy_thermal/        # Original pretrained thermal weights

└── scripts/                       # Utility scripts (GPU checks, etc.)

## Requirements```



```bash---

pip install torch torchvision pillow numpy kaggle tqdm

```## Model Architecture



See `requirements.txt` for details.**Generator**: 

- RGB-capable attention U-Net (3 input channels, 3 output channels)

---- Encoder-decoder structure with skip connections

- Attention mechanism for feature refinement

## Notes- Input: Degraded Mars images

- Output: Enhanced clear images

- **GPU recommended** for faster processing

- Images auto-resized to dimensions divisible by 16**Training**:

- Default weights: `weights/generator_fixed_quick_epoch_2.pth`- Synthetic degradation: blur + downsample + noise

- Sample Mars images included in `input/` for testing- Loss: L1 content loss

- Optimizer: Adam (lr=1e-4)

---- Dataset: ~39k Mars surface images



## Credits---



- Dataset: Mars Surface images from [Kaggle](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-image-set-nasa)## Requirements

- Architecture: Attention U-Net for image enhancement

- Inspired by RealSRGAN workflow```bash

pip install torch torchvision pillow numpy kaggle

---```



## LicenseSee `requirements.txt` for full dependencies.



See original project license.---


## Usage Examples

### Example 1: Quick test on a few images
```bash
# Copy some Mars images to input/
# Run inference
python inference.py --max_images 5
```

### Example 2: Train for more epochs
```bash
# Train for 10 epochs
python src/scripts/train_mars.py --epochs 10 --batch_size 4

# Use the new weights
python inference.py --weights weights_mars/generator_rgb_epoch_10.pth
```

### Example 3: Process your own dataset
```bash
# Put your images in a folder
python inference.py --input_dir path/to/your/images --output_dir path/to/results
```

---

## Notes

- **GPU Recommended**: Training and inference are much faster with CUDA
- **Memory**: For large images, model automatically resizes to max 512px during inference
- **Legacy Code**: Original 1-channel thermal GAN code is in `legacy/` directory
- **Dataset**: Mars Surface and Curiosity images from [Kaggle](https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-image-set-nasa)

---

## Training Progress

Current model status:
- ✅ Epoch 1 trained (avg L1 loss: 0.0269)
- Weights saved to: `weights_mars/generator_rgb_epoch_1.pth`
- Training samples: See `samples_mars/` directory

To continue training or improve results, run more epochs!

---

## Credits

- Mars Dataset: NASA/JPL-Caltech via Kaggle
- Architecture inspired by attention U-Net and RealSRGAN workflows
- Original thermal GAN project (legacy code)

---

## License

See original project license for details.
