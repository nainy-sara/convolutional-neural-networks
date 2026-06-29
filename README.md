# Convolutional Neural Networks (CNN) - Image Denoising

This project demonstrates the use of Convolutional Neural Networks for image denoising using PyTorch. It compares two approaches: a single fixed convolutional layer and a deep CNN architecture.

## Overview

The notebook implements and trains CNNs to remove Poisson noise from the Shepp-Logan phantom test image. It demonstrates:

1. **Single Layer Convolution**: A simple convolutional network with one learnable filter
2. **Multi-Layer CNN**: A deeper network with 5 convolutional layers and PReLU activations

## Project Structure

- **cnn.ipynb**: Main Jupyter notebook containing all code and visualizations

## Requirements

The project requires the following Python libraries:

- `numpy`: Numerical computations
- `torch` (PyTorch): Deep learning framework
- `torch.nn`: Neural network modules
- `scikit-image`: Image processing utilities (Shepp-Logan phantom)
- `opencv-python` (cv2): Computer vision library
- `google-colab`: For Google Colab environment integration

## Installation

To run this notebook in Google Colab:

1. Open the notebook directly in Google Colab using the provided link in the notebook
2. All required libraries are pre-installed in Colab environment

For local development, install dependencies:

```bash
pip install numpy torch torchvision scikit-image opencv-python
```

## Usage

The notebook is designed to run in **Google Colab** and executes the following steps:

### 1. Data Preparation

```python
true_np = shepp_logan_phantom()
noisy_np = np.random.poisson(true_np)
```

- Loads the Shepp-Logan phantom (a standard test image in image processing)
- Adds Poisson noise to simulate realistic noisy data

### 2. Single Layer Convolution Training

The `Convolution_NxN` class creates a simple CNN with:
- A single 31×31 convolutional filter
- No bias terms
- Learnable kernel weights

The network is trained to denoise the image by minimizing MSE loss between the filtered output and the true image.

### 3. Multi-Layer CNN

The `CNN` class implements a deeper architecture:

```
Conv2d(1→8, 7×7) + PReLU
  ↓
Conv2d(8→8, 7×7) + PReLU
  ↓
Conv2d(8→8, 7×7) + PReLU
  ↓
Conv2d(8→8, 7×7) + PReLU
  ↓
Conv2d(8→1, 7×7) + PReLU
```

**Features:**
- 5 convolutional layers with 7×7 kernels
- PReLU activation functions between layers
- 8 feature maps in hidden layers
- Trained with Adam optimizer (lr=0.003) for 500 epochs

### 4. Visualization

The notebook displays:
- Original clean image
- Noisy image
- Filter kernels (before and after training)
- Denoised output at different training epochs

## Key Functions

### `cv2disp(name, image, xpos, ypos)`
Displays normalized image in Google Colab environment.

### `Convolution_NxN(kernel_size)`
Simple CNN module with a single learnable convolutional layer.

### `CNN()`
Multi-layer CNN architecture for image denoising.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Kernel Size (Single Layer) | 31×31 | Filter dimensions |
| Kernel Size (Multi-Layer) | 7×7 | Filter dimensions |
| Hidden Channels (Multi-Layer) | 8 | Number of feature maps |
| Learning Rate | 0.003 | Adam optimizer learning rate |
| Epochs | 500 | Number of training iterations |
| Loss Function | MSE | Mean Squared Error |

## Results

The training process optimizes the convolutional filters to progressively denoise the image:

- **Single Layer**: Learns a single denoising filter
- **Multi-Layer CNN**: Learns hierarchical features for improved denoising quality

## Notes

- The notebook is optimized for **Google Colab** environment
- `cv2.moveWindow()` is disabled as it's not supported in Colab
- Real-time visualization of denoising progress during training
- Training progress is annotated with epoch number on output images

## Future Enhancements

- Experiment with different network architectures (ResNet, U-Net, etc.)
- Test on different image types and noise models
- Implement batch processing for multiple images
- Add metrics for denoising quality (PSNR, SSIM)
- Train on larger datasets

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Shepp-Logan Phantom: https://scikit-image.org/
- CNN for Image Denoising: https://arxiv.org/abs/1608.03981

## License

This project is provided as educational material.

## Author

nainy-sara
