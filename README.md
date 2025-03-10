# Image to Point Cloud

A deep learning project for generating 3D point clouds from 2D images. This repository contains code for training and evaluating a neural network that predicts 3D point cloud representations from single RGB images.

## Overview

This project implements an end-to-end pipeline for training a deep learning model that can predict 3D point clouds from 2D images. The model is trained using a custom dataset of image-point cloud pairs, and uses Chamfer Loss to measure the difference between predicted and ground truth point clouds.

## Features

- Custom dataset loader for image-point cloud pairs
- Point cloud normalization and standardization
- Data augmentation for improved generalization
- Training pipeline with validation
- Learning rate scheduling
- Progress tracking and visualization

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Pillow
- tqdm

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/image-to-pointcloud.git
cd image-to-pointcloud
```

2. Install the required packages:
```bash
pip install torch torchvision numpy pillow tqdm
```

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
  models/
    model_1/
      point_cloud.xyz
      images/
        view_1.jpg
        view_2.jpg
        ...
    model_2/
      point_cloud.xyz
      images/
        view_1.jpg
        view_2.jpg
        ...
    ...
```

Each model folder contains:
- A `point_cloud.xyz` file with the 3D point cloud coordinates
- An `images` folder with multiple views of the 3D model

## Usage

### Training

To train the model, update the `root_dir` path in the script and run:

```python
from model import ImageToPointCloud
from train import create_dataloaders, train_model
import torch

# Set the path to your dataset
root_dir = "path/to/your/dataset/models"

# Create model
model = ImageToPointCloud(num_points=5304)

# Create dataloaders
train_loader, val_loader = create_dataloaders(root_dir, batch_size=8)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, val_loader, num_epochs=100, device=device)
```

### Inference

```python
import torch
from model import ImageToPointCloud
from torchvision import transforms
from PIL import Image

# Load the model
model = ImageToPointCloud(num_points=5304)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prepare the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open('path/to/image.jpg').convert('RGB')
image = transform(image).unsqueeze(0)

# Generate point cloud
with torch.no_grad():
    point_cloud = model(image)

# Save the point cloud
point_cloud = point_cloud.squeeze().numpy()
np.savetxt('output_point_cloud.xyz', point_cloud, delimiter=' ')
```

## Model Architecture

The model architecture is defined in `model.py` and consists of an `ImageToPointCloud` class that transforms image features into 3D point coordinates.

## Loss Function

The Chamfer Loss is used to measure the similarity between the predicted and ground truth point clouds. This loss function is implemented in `loss.py`.

## Training Process

The training process includes:
- Data loading and augmentation
- Forward pass through the network
- Loss calculation using Chamfer Loss
- Backpropagation and optimization
- Learning rate scheduling based on validation performance
- Model checkpointing to save the best model
