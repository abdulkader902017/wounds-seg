# Wound Segmentation using DeepLabV3+ (ResNet50)

This repository contains Python code for **semantic segmentation of chronic wounds** from images. It leverages the **DeepLabV3+** model with a **ResNet50** encoder, trained on a custom dataset of wound images and their corresponding annotations. The goal is to accurately delineate wound boundaries for medical analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Setup](#setup)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Accurate segmentation of wounds is crucial for monitoring healing progress, assessing wound severity, and guiding treatment strategies. This project implements a deep learning approach using a state-of-the-art segmentation model to automate this process.

## Features

-   **Custom Dataset Handling**: `WoundDataset` class to load images and polygonal segmentation masks from COCO-style JSON annotations.
-   **Deep Learning Model**: Utilizes `segmentation_models_pytorch` library for a DeepLabV3+ model with a ResNet50 backbone, pre-trained on ImageNet.
-   **Training Pipeline**: Includes data loading, transformation, model training, loss tracking, learning rate scheduling (`ReduceLROnPlateau`), and early stopping.
-   **Evaluation Metrics**: Calculates **IoU (Jaccard Index)**, **Dice Coefficient**, and **Pixel Accuracy** for model performance assessment.
-   **Visualization**: Functions to display original images, ground truth masks, predicted masks, and overlays for visual inspection of results.
-   **GPU Acceleration**: Supports training and inference on CUDA-enabled GPUs if available.

## Dataset

The model is trained on the **CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients**.
* **Image Directory**: Expected at `dataset_base_path/imgs`
* **Annotation File**: Expected at `dataset_base_path/annotations/annotations.json`

**Note**: You will need to download and set up this dataset in the specified path for the code to run. Update the `dataset_base_path` variable in the script to match your local setup.

## Model Architecture

The segmentation model used is **DeepLabV3+** from the `segmentation_models_pytorch` library.
-   **Encoder**: `resnet50` (pre-trained on ImageNet)
-   **Decoder**: DeepLabV3+ specific decoder for semantic segmentation.
-   **Output**: Single channel with `sigmoid` activation for binary segmentation.

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) # For CUDA 12.1, adjust for your CUDA version or remove --index-url for CPU
pip install Pillow numpy matplotlib scikit-learn segmentation_models_pytorch
