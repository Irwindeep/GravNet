# GravNet

**Detection of Gravitational Waves using Deep Learning**

GravNet is a deep learning project aimed at detecting gravitational waves from astrophysical sources. This repository contains the source code, training files, pre-trained model weights, and data utilities necessary for replicating and extending the experiments.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Acknowledgements](#acknowledgements)

## Overview

Gravitational wave astronomy has opened a new window into the universe. GravNet leverages deep learning techniques to identify gravitational wave signals from noisy data. This project demonstrates how convolutional and other deep neural network architectures can be applied to this emerging field in astrophysics. This project utilizes the following models:
1. CNN, ResNet and DenseNet for Wave Detection and Parameter Estimation.
2. UNet for noise extraction from noisy GW data.
3. Fine Tuning of UNet for Wave Detection and Parameter Estimation.

## Features

- **Deep Learning Models:** Implementation of neural networks for gravitational wave detection.
- **Data Processing:** Scripts to load and preprocess waveform data.
- **Pre-trained Weights:** Ready-to-use model weights to help you get started quickly.
- **Modular Codebase:** Easily extend or modify the code to experiment with different architectures or data preprocessing techniques.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Irwindeep/GravNet.git
   cd GravNet
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Data Loading:**  
  The `load_waveforms.py` script demonstrates how to load and simulate gravitational wave data.  
  Example usage:
  
  ```bash
  python load_waveforms.py
  ```

- **Training and Inference:**  
  While the core model and training routines reside in the `gravnet` folder, refer to the scripts in the `training_files` directory for examples on how to start training your model. Pre-trained model weights are available in the `model_weights` directory for inference or further fine-tuning.

- **Experimentation:**  
  Use the provided code as a baseline to experiment with different deep learning architectures, preprocessing techniques, and hyperparameters.

## File Structure

- **/gravnet:**  
  Contains the implementation of the deep learning model and supporting utilities.

- **/gw_data:**  
  Directory for gravitational wave datasets used for training and evaluation.

- **/model_weights:**  
  Pre-trained weights for the deep learning models. Regression files for Parameter Estmimation and Classification for Wave Detection

- **/training_files:**  
  Scripts and configuration files related to training of models.

- **load_waveforms.py:**  
  A utility script to load and preprocess waveform data.

- **requirements.txt:**  
  Python dependencies required for running the project.

- **.gitignore:**  
  Specifies files and directories to be ignored by Git.

## Acknowledgements

- Special thanks to the contributors and the gravitational wave research community.
- This project builds on the growing field of deep learning in astrophysics.
