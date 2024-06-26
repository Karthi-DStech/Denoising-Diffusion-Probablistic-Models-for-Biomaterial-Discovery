# Denoising-Diffusion-Probablistic-Models-for-Biomaterial-Discovery

**Diffusion Model Training for Image Generation using Biomaterial Topography Dataset**

This project contains the code to train a diffusion model for Biomaterial Discovery. It includes a PyTorch implementation of the U-Net model, several building blocks used in the model architecture, and scripts for training and logging.

**The Generated images can be found in the `Generated Image/` folder. Currently, this repo holds the DDPM-generated images of 32X32 and 64X64 pixel resolution topographies.**

### Real One-by-One Biomaterial Designs 
<img width="720" alt="Screenshot 2024-04-10 at 00 01 35" src="https://github.com/Karthi-DStech/Denoising-Diffusion-Probablistic-Models-for-Biomaterial-Discovery/assets/126179797/59a4b73d-5917-4180-8980-cafec56d43c1">


### Generated One-by-One Biomaterial Designs after 30_000 Epochs
<img width="766" alt="Screenshot 2024-04-10 at 00 00 16" src="https://github.com/Karthi-DStech/Denoising-Diffusion-Probablistic-Models-for-Biomaterial-Discovery/assets/126179797/83c0f320-e92f-4552-84da-826c4ad8ff0d">





### Project Structure

- `models/`: Contains the individual modules used to build the diffusion model.
    - `attention_block.py`: Defines the attention mechanisms.
    - `diffusion_model.py`: The main diffusion model class.
    - `downsampling_block.py`: Modules for downsampling feature maps.
    - `nin_block.py`: Network in network block.
    - `resnet_block.py`: ResNet blocks.
    - `timestep_embedding.py`: Embedding layers for timesteps.
    - `unet.py`: U-Net model architecture.
    - `upsampling_block.py`: Modules for upsampling feature maps.
- `options/`:
    - `base_options.py`: Command-line arguments for the training script.
- `utils/`:
    - `images_utils.py`: Utilities for image handling.
- `train.py`: Script for training the model without TensorBoard logging.
- `updated_train.py`: Script for training the model with TensorBoard logging.

### Requirements

To run the code, you need the following:

- Python 3.8 or above
- PyTorch 1.7 or above
- torchvision
- tqdm
- matplotlib
- Tensorboard 2.7.0

Install the necessary packages using pip:


### Dataset

The training scripts are set up to use the Biomaterial dataset with 2176 Samples, which are loaded from the local machine. If you wish to use a different dataset, you'll need to modify the `images_utils.py` file and potentially the training scripts to handle your dataset's loading and processing.

### Models Saving

The trained models are saved to the disk every 4000 epochs by default. You can change this frequency in the training scripts and the saving frequency will depend upon the scripts (explore train.py and updated_train.py).



