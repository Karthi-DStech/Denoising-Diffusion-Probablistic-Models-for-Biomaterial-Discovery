import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import shutil

from options.base_options import get_arguments


"""
This script contains utility functions for image processing and parsing the arguents.

Functions
---------
sample_batch(batch_size, dataset, device)
    Sample a batch of images from the dataset
    
returns
-------
torch.Tensor
    Batch of images
"""
args = get_arguments()


torch.manual_seed(42)

root = '1by1-TOPO-Resized/'
print(os.path.exists(root))

class_dir = os.path.join(root, 'bio_materials')
os.makedirs(class_dir, exist_ok=True)

for entry in os.scandir(root):
    if entry.is_file() and entry.name.endswith('.png'):  
        shutil.move(entry.path, class_dir)

train_dataset = datasets.ImageFolder(root = root, transform=transforms)

def sample_batch(batch_size, dataset, device):

    sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:batch_size])
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    data_iter = iter(loader)
    images, _ = next(data_iter)
    images = images.to(device) 
    return images



