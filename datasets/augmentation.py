import numbers
import numpy as np
import torch
import torch.nn.functional as F

import cv2 
from PIL import Image 
import torch.nn as nn 

from torchvision import transforms

def train_augmentation(img_size: int, mean: tuple, std: tuple, aug_info: list = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform = add_augmentation(transform=transform, img_size=img_size, aug_info=aug_info)

    return transform

def test_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])

    return transform

def get_color_jitter(s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return color_jitter

def add_augmentation(transform: transforms.Compose, img_size: int, aug_info: list = None):
    augments_dict = {
        'RandomCrop': transforms.RandomCrop((img_size, img_size), padding=4),
        'RandomResizedCrop' : transforms.RandomResizedCrop((img_size, img_size)),
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(),
        'RandomVerticalFlip': transforms.RandomVerticalFlip(),
        'RandomColorJitter' : transforms.RandomApply([get_color_jitter()], p=0.8),
        'GaussianBlur' : GaussianBlur(kernel_size=int(0.1*img_size)),
        'Resize': transforms.Resize((img_size, img_size))
    }
    # insert augmentations
    if aug_info != None:    
        for aug in aug_info:
            transform.transforms.insert(-1, augments_dict[aug])   
    else: 
        for aug in ['RandomResizedCrop', 'RandomHorizontalFlip','RandomColorJitter','GaussianBlur']:
            transform.transforms.insert(-1, augments_dict[aug])   
    
    return transform


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        np.random.seed(0)
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        # img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        # img = self.tensor_to_pil(img)

        return img