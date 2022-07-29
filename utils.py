import random
import os
import os.path as osp
import sys
import math
import copy 
import pprint
import shutil

from tqdm import tqdm
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as T

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
#     plt.axis('off')
    plt.imshow(im)
    plt.title(title)
    
def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
#     plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(title)
    
def ShowHeatMap(im, title, ax=None):
    if ax is None:
        plt.figure()
#     plt.axis('off')
    plt.imshow(im, cmap='inferno')
    plt.title(title)
    
def clamp(x, mean, std):
    upper = torch.from_numpy(np.array((1.0 - mean) / std)).to(x.device)
    lower = torch.from_numpy(np.array((0.0 - mean) / std)).to(x.device)

    if x.shape[1] == 3:  # 3-channel image
        for i in [0, 1, 2]:
            x[0][i] = torch.clamp(x[0][i], min=lower[i], max=upper[i])
    else:
        x = torch.clamp(x, min=lower[0], max=upper[0])
    return x

def load_image_view(data_mean, data_std, image):
    transforms_org = T.Compose([
        T.Resize(data_cfgs['resize']),
        T.CenterCrop(data_cfgs['resize']),
        T.ToTensor(),
#         T.Lambda(lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255)),   # tensor in [0,1]
        T.Lambda(lambda x: x.expand(3,-1,-1))
    ])
    transforms_pred = T.Compose([
        T.Resize(data_cfgs['resize']),
        T.CenterCrop(data_cfgs['resize']),
        T.ToTensor(),
#         T.Lambda(lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255)),   # tensor in [0,1]
        T.Normalize(mean=data_mean, std=data_std),
        T.Lambda(lambda x: x.expand(3,-1,-1))
    ])
    if data_mean is not None and data_std is not None:
        img_ = transforms_pred(image).permute(1,2,0)
    else:
        img_ = transforms_org(image).permute(1,2,0)
    return img_
    

'''load image for momdel input'''
def get_transforms(data_cfgs, if_norm=False, aug=None):
    transforms = [
#         T.Resize((data_cfgs['resize'], data_cfgs['resize']))
        T.Resize(data_cfgs['resize']),
        T.CenterCrop(data_cfgs['resize']),
    ]
    
    if aug is not None:
        transforms += [
            T.Lambda(aug),
            T.CenterCrop(data_cfgs['resize']),
            T.RandomHorizontalFlip(),
        ]
    if if_norm:
        transforms += [
            T.Normalize(mean=data_cfgs['data_mean'], std=data_cfgs['data_std'])
        ]
        
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def image_process(image, data_cfgs, if_norm=False, aug=None):
    transforms = get_transforms(data_cfgs, if_norm, aug)
    return transforms(image)


# '''load image for visualization'''
# def image_process_vis(image, data_cfgs, if_norm=False, aug=None):
#     return image_process(image, data_cfgs, if_norm, aug).expand(3,-1,-1).permute(1,2,0).contiguous()


def img_norm(image, k=1):
    if isinstance(image, np.ndarray):
        image = image.astype('float')
    else:
        image = image.to(torch.float32)
    image = image - image.min()
    image = image / image.max()
    if isinstance(image, np.ndarray):
        image = np.clip(image*k, 0, 1)
    else:
        image = torch.clip(image*k, 0, 1)
    return image


def batch_img_norm(image, k=1):
    if isinstance(image, np.ndarray):
        image = image.astype('float')
        n, h, w = image.shape
        image = image.reshape(n, -1)
        image = image - image.min(axis=1, keepdims=True)
        image = image / (image.max(axis=1, keepdims=True) + 1e-10)
        image = np.clip(image*k, 0, 1)
        image = image.reshape(n, h, w)
    else:
        image = image.to(torch.float32)
        n, h, w = image.size()
        image = image.view(n, -1)
        image = image - image.min(dim=1, keepdim=True)
        image = image / (image.max(dim=1, keepdim=True) + 1e-10)
        image = torch.clip(image*k, 0, 1)
        image = image.view(n, h, w)
    return image