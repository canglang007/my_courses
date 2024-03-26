import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import torchvision.transforms.functional as ff
from augmentation import *
from torchvision import transforms
import torch
import glob
import xml.etree.ElementTree as ET

class HeadSegData(data.Dataset):
    def __init__(self, data_path, crop_size=(512, 512), train=True):
        imgs_list = sorted(glob.glob(data_path + '/DICOM/*/' + '*.png'))
        self.imgs_list=imgs_list

    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512))
        img = self.img_transform(img)

        return img, img_path
    
    def __len__(self):
        return len(self.imgs_list)

    def img_transform(self, img):

        transform_img = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            )

        img = transform_img(img)

        return img




