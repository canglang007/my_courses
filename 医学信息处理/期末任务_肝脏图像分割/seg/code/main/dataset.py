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

class LabelProcessor:

    def __init__(self):

        self.colormap = [
            [0, 0, 0], # background
            [255, 255, 255] # liver
        ]
        
        self.color2label = self.encode_label_pix(self.colormap)

    @staticmethod
    # 形成一个映射的表
    def encode_label_pix(colormap):
        cm2lb = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lb[(cm[0]*256 + cm[1]) * 256 + cm[2]] = i

        return cm2lb

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        label = np.array(self.color2label[idx], dtype='int64')
        return label
    
p = LabelProcessor()

# 数据集的类
class HeadSegData(data.Dataset):
    def __init__(self, data_path, crop_size=(256, 256)):
        # 图片列表是datapath/DICOM/下所有的文件夹内的所有以png结尾的文件
        # glob是一个可以使用通配符的库
        imgs_list = sorted(glob.glob(data_path + '/DICOM/*/' + '*.png'))
        labels_list = sorted(glob.glob(data_path + '/Ground/*/' + '*.png'))
        self.imgs_list=imgs_list
        self.labels_list = labels_list
        self.crop_size = crop_size
        self.augmentations = transforms.Compose([
            RandomHorizontalFlip(),
            RandomRotation(),
        ])
        
    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        label_path = self.labels_list[index]

        # 转变成RGB图
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        # 进行数据增强
        img, label = self.augmentations((img, label))
        # 进行中心遮罩
        img, label = self.center_crop(img, label, self.crop_size)

        img, label = self.img_transform(img, label, index)

        return img, label
    
    def __len__(self):
        return len(self.imgs_list)


    def center_crop(self, img, label, crop_size):
        img = ff.center_crop(img, crop_size)
        label = ff.center_crop(label, crop_size)

        return img, label

    def img_transform(self, img, label, index):
        label = np.array(label)
        # 将array转化为PIL图
        label = Image.fromarray(label.astype('uint8'))

        transform_label = transforms.Compose([
            transforms.ToTensor()]
            )
        # 转为为tensor，以及进行标准化
        transform_img = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            )

        img = transform_img(img)

        label = p.encode_label_img(label)
        
        return img, torch.from_numpy(label)

if __name__ == '__main__':
    image = np.array([
        [[0, 0, 0], [255, 255, 255]],
        [[255, 255, 255], [0, 0, 0]]
    ])


    labels = p.encode_label_img(image)

    print(labels)