# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:04:48 2021

@author: ASUS
"""
import os
import numpy as np
from torch.utils import data
import torch
from .datapro_video import train_files, test_files, augmentation

class VideoDataset(data.Dataset):
    def __init__(self, files, split, aug=True, transform=None):
        super(VideoDataset,self).__init__()
        self.files = files
        self.split = split
        self.aug = aug
        self.transform = transform
        print('Number of {0} images: ::{1} NIFTIs'.format(split, self.__len__()))

    def __getitem__(self, index):
        image_files = self.files[index]['image']
        label_files = self.files[index]['label']
        images, labels = augmentation(image_files, label_files, self.aug)
        image0, image1, image2 = images
        label0, label1, label2 = labels
        BF = images.copy()

        mean = np.mean(image0)
        std = np.std(image0)
        image0 = (image0 - mean)/std
        mean = np.mean(image1)
        std = np.std(image1)
        image1 = (image1 - mean)/std
        mean = np.mean(image2)
        std = np.std(image2)
        image2 = (image2 - mean)/std

        image0 = torch.tensor(image0)
        image1 = torch.tensor(image1)
        image2 = torch.tensor(image2)
        label0 = torch.tensor(label0)
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
        image0 = torch.unsqueeze(image0, dim=0)
        image1 = torch.unsqueeze(image1, dim=0)
        image2 = torch.unsqueeze(image2, dim=0)
        label0 = torch.unsqueeze(label0, dim=0)
        label1 = torch.unsqueeze(label1, dim=0)
        label2 = torch.unsqueeze(label2, dim=0)

        if self.split == 'test':
            image0 = torch.unsqueeze(image0, dim=0)
            image1 = torch.unsqueeze(image1, dim=0)
            image2 = torch.unsqueeze(image2, dim=0)
            label0 = torch.unsqueeze(label0, dim=0)
            label1 = torch.unsqueeze(label1, dim=0)
            label2 = torch.unsqueeze(label2, dim=0)

            return image0.float(), image1.float(), image2.float(), label0.long(), label1.long(), label2.long(), BF

        return image0.float(), image1.float(), image2.float(), label0.long(), label1.long(), label2.long()
        
    def __len__(self):
        return len(self.files)


train_data = VideoDataset(train_files, 'train', aug = True)
test_data  = VideoDataset(test_files, 'test', aug = False)

if __name__ == "__main__":
    image0, image1, image2, label2 = train_data[0]
    print(image0.shape)
    print(image1.shape)
    print(image2.shape)
    print(label2.shape)