import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import cv2
import torchvision.transforms as transforms
from PIL import Image

augmentation = transforms.Compose(

    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.5),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]
)


class patchDataset_add_age(torch.utils.data.Dataset):
    def __init__(self, data, labels,age, transform=augmentation):
        self.transform = transform
        self.data = data
        self.labels = labels
        self.age = age

    def __getitem__(self, idx):
        data_file = self.data[idx]
        labels = self.labels[idx]
        patch_age=self.age[idx]

        data =  Image.open(data_file)

        if self.transform:

            data = self.transform(data)



        return torch.FloatTensor(data), labels,torch.FloatTensor(np.array(patch_age))

    def __len__(self):
        return len(self.data)




