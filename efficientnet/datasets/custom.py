import os
from typing import List

import mlconfig
from torch.utils import data
import pandas as pd

from ..utils import distributed_is_initialized

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CSVparser(data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class2index = {"normal": 0, "abnormal": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filepath = self.df["filepath"][index]
        label = self.class2index[self.df["label"][index]]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)
            return image['image'], label
        else:
            return image, label

@mlconfig.register
class CustomDatasetLoader(data.DataLoader):

    def __init__(self,
                 root: str,
                 image_size: int,
                 train: bool,
                 batch_size: int,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 **kwargs):

        if train:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.CLAHE(),
                A.InvertImg(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

        csv_path = os.path.join(root, "train.csv") if train else os.path.join(root, "valid.csv")

        dataset = CSVparser(csv_path=csv_path, transform=transform)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.distributed.DistributedSampler(dataset)

        super(CustomDatasetLoader, self).__init__(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=(sampler is None),
                                                    sampler=sampler,
                                                    **kwargs)