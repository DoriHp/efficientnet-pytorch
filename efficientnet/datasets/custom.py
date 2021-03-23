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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
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

        self.batch_size = batch_size

        if train:
            transform = A.Compose([
                A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0), ratio=(1.0, 1.0), p=1.0),
                A.HorizontalFlip(p=0.85),
                A.Rotate(limit=20, p=0.6),
                A.CLAHE(clip_limit=4.0, p=0.85),
                A.GaussianBlur(blur_limit=(1, 3), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.Normalize(mean=mean, std=std, p=1.0),
                ToTensorV2(p=1.0),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=image_size, width=image_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(p=1.0),
            ])

        csv_path = os.path.join(root, "train.csv") if train else os.path.join(root, "valid.csv")

        dataset = CSVparser(csv_path=csv_path, transform=transform)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.distributed.DistributedSampler(dataset)

        super(CustomDatasetLoader, self).__init__(dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=train,
                                                    sampler=sampler,
                                                    **kwargs)

if __name__ == '__main__':
    org_img = cv2.imread("test.png")
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
                # A.Resize(height=320, width=320),
                A.RandomResizedCrop(height=320, width=320, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            ])
    img = transform(image=img)
    cv2.imshow("Original", org_img)
    cv2.imshow("Augmented", img['image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
