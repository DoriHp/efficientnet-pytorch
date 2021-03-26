import os
from typing import List

import mlconfig
from torch.utils import data
import pandas as pd

from ..utils import distributed_is_initialized

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import six
import numpy as np
import torch

class DatasetMixin(data.Dataset):

    def __init__(self):
        pass

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_item(i)
        return example

    def get_item(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError

class CSVparser(DatasetMixin):
    def __init__(self, csv_path, transform=None, is_train: bool=False, mixup_prob: float=0.0, label_smoothing: float = 0.0):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class2index = {"normal": 0, "abnormal": 1}
        self.is_train = is_train
        self.mixup_prob = mixup_prob
        self.label_smoothing = label_smoothing

    def __len__(self):
        return len(self.df)

    def get_single_item(self, idx):
        
        filepath = self.df["filepath"][idx]
        label = self.class2index[self.df["label"][idx]]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        if self.transform:
            img = self.transform(image=img)["image"]

        # To NCWH input format
        img = torch.tensor(np.transpose(img, (2, 0, 1)).astype(np.float32))

        if self.is_train and self.label_smoothing > 0:
            if label == 0:
                return img, float(label) + self.label_smoothing
            else:
                return img, float(label) - self.label_smoothing
        else:
            return img, float(label)

    def get_item(self, index):

        img, label = self.get_single_item(index)

        # For mixup
        if self.is_train and np.random.uniform() < self.mixup_prob:
            j = np.random.randint(0, len(self.df))
            p = np.random.uniform()
            img2, label2 = self.get_single_item(j)

            img = img * p + img2 * (1 - p)

            label = label * p + label2 * (1 - p)

        return img, torch.tensor(label, dtype=torch.long)
        

@mlconfig.register
class XRayDatasetLoader(data.DataLoader):

    def __init__(self,
                 root: str,
                 image_size: int,
                 train: bool,
                 batch_size: int,
                 label_smoothing: float,
                 mixup_prob: float,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 **kwargs):

        self.batch_size = batch_size

        if train:
            transform = A.Compose([
                A.Resize(height=image_size, width=image_size, p=1.0),
                A.HorizontalFlip(p=0.85),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.CLAHE(clip_limit=4.0, p=0.85),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.Normalize(mean=mean, std=std, p=1.0),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=image_size, width=image_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
            ])

        csv_path = os.path.join(root, "train.csv") if train else os.path.join(root, "valid.csv")

        dataset = CSVparser(csv_path=csv_path, transform=transform, is_train=train, mixup_prob=mixup_prob, label_smoothing=0)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.distributed.DistributedSampler(dataset)

        super(XRayDatasetLoader, self).__init__(dataset,
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
