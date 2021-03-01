import os
from typing import List

import mlconfig
from PIL import Image
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import pandas as pd

from ..utils import distributed_is_initialized

class CSVLoader(data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {"normal": 0, "abnormal": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df[index, "filename"]
        label = self.class2index[self.df[index, "label"]]
        image = Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

@mlconfig.register
class CustomDatasetLoader():

    def __init__(self,
                 csv_path: str,
                 images_folder: str,
                 image_size: int,
                 train: bool,
                 batch_size: int,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 **kwargs):
        normalize = transforms.Normalize(mean=mean, std=std)

        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size + 32, interpolation=Image.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        dataset = CSVLoader(csv_path=csv_path, images_folder=images_folder, transform=transform)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.distributed.DistributedSampler(dataset)

        super(CustomDatasetLoader, self).__init__(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=(sampler is None),
                                                    sampler=sampler,
                                                    **kwargs)
