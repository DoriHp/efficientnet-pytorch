# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-25 16:52:54
# @Last Modified by:   bao
# @Last Modified time: 2021-03-26 08:22:36
import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch

# --- setup ---
pd.set_option('max_columns', 50)

from typing import Any
import yaml

def save_yaml(filepath: str, content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Union, List


@dataclass
class Flags:
    # General
    debug: bool = True
    outdir: str = "results/det"
    device: str = "cuda:0"

    # Data config
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    # split_mode: str = "all_train"  # all_train or valid20
    seed: int = 111
    target_fold: int = 0  # 0~4
    label_smoothing: float = 0.0
    # Model config
    model_name: str = "resnet18"
    model_mode: str = "normal"  # normal, cnn_fixed supported
    # Training config
    epoch: int = 20
    batchsize: int = 8
    valid_batchsize: int = 16
    num_workers: int = 4
    snapshot_freq: int = 5
    ema_decay: float = 0.999  # negative value is to inactivate ema.
    scheduler_type: str = ""
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    scheduler_trigger: List[Union[int, str]] = field(default_factory=lambda: [1, "iteration"])
    aug_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    mixup_prob: float = -1.0  # Apply mixup augmentation when positive value is set.

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self

flags_dict = {
    "debug": False,  # Change to True for fast debug run!
    "outdir": "results/tmp_debug",
    # Data
    "imgdir_name": "vinbigdata-chest-xray-resized-png-256x256",
    # Model
    "model_name": "resnet18",
    # Training
    "num_workers": 4,
    "epoch": 15,
    "batchsize": 8,
    "scheduler_type": "CosineAnnealingWarmRestarts",
    "scheduler_kwargs": {"T_0": 28125},  # 15000 * 15 epoch // (batchsize=8)
    "scheduler_trigger": [1, "iteration"],
    "aug_kwargs": {
        "HorizontalFlip": {"p": 0.5},
        "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
        "RandomBrightnessContrast": {"p": 0.5},
        "CoarseDropout": {"max_holes": 8, "max_height": 25, "max_width": 25, "p": 0.5},
        "Blur": {"blur_limit": [3, 7], "p": 0.5},
        "Downscale": {"scale_min": 0.25, "scale_max": 0.9, "p": 0.3},
        "RandomGamma": {"gamma_limit": [80, 120], "p": 0.6},
    }
}

import dataclasses

# args = parse()
print("torch", torch.__version__)
flags = Flags().update(flags_dict)
print("flags", flags)
debug = flags.debug
outdir = Path(flags.outdir)
os.makedirs(str(outdir), exist_ok=True)
flags_dict = dataclasses.asdict(flags)
save_yaml(str(outdir / "flags.yaml"), flags_dict)

# Read in the data CSV files
train = pd.read_csv(datadir / "train.csv")
# sample_submission = pd.read_csv(datadir / 'sample_submission.csv')

import cv2
import numpy as np

class VinbigdataTwoClassDataset(DatasetMixin):
    def __init__(self, dataset_dicts, image_transform=None, transform=None, train: bool = True,
                 mixup_prob: float = -1.0, label_smoothing: float = 0.0):
        super(VinbigdataTwoClassDataset, self).__init__(transform=transform)
        self.dataset_dicts = dataset_dicts
        self.image_transform = image_transform
        self.train = train
        self.mixup_prob = mixup_prob
        self.label_smoothing = label_smoothing

    def _get_single_example(self, i):
        d = self.dataset_dicts[i]
        filename = d["file_name"]

        img = cv2.imread(filename)
        if self.image_transform:
            img = self.image_transform(img)
        img = torch.tensor(np.transpose(img, (2, 0, 1)).astype(np.float32))

        if self.train:
            label = int(len(d["annotations"]) > 0)  # 0 normal, 1 abnormal
            if self.label_smoothing > 0:
                if label == 0:
                    return img, float(label) + self.label_smoothing
                else:
                    return img, float(label) - self.label_smoothing
            else:
                return img, float(label)
        else:
            # Only return img
            return img, None

    def get_example(self, i):
        img, label = self._get_single_example(i)
        if self.mixup_prob > 0. and np.random.uniform() < self.mixup_prob:
            j = np.random.randint(0, len(self.dataset_dicts))
            p = np.random.uniform()
            img2, label2 = self._get_single_example(j)
            img = img * p + img2 * (1 - p)
            if self.train:
                label = label * p + label2 * (1 - p)

        if self.train:
            label_logit = torch.tensor([1 - label, label], dtype=torch.float32)
            return img, label_logit
        else:
            # Only return img
            return img

    def __len__(self):
        return len(self.dataset_dicts)


train_loader = DataLoader(
    train_dataset,
    batch_size=flags.batchsize,
    num_workers=flags.num_workers,
    shuffle=True,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=flags.valid_batchsize,
    num_workers=flags.num_workers,
    shuffle=False,
    pin_memory=True,
)

# Select device
device = torch.device(flags.device)

# Build model
predictor = build_predictor(model_name=flags.model_name, model_mode=flags.model_mode)
classifier = Classifier(predictor)
model = classifier

# Train setup
trainer = create_trainer(model, optimizer, device)

ema = EMA(predictor, decay=flags.ema_decay)

def eval_func(*batch):
    loss, metrics = model(*[elem.to(device) for elem in batch])
    # HACKING: report ema value with prefix.
    if flags.ema_decay > 0:
        classifier.prefix = "ema_"
        ema.assign()
        loss, metrics = model(*[elem.to(device) for elem in batch])
        ema.resume()
        classifier.prefix = ""

valid_evaluator = E.Evaluator(
    valid_loader, model, progress_bar=False, eval_func=eval_func, device=device
)

# log_trigger = (10 if debug else 1000, "iteration")
log_trigger = (1, "epoch")
log_report = E.LogReport(trigger=log_trigger)
extensions = [
    log_report,
    E.ProgressBarNotebook(update_interval=10 if debug else 100),  # Show progress bar during training
    E.PrintReportNotebook(),  # Show "log" on jupyter notebook  
    # E.ProgressBar(update_interval=10 if debug else 100),  # Show progress bar during training
    # E.PrintReport(),  # Print "log" to terminal
    E.FailOnNonNumber(),  # Stop training when nan is detected.
]
epoch = flags.epoch
models = {"main": model}
optimizers = {"main": optimizer}
manager = IgniteExtensionsManager(
    trainer, models, optimizers, epoch, extensions=extensions, out_dir=str(outdir),
)
# Run evaluation for valid dataset in each epoch.
manager.extend(valid_evaluator)

# Save predictor.pt every epoch
manager.extend(
    E.snapshot_object(predictor, "predictor.pt"), trigger=(flags.snapshot_freq, "epoch")
)
# Check & Save best validation predictor.pt every epoch
# manager.extend(E.snapshot_object(predictor, "best_predictor.pt"),
#                trigger=MinValueTrigger("validation/module/nll",
#                trigger=(flags.snapshot_freq, "iteration")))

# --- lr scheduler ---
if flags.scheduler_type != "":
    scheduler_type = flags.scheduler_type
    print(f"using {scheduler_type} scheduler with kwargs {flags.scheduler_kwargs}")
    manager.extend(
        LRScheduler(optimizer, scheduler_type, flags.scheduler_kwargs),
        trigger=flags.scheduler_trigger,
    )

manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)

if flags.ema_decay > 0:
    # Exponential moving average
    manager.extend(lambda manager: ema(), trigger=(1, "iteration"))

    def save_ema_model(manager):
        ema.assign()
        torch.save(predictor.state_dict(), outdir / "predictor_ema.pt")
        ema.resume()

    manager.extend(save_ema_model, trigger=(flags.snapshot_freq, "epoch"))

_ = trainer.run(train_loader, max_epochs=epoch)

torch.save(predictor.state_dict(), outdir / "predictor_last.pt")
df = log_report.to_dataframe()
df.to_csv(outdir / "log.csv", index=False)