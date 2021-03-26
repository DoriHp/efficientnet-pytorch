# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-24 08:30:07
# @Last Modified by:   bao
# @Last Modified time: 2021-03-26 09:56:58

import timm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

# Transform
train_aug = A.Compose(
	[  
		A.Resize(528, 528, p=1.0),
		A.HorizontalFlip(p=0.85),
		A.Rotate(limit=20, p=0.6),
		A.CLAHE(clip_limit=4.0, p=0.85),
		A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
		A.Normalize(p=1.0),
		ToTensorV2(p=1.0)
	]
)

val_aug = A.Compose(
	[
		A.Resize(528, 528, p=1.0),
		A.HorizontalFlip(p=0.5),
		A.Normalize(p=1.0),
		ToTensorV2(p=1.0),
	]
)

model = timm.create_model('tf_efficientnet_b6_ns', pretrained=True) # set pretrained=True to use the pretrained weights
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)

# Dataset ins
class Xray(Dataset):
	def __init__(self, csv_path, augs=None, is_train: bool=False, mixup_prob: float=0.5, label_smoothing: float = 0.0):
		self.df = pd.read_csv(csv_path)
		self.augs = augs
		self.mixup_prob = mixup_prob
		self.is_train = is_train

	def __len__(self):
		return(len(self.df))

	def _get_single_item(self, idx):
		img_src = self.df["filepath"][idx]
		image = cv2.imread(img_src)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
		
		target = self.df['label'][idx]

		return image, target 

	def __getitem__(self, idx):
		
		img, label = self._get_single_item(idx)

		# For mixup
		if self.is_train and np.uniform.random() < self.mixup_prob:
			j = np.random.randint(0, len(self.df))
			p = np.random.uniform()
			img2, label2 = self._get_single_item(j)

			img = img * p + img2 * (1 - p)

			label = label * p + label2 * (1 - p)

		if (self.augs):
			transformed = self.augs(image=image)
			image = transformed['image']
		
		return image, torch.tensor(target) 

# ss=F.sigmoid(model(torch.randn(3,3,300,300)))

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def train_one_epoch(train_loader, model, optimizer, criterion, e, epochs):
	losses = AverageMeter()
	scores = AverageMeter()
	model.train()
	global_step = 0
	loop = tqdm(enumerate(train_loader), total=len(train_loader))
	
	for _, (image,labels) in loop:
		image = image.to(device)
		labels = labels.unsqueeze(1)
		labels= labels.to(device)
		output = model(image)
		batch_size = labels.size(0)
		loss = criterion(output,labels.float())
		
		out = F.sigmoid(output)
		outputs = out.cpu().detach().numpy()
		targets = labels.cpu().detach().numpy()
		try:
			auc = roc_auc_score(targets, outputs)
			losses.update(loss.item(), batch_size)
			scores.update(auc.item(), batch_size)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			global_step += 1
		
			loop.set_description(f"Epoch {e+1}/{epochs}")
			loop.set_postfix(loss = loss.item(), auc = auc.item(), stage = 'train')
		
		except ValueError:
			pass
		
	return losses.avg, scores.avg

def val_one_epoch(loader, model, optimizer, criterion):
	losses = AverageMeter()
	scores = AverageMeter()
	model.eval()
	global_step = 0
	loop = tqdm(enumerate(loader), total = len(loader))
	
	for _, (image, labels) in loop:
		image = image.to(device)
		labels = labels.unsqueeze(1)
		labels = labels.to(device)
		batch_size = labels.size(0)
		with torch.no_grad():
			output = model(image)
		loss = criterion(output,labels.float())
		
		out = F.sigmoid(output)
		outputs = out.cpu().detach().numpy()
		targets = labels.cpu().detach().numpy()
		try:
			auc = roc_auc_score(targets, outputs)
			losses.update(loss.item(), batch_size)
			scores.update(auc.item(), batch_size)
			loop.set_postfix(loss = loss.item(), auc = auc.item(), stage = 'valid')
			optimizer.step()
			optimizer.zero_grad()
			global_step += 1

		except ValueError as e:
			print(e)
			pass

	return losses.avg, scores.avg

def fit(model, fold_n, training_batch_size=4, validation_batch_size=32):
	
	train_data = Xray("data/train.csv", augs = train_aug)
	val_data = Xray("data/valid.csv", augs = val_aug)
	writer = SummaryWriter("logs")
	
	train_loader = DataLoader(train_data,
							 shuffle=True,
							 num_workers=8,
							 batch_size=training_batch_size)
	valid_loader = DataLoader(val_data,
							 shuffle=True,
							 num_workers=8,
							 batch_size=validation_batch_size)


	# Multi-GPUS
	num_gpus = torch.cuda.device_count()
	model = nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
	model.to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = True)
	epochs = 100
	
	best_acc = 0
	
	loop = range(epochs)
	for e in loop:
		
		train_loss, train_auc = train_one_epoch(train_loader, model, optimizer, criterion, e, epochs)
		# scheduling step if given
	
		print(f'For epoch {e+1}/{epochs}')
		print(f'average train_loss {train_loss}')
		print(f'average train_auc {train_auc}' )
		
		val_loss, val_auc = val_one_epoch(valid_loader, model, optimizer, criterion)
		
		scheduler.step(val_loss)
		
		print(f'avarage val_loss {val_loss}')
		print(f'avarage val_auc {val_auc}')
		
		
		if (val_auc>best_acc):
			best_acc =val_auc
			print(f'saving model for {best_acc}')
			torch.save(model.state_dict(), OUTPUT_DIR+ f'Fold {fold_n} model with val_acc {best_acc}.pth') 

		writer.add_scalar("train/loss", train_loss, e+1)
		writer.add_scalar("train/acc", train_auc, e+1)

		writer.add_scalar("valid/loss", val_loss, e+1)
		writer.add_scalar("valid/acc", val_auc, e+1)

fit(model,0)