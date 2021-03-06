# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-02 09:46:05
# @Last Modified by:   bao
# @Last Modified time: 2021-03-18 10:43:19

import os
import torch
import argparse
from PIL import Image
from collections import OrderedDict
from datetime import datetime
import ntpath
from tqdm import tqdm

from torchvision import transforms
from torch.autograd import Variable

from efficientnet import models
from efficientnet.models.efficientnet import EfficientNet, params
from efficientnet.models.efficientnet import params

import cv2
import albumentations as A 
from albumentations.pytorch import ToTensorV2

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch', type=str, default='efficientnet_b3')
	parser.add_argument('--weight', type=str, default='weights/best-b3.pth')
	parser.add_argument('--input-size', type=int, help="input image's size", default=512)
	parser.add_argument('--no-cuda', action='store_true')
	parser.add_argument('--num-classes', type=int, default=2)
	parser.add_argument('--img', type=str, help='single input image')
	parser.add_argument('--img-dir', type=str, help='input image folder', default="F:/Dev/X-ray/x-ray original/test")
	parser.add_argument('--save-csv', action='store_true', help='whether to save result to csv file')
	return parser.parse_args()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def load_ckpt(checkpoint_fpath, model):
	checkpoint = torch.load(checkpoint_fpath)

	# Load state_dict that save with nn.DataParallel
	new_state_dict = OrderedDict()

	for k, v in checkpoint['model'].items():
		# name = k[7:]
		# new_state_dict[name] = v
		new_state_dict[k] = v

	model.load_state_dict(new_state_dict)	

	return model

def load_image(image_path, device, transform=None):
    """load image, returns device tensor"""
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform is not None:
    	trans_img = transform(image=image)
    	image = trans_img['image']

    image = torch.unsqueeze(image, 0) # to batch 1    
    return image.to(device)  

def main(args):

	device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

	model = getattr(models, args.arch)(pretrained=(args.weight is None), num_classes=args.num_classes)
	if args.weight is not None:
		model = load_ckpt(args.weight, model)

	model.to(device)
	model.eval()

	image_size = params[args.arch][2]

	if args.img is not None:
		if not os.path.exists(args.img):
			print("Input image must not be empty!")
			return

	if args.img_dir is not None:
		if not os.path.exists(args.img_dir):
			print("Input image folder must not be empty!")

	# Only 1 image:
	if args.img is not None:
		imgs = [img]
	else:
		imgs = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.endswith(".jpg")]
	
	# Preprocessing
	transform = A.Compose([
    	A.Resize(height=args.input_size, width=args.input_size),
    	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    	ToTensorV2()
    ])

	write_string = "image_id,target\n"
	for img_path in tqdm(imgs, total=len(imgs), desc="Processing images"):
		# Preprocess image and transfer to deivce (cuda or cpu)
		img = load_image(img_path, device, transform)
		# Get only filename from path
		img_id = path_leaf(img_path).replace(".jpg", "")

		with torch.no_grad():
			# Inference image
			output = model(img)
			output = torch.nn.functional.softmax(output, dim=1)[0]
			prob = output[1].item() # only get score for abnormal class	
			if args.save_csv:
				write_string += "%s,%.3f\n" %(img_id, prob)

	if args.save_csv:
		with open("classify_csv_" + str(int(datetime.now().timestamp())) + ".csv", "w") as f:
			f.write(write_string)

	print("Completed!")

if __name__ == '__main__':
	args = parse_args()
	main(args)