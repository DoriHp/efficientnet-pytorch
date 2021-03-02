# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-02 09:46:05
# @Last Modified by:   bao
# @Last Modified time: 2021-03-02 11:05:49

import os
import torch
import argparse
from PIL import Image
from collections import OrderedDict
from torchvision import transforms
from torch.autograd import Variable

from efficientnet import models
from efficientnet.models.efficientnet import EfficientNet, params
from efficientnet.models.efficientnet import params


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch', type=str, default='efficientnet_b4')
	parser.add_argument('--weight', type=str, default='weights/best.pth')
	parser.add_argument('--no-cuda', action='store_true')
	parser.add_argument('--num-classes', type=int, default=2)
	parser.add_argument('--img', type=str, required=True)
	return parser.parse_args()

def load_ckpt(checkpoint_fpath, model):
	checkpoint = torch.load(checkpoint_fpath)

	# Load state_dict that save with nn.DataParallel
	new_state_dict = OrderedDict()

	for k, v in checkpoint['model'].items():
		name = k[7:]
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)	

	return model

def load_image(image_path, input_size, device):
    """load image, returns device tensor"""
    
    transform = transforms.Compose([
        transforms.Resize(input_size + 32, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = torch.unsqueeze(image, 0)
    return image.to(device)  

def main(args):

	labels = ["normal", "abnormal"]
	device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

	model = getattr(models, args.arch)(pretrained=(args.weight is None), num_classes=args.num_classes)
	if args.weight is not None:
		model = load_ckpt(args.weight, model)

	model.to(device)

	image_size = params[args.arch][2]

	if not os.path.exists(args.img):
		print("Input image must not be empty!")
		return

	model.eval()

	img = load_image(args.img, 512, device)
	# x = torch.randn(1, 3, 512, 512).to(device)
	with torch.no_grad():
		output = model(img)
		_, index = torch.max(output, 1)
		# Get predict value in percentage format
		percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100

		print(labels[index[0]], "%.5f" %percentage[index[0]].item())


if __name__ == '__main__':
	args = parse_args()
	main(args)