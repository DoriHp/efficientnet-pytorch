# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-01 15:02:26
# @Last Modified by:   bao
# @Last Modified time: 2021-03-01 15:54:21

# Read label from original csv file, then divide dataset in to 2 classes dataset (normal and abnormal)
# Output file format
#  filepath | label |
#  ...      | ...   |
#  ...      | ...   |
#  ...      | ...   |

import os
from tqdm import tqdm

def generator(root_folder):
	"""
		Args:
		- root_folder: folder that contains all image & label files (YOLOv5 format)
		Output:
		- train.csv and valid.csv has require output
	"""

	# Read original train.csv
	train_dir = os.path.join(root_folder, "train")
	valid_dir = os.path.join(root_folder, "valid")

	train_list = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".txt")]
	valid_list = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith(".txt")]

	with open("../../data/train.csv", "w") as writer:
		write_string = "filename,label\n"
		for f in tqdm(train_list, total=len(train_list), desc="Generating train set"):
			if os.stat(f).st_size == 0:
				write_string += "%s,normal\n" %f.replace(".txt", ".jpg")
			else:
				write_string += "%s,abnormal\n" %f.replace(".txt", ".jpg")
		writer.write(write_string)

	with open("../../data/valid.csv", "w") as writer:
		write_string = "filename,label\n"
		for f in tqdm(valid_list, total=len(valid_list), desc="Generating valid set"):
			if os.stat(f).st_size == 0:
				write_string += "%s,normal\n" %f.replace(".txt", ".jpg")
			else:
				write_string += "%s,abnormal\n" %f.replace(".txt", ".jpg")
		writer.write(write_string)

if __name__ == '__main__':
	generator("F:/Dev/yolov5/dataset")