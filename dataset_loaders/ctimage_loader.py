import os
import numpy as np
import pandas as pd
import parse
from PIL import Image
from skimage.io import imread

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split


def load():
	# Read binary data and labels csv
	#im_data = np.load('../../data/CT medical im/ctmedicalimage/full_archive.npz', allow_pickle=True)
	im_path = '../../data/ctmedicalimage/tiff_images'

	'''
	overview_df = pd.read_csv('../../data/CT medical im/ctmedicalimage/overview.csv')
	overview_df.columns = ['idx']+list(overview_df.columns[1:])
	overview_df['Contrast'] = overview_df['Contrast'].map(lambda x: 1 if x else 0)
	
	print(im_data['idx'], len(im_data['idx']))

	# Keep only the images of shape (512, 512)
	images = []
	labels = []
	print(im_data['image'].shape)
	for i, im in enumerate(im_data['image']):
		if im.shape == (512, 512):
			images.append(im)
			labels.append(overview_df.iloc[i]['Contrast'])
	images = np.array(images, dtype=np.float16)
	'''

	images = []
	labels = []
	shapes = set()
	for fname in os.listdir(im_path):
		#fname = fname_path.split('/')[-1]
		# Image
		image = imread(os.path.join(os.path.abspath(im_path), fname))
		image = np.expand_dims(image, axis=0).astype('float32') 
		shapes.add(image.shape)
		images.append(image)
		# Label
		_, _, contrast = parse.parse('ID_{}_AGE_{}_CONTRAST_{}_CT.tif', fname)
		labels.append(int(contrast))

	# Convert to Pytorch tensors, then dataset
	tensor_x = torch.tensor(images)
	tensor_y = torch.tensor(labels)
	
	full_dataset = TensorDataset(tensor_x, tensor_y)

	# Split to train-test sets
	len_train = int(0.2*len(tensor_x))
	len_test = len(tensor_x) - len_train
	train_dataset, test_dataset = random_split(full_dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))

	train_dataloader = DataLoader(train_dataset)
	test_dataloader = DataLoader(test_dataset)

	return [train_dataloader, test_dataloader]