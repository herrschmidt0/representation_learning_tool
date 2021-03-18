import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def load():

	# Load files
	path = '../../data/rsd_processed/undersampled/mfcc_40_delta_split_05_0125'

	data = np.load(os.path.join(path, 'rsd_s0.5_mfcc_40_d.npy'), allow_pickle=True)
	labels = np.load(os.path.join(path, 'labels_rsd.npy'))

	# Obtain a list of spectrograms
	arr_flattened = []
	labels_new = []
	for i, obj in enumerate(data):
		for spec in obj:
			arr_flattened.append(spec)
			labels_new.append(labels[i])

	# Label distubution
	hist = np.histogram(np.argmax(labels_new, axis=1), bins=8)
	print("Label distribution:", hist, '\nTotal (label/data):', len(labels_new), len(arr_flattened))

	# Normalize (Per-image normalization)
	for i, spect in enumerate(arr_flattened):
		arr_flattened[i] = (spect-np.mean(spect))/np.std(spect)

	# Convert to numpy, add channel axis
	data = np.array(arr_flattened, dtype=np.float16)
	data = np.expand_dims(data, axis=1)

	# Convert to Pytorch DataLoader
	tensor_x = torch.from_numpy(data)
	tensor_x = tensor_x.float()
	tensor_y = torch.FloatTensor(labels_new)
	tensor_y = torch.argmax(tensor_y, dim=1)

	# Dataset
	full_dataset = TensorDataset(tensor_x, tensor_y)

	# Split to train-test sets
	len_train = int(0.9*len(tensor_x))
	len_test = len(tensor_x) - len_train
	train_dataset, test_dataset = random_split(full_dataset, [len_train, len_test])

	# Create dataloaders
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
	test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)

	return [train_dataloader, test_dataloader]