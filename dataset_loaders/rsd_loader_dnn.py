import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def load():

	# Load files
	path = '../../data/rsd_processed/undersampled/mfcc_40_delta_aggr4_unsplit'

	data = np.load(os.path.join(path, 'rsd_mfcc_40_d_aggr4.npy'))
	labels = np.load(os.path.join(path, 'labels_rsd.npy'))

	# Normalize
	scaler = StandardScaler()
	data = scaler.fit_transform(data)

	# Convert to Pytorch DataLoader
	tensor_x = torch.from_numpy(data)
	tensor_x = tensor_x.float()
	tensor_y = torch.from_numpy(labels)
	tensor_y = torch.argmax(tensor_y, dim=1)

	# Dataset
	full_dataset = TensorDataset(tensor_x, tensor_y)

	# Split to train-test sets
	len_train = int(0.8*len(tensor_x))
	len_test = len(tensor_x) - len_train
	train_dataset, test_dataset = random_split(full_dataset, [len_train, len_test])

	# Create dataloaders
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
	test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)

	return [train_dataloader, test_dataloader]