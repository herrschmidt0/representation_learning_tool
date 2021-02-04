import parse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, QConfig, MinMaxObserver, default_observer

def quantize_static_post(model, dataloader, params):

	device = torch.device('cpu')
	model.to(device)

	# Quantize with Pytorch utilities
	model.eval()

	#model.qconfig = torch.quantization.default_qconfig
	model.qconfig = torch.quantization.get_default_qconfig('fbgemm') # 'qnnpack' for ARM
	#observer = MinMaxObserver.with_args(dtype=torch.quint8 , qscheme=torch.per_tensor_affine )
	#model.qconfig = QConfig(activation=observer, weight=observer)

	print(model.qconfig)
	torch.quantization.prepare(model, inplace=True)

	# Calibrate with the training set
	#test(model, X_test, y_test)

	with torch.no_grad():
		cnt = 0
		for i, data in enumerate(dataloader, 0):
			# get minibatch
			batch_inputs, batch_labels = data

			# forward 
			outputs = model(batch_inputs.float())

			cnt += 1
			if cnt > 10:
			 	break


	torch.quantization.convert(model, inplace=True)
	print('Post Training Quantization: Convert done')

	return model