import parse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, QConfig, MinMaxObserver, default_observer

def quantize_dynamic(model, params):

	device = torch.device('cpu')
	model.to(device)
	model.eval()

	model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

	return model


def quantize_static_post(model, dataloader, params):

	device = torch.device('cpu')
	model.to(device)

	# Quantize with Pytorch utilities
	model.eval()
	
	if 'observer' in params:
		if params['observer'] == 0:
			observer = torch.quantization.MinMaxObserver
		elif params['observer'] == 1:
			observer = torch.quantization.MovingAverageMinMaxObserver
		elif params['observer'] == 2:
			observer = torch.quantization.PerChannelMinMaxObserver
		elif params['observer'] == 3:
			observer = torch.quantization.MovingAveragePerChannelMinMaxObserver
		elif params['observer'] == 4:
			observer = torch.quantization.HistogramObserver

		#observer = MinMaxObserver.with_args(dtype=torch.quint8 , qscheme=torch.per_tensor_affine )
		model.qconfig = QConfig(activation=observer.with_args(dtype=torch.quint8), weight=observer.with_args(dtype=torch.qint8))
	else:
		model.qconfig = torch.quantization.get_default_qconfig('fbgemm') # 'qnnpack' for ARM

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


def quantize_qat(model, train_loader, train_fn, params):

	# Move device to CPU
	device = torch.device('cpu')
	model.to(device)
	
	# Reset weights
	def weight_init(l):
		if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
			torch.nn.init.xavier_uniform(l.weight.data)
	model.apply(weight_init)
	
	# Quantize
	quantization_config = torch.quantization.get_default_qconfig("fbgemm")

	quantized_model = torch.quantization.quantize_qat(model=model, run_fn=train_fn, \
		run_args=[train_loader, device], inplace=False)

	quantized_model.eval()
	return quantized_model