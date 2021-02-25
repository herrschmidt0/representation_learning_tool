import parse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, QConfig, MinMaxObserver, default_observer

class QuantStubbedModel(nn.Module):
	def __init__(self, model_fp32):
		super(QuantStubbedModel, self).__init__()

		self.model_fp32 = model_fp32
		self.quant = QuantStub()
		self.dequant = DeQuantStub()

	def forward(self, x):
		x = self.quant(x)
		x = self.model_fp32(x)
		x = self.dequant(x)
		return x

def quantize_dynamic(model, params):

	device = torch.device('cpu')
	model.to(device)
	model.eval()

	model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

	return model


def quantize_static_post(model, dataloader, params):

	quant_model = QuantStubbedModel(model)

	device = torch.device('cpu')
	quant_model.to(device)

	# Quantize with Pytorch utilities
	quant_model.eval()
	
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
		quant_model.qconfig = QConfig(activation=observer.with_args(dtype=torch.quint8), weight=observer.with_args(dtype=torch.qint8))
	else:
		quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm') # 'qnnpack' for ARM

	print(quant_model.qconfig)
	torch.quantization.prepare(quant_model, inplace=True)

	# Calibrate with the training set
	with torch.no_grad():
		cnt = 0
		for i, data in enumerate(dataloader, 0):
			# get minibatch
			batch_inputs, batch_labels = data

			# forward 
			outputs = quant_model(batch_inputs.float())

			cnt += 1
			if cnt > 10:
			 	break


	torch.quantization.convert(quant_model, inplace=True)
	print('Post Training Quantization: Convert done')

	return quant_model


def quantize_qat(model, train_loader, train_fn, params):

	# Insert QuantStub
	quant_model = QuantStubbedModel(model)

	# Move device to CPU
	cuda = torch.device('cuda')
	cpu = torch.device('cpu')
	#quant_model.to(device)
	
	# Reset weights
	def weight_init(l):
		if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
			torch.nn.init.xavier_uniform(l.weight.data)
	quant_model.apply(weight_init)
	
	# Quantize
	quant_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

	torch.quantization.prepare_qat(quant_model, inplace=True)
	quant_model.train()
	quant_model = train_fn(quant_model, [train_loader, cuda, 15])
	quant_model.to(cpu)

	torch.quantization.convert(quant_model, inplace=True)

	#quantized_model = torch.quantization.quantize_qat(model=quant_model, run_fn=train_fn, \
	#	run_args=[train_loader, device, 7], inplace=False)

	quant_model.eval()
	return quant_model