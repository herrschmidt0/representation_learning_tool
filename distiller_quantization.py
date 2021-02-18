import copy
import sys
import torch
sys.path.append('./distiller')
import distiller
import distiller.apputils as apputils
from distiller.data_loggers import collect_quant_stats
from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode


def quantize_static_post(model, train_loader, params):
	
	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'
	model.to(device)

	# Collect activation stats
	def eval_for_stats(model):
		for i, data in enumerate(train_loader, 0):
			# get minibatch
			batch_inputs, batch_labels = data
			# Move tensors to Device
			batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

			# forward 
			outputs = model(batch_inputs.float())

			if i > 10:
			 	break
	quant_stats = collect_quant_stats(model, eval_for_stats, save_dir=None)

	if params['mode'] == 0:
		mode = LinearQuantMode.SYMMETRIC
	elif params['mode'] == 1:
		mode = LinearQuantMode.SYMMETRIC_RESTRICTED
	elif params['mode'] == 2:
		mode = LinearQuantMode.ASYMMETRIC_UNSIGNED
	elif params['mode'] == 3:
		mode = LinearQuantMode.ASYMMETRIC_SIGNED


	# Define the quantizer
	quantizer = PostTrainLinearQuantizer(copy.deepcopy(model),
	                                  bits_activations=params['bits-act'], bits_parameters=params['bits-weight'],
	                                  mode=mode,
	                                  per_channel_wts=params['per-channel'],
	                                  model_activation_stats=quant_stats)

	# Get dummy input
	for images, _ in train_loader:  
		sample = images[0]
		break
	dummy_input = distiller.get_dummy_input(input_shape=(32, *sample.shape))

	# Quantize
	quantizer.prepare_model(dummy_input=dummy_input)

	# Convert to Pytorch model
	#pyt_model = quantizer.convert_to_pytorch(dummy_input)

	#return pyt_model.to('cpu')
	return quantizer.model.to(device)