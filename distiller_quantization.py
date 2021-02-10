import copy
import sys
sys.path.append('./distiller')
import distiller
import distiller.apputils as apputils
from distiller.data_loggers import collect_quant_stats
from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode


def quantize_static_post(model):

	#quant_stats = collect_quant_stats(model, eval_for_stats, save_dir=None)

	params = [
	  [8], #16
	  [8], #16
	  [LinearQuantMode.ASYMMETRIC_SIGNED, 
	   LinearQuantMode.ASYMMETRIC_UNSIGNED], # LinearQuantMode.SYMMETRIC, 
	  [False, True]
	]

	# Define the quantizer
	quantizer = PostTrainLinearQuantizer(copy.deepcopy(model),
	                                  bits_activations=8, bits_parameters=8,
	                                  mode=LinearQuantMode.ASYMMETRIC_SIGNED,
	                                  per_channel_wts=False)

	return model
	