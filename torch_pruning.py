import numpy as np
import torch
from torch.nn.utils import prune

class ThresholdPruning(prune.BasePruningMethod):
	PRUNING_TYPE = "unstructured"

	def __init__(self, threshold, abs_val):
		self.threshold = threshold
		self.abs_val = abs_val

	def compute_mask(self, tensor, default_mask):
		if self.abs_val:
			return torch.abs(tensor) > self.threshold
		else:
			return tensor > self.threshold

'''
class NodeWisePruning(prune.BasePruningMethod):
	PRUNING_TYPE = "unstructured"

	def __init__(self, threshold, shape):
		self.threshold = threshold
		self.shape = shape

	def compute_mask(self, tensor, default_mask):

		tensor = torch.reshape(tensor, self.shape)
		mask = torch.zeros_like(tensor)
		tensor_np = (tensor.cpu()).detach().numpy()
		
		for i, row in enumerate(tensor_np):
			abs_sum = np.sum(np.absolute(row))

			#print(abs_sum, self.threshold)
			if abs_sum > self.threshold:
				mask[i, :] = 1

		pruned = tensor * mask.float()
		#print(pruned)
		return torch.reshape(pruned, (-1,))
'''