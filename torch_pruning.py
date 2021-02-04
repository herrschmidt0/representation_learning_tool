import parse
import numpy as np
import networkx as nx

import torch
from torch.nn.utils import prune

from utils import ChannelGraph

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

def magnitude_pruning(model, params):
	for child in model.children():
		if isinstance(child, torch.nn.Linear):

			if params['percentage']:
				weights = (child.weight.cpu()).detach().numpy()
				threshold = np.percentile(np.abs(weights), int(params['threshold_1']))
			else:
				threshold = params['threshold_1']

			parameters_to_prune = [(child, "weight")] 				
			prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, 
				threshold=threshold, abs_val=params['absolute'])

			prune.remove(child, "weight")
	return model

def nodewise_pruning(model, params):
	for child in model.children():
		if isinstance(child, torch.nn.Linear):

			# Calculate threshold
			if params['percentage']:
				weights = (child.weight.cpu()).detach().numpy()
				row_sums = [np.sum(np.absolute(row)) for row in weights]
				threshold = np.percentile(row_sums, int(params['threshold']))
			else:
				threshold = params['threshold']

			
			mask = torch.zeros_like(child.weight)
			tensor_np = (child.weight.cpu()).detach().numpy()
			
			for i, row in enumerate(tensor_np):
				abs_sum = np.sum(np.absolute(row))
				if abs_sum > threshold:
					mask[i, :] = 1

			prune.custom_from_mask(child, 'weight', mask)
			prune.remove(child, "weight")

	return model

def filter_pruning(model, params):

	for child in model.children():
		if isinstance(child, torch.nn.Conv2d):

			tensor_np = (child.weight.cpu()).detach().numpy()

			# Set norm calculator
			if params['norm'] == 'L1 norm':
				fun_norm = lambda x: np.linalg.norm(x.reshape(-1), ord=1)
			elif params['norm'] == 'Greatest singular value':
				fun_norm = lambda x: np.linalg.svd(x, compute_uv=False) [0]
			else:
				print('Norm calculator not implemented!')

			# Calculate threshold
			if not params['percentage']:
				threshold = float(params['threshold'])
			else:
				norms = []
				for channel in range(tensor_np.shape[0]):
					for in_filter in range(tensor_np.shape[1]):
						norms.append(fun_norm(tensor_np[channel][in_filter]))
				threshold = np.percentile(norms, float(params['threshold']))
				print(threshold, norms)

			# Create pruning mask
			mask = torch.zeros_like(child.weight)
			for channel in range(tensor_np.shape[0]):
				for in_filter in range(tensor_np.shape[1]):

					norm = fun_norm(tensor_np[channel][in_filter])
					if norm > threshold:
						mask[channel, in_filter,:,:] = 1

			# Apply pruning
			prune.custom_from_mask(child, 'weight', mask)
			prune.remove(child, "weight")

	return model


def graph_pruning(model, params):

	def paths_to_filters(path):

		# Extract filters that lie on these paths
		path_filters = {}
		for path in paths:
			for i in range(len(path['path'])-1):
				node1 = parse.parse("l{}_n{}", path['path'][i])
				node2 = parse.parse("l{}_n{}", path['path'][i+1])
			
				l1, n1 = map(int, node1)
				l2, n2 = map(int, node2)

				if l2 in path_filters:
					path_filters[l2].append((n2, n1))
				else:
					path_filters[l2] = [(n2, n1)]
		return path_filters

	def create_mask(weight, filters, layer_id):
		mask = torch.zeros_like(weight)
		for filt in filters[layer_id]:
			mask[filt[0], filt[1], :, :] = 1
		return mask			

	# Construct channel graph
	G = ChannelGraph(model)

	if params['percentage'] and not params['sep_perc']:
		
		perc = 100
		while perc > int(params['value']):
			
			# Find smallest path
			paths = G.getSmallestNPaths(1)
			print(paths)

			# Prune smallest path
			path_filters = paths_to_filters(paths)
			#print(path_filters)
			layer_id = 1
			for child in model.children():
				if isinstance(child, torch.nn.Conv2d):

					# Create mask
					mask = torch.ones_like(child.weight)
					for filt in path_filters[layer_id]:
						mask[filt[0], filt[1], :, :] = 0
					
					# Apply pruning
					prune.custom_from_mask(child, 'weight', mask)
					prune.remove(child, "weight")

					layer_id += 1

			# Remove from channel graph
			G.excludePath(paths[0]['path'])

			# Recalculate percentage
			nr_non_zero = 0
			nr_overall = 0
			for child in model.children():
				if isinstance(child, torch.nn.Conv2d):
					tensor_np = (child.weight.cpu()).detach().numpy()

					nr_non_zero += np.count_nonzero(tensor_np)
					nr_overall += tensor_np.size

			perc = nr_non_zero/nr_overall * 100
			print(perc, '\n')

	elif params['percentage'] and params['sep_perc']:

		percents = list(map(int, params['value'].split(',')))
		target_reached = False
		while not target_reached:
			
			# Find smallest path
			paths = G.getSmallestNPaths(1)
			if len(paths) == 0:
				break
			
			# Remove from channel graph
			G.excludePath(paths[0]['path'])

			# Check if all layers allow to prune this path
			path_filters = paths_to_filters(paths)
			pruning_allowed = True
			layer_id = 1
			with torch.no_grad():
				for child in model.children():
					if isinstance(child, torch.nn.Conv2d):

						# Create mask
						mask = torch.ones_like(child.weight)
						for filt in path_filters[layer_id]:
							mask[filt[0], filt[1], :, :] = 0

						# Weights to binary matrix
						weight_bin = torch.abs(torch.sign(child.weight)).int()

						# Bitwise - And
						res_bin = torch.logical_and(mask, weight_bin)

						# Check if layer allows pruning
						allowed = bool((torch.count_nonzero(res_bin)/torch.numel(res_bin) * 100) > percents[layer_id-1])
						pruning_allowed = pruning_allowed & allowed

						layer_id += 1 

			# Prune, if allowed
			if pruning_allowed:			
				
				#print(path_filters)
				layer_id = 1
				for child in model.children():
					if isinstance(child, torch.nn.Conv2d):

						# Create mask
						mask = torch.ones_like(child.weight)
						for filt in path_filters[layer_id]:
							mask[filt[0], filt[1], :, :] = 0

						# Apply pruning
						prune.custom_from_mask(child, 'weight', mask)
						prune.remove(child, "weight")

						layer_id += 1

			# Check if all layers have reached the target percentage
			target_reached = True
			layer_id = 1
			with torch.no_grad():
				for child in model.children():
					if isinstance(child, torch.nn.Conv2d):
						target_reached &= bool(torch.count_nonzero(child.weight)/torch.numel(child.weight) * 100 <= percents[layer_id-1])

				print([float(torch.count_nonzero(child.weight)/torch.numel(child.weight)*100) for child in model.children() if isinstance(child, torch.nn.Conv2d)])

	else:
		# Extract first n heaviest paths
		paths = G.getHeaviestNPaths(int(params['value']))

		path_filters = paths_to_filters(paths)

		# Prune all paths outside the kept ones
		layer_id = 1
		for child in model.children():
			if isinstance(child, torch.nn.Conv2d):

				# Create mask
				mask = create_mask(child.weight, path_filters, layer_id)

				# Apply pruning
				prune.custom_from_mask(child, 'weight', mask)
				prune.remove(child, "weight")

				layer_id += 1

	return model