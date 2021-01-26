import json
import sys
import os
import traceback
import copy
import parse
import numpy as np
import networkx as nx

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import prune

import config
from torch_pruning import ThresholdPruning
from utils import ChannelGraph


class ModelTransformer(QTabWidget):

	signal_model_ready = pyqtSignal(object)

	def __init__(self):
		super().__init__()

		self.tab_reset = self.ResetTab()
		self.tab_pruning = self.PruningTab(self)

		self.addTab(self.tab_reset, 'Reset')
		self.addTab(self.tab_pruning, 'Pruning')

		self.tab_reset.button_reset.clicked.connect(self.reset_model)

	def receive_model(self, model):
		self.original_model = copy.deepcopy(model)
		self.model = model
		print('ModelTransformer received the model.')

		self.tab_pruning.update(model)

	def reset_model(self):
		self.model = copy.deepcopy(self.original_model)
		self.tab_pruning.update(self.model)
		self.signal_model_ready.emit(self.model)

	class ResetTab(QWidget):
		def __init__(self):
			super().__init__()		
			layout = QVBoxLayout()
			self.setLayout(layout)

			self.button_reset = QPushButton('Reset model')
			self.button_reset.setMaximumWidth(100)
			layout.addWidget(self.button_reset)

	class PruningTab(QWidget):

			method = None
			methods = [
				'Magnitude pruning', 
				'Node-wise pruning', 
				'Filter-pruning',
				'Graph pruning'
			]

			def __init__(self, outer):
				super().__init__()
				self.outer = outer

				layout = QVBoxLayout()
				self.setLayout(layout)


				def pruning_changed(i):
					self.method = i
					self.controls.setCurrentIndex(i)

				label_pruningtype = QLabel('Pruning type')
				self.dropwdown_pruningtype = QComboBox()
				self.dropwdown_pruningtype.setMaximumWidth(250)
				self.dropwdown_pruningtype.addItems(self.methods)
				self.dropwdown_pruningtype.currentIndexChanged.connect(pruning_changed)
				self.dropwdown_pruningtype.setCurrentIndex(0)
				self.method = 0

				self.controls = QStackedWidget()
				self.controls.addWidget(self.MagnitudePruningControls())
				self.controls.addWidget(self.NodeWisePruningControls())
				self.controls.addWidget(self.FilterNormPruning())
				self.controls.addWidget(self.GraphPruningControls())

				self.button_execute = QPushButton('Prune model!')
				self.button_execute.setMaximumWidth(180)
				self.button_execute.clicked.connect(lambda: self.prune_model(self.controls.currentWidget().getValues()))
				self.button_execute.setVisible(False)

				layout.addWidget(label_pruningtype)
				layout.addWidget(self.dropwdown_pruningtype)
				layout.addWidget(self.controls)
				layout.addWidget(self.button_execute)

		
			class MagnitudePruningControls(QWidget):
				def __init__(self):
					super().__init__()

					layout = QVBoxLayout()
					self.setLayout(layout)

					layout.addWidget(QLabel('Unstructured magnitude pruning of dense layer weights.'))

					self.checkbox_absolute = QCheckBox('Absolute value')
					self.checkbox_absolute.setChecked(True)

					self.checkbox_percentage = QCheckBox('Percentage')
					self.checkbox_percentage.setChecked(True)

					label_threshold = QLabel('Magnitude threshold (value/percentage):')
					self.edit_threshold_1 = QLineEdit()
					#onlyInt = QIntValidator()
					#self.edit_threshold_1.setValidator(onlyInt)
					self.edit_threshold_1.setMaximumWidth(100)

					self.edit_threshold_2 = QLineEdit()
					#self.edit_threshold_2.setValidator(onlyInt)
					self.edit_threshold_2.setMaximumWidth(100)
					self.edit_threshold_2.setVisible(False)

					layout.addWidget(self.checkbox_absolute)
					layout.addWidget(self.checkbox_percentage)
					layout.addWidget(label_threshold)
					layout.addWidget(self.edit_threshold_1)
					#layout.addWidget(self.edit_threshold_2)

					def checkbox_abs_changed():
						self.edit_threshold_2.setVisible(not self.edit_threshold_2.isVisible())
					self.checkbox_absolute.stateChanged.connect(checkbox_abs_changed)


				def getValues(self):
					threshold_2 = None
					if len(self.edit_threshold_2.text())>0:
						threshold_2 = float(self.edit_threshold_2.text())
					params = {
						'absolute': self.checkbox_absolute.isChecked(),
						'threshold_1': float(self.edit_threshold_1.text()),
						'threshold_2': threshold_2,
						'percentage': self.checkbox_percentage.isChecked()
					}
					return params

			class NodeWisePruningControls(QWidget):
				def __init__(self):
					super().__init__()

					layout = QVBoxLayout()
					self.setLayout(layout)

					layout.addWidget(QLabel('Pruning of whole nodes with smaller total weight sum (dense layers).'))

					self.checkbox_perc = QCheckBox('Percentage')
					self.checkbox_perc.setChecked(True)
					label_threshold = QLabel('Threshold (value/percentage):')
					self.edit_threshold = QLineEdit()
					self.edit_threshold.setMaximumWidth(150)

					layout.addWidget(self.checkbox_perc)
					layout.addWidget(label_threshold)
					layout.addWidget(self.edit_threshold)

				def getValues(self):
					return {
						'percentage': self.checkbox_perc.isChecked(),
						'threshold': self.edit_threshold.text()
					}

			class FilterNormPruning(QWidget):
				def __init__(self):
					super().__init__()

					layout = QVBoxLayout()
					self.setLayout(layout)

					layout.addWidget(QLabel('Filter pruning of Conv2D layers based on individual filter norms/measures.'))

					label_combo = QLabel('Select norm/measure:')
					self.combo_measure = QComboBox()
					self.combo_measure.setMaximumWidth(180)
					self.combo_measure.addItems(['L1 norm', 'Greatest singular value'])

					label_value = QLabel('Value:')
					self.edit_value = QLineEdit()
					self.edit_value.setMaximumWidth(150)

					self.checkbox_perc = QCheckBox('Percentage')
					self.checkbox_perc.setChecked(True)

					layout.addWidget(label_combo)
					layout.addWidget(self.combo_measure)
					layout.addWidget(label_value)
					layout.addWidget(self.edit_value)
					layout.addWidget(self.checkbox_perc)

				def getValues(self):
					return {
						'threshold': self.edit_value.text(),
						'percentage': self.checkbox_perc.isChecked(),
						'norm': self.combo_measure.currentText()
					}

			class GraphPruningControls(QWidget):
				def __init__(self):
					super().__init__()

					layout = QVBoxLayout()
					self.setLayout(layout)

					layout.addWidget(QLabel('Filter pruning of Conv2D layers: keep n paths (from diff input channels to different output channels) with highest sum of weights. \n When percentage is checked: prune smallest paths until overall non-zero percentage drops to specified value.'))
	
					self.checkbox_percentage = QCheckBox('Percentage')
					self.checkbox_percentage.setChecked(True)

					self.checkbox_separate_perc = QCheckBox('Separate percentages for each layer')
					self.checkbox_separate_perc.setChecked(False)

					label_value = QLabel('Percentage(s) [comma separated] / Nr of paths:')
					self.edit_value = QLineEdit()
					self.edit_value.setMaximumWidth(200)		

					layout.addWidget(label_value)
					layout.addWidget(self.checkbox_percentage)
					layout.addWidget(self.checkbox_separate_perc)
					layout.addWidget(self.edit_value)

				def getValues(self):
					return {
						'percentage': self.checkbox_percentage.isChecked(),
						'sep_perc': self.checkbox_separate_perc.isChecked(),
						'value': self.edit_value.text()
					}						

########################################################################

			def prune_model(self, params):

				# Apply pruning
				if self.method == 0:
				
					for child in self.model.children():
						if isinstance(child, torch.nn.Linear):

							if params['percentage']:
								weights = (child.weight.cpu()).detach().numpy()
								threshold = np.percentile(np.abs(weights), int(params['threshold_1']))
							else:
								threshold = params['threshold_1']

							print(threshold)
							parameters_to_prune = [(child, "weight")] 				
							prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, 
								threshold=threshold, abs_val=params['absolute'])

							prune.remove(child, "weight")

				elif self.method == 1:

					for child in self.model.children():
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

							# Prune
							#parameters_to_prune = [(child, 'weight')]
							#prune.global_unstructured(parameters_to_prune, pruning_method=NodeWisePruning, threshold=threshold, shape=child.weight.shape)

							prune.custom_from_mask(child, 'weight', mask)
							prune.remove(child, "weight")

				elif self.method == 2:

					for child in self.model.children():
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

				elif self.method == 3:

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
					G = ChannelGraph(self.model)

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
							for child in self.model.children():
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
							for child in self.model.children():
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
								for child in self.model.children():
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
								for child in self.model.children():
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
								for child in self.model.children():
									if isinstance(child, torch.nn.Conv2d):
										target_reached &= bool(torch.count_nonzero(child.weight)/torch.numel(child.weight) * 100 <= percents[layer_id-1])

								print([float(torch.count_nonzero(child.weight)/torch.numel(child.weight)*100) for child in self.model.children() if isinstance(child, torch.nn.Conv2d)])

					else:
						# Extract first n heaviest paths
						paths = G.getHeaviestNPaths(int(params['value']))

						path_filters = paths_to_filters(paths)

						# Prune all paths outside the kept ones
						layer_id = 1
						for child in self.model.children():
							if isinstance(child, torch.nn.Conv2d):

								# Create mask
								mask = create_mask(child.weight, path_filters, layer_id)

								# Apply pruning
								prune.custom_from_mask(child, 'weight', mask)
								prune.remove(child, "weight")

								layer_id += 1

				# Send model to tester/visualizer
				self.outer.signal_model_ready.emit(self.model)

			def update(self, model):
				self.model = model
				self.button_execute.setVisible(True)



				'''
				node_attr = dict(style='filled',
	                 shape='circle',
	                 align='left',
	                 fontsize='12',
	                 ranksep='0.1',
	                 height='0.2')
				
				dot = Graph(node_attr=node_attr, graph_attr=dict(size="12,12"), format='png')

				inner_model = next(model.children())
				for i, layer in enumerate(inner_model.children()):

					if isinstance(layer, nn.Linear):
						weights = next(layer.parameters())
						in_features = weights.shape[1]
						out_features = weights.shape[0]


						for m in range(in_features):
							for n in range(m, out_features):
								dot.edge('n_'+str(i)+'_'+str(m), 'n_'+str(i+1)+'_'+str(n))			

						weights = (weights.cpu()).detach().numpy()
						pos_threshold = np.percentile(weights, 95)
						neg_threshold = np.percentile(weights, 05)
						
						for m, row in enumerate(weights):
							for n, w in enumerate(row):
								if w > pos_threshold:
									
									print('Positive:', w)

									if i == 0:
										with dot.subgraph(name='layer0') as inp:
												for j in range(in_features):
													inp.node('n_'+str(i)+'_'+str(j))
									

									with dot.subgraph(name='layer'+str(i+1)) as layer:
											for j in range(out_features):
												layer.node('n_'+str(i+1)+'_'+str(j))
									
									dot.edge('n_'+str(i)+'_'+str(m), 'n_'+str(i+1)+'_'+str(n))
									
								if w < neg_threshold:
									print('Negative:', w)



				resize_graph(dot)
				png = dot.render()

				canvas = QPixmap(png)
				self.widget_canvas.setPixmap(canvas)
				self.widget_canvas.show()
				'''