import json
import sys
import os
import traceback
import copy
import numpy as np

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
from torch_pruning import ThresholdPruning, NodeWisePruning


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

			def __init__(self, outer):
				super().__init__()
				self.outer = outer

				layout = QVBoxLayout()
				self.setLayout(layout)


				def pruning_changed(i):
					print(i)
					self.method = i
					self.controls.setCurrentIndex(i)

				label_pruningtype = QLabel('Pruning type')
				self.dropwdown_pruningtype = QComboBox()
				self.dropwdown_pruningtype.setMaximumWidth(250)
				self.dropwdown_pruningtype.addItems(['Magnitude pruning', 'Node-wise pruning', 'Filter-pruning'])
				self.dropwdown_pruningtype.currentIndexChanged.connect(pruning_changed)
				self.dropwdown_pruningtype.setCurrentIndex(0)
				self.method = 0

				self.controls = QStackedWidget()
				self.controls.addWidget(self.MagnitudePruningControls())
				self.controls.addWidget(self.NodeWisePruningControls())
				self.controls.addWidget(self.FilterNormPruning())

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

					layout.addWidget(label_combo)
					layout.addWidget(self.combo_measure)
					layout.addWidget(label_value)
					layout.addWidget(self.edit_value)

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

							# Prune
							parameters_to_prune = [(child, 'weight')]
							#prune.global_unstructured(parameters_to_prune, pruning_method=NodeWisePruning, threshold=threshold, shape=child.weight.shape)

							#tensor = torch.reshape(tensor, self.shape)
							mask = torch.zeros_like(child.weight)
							tensor_np = (child.weight.cpu()).detach().numpy()
							
							for i, row in enumerate(tensor_np):
								abs_sum = np.sum(np.absolute(row))

								#print(abs_sum, self.threshold)
								if abs_sum > threshold:
									mask[i, :] = 1

							#pruned = child.weight * mask.float()

							prune.custom_from_mask(child, 'weight', mask)

							#print(child.weight)
							prune.remove(child, "weight")

				elif self.method == 2:

					pass

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