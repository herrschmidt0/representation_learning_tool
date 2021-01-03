import json
import sys
import os
import traceback
from graphviz import Graph
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
from torch_pruning import ThresholdPruning

class ModelTransformer(QTabWidget):

	signal_model_ready = pyqtSignal(object)

	def __init__(self):
		super().__init__()

		self.tab_pruning = self.PruningTab(self)

		self.addTab(self.tab_pruning, 'Pruning')

	def receive_model(self, model):
		self.model = model
		print('ModelTransformer received the model.')

		self.tab_pruning.update(model)

	class PruningTab(QWidget):

			pruning_type = None
			pruning_threshold = 5


			def __init__(self, outer):
				super().__init__()
				self.outer = outer

				layout = QVBoxLayout()
				self.setLayout(layout)


				def pruning_changed(i):
					pass

				label_pruningtype = QLabel('Pruning type')
				self.dropwdown_pruningtype = QComboBox()
				self.dropwdown_pruningtype.setMaximumWidth(250)
				self.dropwdown_pruningtype.addItems(['Magnitude - absolute value', 'Magnitude - real value'])
				self.dropwdown_pruningtype.currentIndexChanged.connect(pruning_changed)

				self.button_execute = QPushButton('Prune model!')
				self.button_execute.setMaximumWidth(180)
				self.button_execute.clicked.connect(self.prune_model)
				self.button_execute.setVisible(False)

				label_results = QLabel('Results:')
				self.container_results = QLabel()

				layout.addWidget(label_pruningtype)
				layout.addWidget(self.dropwdown_pruningtype)
				layout.addWidget(self.button_execute)

				self.widget_canvas = QLabel()			
				layout.addWidget(self.widget_canvas)
		
			
			def prune_model(self):
				
				layer1 = next(self.model.children())
				for params in layer1.parameters():
					print(params.shape)

				# Apply pruning
				parameters_to_prune = [(child, "weight") for child in self.model.children() if isinstance(child, torch.nn.Linear)]
				'''
				weights = (weights.cpu()).detach().numpy()
				pos_threshold = np.percentile(weights, 95)
				pruning_method = ThresholdPruning()
				'''
				prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=0.02)


				for child in self.model.children():
					if isinstance(child, torch.nn.Linear):
						prune.remove(child, "weight")


				layer1 = next(self.model.children())
				for params in layer1.parameters():
					print(params.shape)

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