import json
import sys
import os
import traceback
from graphviz import Graph
import numpy as np
from sklearn.metrics import classification_report

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from torchviz import make_dot, resize_graph
from framelayout import FrameLayout
from torch_pruning import ThresholdPruning
import config


class Visualizer(QTabWidget):

	def __init__(self):
		super().__init__()

		self.tab_overview = self.OverviewTab()
		self.tab_testresults = self.TestResultsTab()
		self.tab_graph = self.GraphTab()
		self.tab_pruning = self.PruningTab()

		self.addTab(self.tab_overview, 'Overview')
		self.addTab(self.tab_testresults, 'Test results')
		self.addTab(self.tab_graph, 'Graph')
		self.addTab(self.tab_pruning, 'Pruning')
		self.addTab(QLabel('viz 3'), 'Tab 3')

		self.currentChanged.connect(self.updateSizes)

	def updateSizes(self):
	    for i in range(self.count()):
	        self.widget(i).setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

	    current = self.currentWidget()
	    current.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

	#@pyqtSlot(object, object)
	def receive_dataset_model(self, dataset, model):
		self.model = model
		self.dataset = dataset

		self.tab_overview.update(model)
		self.tab_testresults.update(dataset, model)
		self.tab_graph.update(model)
		self.tab_pruning.update(model)


	class OverviewTab(QWidget):
		def __init__(self):
			super().__init__()

			self.layout = QVBoxLayout()
			self.setLayout(self.layout)

		def update(self, model):
			
			self.layout.addWidget(QLabel('Layers:'))
			for name, parameter in model.named_parameters():
				
				container = FrameLayout(title=name)
				container.addWidget(QLabel("Parameter shape: %s" % str(parameter.shape)))

				if len(parameter.shape) == 4:
					container.addWidget(QLabel('Browse filters:'))
					combobox_filters = QComboBox()
					combobox_filters.addItems([('Filter '+str(i)) for i in range(parameter.shape[0])])
					container.addWidget(combobox_filters)
					label_weights = QLabel()
					container.addWidget(label_weights)

					def filter_selected(i):
						print(parameter.shape)
						iiid_filter = parameter.narrow(1, 0, parameter.shape[1])
						iiid_filter_print = Variable(iiid_filter).data[0]
						label_weights.setText(iiid_filter_print)
					combobox_filters.currentIndexChanged.connect(filter_selected)

				self.layout.addWidget(container)

	
	class TestResultsTab(QWidget):
		batch_size = 32

		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			self.label_results = QLabel()
			layout.addWidget(self.label_results)

		def update(self, dataset, model):

			# Define dataset loader
			if isinstance(dataset, str):
				dataset = dataset.lower()
				if dataset in config_dataset:
					test_dataset = config_dataset[dataset]["datasets"][1]
				
					test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
					                                          batch_size=self.batch_size, 
					                                          shuffle=True)
					transform = config_dataset[dataset]["transform"]

			# Test model on dataset
			model.eval()
			y_true = []
			y_pred = []
			with torch.no_grad():
				correct = 0
				total = 0
				for images, labels in test_loader:
					images = transform(images)
					images = images.to(device)
					labels = labels.to(device)

					outputs = model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

					y_true += labels.tolist()
					y_pred += predicted.tolist()
					#print(labels.data, predicted)

				accuracy = correct / total

			# Display results
			report = classification_report(y_true, y_pred)
			self.label_results.setText(report)	

	class GraphTab(QWidget):
		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			self.widget_canvas = QLabel()			
			layout.addWidget(self.widget_canvas)
			#self.painter = QPainter(self.widget_canvas.pixmap())

		def update(self, model):

			#inner_model = next(model.children())
			layer1 = next(model.children())

			if isinstance(layer1, nn.Linear):
				in_features = (next(layer1.parameters())).shape[1]
				x = torch.randn(16, in_features).to(device)
			elif isinstance(layer1, nn.Conv2d):
				x = torch.randn(16, layer1.shape[1], 50, 50).to(device)

			y = model(x)
			graph = make_dot(y, params=dict(model.named_parameters()))

			canvas = QPixmap(graph)
			self.widget_canvas.setPixmap(canvas)
			self.widget_canvas.show()

			'''
			pen = QPen()
			pen.setWidth(40)
			pen.setColor(QColor('red'))
			self.painter.setPen(pen)

			self.painter.drawPoint(200, 150)
			self.painter.end()
			'''


	class PruningTab(QWidget):
		dataset = None
		pruning_type = None
		pruning_threshold = 5
		batch_size = 32

		def __init__(self):
			super().__init__()
			
			layout = QVBoxLayout()
			self.setLayout(layout)

			def dataset_changed(i):
				# which dataset was selected?
				# MNIST dataset 
				test_dataset = torchvision.datasets.MNIST(root='../data', 
				                                          train=False, 
				                                          transform=transforms.ToTensor(),
				                                          download=True)
				# Data loader
				self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
				                                          batch_size=self.batch_size, 
				                                          shuffle=True)
				self.transform = lambda x: x.reshape(-1, 28*28)


			label_dataset = QLabel('Dataset:')
			self.dropwdown_dataset = QComboBox()
			self.dropwdown_dataset.addItem('MNIST-1D')
			self.dropwdown_dataset.addItem('MNIST-2D')
			dataset_changed(0)
			self.dropwdown_dataset.currentIndexChanged.connect(dataset_changed)


			def pruning_changed(i):
				pass

			label_pruningtype = QLabel('Pruning type')
			self.dropwdown_pruningtype = QComboBox()
			self.dropwdown_pruningtype.addItems(['Magnitude - absolute value', 'Magnitude - real value'])
			self.dropwdown_pruningtype.currentIndexChanged.connect(pruning_changed)

			self.button_execute = QPushButton('Prune model!')
			self.button_execute.setMaximumWidth(180)
			self.button_execute.clicked.connect(self.prune_model)
			self.button_execute.setVisible(False)

			label_results = QLabel('Results:')
			self.container_results = QLabel()

			layout.addWidget(label_dataset)
			layout.addWidget(self.dropwdown_dataset)
			layout.addWidget(label_pruningtype)
			layout.addWidget(self.dropwdown_pruningtype)
			layout.addWidget(self.button_execute)
			layout.addWidget(label_results)
			layout.addWidget(self.container_results)

			self.widget_canvas = QLabel()			
			layout.addWidget(self.widget_canvas)

			
		
		def prune_model(self):
			
			# Apply pruning

			# Test
			self.model.eval()
			with torch.no_grad():
				correct = 0
				total = 0
				for images, labels in self.test_loader:
					images = self.transform(images)
					images = images.to(device)
					labels = labels.to(device)

					outputs = self.model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

					print(labels.data, predicted)

				accuracy = correct / total

			# Display results
			self.container_results.setText('Accuracy: %.3f' % accuracy) 

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