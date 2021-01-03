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

		self.addTab(self.tab_overview, 'Overview')
		self.addTab(self.tab_testresults, 'Test results')
		self.addTab(self.tab_graph, 'Graph')
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
		print('Visualizer received the model (& dataset).')

		self.tab_overview.update(model)
		self.tab_testresults.update(dataset, model)
		self.tab_graph.update(model)


	class OverviewTab(QWidget):
		def __init__(self):
			super().__init__()

			self.layout = QVBoxLayout()
			self.setLayout(self.layout)

		def update(self, model):
			
			# Remove previous data
			for i in reversed(range(self.layout.count())): 
				self.layout.itemAt(i).widget().setParent(None)

			# Insert new layer information
			self.layout.addWidget(QLabel('Layers:'))

			for child in model.children():
				

				if isinstance(child, torch.nn.Linear):
					layer_type = 'Linear'
				elif isinstance(child, torch.nn.Conv2d):
					layer_type = '2D Convolutional'
				elif isinstance(child, torch.nn.ReLU):
					layer_type = 'ReLU activation'
				elif isinstance(child, torch.nn.Dropout2d):
					layer_type = '2D Dropout'
				else:
					layer_type = 'Unknown'
					'''
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
					'''
				name = layer_type + ' layer'
				container = FrameLayout(title=name)
				container.addWidget(QLabel('Layer type: %s' % layer_type))

				for name, parameter in child.named_parameters():

					container.addWidget(QLabel('Parameter name: {0}'.format(name)))
					container.addWidget(QLabel('Parameter shape: {0}'.format(parameter.shape,)))

					param_np = (parameter.cpu()).detach().numpy()
					container.addWidget(QLabel('Mean of weights: {:.3f}'.format(np.mean(param_np))))
					container.addWidget(QLabel('Std of weights: {:.3f}'.format(np.std(param_np))))
					container.addWidget(QLabel('Min of weights: {:.3f}'. format(np.min(param_np))))
					container.addWidget(QLabel('Max of weights: {:.3f}'. format(np.max(param_np))))
					container.addWidget(QLabel(' '))

				self.layout.addWidget(container)


			print('Model overview updated.')

	
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
				if dataset in config.config:
					test_dataset = config.config[dataset]["datasets"][1]
				
					test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
					                                          batch_size=self.batch_size, 
					                                          shuffle=True)
					transform = config.config[dataset]["transform"]

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

				accuracy = correct / total

			# Display results
			report = classification_report(y_true, y_pred)
			self.label_results.setText(report)
			print('Test results updated.')

	class GraphTab(QWidget):
		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			self.widget_canvas = QLabel()			
			layout.addWidget(self.widget_canvas)
			#self.painter = QPainter(self.widget_canvas.pixmap())

		def update(self, model):

			# Find input size
			layer1 = next(model.children())

			if isinstance(layer1, nn.Linear):
				for name, param in layer1.named_parameters():
					if name == 'weight':
						in_features = param.shape[1]
						break
				x = torch.randn(16, in_features).to(device)
			elif isinstance(layer1, nn.Conv2d):
				for name, param in layer1.named_parameters():
					if name == 'weight':
						in_channels = param.shape[1]
						break
				x = torch.randn(16, in_channels, 50, 50).to(device)

			y = model(x)
			graph = make_dot(y, params=dict(model.named_parameters()))

			canvas = QPixmap(graph)
			self.widget_canvas.setPixmap(canvas)
			self.widget_canvas.show()

			print('Computation graph updated.')

			'''
			pen = QPen()
			pen.setWidth(40)
			pen.setColor(QColor('red'))
			self.painter.setPen(pen)

			self.painter.drawPoint(200, 150)
			self.painter.end()
			'''
