import json
import sys
import os
import traceback
import importlib.util
from graphviz import Graph
import numpy as np
from sklearn.metrics import classification_report
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
#import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from torchviz import make_dot, resize_graph
from framelayout import FrameLayout
from torch_pruning import ThresholdPruning
import config
from utils import MplCanvas, StackedWidgetWithComboBox, ChannelGraph, draw_neural_net, get_testloader


class Visualizer(QTabWidget):

	def __init__(self):
		super().__init__()

		self.tab_overview = self.OverviewTab()
		self.tab_testresults = self.TestResultsTab()
		self.tab_comp_graph = self.ComputationGraphTab()
		self.tab_channel_graph = self.ChannelGraphTab()

		self.addTab(self.tab_overview, 'Overview')
		self.addTab(self.tab_testresults, 'Test results')
		self.addTab(self.tab_comp_graph, 'Computation Graph')
		self.addTab(self.tab_channel_graph, 'Channel Graph')

		self.currentChanged.connect(self.updateSizes)

	def updateSizes(self):
	    for i in range(self.count()):
	        self.widget(i).setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

	    current = self.currentWidget()
	    current.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

	#@pyqtSlot(object, object)
	def receive_dataset_model(self, dataset, model, params=dict()):
		self.model = model
		self.dataset = dataset
		print('Visualizer received the model (& dataset).')

		self.tab_overview.update(model)
		self.tab_testresults.update(dataset, model, params)
		self.tab_comp_graph.update(dataset, model, params)
		self.tab_channel_graph.update(model)

######### OVERVIEW TAB #####################################################

	class OverviewTab(QWidget):
		def __init__(self):
			super().__init__()

			self.layout = QVBoxLayout()
			self.setLayout(self.layout)

		def update(self, model):
			
			#print(self.dataset)

			# Remove previous data
			for i in reversed(range(self.layout.count())): 
				self.layout.itemAt(i).widget().setParent(None)

			# Summary data
			self.label_totalparam = QLabel()
			total_nr_params = 0
			self.layout.addWidget(self.label_totalparam)

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
				elif isinstance(child, torch.nn.Flatten):
					layer_type = 'Flattening'
				elif 'DeQuantStub' in str(child):
					layer_type = 'DeQuantStub'
				elif 'QuantStub' in str(child):
					layer_type = 'QuantStub'
				#elif 'QuantizedLinear' in str(child):
					#layer_type = 'QuantizedLinear'
				else:
					layer_type = str(child)

				name = layer_type + ' layer'
				container = FrameLayout(title=name)
				container.addWidget(QLabel('Layer type: %s' % layer_type))

				for name, parameter in child.named_parameters():

					# Parameter name, shape
					container.addWidget(QLabel('Parameter name: {0}'.format(name)))
					container.addWidget(QLabel('Parameter shape: {0}'.format(parameter.shape,)))
					total_nr_params += torch.numel(parameter)

					# Statistical data about the weight matrices/vectors
					param_np = (parameter.cpu()).detach().numpy()

					weight_statistics = self.WeightStatistics(param_np)
					container.addWidget(weight_statistics)

					# Selector for further diagrams/tables/info 
					stacked_combo = StackedWidgetWithComboBox()
					container.addWidget(stacked_combo)
					

					if isinstance(child, torch.nn.Linear):

						weights_dist = self.WeightDistribution(param_np)
						stacked_combo.addItem(weights_dist, 'Distribution of weights')

						if name == 'weight':
							node_sums = self.WeightSumsNodeWise(param_np)
							stacked_combo.addItem(node_sums, 'Sum of weights for each node')				

					elif isinstance(child, torch.nn.Conv2d):
						if name == 'weight':

							table_norms = self.FilterNormsTable(param_np)
							stacked_combo.addItem(table_norms, 'L1 norms of filters')

							table_svds = self.FilterSVDsTable(param_np)
							stacked_combo.addItem(table_svds, 'Singular values of filters')

						if name == 'bias':
							weights_dist = self.WeightDistribution(param_np)
							stacked_combo.addItem(weights_dist, 'Distribution of weights')


					container.addWidget(QLabel(' '))
				self.layout.addWidget(container)

			self.label_totalparam.setText('Total number of parameters: {}'.format(total_nr_params))			
			print('Model overview updated.')

	
		class WeightStatistics(QWidget):
			def __init__(self, weight):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				nonzero_weights = np.count_nonzero(weight)
				total_weights = weight.size
				layout.addWidget(QLabel('Number of non-zero weights: {} ({:.1f}%)'.format(nonzero_weights, 100*nonzero_weights/total_weights)))
				layout.addWidget(QLabel('Mean of weights: {:.3f}'.format(np.mean(weight))))
				layout.addWidget(QLabel('Std of weights: {:.3f}'.format(np.std(weight))))
				layout.addWidget(QLabel('Min of weights: {:.3f}'.format(np.min(weight))))
				layout.addWidget(QLabel('Max of weights: {:.3f}'.format(np.max(weight))))

		class WeightDistribution(QWidget):
			def __init__(self, weight):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				percentile_90 = np.percentile(weight, 90)
				percentile_10 = np.percentile(weight, 10)

				weights_dist = MplCanvas(self, width=5, height=4)
				weights_dist.axes.hist(weight.reshape(-1), bins=50, range=(percentile_10, percentile_90))
				weights_dist.axes.set_title('Distribution of weights (from 10 percentile to 90)')

				layout.addWidget(weights_dist)

		class WeightSumsNodeWise(QWidget):
			def __init__(self, weight):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				row_sums = [np.sum(np.absolute(row)) for row in weight]
				node_sums = MplCanvas(self, width=5, height=4)
				node_sums.axes.plot(row_sums)
				node_sums.axes.set_title('Sum of weights for each node')

				layout.addWidget(node_sums)

		class FilterNormsTable(QTableWidget):
			def __init__(self, weight, norm=1):
				super().__init__()

				nr_of_channels = weight.shape[0]
				nr_of_input_channels = weight.shape[1]

				self.setRowCount(nr_of_channels)
				self.setColumnCount(nr_of_input_channels)

				for i in range(nr_of_channels):
					for j in range(nr_of_input_channels):
						l1_norm = np.linalg.norm(np.reshape(weight[i][j], (-1,)), ord=norm)
						cell = QTableWidgetItem("{:.2f}".format(l1_norm))
						cell.setFlags(Qt.ItemIsEnabled)
						self.setItem(i, j, cell)							

		class FilterSVDsTable(QTableWidget):
			def __init__(self, weight, norm=1):
				super().__init__()

				nr_of_channels = weight.shape[0]
				nr_of_input_channels = weight.shape[1]

				self.setRowCount(nr_of_channels)
				self.setColumnCount(nr_of_input_channels)

				for i in range(nr_of_channels):
					for j in range(nr_of_input_channels):
						
						svds = np.linalg.svd(weight[i][j], compute_uv=False)

						cell = QTableWidgetItem(str(["{:.2f}".format(sv) for sv in svds]))
						cell.setFlags(Qt.ItemIsEnabled)
						self.setItem(i, j, cell)	

######### TEST RESULT TAB ####################################################################################################3

	class TestResultsTab(QWidget):
		batch_size = 32

		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			self.label_results = QLabel()
			layout.addWidget(self.label_results)

		def update(self, dataset, model, params):

			# CPU or GPU
			if torch.cuda.is_available() and \
				('quantized_cpu' not in params or ('quantized_cpu' in params and not params['quantized_cpu'])):
				device = 'cuda'
			else:
				device = 'cpu'
			print('Device:', device)
			#model.to(device)

			# Get dataset loader
			if isinstance(dataset, str):
				dataset = dataset.lower()
				if dataset in config.config:
					test_loader = get_testloader(dataset, self.batch_size)
					#transform = config.config[dataset]["transform"]

			# Test model on dataset
			model.eval()
			y_true = []
			y_pred = []
			with torch.no_grad():
				correct = 0
				total = 0
				for data, labels in test_loader:
					data = data.to(device)
					labels = labels.to(device)

					outputs = model(data)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

					y_true += labels.tolist()
					y_pred += predicted.tolist()

				accuracy = correct / total

			# Display results
			report = classification_report(y_true, y_pred, digits=3)
			self.label_results.setText(report)
			print('Test results updated.')

######## COMP GRAPH TAB ##########################################

	class ComputationGraphTab(QWidget):
		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			self.widget_canvas = QLabel()			
			layout.addWidget(self.widget_canvas)

		def update(self, dataset, model, params):

			# CPU or GPU
			if torch.cuda.is_available() and \
				('quantized_cpu' not in params or ('quantized_cpu' in params and not params['quantized_cpu'])):
				device = 'cuda'
			else:
				device = 'cpu'

			# Get input shape
			test_loader = get_testloader(dataset, 4)
			for test_images, _ in test_loader:  
				sample = test_images[0]
				break

			x = torch.randn(16, *sample.shape).to(device)

			# Find input size
			'''
			for layer in model.children():
				print(layer)
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.QuantizedLinear):		
					in_features = layer.weight.shape[1]				
					x = torch.randn(16, in_features).to(device)
					break
				elif isinstance(layer, nn.Conv2d):

					# Check if dataset is built-in or custom created from script
					test_loader = get_testloader(dataset, 4)
					for test_images, _ in test_loader:  
						sample = test_images[0]

					x = torch.randn(16, sample.shape[0], sample.shape[1], sample.shape[2]).to(device)
					break
			'''
			y = model(x)
			graph = make_dot(y, params=dict(model.named_parameters()))

			canvas = QPixmap(graph)
			self.widget_canvas.setPixmap(canvas)
			self.widget_canvas.show()

			print('Computation graph updated.')

####### CHANNEL GRAPH TAB ##########################

	class ChannelGraphTab(QWidget):
		def __init__(self):
			super().__init__()

			self.layout = QVBoxLayout()
			self.setLayout(self.layout)

			self.canvas_graph = MplCanvas(width=5, height=15)
			
			self.label_longestpaths = QLabel()

			self.layout.addWidget(self.canvas_graph)
			self.layout.addWidget(self.label_longestpaths)

		def update(self, model):

			has_conv_layers = np.any([isinstance(layer, torch.nn.Conv2d) for layer in model.children()])

			if has_conv_layers:
				# Construct graph
				G = ChannelGraph(model)

				# Heaviest paths
				last_conv_layer, last_conv_layer_id = G.getLastConvLayer()
				text_longestpaths = ''
				source = "l0_n0"
				for channel in range(last_conv_layer.weight.shape[0]):
					
					dest = "l{}_n{}".format(last_conv_layer_id, channel)
					#shortest_path = nx.dijkstra_path(G, source, dest)
					#print('Shortest path:', shortest_path)

					# Try Dijkstra (fail if there is no path to dest)
					try:
						heaviest_path = nx.dijkstra_path(G, source, dest, weight=lambda s,e,d: 1/d['weight'])
						path_dist = np.sum([G[heaviest_path[i]][heaviest_path[i+1]]['weight'] for i in range(len(heaviest_path)-1) ])
						text_longestpaths += 'Longest path from "{}" to "{}" is:  {:.2f} ({})\n'.format(source, dest, path_dist, heaviest_path)
					except:
						pass
						#print('No path.')

				self.label_longestpaths.setText(text_longestpaths)

				# Draw graph
				# https://gist.github.com/craffel/2d727968c3aaebd10359
				# https://stackoverflow.com/questions/58511546/in-python-is-there-a-way-to-use-networkx-to-display-a-neural-network-in-the-sta
				'''
				self.canvas_graph.update()
				pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
				nx.draw(G, with_labels=True, node_size=900, pos=pos, ax=self.canvas_graph.axes)
				'''
				ax = self.canvas_graph.figure.add_subplot(111)
				ax.clear()
				#ax = self.canvas_graph.axes
				#ax.clear()
				#ax.set_axis_off()

				draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, G.getLayerSizes(), G)
				self.canvas_graph.draw_idle()
				#self.canvas_graph.axes.draw()
				#self.canvas_graph.show()
				#self.canvas_graph.draw()
			else:
				self.label_longestpaths.setText('No Conv layers in the model.')
				