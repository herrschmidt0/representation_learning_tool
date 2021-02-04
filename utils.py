from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import parse
import numpy as np
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
import importlib.util

import config

class MplCanvas(FigureCanvasQTAgg):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		self.figure = Figure(figsize=(width, height))
		self.figure.tight_layout()

		self.axes = self.figure.subplots()
		#self.axes = self.figure.add_subplot(111)
		super(MplCanvas, self).__init__(self.figure)
	def update(self):
		self.axes.clear()

class StackedWidgetWithComboBox(QWidget):
	def __init__(self):
		super().__init__()
		layout = QVBoxLayout()
		self.setLayout(layout)

		self.combobox = QComboBox()
		self.combobox.setMaximumWidth(250)
		self.stackedw = QStackedWidget()

		self.combobox.currentIndexChanged.connect(lambda i: self.stackedw.setCurrentIndex(i))

		layout.addWidget(self.combobox)
		layout.addWidget(self.stackedw)

	def addItem(self, widget, name):
		self.combobox.addItem(name)
		self.stackedw.addWidget(widget)

		if self.stackedw.count() == 1:
			self.stackedw.setCurrentIndex(0)
			self.combobox.setCurrentIndex(0)


class ChannelGraph(nx.DiGraph):
	excluded_paths = set()

	def __init__(self, model, edge_weight='l1_norm'):
		super().__init__()
		self.model = model

		# Construct graph
		for i, layer in enumerate(model.children()):
			if isinstance(layer, torch.nn.Conv2d):

				self.last_conv_layer = layer
				self.last_conv_layer_id = i+1
				weight_matrix = (layer.weight.cpu()).detach().numpy()
				# Add input channels
				if i == 0:
					for in_channel in range(weight_matrix.shape[1]):
						self.add_node("l{}_n{}".format(i, in_channel))

				# Add channels
				for out_channel in range(weight_matrix.shape[0]):
					self.add_node("l{}_n{}".format(i+1, out_channel))

				# Add edges
				for in_channel in range(weight_matrix.shape[1]):
					for out_channel in range(weight_matrix.shape[0]):
						edge_weight = np.linalg.norm(weight_matrix[out_channel][in_channel].reshape(-1), ord=1)
						filter_nonzero = np.count_nonzero(weight_matrix[out_channel][in_channel])
						if filter_nonzero > 0:
							in_node = "l{}_n{}".format(i, in_channel)
							out_node = "l{}_n{}".format(i+1, out_channel)
							self.add_edge(in_node, out_node, weight=edge_weight)

	#Remove path
	def excludePath(self, path):
		self.excluded_paths.add(path)


	def getPathsFromInputToAllOutput(self, n, weight_fun):
		results = []
		source = "l0_n0"
		for channel in range(self.last_conv_layer.weight.shape[0]):
			
			dest = "l{}_n{}".format(self.last_conv_layer_id, channel)
			path_excluded = True

			paths = nx.shortest_simple_paths(self, source, dest, weight=weight_fun)
			for path in paths:
				#path = nx.dijkstra_path(self, source, dest, weight=weight_fun)
				path = tuple(path)
				#print('No path from {} to {}.'.format(source, dest))

				if path not in self.excluded_paths:
					break

			#print(path, path in self.excluded_paths)
			if path not in self.excluded_paths:
				path_dist = np.sum([self[path[i]][path[i+1]]['weight'] for i in range(len(path)-1) ])
				results.append({
					'sum_weights': path_dist,
					'path': path
				})

		return results 

	# Distinct source & end nodes
	def getHeaviestNPaths(self, n):
		# Get paths, then sort, then get first n
		paths = self.getPathsFromInputToAllOutput(n, lambda s,e,d: 1/d['weight'])
		paths = sorted(paths, key=lambda x: -x['sum_weights'])[:n]
		
		# Exclude these paths from the graph
		'''
		for path in paths:
			self.excluded_paths.add(path)
		'''
		return paths

	def getSmallestNPaths(self, n):
		# Get paths, then sort, then get first n
		paths = self.getPathsFromInputToAllOutput(n, lambda s,e,d: d['weight'])
		paths = sorted(paths, key=lambda x: x['sum_weights'])[:n]
		
		# Exclude these paths from the graph
		'''
		for path in paths:
			self.excluded_paths.add(path)
		'''
		return paths

	def getLayerSizes(self):
		sizes = []
		for i, layer in enumerate(self.model.children()):
			if isinstance(layer, torch.nn.Conv2d):
				if i==0:
					sizes.append(layer.weight.shape[1])
				sizes.append(layer.weight.shape[0])
		return sizes

	def getLastConvLayer(self):
		return self.last_conv_layer, self.last_conv_layer_id


def draw_neural_net(ax, left, right, bottom, top, layer_sizes, graph):

	n_layers = len(layer_sizes)
	v_spacing = (top - bottom)/float(max(layer_sizes))
	h_spacing = (right - left)/float(len(layer_sizes) - 1)
	
	# Nodes
	for n, layer_size in enumerate(layer_sizes):
	    layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
	    for m in range(layer_size):
	        circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
	                            color='w', ec='k', zorder=4)
	        ax.add_artist(circle)

	# Edges
	for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
		layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
		layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
		
		for m in range(layer_size_a):
			for o in range(layer_size_b):
				
				node1 = "l{}_n{}".format(n, m)
				node2 = "l{}_n{}".format(n+1, o)	
				if graph.has_edge(node1, node2):
					line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
					                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
					ax.add_artist(line)


def load_module_from_path(path):
	spec = importlib.util.spec_from_file_location("module.name", path)
	foo = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(foo)
	return foo


def get_trainloader(dataset):

	if isinstance(config.config[dataset]['datasets'], list):
		batch_size = 32
		train_dataset = config.config[dataset]["datasets"][0]
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
				                                   batch_size=batch_size, 
				                                   shuffle=True)
	elif isinstance(config.config[dataset]['datasets'], str):
		loader_path = config.config[dataset]['datasets']
		
		spec = importlib.util.spec_from_file_location("module.name", loader_path)
		foo = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(foo)
		train_loader, test_loader = foo.load()

	return train_loader


def get_testloader(dataset, batch_size):

	# Check if dataset is built-in or custom created from script
	if isinstance(config.config[dataset]['datasets'], list):
		test_dataset = config.config[dataset]["datasets"][1]
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
				                                   batch_size=batch_size, 
				                                   shuffle=True)
	elif isinstance(config.config[dataset]['datasets'], str):
		loader_path = config.config[dataset]['datasets']
		
		spec = importlib.util.spec_from_file_location("module.name", loader_path)
		foo = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(foo)
		train_loader, test_loader = foo.load()

	return test_loader

