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
from utils import ChannelGraph, get_trainloader, get_testloader, load_module_from_path
from torch_pruning import magnitude_pruning, nodewise_pruning, filter_pruning, graph_pruning
import torch_quantization as torchquant
import distiller_quantization as distillerquant
import tensorrt_quantization as tensorrtquant

class ModelTransformer(QTabWidget):

	signal_model_ready = pyqtSignal(object, object)

	def __init__(self):
		super().__init__()

		self.tab_reset = self.ResetTab()
		self.tab_pruning = self.PruningTab(self)
		self.tab_quant_torch = self.QuantizationPytorch(self)
		self.tab_quant_distiller = self.QuantizationDistiller(self)
		self.tab_xnor = self.XNOR(self)
		self.tab_tensorrt = self.QuantizationTensorRT(self)

		self.addTab(self.tab_reset, 'Reset')
		self.addTab(self.tab_pruning, 'Pruning(PyTorch)')
		self.addTab(self.tab_quant_torch, 'Quantization(PyTorch)')
		self.addTab(QWidget(), 'Pruning(Distiller)')
		self.addTab(self.tab_quant_distiller, 'Quantization(Distiller)')
		self.addTab(self.tab_xnor, 'XNOR')
		self.addTab(self.tab_tensorrt, 'Quantization(TensorRT)')

		self.tab_reset.button_reset.clicked.connect(self.reset_model)

	def receive_model(self, dataset, model):
		self.dataset = dataset
		self.original_model = copy.deepcopy(model)
		self.model = model
		print('ModelTransformer received the model.')

		self.tab_pruning.update(model)
		self.tab_quant_torch.update(dataset, model)
		self.tab_quant_distiller.update(dataset, model)
		self.tab_xnor.update(dataset, model)
		self.tab_tensorrt.update(dataset, model)

	def reset_model(self):
		self.model = copy.deepcopy(self.original_model)
		self.tab_pruning.update(self.model)
		self.tab_quant_torch.update(self.dataset, self.model)
		self.tab_quant_distiller.update(self.dataset, self.model)
		self.tab_xnor.update(self.dataset, self.model)
		self.tab_tensorrt.update(self.dataset, self.model)

		self.signal_model_ready.emit(self.model, dict())


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

			self.controls = QStackedWidget()
			self.controls.addWidget(self.MagnitudePruningControls())
			self.controls.addWidget(self.NodeWisePruningControls())
			self.controls.addWidget(self.FilterNormPruning())
			self.controls.addWidget(self.GraphPruningControls())

			label_pruningtype = QLabel('Pruning type')
			self.dropwdown_pruningtype = QComboBox()
			self.dropwdown_pruningtype.setMaximumWidth(250)
			self.dropwdown_pruningtype.addItems(self.methods)
			self.dropwdown_pruningtype.currentIndexChanged.connect(self.controls.setCurrentIndex)
			self.dropwdown_pruningtype.setCurrentIndex(0)

			self.button_execute = QPushButton('Prune model!')
			self.button_execute.setMaximumWidth(180)
			self.button_execute.clicked.connect(lambda: self.prune_model(self.dropwdown_pruningtype.currentIndex(), \
				self.controls.currentWidget().getValues()))
			self.button_execute.setVisible(False)

			layout.addWidget(label_pruningtype)
			layout.addWidget(self.dropwdown_pruningtype)
			layout.addWidget(self.controls)
			layout.addWidget(self.button_execute)

		def prune_model(self, methodid, params):

			# Magnitude pruning
			if methodid == 0:
				self.model = magnitude_pruning(self.model, params)

			# Node-pruning
			elif methodid == 1:
				self.model = nodewise_pruning(self.model, params)
				
			# Filter pruning
			elif methodid == 2:
				self.model = filter_pruning(self.model, params)

			# Smallest path pruning
			elif methodid == 3:
				self.model = graph_pruning(self.model, params)
			
			# Send model to tester/visualizer
			self.outer.signal_model_ready.emit(self.model, dict())

		def update(self, model):
			self.model = model
			self.button_execute.setVisible(True)

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


	class QuantizationPytorch(QTabWidget):

		methods = [
			'Post Training Static',
			'Dynamic',
			'Quantization Aware Training'
		]

		def __init__(self, outer):
			super().__init__()
			self.outer = outer
			layout = QVBoxLayout()
			self.setLayout(layout)

			self.label_support = QLabel()
			self.label_support.setText('Supported quantizer engines: {}'.format(torch.backends.quantized.supported_engines))

			self.controls = QStackedWidget()
			self.controls.addWidget(self.PostStatic())
			self.controls.addWidget(self.Dynamic())
			self.controls.addWidget(self.QAT())

			label_quanttype = QLabel('Quantization type')
			self.dropwdown_quanttype = QComboBox()
			self.dropwdown_quanttype.setMaximumWidth(250)
			self.dropwdown_quanttype.addItems(self.methods)
			self.dropwdown_quanttype.currentIndexChanged.connect(self.controls.setCurrentIndex)
			self.dropwdown_quanttype.setCurrentIndex(0)

			self.button_execute = QPushButton('Quantize model!')
			self.button_execute.setMaximumWidth(180)
			self.button_execute.clicked.connect(lambda: self.execute(self.dropwdown_quanttype.currentIndex(), \
				self.controls.currentWidget().getValues()))
			self.button_execute.setVisible(False)

			layout.addWidget(self.label_support)
			layout.addWidget(label_quanttype)
			layout.addWidget(self.dropwdown_quanttype)
			layout.addWidget(self.controls)
			layout.addWidget(self.button_execute)

		def update(self, dataset, model):
			self.dataset = dataset
			self.model = model
			self.button_execute.setVisible(True)

		def execute(self, methodid, params):

			if methodid == 0:
				train_loader = get_trainloader(self.dataset)
				self.model = torchquant.quantize_static_post(self.model, train_loader, params)
			elif methodid == 1:
				self.model = torchquant.quantize_dynamic(self.model, params)
			elif methodid == 2:
				train_loader = get_trainloader(self.dataset)
				train_module = load_module_from_path(config.config[self.dataset]['train'])
				self.model = torchquant.quantize_qat(self.model, train_loader=train_loader, \
					train_fn=train_module.train, params=params)

			# Send model to tester/visualizer
			params['quantized_cpu'] = True
			self.outer.signal_model_ready.emit(self.model, params)

		class PostStatic(QWidget):

			observers = [
				'MinMax Observer',
				'Moving Average MinMax Observer',
				'Per Channel MinMax Observer',
				'Moving Average Per Channel MinMax Observer',
				'Histogram Observer'
			]

			def __init__(self):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				desc = QLabel('Post Training Static Quantization in Pytorch.' + \
					' Only supports 8-bit integer target weights on the CPU.')
				layout.addWidget(desc)

				label_observer = QLabel('Observer:')
				self.combobox_observers = QComboBox()
				self.combobox_observers.addItems(self.observers)
				self.combobox_observers.setMaximumWidth(250)

				self.checkbox_fuse = QCheckBox('Fuse modules')

				layout.addWidget(label_observer)
				layout.addWidget(self.combobox_observers)
				layout.addWidget(self.checkbox_fuse)

			def getValues(self):
				return {
					'observer': self.combobox_observers.currentIndex(),
					'fuse': self.checkbox_fuse.isChecked()
				}

		class Dynamic(QWidget):
			def __init__(self):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				desc = QLabel('Dynamic Quantization in Pytorch. Only supports 8-bit integer target weights on the CPU.')
				layout.addWidget(desc)

			def getValues(self):
				return dict()

		class QAT(QWidget):
			def __init__(self):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				desc = QLabel('Quantization Aware Training. Re-training is required.' + \
					' Only supports 8-bit integer target weights on the CPU.' )
				layout.addWidget(desc)

			def getValues(self):
				return dict()


	class QuantizationDistiller(QTabWidget):
		methods = [
			'Post Training Static'
		]

		def __init__(self, outer):
			super().__init__()
			self.outer = outer
			layout = QVBoxLayout()
			self.setLayout(layout)

			from torch.onnx.symbolic_helper import _export_onnx_opset_version as opset_versionx
			self.label = QLabel()
			self.label.setText('Quantization using the Distiller library. ' + \
				'ONNX opset version: ' + str(opset_versionx))

			self.controls = QStackedWidget()
			self.controls.addWidget(self.PostStatic())

			label_quanttype = QLabel('Quantization type')
			self.dropwdown_quanttype = QComboBox()
			self.dropwdown_quanttype.setMaximumWidth(250)
			self.dropwdown_quanttype.addItems(self.methods)
			self.dropwdown_quanttype.currentIndexChanged.connect(self.controls.setCurrentIndex)
			self.dropwdown_quanttype.setCurrentIndex(0)

			self.button_execute = QPushButton('Quantize model!')
			self.button_execute.setMaximumWidth(180)
			self.button_execute.clicked.connect(lambda: self.execute(self.dropwdown_quanttype.currentIndex(), \
				self.controls.currentWidget().getValues()))
			self.button_execute.setVisible(False)

			layout.addWidget(self.label)
			layout.addWidget(label_quanttype)
			layout.addWidget(self.dropwdown_quanttype)
			layout.addWidget(self.controls)
			layout.addWidget(self.button_execute)

		def update(self, dataset, model):
			self.dataset = dataset
			self.model = model
			self.button_execute.setVisible(True)

		def execute(self, methodid, params):

			if methodid == 0:
				train_loader = get_trainloader(self.dataset)
				self.model = distillerquant.quantize_static_post(self.model, train_loader, params)

			params['quantized_cpu'] = True
			self.outer.signal_model_ready.emit(self.model, params)

		class PostStatic(QWidget):
			def __init__(self):
				super().__init__()
				layout = QVBoxLayout()
				self.setLayout(layout)

				label_mode = QLabel('Quantization mode')
				self.dropdown_mode = QComboBox()
				self.dropdown_mode.setMaximumWidth(200)
				self.dropdown_mode.addItems(['Symmetric', 'Symmetric-Restricted', 'Assymetric-Unsigned', 'Assymetric-Signed'])

				label_bits_weights = QLabel('Weight bits')
				self.edit_bits_weights = QLineEdit()
				self.edit_bits_weights.setMaximumWidth(50)

				label_bits_act = QLabel('Activation bits')
				self.edit_bits_act = QLineEdit()
				self.edit_bits_act.setMaximumWidth(50)

				self.check_per_channel = QCheckBox('Per-channel quantization')

				layout.addWidget(label_mode)
				layout.addWidget(self.dropdown_mode)
				layout.addWidget(label_bits_weights)
				layout.addWidget(self.edit_bits_weights)
				layout.addWidget(label_bits_act)
				layout.addWidget(self.edit_bits_act)
				layout.addWidget(self.check_per_channel)


			def getValues(self):
				return {
					'mode': self.dropdown_mode.currentIndex(),
					'bits-weight': int(self.edit_bits_weights.text()),
					'bits-act': int(self.edit_bits_act.text()),
					'per-channel': self.check_per_channel.isChecked()
				}

	class XNOR(QWidget):
		
		def __init__(self, outer):
			super().__init__()
			self.outer = outer
			layout = QVBoxLayout()
			self.setLayout(layout)

			self.label_info = QLabel('PyTorch implementation of "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks".' + \
					' Applicable only to Convolutional Nets.')
			layout.addWidget(self.label_info)

			self.button_execute = QPushButton('Quantize model!')
			self.button_execute.setMaximumWidth(150)
			self.button_execute.setVisible(False)
			layout.addWidget(self.button_execute)

		def update(self, dataset, model):
			self.dataset = dataset
			self.model = model
			self.button_execute.setVisible(True)


	class QuantizationTensorRT(QWidget):

		def __init__(self, outer):
			super().__init__()
			self.outer = outer
			layout = QVBoxLayout()
			self.setLayout(layout)

			label_info = QLabel('TensorRT low-precision inference. The model is converted to an internal, low-level representation.')
			layout.addWidget(label_info)

			label_dropdown = QLabel('Low-precision type:')
			layout.addWidget(label_dropdown)
			self.dropdown = QComboBox()
			self.dropdown.setMaximumWidth(200)
			self.dropdown.addItems(['16-bit float', '8-bit integer'])
			layout.addWidget(self.dropdown)

			self.button_execute = QPushButton('Quantize model!')
			self.button_execute.setMaximumWidth(150)
			self.button_execute.setVisible(False)
			self.button_execute.clicked.connect(self.execute)
			layout.addWidget(self.button_execute)

			self.label_results = QLabel('Results:')
			layout.addWidget(self.label_results)

		def update(self, dataset, model):
			self.dataset = dataset
			self.model = model
			self.button_execute.setVisible(True)

		def get_values(self):
			return {
				'method': self.dropdown.currentIndex()
			}

		def execute(self):

			train_loader = get_testloader(self.dataset, batch_size=32)
			inference_res = tensorrtquant.tensorrt_compression(self.model, train_loader, self.get_values())

			text_results = 'Results:\nAccuracy: {:.3f}\nMacro average - Precision: {:.3f}, Recall: {:.3f}\nWeighted average - Precision: {:.3f}, Recall: {:.3f}'.format(
				inference_res['accuracy'], inference_res['macro avg']['precision'], inference_res['macro avg']['recall'],
				inference_res['weighted avg']['precision'], inference_res['weighted avg']['recall'])
			self.label_results.setText(text_results)
			#self.outer.signal_model_ready.emit(self.model, params)