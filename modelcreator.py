import json
import sys
sys.path.append('trainers')
sys.path.append('dataset_loaders')
import os
import traceback
import importlib
import importlib.util

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import config
from utils import get_trainloader

class ModelCreator(QTabWidget):

	signal_model_ready = pyqtSignal(object)

	def __init__(self):
		super().__init__()
		
		self.editor = self.EditorWidget(self)
		self.importer = self.ModelImporter(self)

		self.addTab(self.editor, "Editor")
		self.addTab(self.importer, "Model importer")

		self.currentChanged.connect(self.updateSizes)

	def __del__(self):
		os.remove('network_def.py')		

	def updateSizes(self):
	    for i in range(self.count()):
	        self.widget(i).setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

	    current = self.currentWidget()
	    current.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

	def receive_dataset_model(self, dataset):
		self.dataset = dataset

		if 'network-def' in config.config[dataset]:
			model_path = config.config[dataset]['network-def']	
			with open(model_path, 'r') as f:
				code_network_def = f.read()

			self.editor.textedit_network.setText(code_network_def)
			self.importer.textedit_network.setText(code_network_def)

		if 'train' in config.config[dataset]:
			trainer_path = config.config[dataset]['train'] 
			with open(trainer_path, 'r') as f:
				train_code = f.read()
			self.editor.textedit_train.setText(train_code)

		if 'weights' in config.config[dataset]:
			weights_path = config.config[dataset]['weights']
			model = self.importer.construct_model(weights_path)

			if model != -1:
				self.signal_model_ready.emit(model)


	class EditorWidget(QWidget):

		def __init__(self, outer):
			super().__init__()
			self.outer = outer

			layout = QVBoxLayout()
			self.setLayout(layout)


			# Add network definition
			label_network = QLabel('Network definition')
			self.textedit_network = QTextEdit()
			self.textedit_network.setFixedHeight(300)

			# Add trainer
			label_train = QLabel('Train model')
			self.textedit_train = QTextEdit()
			self.textedit_train.setFixedHeight(300)
		
			# Nr of epochs
			label_epochs = QLabel('Number of epochs:')
			self.edit_epochs = QLineEdit()
			self.edit_epochs.setMaximumWidth(75)

			# Execute button
			button_run = QPushButton("Train model!")
			button_run.setMaximumWidth(180)
			button_run.clicked.connect(self.fun_train_model)

			self.button_export = QPushButton("Save trained weights")
			self.button_export.setMaximumWidth(180)
			self.button_export.clicked.connect(self.fun_export_model)
			self.button_export.setVisible(False)

			# Training info
			self.label_epoch = QLabel('Epoch: -')
			self.label_loss = QLabel('Loss: -')

			layout.addWidget(label_network)
			layout.addWidget(self.textedit_network)
			layout.addWidget(label_train)
			layout.addWidget(self.textedit_train)
			layout.addWidget(label_epochs)
			layout.addWidget(self.edit_epochs)
			layout.addWidget(button_run)
			layout.addWidget(self.label_epoch)
			layout.addWidget(self.label_loss)
			layout.addWidget(self.button_export)

		def fun_train_model(self):

			# Save the edited network definition to temporary module (network_def.py)
			'''
			with open('code-segments.json') as json_file:
				data = json.load(json_file)
				imports = data['pytorch']['imports']
			'''

			code_definition = self.textedit_network.toPlainText()
			code = '\nglobal Model\n' + code_definition
			print('writing code to network_def', len(code))
			with open('network_def.py', 'w') as f:
				f.write(code)

			# Save the edited training code block to temporary module (train.py)
			code_train = self.textedit_train.toPlainText()
			code = '\nglobal Model\n' + code_train
			with open('train.py', 'w') as f:
				f.write(code)

			# Create and run trainer thread
			self.thread = QThread()
			self.worker = self.TrainWorker(self.outer.dataset, int(self.edit_epochs.text()))

			self.worker.moveToThread(self.thread)

			self.thread.started.connect(self.worker.run)
			self.worker.finished.connect(self.thread.quit)
			self.worker.finished.connect(self.worker.deleteLater)
			self.worker.finished_with_model.connect(self.fun_train_finished)
			self.thread.finished.connect(self.thread.deleteLater)

			self.thread.start()

		def fun_train_finished(self, model):

			self.outer.signal_model_ready.emit(model)
			self.model = model
			self.button_export.setVisible(True)

		def fun_export_model(self):

			filename, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Model Files (*pt *pth)")

			torch.save(self.model.state_dict(), filename)


		class TrainWorker(QObject):
			finished = pyqtSignal()
			finished_with_model = pyqtSignal(object)
			progress = pyqtSignal(int)

			def __init__(self, dataset, epochs):
				super().__init__()
				self.dataset = dataset
				self.epochs = epochs	

			def run(self):
				import network_def
				import train

				# Check if dataset is built-in or custom created from script
				train_loader = get_trainloader(self.dataset)

				model = network_def.Model()
				model = model.to(device)

				model = train.train(model, [train_loader, device, self.epochs])

				self.finished_with_model.emit(model)

	class ModelImporter(QWidget):

		def __init__(self, outerInstance):
			super().__init__()
			self.outerInstance = outerInstance

			self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred);
			self.resize(self.minimumSizeHint());
			self.adjustSize()

			layout = QVBoxLayout()
			self.setLayout(layout)

			# Add network definition
			label_network = QLabel('Network definition')
			self.textedit_network = QTextEdit()
			self.textedit_network.setFixedHeight(500)

			# Model buttons
			widget_import_buttons = QWidget()
			layout_import_buttons = QHBoxLayout()
			widget_import_buttons.setLayout(layout_import_buttons)

			# Load state dict file
			button_import = QPushButton('Import model weights')
			button_import.clicked.connect(self.fun_import_model)
			button_import.setMaximumWidth(200)
			layout_import_buttons.addWidget(button_import)

			# Load network code
			button_load_networkdef = QPushButton('Load network definition')
			button_load_networkdef.clicked.connect(self.fun_load_networkdef)
			button_load_networkdef.setMaximumWidth(200)
			layout_import_buttons.addWidget(button_load_networkdef)

			# Save network code to file
			button_save_networkdef = QPushButton('Save network definition')
			button_save_networkdef.clicked.connect(self.fun_save_networkdef)
			button_save_networkdef.setMaximumWidth(200)
			layout_import_buttons.addWidget(button_save_networkdef)

			# Info label about imported weights file
			self.label_imported_weights = QLabel()

			# Add ui elements
			layout.addWidget(label_network)
			layout.addWidget(self.textedit_network)
			layout.addWidget(widget_import_buttons)
			layout.addWidget(self.label_imported_weights)

		def fun_import_model(self):
			
			fname = QFileDialog.getOpenFileName(self, 'Open file', '\\',"Model files (*.pth *.pt *.onnx)")
			
			model = self.construct_model(fname[0])

			if model != -1:
				self.outerInstance.signal_model_ready.emit(model)

		def construct_model(self, path):
			print('Constructing model from weights...')
			
			# Pytorch state dict
			extension = os.path.splitext(path)[1]
			if extension == '.pth' or extension == '.pt':
				self.label_imported_weights.setText('Weights file: ' + path)

				with open('code-segments.json') as json_file:
					data = json.load(json_file)
					imports = data['pytorch']['imports']
				
				code_definition = self.textedit_network.toPlainText()
				code = imports + '\nglobal Model\n' + code_definition

				with open('network_def.py', 'w') as f:
					f.write(code)
				import network_def
				importlib.reload(network_def)

				try:
					model = network_def.Model()
					model = model.to(device)
					#model = torch.nn.DataParallel(model)

					data = torch.load(path)
					#model.load_state_dict(data['net'])
					model.load_state_dict(data)
					model.eval()
				except Exception as e:
					print('Exception:', e)

				return model
			return -1

		def fun_load_networkdef(self):
			fname = QFileDialog.getOpenFileName(self, 'Open file', '\\',"Source files (*.py)")

			with open(fname[0], 'r') as f:
				self.textedit_network.setText(f.read())

		def fun_save_networkdef(self):
			fname = QFileDialog.getSaveFileName(self, 'Save file', '\\',"Source files (*.py)")

			source_code = self.textedit_network.toPlainText()
			with open(fname[0], 'w') as f:
				f.write(source_code)