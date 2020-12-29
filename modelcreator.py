import json
import sys
import os
import traceback

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelCreator(QTabWidget):

	signal_model_ready = pyqtSignal(object)

	def __init__(self):
		super().__init__()
		
		self.tab1 = self.EditorWidget()
		self.tab2 = self.ModelImporter(self)

		self.addTab(self.tab1, "Editor")
		self.addTab(self.tab2, "Model importer")

		self.currentChanged.connect(self.updateSizes)

	def updateSizes(self):
	    for i in range(self.count()):
	        self.widget(i).setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

	    current = self.currentWidget()
	    current.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

	class EditorWidget(QWidget):

		def __init__(self):
			super().__init__()


			layout = QVBoxLayout()
			self.setLayout(layout)

			# Add dataset loader
			self.label_datasetloader = QLabel('Dataset loader')
			self.textedit_datasetloader = QTextEdit()
			self.textedit_datasetloader.setFixedHeight(300)

			# Add network definition
			self.label_network = QLabel('Network definition')
			self.textedit_network = QTextEdit()
			self.textedit_network.setFixedHeight(300)

			# Add trainer
			self.label_train = QLabel('Train model')
			self.textedit_train = QTextEdit()
			self.textedit_train.setFixedHeight(300)
		

			# Execute button
			button_run = QPushButton("Execute")
			#button_run.clicked.connect()


			layout.addWidget(self.label_datasetloader)
			layout.addWidget(self.textedit_datasetloader)
			layout.addWidget(self.label_network)
			layout.addWidget(self.textedit_network)
			layout.addWidget(self.label_train)
			layout.addWidget(self.textedit_train)
			layout.addWidget(button_run)


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

			button_load_networkdef = QPushButton('Load network definition')
			button_load_networkdef.clicked.connect(self.fun_load_networkdef)
			button_load_networkdef.setMaximumWidth(200)
			layout_import_buttons.addWidget(button_load_networkdef)

			button_save_networkdef = QPushButton('Save network definition')
			button_save_networkdef.clicked.connect(self.fun_save_networkdef)
			button_save_networkdef.setMaximumWidth(200)
			layout_import_buttons.addWidget(button_save_networkdef)


			layout.addWidget(label_network)
			layout.addWidget(self.textedit_network)
			layout.addWidget(widget_import_buttons)

		def fun_import_model(self):
			
			fname = QFileDialog.getOpenFileName(self, 'Open file', '\\',"Model files (*.pth *.pt *.onnx)")
			extension = os.path.splitext(fname[0])[1]

			# Pytorch state dict
			if extension == '.pth' or extension == '.pt':
				
				with open('code-segments.json') as json_file:
					data = json.load(json_file)
					imports = data['pytorch']['imports']
				
				code_definition = self.textedit_network.toPlainText()
				code = imports + '\nglobal Model\n' + code_definition

				with open('network_def.py', 'w') as f:
					f.write(code)

				from network_def import Model
				'''
				try:
					exec(code)
				except Exception as e:
					print(e)
					traceback.print_exc(file=sys.stdout)
				'''
				try:
					model = Model()
					model = model.to(device)
					#model = torch.nn.DataParallel(model)

					data = torch.load(fname[0])
					model.load_state_dict(data['net'])
					#model = model.to(device)
					model.eval()
				except Exception as e:
					print(e)

				#print(type(model.named_parameters()))
				self.outerInstance.signal_model_ready.emit(model)


		def fun_load_networkdef(self):
			fname = QFileDialog.getOpenFileName(self, 'Open file', '\\',"Source files (*.py)")

			with open(fname[0], 'r') as f:
				self.textedit_network.setText(f.read())

		def fun_save_networkdef(self):
			fname = QFileDialog.getSaveFileName(self, 'Save file', '\\',"Source files (*.py)")

			source_code = self.textedit_network.toPlainText()
			with open(fname[0], 'w') as f:
				f.write(source_code)