import json
import sys
import os
import traceback

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from visualizer import Visualizer
from modelcreator import ModelCreator
from dataset import DatasetLoader

class MainWindow(QMainWindow):

	def __init__(self):
		super().__init__()

		# Scroll area inner contents
		body = QWidget()
		body_layout = QVBoxLayout()
		body.setLayout(body_layout)

		self.dataset_loader = DatasetLoader()
		self.model_creator = ModelCreator()
		self.visualizer = Visualizer()

		body_layout.addWidget(self.dataset_loader)
		body_layout.addWidget(self.model_creator)
		body_layout.addWidget(self.visualizer)

		self.dataset_loader.signal_dataset_ready.connect(self.receive_dataset)
		self.model_creator.signal_model_ready.connect(self.receive_model)

		# Add scrollbar
		scrollArea = QScrollArea()
		scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		scrollArea.setWidgetResizable(True)
		scrollArea.setWidget(body)
		#scrollArea.setFixedHeight(600)

		# Central widget
		centralw = QWidget()
		mainLayout = QVBoxLayout()
		mainLayout.addWidget(scrollArea)

		centralw.setLayout(mainLayout)

		self.setWindowTitle("Representation learning tool")	
		#win.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowType_Mask)
		self.showMaximized()
		self.setGeometry(50, 50, 1200, 800)
		self.setCentralWidget(centralw)
		self.show()

		# Load config
		with open('config.json') as json_file:
			self.config = json.load(json_file)

	@pyqtSlot(object)
	def receive_dataset(self, dataset):
		self.dataset = dataset

		if isinstance(dataset, str):
			dataset = dataset.lower()
			if dataset in self.config:
				self.model_path = self.config[dataset]['network-def']
				self.weights_path = self.config[dataset]['weights']

				self.model_creator.receive_dataset_model(self.dataset, self.model_path, self.weights_path)

	@pyqtSlot(object)
	def receive_model(self, model):

		self.visualizer.receive_dataset_model(self.dataset, model)

if __name__ == '__main__':
	app = QApplication([])
	win = MainWindow()
	win.show()
	app.exec_()