import json
import sys
import os

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class DatasetLoader(QTabWidget):

	signal_dataset_ready = pyqtSignal(object)

	def __init__(self):
		super().__init__()

		self.editorTab = self.Editor()
		self.selectorTab = self.Selector()

		self.addTab(self.editorTab, "Editor")
		self.addTab(self.selectorTab, "Select dataset")

		self.currentChanged.connect(self.updateSizes)
		self.selectorTab.dropwdown_dataset.currentIndexChanged.connect(self.dataset_selected)

	def updateSizes(self):
	    for i in range(self.count()):
	        self.widget(i).setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

	    current = self.currentWidget()
	    current.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

	def dataset_selected(self, i):
		dataset = self.selectorTab.dropwdown_dataset.currentText()
		self.signal_dataset_ready.emit(dataset) 


	class Editor(QWidget):
		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			# Add dataset loader
			self.label_datasetloader = QLabel('Dataset loader')
			self.textedit_datasetloader = QTextEdit()
			self.textedit_datasetloader.setFixedHeight(300)

			layout.addWidget(self.label_datasetloader)
			layout.addWidget(self.textedit_datasetloader)


	class Selector(QWidget):
		def __init__(self):
			super().__init__()
			
			layout = QVBoxLayout()
			self.setLayout(layout)

			self.dropwdown_dataset = QComboBox()
			self.dropwdown_dataset.addItem('MNIST-1D')
			self.dropwdown_dataset.addItem('MNIST-2D')

			layout.addWidget(self.dropwdown_dataset)