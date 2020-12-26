import json
import sys
import os
import traceback

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import torch
from torch.autograd import Variable

from torchviz import make_dot
from framelayout import FrameLayout

class Visualizer(QTabWidget):

	def __init__(self):
		super().__init__()

		self.tab_overview = self.OverviewTab()
		self.tab_graph = self.GraphTab()

		self.addTab(self.tab_overview, 'Overview')
		self.addTab(self.tab_graph, 'Graph')
		self.addTab(QLabel('viz 2'), 'Tab 2')
		self.addTab(QLabel('viz 3'), 'Tab 3')

		self.currentChanged.connect(self.updateSizes)

	def updateSizes(self):
	    for i in range(self.count()):
	        self.widget(i).setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

	    current = self.currentWidget()
	    current.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

	@pyqtSlot(object)
	def receive_model(self, model):
		self.model = model

		self.tab_overview.update(model)
		self.tab_graph.update(model)


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

	class GraphTab(QWidget):
		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			
			self.widget_canvas = QLabel()			
			layout.addWidget(self.widget_canvas)
			#self.painter = QPainter(self.widget_canvas.pixmap())

		def update(self, model):

			x = torch.randn(16, 3, 50, 50)
			y = model(x)

			graph = make_dot(y, params=dict(model.named_parameters()))

			print(type(graph))

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