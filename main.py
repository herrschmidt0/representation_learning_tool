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

class MainWindow(QMainWindow):

	def __init__(self):
		super().__init__()


		# Scroll area inner contents
		body = QWidget()
		body_layout = QVBoxLayout()
		body.setLayout(body_layout)

		self.model_creator = ModelCreator()

		self.visualizer = Visualizer()

		body_layout.addWidget(self.model_creator)
		body_layout.addWidget(self.visualizer)

		self.model_creator.signal_model_ready.connect(self.visualizer.receive_model)

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

class ModelCreator(QTabWidget):

	signal_model_ready = pyqtSignal(list)

	def __init__(self):
		super().__init__()
		
		self.tab1 = self.EditorWidget()
		self.tab2 = self.ModelImporter(self)

		self.addTab(self.tab1, "Editor")
		self.addTab(self.tab2, "Model importer")


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
			self.label_network = QLabel('Network definition')
			self.textedit_network = QTextEdit()
			self.textedit_network.setFixedHeight(500)

			# Load state dict file
			self.button_import = QPushButton('Import model')
			self.button_import.clicked.connect(self.open_browser)

			layout.addWidget(self.label_network)
			layout.addWidget(self.textedit_network)
			layout.addWidget(self.button_import)

		def open_browser(self):
			
			fname = QFileDialog.getOpenFileName(self, 'Open file', '\\',"Model files (*.pth *.onnx)")
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
					model = torch.nn.DataParallel(model)

					data = torch.load(fname[0])
					model.load_state_dict(data['net'])
				except Exception as e:
					print(e)

				print(type(model.named_parameters()))
				self.outerInstance.signal_model_ready.emit(list(model.named_parameters()))

				for name, param in model.named_parameters():
					print(name, param.shape)

				model.eval()

class Visualizer(QTabWidget):

	def __init__(self):
		super().__init__()

		self.tab_overview = self.OverviewTab()
		self.tab_graph = self.GraphTab()

		self.addTab(self.tab_overview, 'Overview')
		self.addTab(self.tab_graph, 'Graph')
		self.addTab(QLabel('viz 2'), 'Tab 2')
		self.addTab(QLabel('viz 3'), 'Tab 3')


	@pyqtSlot(list)
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
			for name, parameter in model:
				
				label = QLabel(name)
				self.layout.addWidget(label)

	class GraphTab(QWidget):
		def __init__(self):
			super().__init__()

			layout = QVBoxLayout()
			self.setLayout(layout)

			self.widget_canvas = QLabel()
			canvas = QPixmap(400, 300)
			self.widget_canvas.setPixmap(canvas)
			
			layout.addWidget(self.widget_canvas)

			self.painter = QPainter(self.widget_canvas.pixmap())
		
		def update(self, model):

			pen = QPen()
			pen.setWidth(40)
			pen.setColor(QColor('red'))
			self.painter.setPen(pen)

			self.painter.drawPoint(200, 150)
			self.painter.end()



if __name__ == '__main__':
	app = QApplication([])
	win = MainWindow()
	win.show()
	app.exec_()