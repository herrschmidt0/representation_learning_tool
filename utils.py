from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height))
		fig.tight_layout()

		self.axes = fig.add_subplot(111)
		super(MplCanvas, self).__init__(fig)


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