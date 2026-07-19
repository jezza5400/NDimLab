import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QVBoxLayout


class MatrixLineEdit(QLineEdit):
	"""Custom QLineEdit that catches symbols like '*' and converts them to math ones like '·'."""

	def keyPressEvent(self, arg__1: QKeyEvent) -> None:
		if arg__1.text() == "*":
			self.insert("·")
		else:
			super().keyPressEvent(arg__1)


class MatrixWidget(QWidget):
	def __init__(self, rows=3, cols=3, parent=None):
		super().__init__(parent)
		self.rows = rows
		self.cols = cols

		self.grid = QGridLayout(self)
		self.grid.setSpacing(6)
		self.grid.setContentsMargins(14, 8, 14, 8)

		self.cells = []

		cell_style = """
			QLineEdit {
				border: 1px solid palette(mid);
				background-color: palette(base);
				color: palette(text);
				font-family: 'Courier New', monospace;
				font-size: 14px;
				font-weight: bold;
				min-width: 45px;
				max-width: 90px;
				height: 26px;
			}
			QLineEdit:focus {
				border: 1px solid palette(highlight);
				background-color: palette(window);
			}
		"""

		for r in range(self.rows):
			row_cells = []
			for c in range(self.cols):
				cell = MatrixLineEdit()
				cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
				cell.setStyleSheet(cell_style)
				self.grid.addWidget(cell, r, c)
				row_cells.append(cell)
			self.cells.append(row_cells)

	def paintEvent(self, event):
		super().paintEvent(event)
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)

		text_color = self.palette().color(self.foregroundRole())

		pen = QPen(text_color, 2)
		painter.setPen(pen)

		w = self.width()
		h = self.height()
		bracket_depth = 8
		offset = 2

		# Left Bracket [
		painter.drawLine(offset + bracket_depth, offset, offset, offset)
		painter.drawLine(offset, offset, offset, h - offset)
		painter.drawLine(offset, h - offset, offset + bracket_depth, h - offset)

		# Right Bracket ]
		painter.drawLine(w - offset - bracket_depth, offset, w - offset, offset)
		painter.drawLine(w - offset, offset, w - offset, h - offset)
		painter.drawLine(w - offset, h - offset, w - offset - bracket_depth, h - offset)

	def get_matrix_data(self):
		return [[cell.text() for cell in row] for row in self.cells]


class MainWindow(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Matrix Demo")

		layout = QVBoxLayout(self)

		self.matrix = MatrixWidget(rows=3, cols=3)
		layout.addWidget(self.matrix, alignment=Qt.AlignmentFlag.AlignCenter)

		self.matrix2 = MatrixWidget(rows=4, cols=4)
		layout.addWidget(self.matrix2, alignment=Qt.AlignmentFlag.AlignCenter)


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.resize(300, 200)
	window.show()
	sys.exit(app.exec())
