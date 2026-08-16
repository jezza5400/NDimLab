from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QLineEdit, QWidget

from lib.math_eval import Evaluator


class MatrixLineEdit(QLineEdit):
	"""Custom QLineEdit that catches symbols like '*' and converts them to math ones like '·',
	and dynamically outlines red on invalid mathematical input.
	"""

	VALID_CHARACTERS = frozenset({"+", "-", "*", "/", ".", "(", ")"})

	has_error_changed = Signal(bool)

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self._has_error: bool = False

		self.setStyleSheet(self.base_style())

		self.textChanged.connect(self.validate_current_text)

	def base_style(self) -> str:
		pal = self.palette()
		base = pal.color(pal.currentColorGroup(), pal.ColorRole.Base).name()
		text = pal.color(pal.currentColorGroup(), pal.ColorRole.Text).name()
		return f"""
			MatrixLineEdit {{
				border: 2px solid #a0a0a0;
				border-radius: 4px;
				padding: 4px;
				background-color: {base};
				color: {text};
			}}
			MatrixLineEdit[has_error="true"] {{
				border: 2px solid #ef4444;
				background-color: #fef2f2;
				color: #7f1d1d;
			}}
			MatrixLineEdit:focus {{
				border: 2px solid #3b82f6;
			}}
			MatrixLineEdit[has_error="true"]:focus {{
				border: 2px solid #dc2626;
			}}
		"""

	def set_has_error(self, value: bool) -> None:
		"""Updates the error state, forces a layout paint cycle, and emits a signal."""
		if self._has_error != value:
			self._has_error = value

			self.setProperty("has_error", value)

			self.style().unpolish(self)
			self.style().polish(self)

			self.has_error_changed.emit(value)

	def validate_current_text(self, text: str) -> None:
		"""Runs input through the evaluator to update the error state."""
		if not text.strip():
			self.set_has_error(False)
			return

		try:
			Evaluator.evaluate_expression(text)
			self.set_has_error(False)
		except ValueError:
			self.set_has_error(True)

	def keyPressEvent(self, arg__1: QKeyEvent) -> None:
		key = arg__1.key()
		text = arg__1.text()

		is_nav_key = key in (
			Qt.Key.Key_Backspace,
			Qt.Key.Key_Delete,
			Qt.Key.Key_Left,
			Qt.Key.Key_Right,
			Qt.Key.Key_Up,
			Qt.Key.Key_Down,
			Qt.Key.Key_Home,
			Qt.Key.Key_End,
			Qt.Key.Key_Return,
			Qt.Key.Key_Enter,
			Qt.Key.Key_Tab,
		)

		has_ctrl = arg__1.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)

		if is_nav_key or has_ctrl:
			super().keyPressEvent(arg__1)

		elif text == "*":
			self.insert("·")

		elif text and (text.isdigit() or text.isalpha() or text in self.VALID_CHARACTERS):
			super().keyPressEvent(arg__1)


class MatrixWidget(QWidget):
	def __init__(self, rows=3, cols=None, parent=None):
		"""If cols=None, cols will be set to rows"""
		super().__init__(parent)
		self.rows = rows
		self.cols = rows if cols is None else cols

		self.grid = QGridLayout(self)
		self.grid.setSpacing(6)
		self.grid.setContentsMargins(14, 8, 14, 8)

		self.cells = []

		extra_style = """
			MatrixLineEdit {
				font-family: 'Courier New', monospace;
				font-size: 14px;
				font-weight: bold;
				min-width: 45px;
				max-width: 90px;
				height: 26px;
			}
		"""

		for r in range(self.rows):
			row_cells = []
			for c in range(self.cols):
				cell = MatrixLineEdit()
				cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
				cell.setStyleSheet(cell.base_style() + extra_style)
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

		painter.drawLine(offset + bracket_depth, offset, offset, offset)
		painter.drawLine(offset, offset, offset, h - offset)
		painter.drawLine(offset, h - offset, offset + bracket_depth, h - offset)

		painter.drawLine(w - offset - bracket_depth, offset, w - offset, offset)
		painter.drawLine(w - offset, offset, w - offset, h - offset)
		painter.drawLine(w - offset, h - offset, w - offset - bracket_depth, h - offset)

	def get_matrix_data(self, default_val: float = 0.0) -> NDArray:
		"""Extracts matrix contents into float32 NumPy array.

		If a cell is empty or contains an invalid string, it falls back to `default_val`.
		"""
		data = []
		for row in self.cells:
			row_data = []
			for cell in row:
				try:
					value = Evaluator.evaluate_expression(cell.text().strip())
				except ValueError:
					value = default_val
				row_data.append(value)
			data.append(row_data)

		return np.array(data, dtype=np.float32)

	def get_matrix_text(self) -> list[list[str]]:
		"""Raw, unevaluated cell text (e.g. 'sin(90)'), same shape as get_matrix_data()."""
		return [[cell.text() for cell in row] for row in self.cells]


class LiveMatrixWidget(MatrixWidget):
	"""MatrixWidget that reports its numeric contents on every edit via on_change."""

	def __init__(self, rows: int, cols: int, on_change: Callable[[NDArray], None] | None = None, parent: QWidget | None = None) -> None:
		super().__init__(rows, cols, parent)
		self.on_change = on_change
		for row in self.cells:
			for cell in row:
				cell.editingFinished.connect(self._emit_change)

	def _emit_change(self) -> None:
		if self.on_change:
			self.on_change(self.get_matrix_data())

	def set_values(self, data: NDArray) -> None:
		"""Populate cells from an (rows, cols) array without triggering on_change."""
		for r, row in enumerate(self.cells):
			for c, cell in enumerate(row):
				cell.blockSignals(True)
				cell.setText(f"{float(data[r, c]):g}")
				cell.blockSignals(False)

	def set_text_values(self, text_grid) -> None:
		"""Populate cells from (rows, cols) grid of raw strings to redisplay a transformation's original expressions instead of their evaluated value."""
		for r, row in enumerate(self.cells):
			for c, cell in enumerate(row):
				cell.blockSignals(True)
				cell.setText(str(text_grid[r][c]))
				cell.blockSignals(False)
