from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QGridLayout, QLineEdit, QWidget

from lib.math_eval import Evaluator


class MatrixLineEdit(QLineEdit):
	VALID_CHARACTERS = frozenset({"+", "-", "*", "/", ".", "(", ")"})

	has_error_changed = Signal(bool)

	def __init__(self, parent: QWidget | None = None) -> None:
		"""
		Initialize a MatrixLineEdit widget.

		This custom QLineEdit normalizes certain symbols (e.g., '*' → '·'),
		validates mathematical expressions using the Evaluator, and visually
		indicates errors by applying a red outline.

		Args:
			parent (QWidget | None): Optional parent widget.
		"""
		super().__init__(parent)
		self._has_error: bool = False

		self.setStyleSheet(self.base_style())

		self.textChanged.connect(self.validate_current_text)

	def base_style(self) -> str:
		"""
		Build and return the base stylesheet used for normal and error states.

		Returns:
			**css:** `str`
			A Qt stylesheet string defining normal, error, and focus styles.
		"""
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
		"""
		Update the internal error state, refresh styling, and emit a change signal.

		Args:
			value (bool): Whether the current text is considered invalid.
		"""
		"""Updates the error state, forces a layout paint cycle, and emits a signal."""
		if self._has_error != value:
			self._has_error = value

			self.setProperty("has_error", value)

			self.style().unpolish(self)
			self.style().polish(self)

			self.has_error_changed.emit(value)

	def validate_current_text(self, text: str) -> None:
		"""
		Validate the current text using the Evaluator.
		Empty text clears the error state.

		Args:
			text (str): The raw text from the line edit.

		Returns:
			**result:** `None`
			No return value; updates error state based on expression validity.
		"""
		if not text.strip():
			self.set_has_error(False)
			return

		try:
			Evaluator.evaluate_expression(text)
			self.set_has_error(False)
		except ValueError:
			self.set_has_error(True)

	def keyPressEvent(self, arg__1: QKeyEvent) -> None:
		"""
		Handle key presses, allowing navigation keys, converting '*' to '·',
		and filtering allowed characters.

		Args:
			arg__1 (QKeyEvent): The incoming key event.
		"""
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
	def __init__(self, rows: int = 3, cols: int | None = None, parent: QWidget | None = None):
		"""
		A grid-based matrix input widget composed of MatrixLineEdit cells.
		Supports arbitrary row/column sizes and provides methods to extract
		numeric or raw text matrix data.

		Args:
			rows (int): Number of matrix rows.
			cols (int | None): Number of columns; defaults to rows if None.
			parent (QWidget | None): Optional parent widget.

		Returns:
			**instance:** `MatrixWidget`
			A fully constructed matrix input widget.
		"""
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

	def paintEvent(self, event: QPaintEvent) -> None:
		"""
		Paint decorative matrix brackets around the widget using QPainter.

		Args:
			event (QPaintEvent): The paint event.
		"""
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
		"""
		Extract matrix contents into a float32 NumPy array.

		Empty or invalid cells fall back to `default_val`.

		Args:
			default_val (float): Value used when a cell cannot be evaluated.

		Returns:
			**matrix:** `NDArray`
			A NumPy array of shape (rows, cols) containing evaluated float values.
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
		"""
		Return raw, unevaluated text from each matrix cell.

		Returns:
			**text_grid:** `list[list[str]]`
			A nested list of strings matching the matrix shape.
		"""
		return [[cell.text() for cell in row] for row in self.cells]


class LiveMatrixWidget(MatrixWidget):
	"""MatrixWidget that reports its numeric contents on every edit via on_change."""

	def __init__(self, rows: int, cols: int, on_change: Callable[[NDArray], None] | None = None, parent: QWidget | None = None) -> None:
		"""
		A MatrixWidget that emits its evaluated numeric matrix contents whenever
		any cell finishes editing. Useful for live-updating transformations or
		linked UI components.

		Args:
			rows (int): Number of matrix rows.
			cols (int): Number of matrix columns.
			on_change (Callable[[NDArray], None] | None): Callback invoked with the evaluated matrix whenever a cell changes.
			parent (QWidget | None): Optional parent widget.

		Returns:
			**instance:** `LiveMatrixWidget`
			A matrix widget with automatic change notifications.
		"""
		super().__init__(rows, cols, parent)
		self.on_change = on_change
		for row in self.cells:
			for cell in row:
				cell.editingFinished.connect(self._emit_change)

	def _emit_change(self) -> None:
		"""
		Emit the on_change callback with the current evaluated matrix data.
		"""
		if self.on_change:
			self.on_change(self.get_matrix_data())

	def set_values(self, data: NDArray) -> None:
		"""
		Populate matrix cells from a numeric array without triggering on_change.

		Args:
			data (NDArray): A (rows, cols) array of numeric values.
		"""
		for r, row in enumerate(self.cells):
			for c, cell in enumerate(row):
				cell.blockSignals(True)
				cell.setText(f"{float(data[r, c]):g}")
				cell.blockSignals(False)

	def set_text_values(self, text_grid) -> None:
		"""
		Populate matrix cells from a raw text grid, preserving original expressions
		instead of evaluated values.

		Args:
			text_grid (list[list[str]]): A grid of raw string expressions.
		"""
		for r, row in enumerate(self.cells):
			for c, cell in enumerate(row):
				cell.blockSignals(True)
				cell.setText(str(text_grid[r][c]))
				cell.blockSignals(False)
