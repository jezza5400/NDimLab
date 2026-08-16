from collections.abc import Callable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QPushButton, QWidget


class ColorSwatchButton(QPushButton):
	"""Small button showing a color swatch; opens a color picker on click."""

	def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.setFixedSize(28, 22)
		self.color: QColor = color
		self.color_changed: Callable[[QColor], None] | None = None
		self._refresh_style()
		self.clicked.connect(self._pick_color)

	def _refresh_style(self) -> None:
		self.setStyleSheet(f"background-color: {self.color.name()}; border: 1px solid palette(mid);")

	def _pick_color(self) -> None:
		new_color = QColorDialog.getColor(self.color, self, "Pick Color")
		if new_color.isValid():
			self.color = new_color
			self._refresh_style()
			if self.color_changed:
				self.color_changed(self.color)
