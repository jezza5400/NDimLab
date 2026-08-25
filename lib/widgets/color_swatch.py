from collections.abc import Callable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QPushButton, QWidget


class ColorSwatchButton(QPushButton):
	"""Small button showing a color swatch; opens a color picker on click."""

	def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
		"""
		A small clickable color-swatch button that displays a QColor and opens a
		QColorDialog when clicked. When the user selects a new color, the button
		updates its appearance and optionally notifies a callback.

		Args:
			color (QColor): The initial color displayed by the swatch.
			parent (QWidget | None): Optional parent widget.

		Returns:
			**instance:** `ColorSwatchButton`
			A button widget that visually represents and edits a color value.
		"""
		super().__init__(parent)
		self.setFixedSize(28, 22)
		self.color: QColor = color
		self.color_changed: Callable[[QColor], None] | None = None
		self._refresh_style()
		self.clicked.connect(self._pick_color)

	def _refresh_style(self) -> None:
		"""
		Update the button's stylesheet to reflect the current color.
		"""
		self.setStyleSheet(f"background-color: {self.color.name()}; border: 1px solid palette(mid);")

	def _pick_color(self) -> None:
		"""
		Open a QColorDialog to allow the user to select a new color. If a valid
		color is chosen, the swatch updates and the `color_changed` callback is
		invoked (if provided).
		"""
		new_color = QColorDialog.getColor(self.color, self, "Pick Color")
		if new_color.isValid():
			self.color = new_color
			self._refresh_style()
			if self.color_changed:
				self.color_changed(self.color)
