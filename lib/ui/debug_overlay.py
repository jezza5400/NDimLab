from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DebugOverlay(QWidget):
	def __init__(self, parent: QWidget | None = None) -> None:
		"""
		A lightweight translucent overlay widget that displays real-time debug
		metrics such as tick duration and OpenGL frame timing. It is drawn above
		the main OpenGL widget and ignores mouse events.

		Args:
			parent (QWidget | None): Optional parent widget.
		"""
		super().__init__(parent)

		self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
		self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(6, 4, 6, 4)
		layout.setSpacing(2)

		label_style = "color: #00FF00; background: transparent;"
		self.tick_duration = QLabel("Tick Duration: _")
		self.tick_duration.setStyleSheet(label_style)
		self.gl_paint_interval = QLabel("GL Frame Interval: _")
		self.gl_paint_interval.setStyleSheet(label_style)

		layout.addWidget(self.tick_duration)
		layout.addWidget(self.gl_paint_interval)

		self.setFixedSize(170, 50)

	def paintEvent(self, event: QPaintEvent) -> None:
		"""
		Paint a translucent rounded background behind the debug text labels.

		Args:
			event (QPaintEvent): The paint event.
		"""
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)
		painter.setPen(Qt.PenStyle.NoPen)
		painter.setBrush(QColor(20, 20, 20, 150))
		painter.drawRoundedRect(self.rect(), 4, 4)
		super().paintEvent(event)

	def update_metrics(self, text: str) -> None:
		"""
		Update the tick-duration label with new diagnostic text.

		Args:
			text (str): The text to display for tick duration.
		"""
		self.tick_duration.setText(text)

	def update_gl_metrics(self, text: str) -> None:
		"""
		Update the OpenGL frame-interval label with new diagnostic text.

		Args:
			text (str): The text to display for GL frame timing.
		"""
		self.gl_paint_interval.setText(text)
