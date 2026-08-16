from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DebugOverlay(QWidget):
	def __init__(self, parent=None) -> None:
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

	def paintEvent(self, event) -> None:
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)
		painter.setPen(Qt.PenStyle.NoPen)
		painter.setBrush(QColor(20, 20, 20, 150))
		painter.drawRoundedRect(self.rect(), 4, 4)
		super().paintEvent(event)

	def update_metrics(self, text) -> None:
		self.tick_duration.setText(text)

	def update_gl_metrics(self, text) -> None:
		self.gl_paint_interval.setText(text)
