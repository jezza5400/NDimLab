from typing import cast
import math
from PySide6.QtOpenGL import QOpenGLWindow
import sys
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat, QWindow, QKeyEvent
from PySide6.QtWidgets import QApplication
import moderngl as mgl
import numpy as np


def load_shader(path: Path) -> str:
	return Path(path).read_text(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
VERTEX_SHADER = load_shader(SCRIPT_DIR / "shaders" / "basic.vert")
FRAGMENT_SHADER = load_shader(SCRIPT_DIR / "shaders" / "basic.frag")


class OpenGLWindow(QOpenGLWindow):
	def __init__(self, parent: QWindow | None = None) -> None:
		super().__init__(parent=parent)
		self.setTitle("OpenGL Stuff")

		self.angle = 0
		self.ctx = cast(mgl.Context, None)
		self.prog = cast(mgl.Program, None)
		self.vbo = cast(mgl.Buffer, None)
		self.vao = cast(mgl.VertexArray, None)

	def initializeGL(self) -> None:
		self.makeCurrent()
		self.ctx = mgl.create_context()
		self.ctx.enable(mgl.DEPTH_TEST)
		self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

		# fmt: off
		VERTEX_DATA = np.array([
			-0.75, -0.75, 0.0, 1.0, 0.0, 0.0,  # Bottom-left vertex (red)
			0.0, 0.75, 0.0, 0.0, 1.0, 0.0,  # Top vertex (green)
			0.75, -0.75, 0.0, 0.0, 0.0, 1.0,  # Bottom-right vertex (blue)
		], dtype="f4")
		# fmt: on

		self.trans = np.eye(4, dtype="f4")

		self.trans_uniform = cast(mgl.Uniform, self.prog["trans"])
		self.aspect_uniform = cast(mgl.Uniform, self.prog["aspect"])

		self.trans_uniform.write(self.trans.tobytes())
		self.aspect_uniform.value = 1.0

		self.vbo = self.ctx.buffer(VERTEX_DATA.tobytes())

		self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f", "inPosition", "inColour")])
		# self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 12x", "inPosition")])

		# Timers
		self.timer = QTimer()
		self.timer.timeout.connect(self.fixed_update)
		self.timer.start(16)

	def paintGL(self) -> None:
		"""Main rendering call handled via ModernGL."""
		self.makeCurrent()

		# Clear color and depth buffers (equiv to glClearColor + glClear)
		self.ctx.clear(0.0, 0.0, 0.0, 1.0)

		# Draw the triangle array
		self.vao.render(mgl.TRIANGLES)

	def resizeGL(self, w: int, h: int) -> None:
		"""Handles viewport sizing changes on window adjustments."""
		if not self.ctx or self.aspect_uniform is None:
			return

		ratio = self.devicePixelRatio()

		if self.ctx:
			self.ctx.viewport = (0, 0, int(w * ratio), int(h * ratio))

		self.aspect_uniform.value = float(w) / float(h) if h > 0 else 1.0

	def keyPressEvent(self, arg__1: QKeyEvent) -> None:
		key = arg__1.key()
		if key == Qt.Key.Key_Escape:
			self.close()
		elif key == Qt.Key.Key_W:
			# Wireframe mode toggle
			self.ctx.wireframe = True
			self.update()
		elif key == Qt.Key.Key_S:
			# Solid mode toggle
			self.ctx.wireframe = False
			self.update()

		super().keyPressEvent(arg__1)

	def fixed_update(self) -> None:
		self.angle += 0.01

		s = math.sin(self.angle)
		c = math.cos(self.angle)

		# Pre transpose later
		self.trans = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="f4").T

		self.trans_uniform.write(self.trans.tobytes())

		self.update()


fmt: QSurfaceFormat = QSurfaceFormat()
fmt.setSamples(4)  # 4x multisampling for anti-aliasing
fmt.setVersion(4, 6)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
fmt.setDepthBufferSize(24)  # 24-bit depth buffer for self.ctx.enable(mgl.DEPTH_TEST) sorting
QSurfaceFormat.setDefaultFormat(fmt)

app = QApplication()
window = OpenGLWindow()
window.show()
sys.exit(app.exec())
