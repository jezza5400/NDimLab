from shutil import move
from dataclasses import dataclass
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
GRID_VERT_SHADER = load_shader(SCRIPT_DIR / "shaders" / "grid.vert")
GRID_FRAG_SHADER = load_shader(SCRIPT_DIR / "shaders" / "grid.frag")
TEXTURE_VERT = load_shader(SCRIPT_DIR / "shaders" / "texture.vert")
TEXTURE_FRAG = load_shader(SCRIPT_DIR / "shaders" / "texture.frag")

# fmt: off
BG_VERTS = np.array([
	-1.0, -1.0, 1.0, -1.0, 1.0, 1.0,
	-1.0, -1.0, 1.0, 1.0, -1.0, 1.0
], dtype="f4")
# fmt: on


class OpenGLWindow(QOpenGLWindow):
	def __init__(self, parent: QWindow | None = None) -> None:
		super().__init__(parent=parent)
		self.setTitle("OpenGL Stuff")

		self.zoom_level = 30
		self.OG_ZOOM = self.zoom_level
		self.camera_pos = (0, 0)

		self.ctx = cast(mgl.Context, None)

		# --- Background Grid ---
		self.bg_prog = cast(mgl.Program, None)
		self.bg_vbo = cast(mgl.Buffer, None)
		self.bg_vao = cast(mgl.VertexArray, None)

		# --- Framebuffer and texture storage
		self.fbo = cast(mgl.Framebuffer, None)
		self.grid_texture = cast(mgl.Texture, None)

		# --- Shader and VAO to draw static texture
		self.tex_prog = cast(mgl.Program, None)
		self.tex_vao = cast(mgl.VertexArray, None)

	def initializeGL(self) -> None:
		self.makeCurrent()
		self.ctx = mgl.create_context()
		self.ctx.enable(mgl.DEPTH_TEST)

		self.bg_prog = self.ctx.program(vertex_shader=GRID_VERT_SHADER, fragment_shader=GRID_FRAG_SHADER)
		self.u_bg_resolution = cast(mgl.Uniform, self.bg_prog["u_resolution"])
		self.u_bg_zoom = cast(mgl.Uniform, self.bg_prog["u_zoom"])
		self.u_bg_camera_pos = cast(mgl.Uniform, self.bg_prog["u_camera_pos"])

		self.bg_vbo = self.ctx.buffer(BG_VERTS.tobytes())
		self.bg_vao = self.ctx.vertex_array(self.bg_prog, [(self.bg_vbo, "2f", "inPosition")])

		self.tex_prog = self.ctx.program(vertex_shader=TEXTURE_VERT, fragment_shader=TEXTURE_FRAG)
		self.tex_vao = self.ctx.vertex_array(self.tex_prog, [(self.bg_vbo, "2f", "inPosition")])

		# --- Timers ---
		# self.timer = QTimer()
		# self.timer.timeout.connect(self.fixed_update)
		# self.timer.start(16)

	def paintGL(self) -> None:
		self.makeCurrent()

		# Clear color and depth buffers of on-screen framebuffer
		self.ctx.screen.use()
		self.ctx.clear(0.0, 0.0, 0.0, 1.0)

		# Pass 1: Render static baked background image
		self.grid_texture.use(location=0)

		# Disable depth testing while drawing background to avoid masking foreground elements
		self.ctx.disable(mgl.DEPTH_TEST)
		self.tex_vao.render(mgl.TRIANGLES)
		self.ctx.enable(mgl.DEPTH_TEST)

	def resizeGL(self, w: int, h: int) -> None:
		ratio = self.devicePixelRatio()
		phys_w, phys_h = int(w * ratio), int(h * ratio)
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		# Rebake background grid texture
		self.bake_grid(phys_w, phys_h)

	def keyPressEvent(self, arg__1: QKeyEvent) -> None:
		moved = False
		zoom_in_factor = 1.1

		if arg__1.modifiers() & Qt.KeyboardModifier.ControlModifier and arg__1.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
			self.zoom_level *= zoom_in_factor
			moved = True
		if arg__1.modifiers() & Qt.KeyboardModifier.ControlModifier and arg__1.key() == Qt.Key.Key_Minus:
			self.zoom_level *= 1 / zoom_in_factor
			moved = True
		if arg__1.modifiers() & Qt.KeyboardModifier.ControlModifier and arg__1.key() == Qt.Key.Key_0:
			self.zoom_level = self.OG_ZOOM
			moved = True
		if arg__1.key() == Qt.Key.Key_W:
			self.camera_pos = (self.camera_pos[0], self.camera_pos[1] + 1)
			moved = True
		if arg__1.key() == Qt.Key.Key_A:
			self.camera_pos = (self.camera_pos[0] - 1, self.camera_pos[1])
			moved = True
		elif arg__1.key() == Qt.Key.Key_S:
			self.camera_pos = (self.camera_pos[0], self.camera_pos[1] - 1)
			moved = True
		if arg__1.key() == Qt.Key.Key_D:
			self.camera_pos = (self.camera_pos[0] + 1, self.camera_pos[1])
			moved = True

		if moved:
			self.bake_grid()
			self.update()

		super().keyPressEvent(arg__1)

	def bake_grid(self, w: int | None = None, h: int | None = None) -> None:
		if not w or not h:
			ratio = self.devicePixelRatio()
			w, h = int(self.width() * ratio), int(self.height() * ratio)

		# Clean up old texture assets
		if self.grid_texture:
			self.grid_texture.release()
			self.fbo.release()

		# Create blank texture matching current window size
		self.grid_texture = self.ctx.texture((w, h), components=4)
		self.grid_texture.filter = (mgl.NEAREST, mgl.NEAREST)

		# Create framebuffer container and link texture
		self.fbo = self.ctx.framebuffer(color_attachments=[self.grid_texture])

		# Temporarily bind to offscreen framebuffer
		self.fbo.use()
		self.fbo.clear(0.0, 0.0, 0.0, 1.0)

		# Pass uniforms to the texture
		self.u_bg_resolution.value = (w, h)
		self.u_bg_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
		self.u_bg_zoom.value = self.zoom_level

		# Render the grid
		self.bg_vao.render(mgl.TRIANGLES)

	def fixed_update(self) -> None:
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
window.requestActivate()
sys.exit(app.exec())
