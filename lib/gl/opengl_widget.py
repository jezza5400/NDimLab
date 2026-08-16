import sys
from typing import cast

import moderngl as mgl
import numpy as np
from PySide6.QtCore import (
	QElapsedTimer,
	QPointF,
	Qt,
)
from PySide6.QtGui import (
	QCloseEvent,
	QKeyEvent,
	QMouseEvent,
	QResizeEvent,
	QWheelEvent,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from lib import BG_VERTS, GRID_FRAG_SHADER, GRID_VERT_SHADER, POLY_FRAG, POLY_VERT, TEXTURE_FRAG, TEXTURE_VERT, ZOOM_IN_FACTOR_KEY, ZOOM_IN_FACTOR_TRACKPAD, ZOOM_IN_FACTOR_WHEEL
from lib.scene.entity import SceneEntity


class OpenGLWidget(QOpenGLWidget):
	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent=parent)

		self.zoom_level: float = 30
		self.OG_ZOOM = self.zoom_level
		self.camera_pos: tuple[int | float, int | float] = (0, 0)
		self.mouse_pos: QPointF | None = None
		self.is_panning = False
		self.is_zooming = False
		self.pressed_keys = set()

		self._paint_clock = QElapsedTimer()
		self._paint_clock.start()
		self.last_frame_ms: float | None = None

		self.ctx: mgl.Context | None = None

		self.bg_prog: mgl.Program | None = None
		self.bg_vbo: mgl.Buffer | None = None
		self.bg_vao: mgl.VertexArray | None = None

		self.fbo: mgl.Framebuffer | None = None
		self.grid_texture: mgl.Texture | None = None

		self.tex_prog: mgl.Program | None = None
		self.tex_vao: mgl.VertexArray | None = None

		self.poly_prog: mgl.Program | None = None
		self.poly_ssbo: mgl.Buffer | None = None
		self.poly_trans_ssbo: mgl.Buffer | None = None
		self.poly_vao: mgl.VertexArray | None = None

		# Populated by whatever owns this widget (e.g. NDimLabWindow shares its
		# scene_entities list by reference so appends/removes stay in sync, and
		# keeps z_order_enabled synced via a setter). OpenGLWidget never needs to
		# know about its owner's type - keeps this class import-independent of it.
		self.scene_entities: list[SceneEntity] = []
		self.z_order_enabled: bool = False

		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.setMinimumSize(200, 200)

	def initializeGL(self) -> None:
		self.makeCurrent()
		self.ctx = mgl.create_context()
		self.ctx.enable(mgl.PROGRAM_POINT_SIZE)

		self.screen_fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())

		self.bg_prog = self.ctx.program(vertex_shader=GRID_VERT_SHADER, fragment_shader=GRID_FRAG_SHADER)
		self.u_bg_resolution = cast(mgl.Uniform, self.bg_prog["u_resolution"])
		self.u_bg_zoom = cast(mgl.Uniform, self.bg_prog["u_zoom"])
		self.u_bg_camera_pos = cast(mgl.Uniform, self.bg_prog["u_camera_pos"])

		self.bg_vbo = self.ctx.buffer(BG_VERTS.tobytes())
		self.bg_vao = self.ctx.vertex_array(self.bg_prog, [(self.bg_vbo, "2f", "inPosition")])

		self.tex_prog = self.ctx.program(vertex_shader=TEXTURE_VERT, fragment_shader=TEXTURE_FRAG)
		self.tex_vao = self.ctx.vertex_array(self.tex_prog, [(self.bg_vbo, "2f", "inPosition")])

		self.poly_prog = self.ctx.program(vertex_shader=POLY_VERT, fragment_shader=POLY_FRAG)
		self.poly_vao = self.ctx.vertex_array(self.poly_prog, [])
		self.poly_ssbo = self.ctx.buffer(reserve=16)
		self.poly_ssbo.bind_to_storage_buffer(binding=0)
		self.poly_trans_ssbo = self.ctx.buffer(reserve=16)
		self.poly_trans_ssbo.bind_to_storage_buffer(binding=1)
		self.u_poly_dimension = cast(mgl.Uniform, self.poly_prog["u_dimension"])
		self.u_poly_pointColor = cast(mgl.Uniform, self.poly_prog["u_pointColor"])
		self.u_poly_camera_pos = cast(mgl.Uniform, self.poly_prog["u_camera_pos"])
		self.u_poly_zoom = cast(mgl.Uniform, self.poly_prog["u_zoom"])
		self.u_poly_resolution = cast(mgl.Uniform, self.poly_prog["u_resolution"])

		if self.width() > 0 and self.height() > 0:
			self.bake_grid()

	def paintGL(self) -> None:
		if self.ctx is None:
			return
		assert self.bg_vao is not None
		assert self.tex_vao is not None
		assert self.poly_vao is not None

		self.last_frame_ms = self._paint_clock.restart()

		self.makeCurrent()

		self.screen_fbo.use()
		self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

		ratio = self.devicePixelRatioF()
		w, h = int(self.width() * ratio), int(self.height() * ratio)

		# The grid is a flat full-screen quad behind everything - never depth-test it.
		self.ctx.disable(mgl.DEPTH_TEST)
		if self.is_panning or self.is_zooming or self.grid_texture is None:
			self.u_bg_resolution.value = (w, h)
			self.u_bg_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
			self.u_bg_zoom.value = self.zoom_level
			self.bg_vao.render(mgl.TRIANGLES)
		else:
			self.grid_texture.use(location=0)
			self.tex_vao.render(mgl.TRIANGLES)

		self.u_poly_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
		self.u_poly_zoom.value = self.zoom_level
		self.u_poly_resolution.value = (w, h)

		if self.z_order_enabled:
			self.ctx.depth_func = "<="
			self.ctx.enable(mgl.DEPTH_TEST)

		for entity in self.scene_entities:
			points_data = np.ascontiguousarray(entity.points[:, : entity.dimension], dtype=np.float32).tobytes()

			if self.poly_ssbo is None or self.poly_ssbo.size < len(points_data):
				self.poly_ssbo = self.ctx.buffer(points_data)
				self.poly_ssbo.bind_to_storage_buffer(binding=0)
			else:
				self.poly_ssbo.write(points_data)

			render_matrix = entity.get_render_matrix()
			trans_data = render_matrix.tobytes()

			if self.poly_trans_ssbo is None or self.poly_trans_ssbo.size < len(trans_data):
				self.poly_trans_ssbo = self.ctx.buffer(trans_data)
				self.poly_trans_ssbo.bind_to_storage_buffer(binding=1)
			else:
				self.poly_trans_ssbo.write(trans_data)

			self.u_poly_dimension.write(np.uint32(entity.dimension).tobytes())
			self.u_poly_pointColor.write(entity.get_color_array().tobytes())

			primitive = mgl.LINE_LOOP if entity.is_polygon else mgl.POINTS
			self.poly_vao.render(primitive, vertices=entity.points.shape[0])

		self.ctx.disable(mgl.DEPTH_TEST)

	def resizeGL(self, w: int, h: int) -> None:
		if self.ctx is None:
			return

		ratio = self.devicePixelRatioF()
		phys_w, phys_h = int(w * ratio), int(h * ratio)
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		self.screen_fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		self.bake_grid(phys_w, phys_h)

	def resizeEvent(self, e: QResizeEvent) -> None:
		super().resizeEvent(e)

		main_window = self.window()

		update_func = getattr(main_window, "update_gl_overlay", None)
		overlay = getattr(main_window, "overlay", None)

		if update_func and overlay and overlay.isVisible():
			update_func()

	def closeEvent(self, event: QCloseEvent) -> None:
		if self.grid_texture is not None:
			self.grid_texture.release()
			self.grid_texture = None
		self.ctx = None

		super().closeEvent(event)

	def zoom(self, multiplier: float | None = None, reset_zoom: bool = True, reset_camera_pos: bool = False, bake: bool = True) -> None:
		"""When bake is True grid will be baked and paintGL will be called"""
		if reset_zoom:
			self.zoom_level = self.OG_ZOOM
		if multiplier:
			self.zoom_level *= multiplier
		if reset_camera_pos:
			self.camera_pos = (0, 0)
		if bake:
			self.bake_grid()
			self.update()

	def keyPressEvent(self, event: QKeyEvent) -> None:
		moved = False
		is_zoom_key = False

		if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
			self.zoom_level *= ZOOM_IN_FACTOR_KEY
			moved = True
			is_zoom_key = True
		elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Minus:
			self.zoom_level *= 1 / ZOOM_IN_FACTOR_KEY
			moved = True
			is_zoom_key = True
		elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_0:
			self.zoom_level = self.OG_ZOOM
			moved = True

		if moved:
			self.pressed_keys.add(event.key())
			if is_zoom_key:
				self.is_zooming = True
				self.update()
			else:
				self.bake_grid()
				self.update()
			event.accept()
			return

		super().keyPressEvent(event)

	def keyReleaseEvent(self, event: QKeyEvent) -> None:
		self.pressed_keys.discard(event.key())
		if self.is_zooming and not (Qt.Key.Key_Minus in self.pressed_keys or Qt.Key.Key_Plus in self.pressed_keys or Qt.Key.Key_Equal in self.pressed_keys):
			self.is_zooming = False
			self.bake_grid()
			self.update()

		super().keyReleaseEvent(event)

	def mousePressEvent(self, event: QMouseEvent) -> None:
		self.mouse_pos = event.position()

		super().mousePressEvent(event)

	def mouseReleaseEvent(self, event: QMouseEvent) -> None:
		if event.button() == Qt.MouseButton.LeftButton and self.is_panning:
			self.is_panning = False
			self.bake_grid()
			self.update()

		super().mouseReleaseEvent(event)

	def wheelEvent(self, event: QWheelEvent) -> None:
		pixel = event.pixelDelta()
		angle = event.angleDelta()

		px, py = pixel.x(), pixel.y()
		ax, ay = angle.x(), angle.y()

		is_trackpad = px != 0 or py != 0 or ax != 0 or (ay % 120 != 0)

		if is_trackpad:
			dy = 1 + (ay * ZOOM_IN_FACTOR_TRACKPAD)
		else:
			dy = 1 + (ay / 120.0) * ZOOM_IN_FACTOR_WHEEL

		self.zoom_level *= dy
		self.is_zooming = True
		self.update()

		event.accept()

	def mouseMoveEvent(self, event: QMouseEvent) -> None:
		if self.mouse_pos is None:
			return

		if event.buttons() == Qt.MouseButton.LeftButton:
			self.is_panning = True
			mouse_mov = [event.position().x() - self.mouse_pos.x(), event.position().y() - self.mouse_pos.y()]
			mouse_mov = [x / self.zoom_level / self.devicePixelRatioF() for x in mouse_mov]
			self.mouse_pos = event.position()
			self.camera_pos = (self.camera_pos[0] - mouse_mov[0], self.camera_pos[1] + mouse_mov[1])
			self.update()

		super().mouseMoveEvent(event)

	def bake_grid(self, w: int | None = None, h: int | None = None) -> None:
		if self.ctx is None:
			print("Cannot bake grid when context is None, Returning.", file=sys.stderr)
			return
		assert self.bg_vao is not None

		if w is None or h is None:
			ratio = self.devicePixelRatioF()
			w, h = int(self.width() * ratio), int(self.height() * ratio)

		if self.grid_texture is not None:
			self.grid_texture.release()
		if self.fbo is not None:
			self.fbo.release()

		self.grid_texture = self.ctx.texture((w, h), components=4)
		self.grid_texture.filter = (mgl.NEAREST, mgl.NEAREST)

		self.fbo = self.ctx.framebuffer(color_attachments=[self.grid_texture])

		self.fbo.use()
		self.fbo.clear(0.0, 0.0, 0.0, 1.0)

		self.u_bg_resolution.value = (w, h)
		self.u_bg_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
		self.u_bg_zoom.value = self.zoom_level

		self.bg_vao.render(mgl.TRIANGLES)

	def fixed_update(self) -> None:
		self.update()

	def time_since_last_paint(self) -> int:
		"""Milliseconds elapsed since paintGL() last ran."""
		return self._paint_clock.elapsed()
