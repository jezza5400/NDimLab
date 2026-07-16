from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtCore import Qt, QPointF, QLineF, QTimer, QElapsedTimer, QRectF, QRect
from PySide6.QtGui import QAction, QPen, QPolygonF, QPainter, QColor, QWheelEvent, QSurfaceFormat, QKeyEvent, QMouseEvent, QCloseEvent, QInputDevice
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem, QLabel, QHBoxLayout
from random import randint
from typing import cast
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import sys
import moderngl as mgl
import numpy as np
from numpy.typing import NDArray
from math import cos, sin, pi, log10, floor, isclose, ceil
from pathlib import Path


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

ZOOM_IN_FACTOR_KEY = 1.1
ZOOM_IN_FACTOR_WHEEL = 0.1
ZOOM_IN_FACTOR_TRACKPAD = 0.001


class OpenGLWidget(QOpenGLWidget):
	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent=parent)

		self.zoom_level = 30
		self.OG_ZOOM = self.zoom_level
		self.camera_pos: tuple[int | float, int | float] = (0, 0)
		self.mouse_pos = cast(QPointF, None)
		self.mouse_pressed: bool = False
		self.is_panning = False
		self.is_zooming = False
		self.pressed_keys = set()

		self._paint_clock = QElapsedTimer()
		self._paint_clock.start()
		self.last_frame_ms: float | None = None

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

		# --- Widget setup ---
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.setMinimumSize(200, 200)

	def initializeGL(self) -> None:
		self.makeCurrent()
		self.ctx = mgl.create_context()
		self.ctx.enable(mgl.DEPTH_TEST)

		self.screen_fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())

		self.bg_prog = self.ctx.program(vertex_shader=GRID_VERT_SHADER, fragment_shader=GRID_FRAG_SHADER)
		self.u_bg_resolution = cast(mgl.Uniform, self.bg_prog["u_resolution"])
		self.u_bg_zoom = cast(mgl.Uniform, self.bg_prog["u_zoom"])
		self.u_bg_camera_pos = cast(mgl.Uniform, self.bg_prog["u_camera_pos"])

		self.bg_vbo = self.ctx.buffer(BG_VERTS.tobytes())
		self.bg_vao = self.ctx.vertex_array(self.bg_prog, [(self.bg_vbo, "2f", "inPosition")])

		self.tex_prog = self.ctx.program(vertex_shader=TEXTURE_VERT, fragment_shader=TEXTURE_FRAG)
		self.tex_vao = self.ctx.vertex_array(self.tex_prog, [(self.bg_vbo, "2f", "inPosition")])

		# initial layout-baking
		if self.width() > 0 and self.height() > 0:
			self.bake_grid()

		# --- Timers ---
		# self.timer = QTimer()
		# self.timer.timeout.connect(self.fixed_update)
		# self.timer.start(16)

	def paintGL(self) -> None:
		if self.ctx is None:
			return

		self.last_frame_ms = self._paint_clock.restart()

		self.makeCurrent()

		# Clear color and depth buffers of on-screen framebuffer
		self.screen_fbo.use()
		self.ctx.clear(0.0, 0.0, 0.0, 1.0)

		if self.is_panning or self.is_zooming or self.grid_texture is None:
			ratio = self.devicePixelRatioF()
			w, h = int(self.width() * ratio), int(self.height() * ratio)

			self.u_bg_resolution.value = (w, h)
			self.u_bg_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
			self.u_bg_zoom.value = self.zoom_level

			self.ctx.disable(mgl.DEPTH_TEST)
			self.bg_vao.render(mgl.TRIANGLES)
			self.ctx.enable(mgl.DEPTH_TEST)
		else:
			# Pass 1: Render static baked background image
			self.grid_texture.use(location=0)

			# Disable depth testing while drawing background to avoid masking foreground elements
			self.ctx.disable(mgl.DEPTH_TEST)
			self.tex_vao.render(mgl.TRIANGLES)
			self.ctx.enable(mgl.DEPTH_TEST)

		# Render foreground shapes here

	def resizeGL(self, w: int, h: int) -> None:
		if self.ctx is None:
			return

		ratio = self.devicePixelRatioF()
		phys_w, phys_h = int(w * ratio), int(h * ratio)
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		# re-detect FBO (in case Qt swapped FBOs on resize)
		self.screen_fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		# Rebake background grid texture
		self.bake_grid(phys_w, phys_h)

	def closeEvent(self, event: QCloseEvent) -> None:
		if self.grid_texture:
			self.grid_texture.release()
			self.grid_texture = None
		self.ctx = None

		super().closeEvent(event)

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
		elif event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier) and event.key() == Qt.Key.Key_ParenRight:
			self.camera_pos = (0, 0)
			moved = True
		elif event.key() == Qt.Key.Key_W:
			self.camera_pos = (self.camera_pos[0], self.camera_pos[1] + 1)
			moved = True
		elif event.key() == Qt.Key.Key_A:
			self.camera_pos = (self.camera_pos[0] - 1, self.camera_pos[1])
			moved = True
		elif event.key() == Qt.Key.Key_S:
			self.camera_pos = (self.camera_pos[0], self.camera_pos[1] - 1)
			moved = True
		elif event.key() == Qt.Key.Key_D:
			self.camera_pos = (self.camera_pos[0] + 1, self.camera_pos[1])
			moved = True

		if moved:
			self.pressed_keys.add(event.key())
			if is_zoom_key:
				self.is_zooming = True
				self.update()
			else:
				self.bake_grid()
				self.update()

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

		# Any pixelDelta, x-axis scroll, or Non-120-multiple angleDelta -> trackpad
		is_trackpad = px != 0 or py != 0 or ax != 0 or (ay % 120 != 0)

		if is_trackpad:
			dy = 1 + (ay * ZOOM_IN_FACTOR_TRACKPAD)
		else:
			dy = 1 + (ay / 120.0) * ZOOM_IN_FACTOR_WHEEL

		self.zoom_level *= dy
		self.is_zooming = True
		self.update()

		# print("pixelDelta:", pixel)
		# print("angleDelta:", angle)
		# print("trackpad:", is_trackpad)
		# print("dy:", dy)

		event.accept()

	def mouseMoveEvent(self, event: QMouseEvent) -> None:
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

		if not w or not h:
			ratio = self.devicePixelRatioF()
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

	def time_since_last_paint(self) -> int:
		"""Milliseconds elapsed since paintGL() last ran."""
		return self._paint_clock.elapsed()


class NDimLabWindow(QMainWindow):
	def __init__(self, begin_paused: bool = False) -> None:
		super().__init__()
		self.setWindowTitle("NDimLab")
		self.paused: bool = begin_paused
		self.scene_entities: list[SceneEntity] = []
		self._has_reset = True
		self._debug_mode = False

		# --- MenuBar Actions ---
		pause_action = QAction("&Pause", self)
		pause_action.setCheckable(True)
		pause_action.toggled.connect(self.pause_button_clicked)

		self.physics_step_action = QAction("&Step", self)
		self.physics_step_action.triggered.connect(self.physics_step_clicked)
		self.physics_step_action.setEnabled(False)
		pause_action.setChecked(self.paused)

		debug_action = QAction("&Debug", self)
		debug_action.setCheckable(True)
		debug_action.toggled.connect(self.toggle_debug)
		debug_action.setChecked(self._debug_mode)
		debug_action.setShortcut("F3")

		# --- MenuBar ---
		menuBar = self.menuBar()
		pause_menu = menuBar.addMenu("&Menu")
		pause_menu.addAction(pause_action)
		pause_menu.addAction(self.physics_step_action)
		pause_menu.addAction(debug_action)

		# --- Sidebar ---
		sidebar = QWidget()
		sidebar_layout = QVBoxLayout(sidebar)
		sidebar_layout.addWidget(QLabel("<h2>Matrixes here</h2>"))
		sidebar_layout.addWidget(QLabel("Dummy text"))
		sidebar_layout.addStretch()

		# --- OpenGL ---
		self.opengl_widget = OpenGLWidget()

		# --- Central widget + layout ---
		central_widget = QWidget()
		main_layout = QHBoxLayout(central_widget)
		self.setCentralWidget(central_widget)
		main_layout.addWidget(sidebar, stretch=1)
		main_layout.addWidget(self.opengl_widget, stretch=4)

		# --- Debug Overlay ---
		self.overlay = DebugOverlay(self.opengl_widget)
		self.overlay.hide()

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()
		self._tick_duration_samples: deque[tuple[float, float]] = deque()
		self._tick_duration_window_timer = QElapsedTimer()
		self._tick_duration_window_timer.start()

		self.accumulator = 0.0
		self.dt = 1.0 / 60.0

		# self.frame_timer = QTimer()
		# self.frame_timer.timeout.connect(self.tick)
		# self.frame_timer.start(0)

		self.gl_interval_timer = QTimer()
		self.gl_interval_timer.timeout.connect(self.update_gl_overlay)
		self.gl_interval_timer.start(250)

	def update_gl_overlay(self) -> None:
		widget = self.opengl_widget
		since_last = widget.time_since_last_paint()

		IDLE_THRESHOLD_MS = 250

		if widget.last_frame_ms is None or since_last > IDLE_THRESHOLD_MS:
			self.overlay.update_gl_metrics(f"GL: idle ({since_last} ms)")
		else:
			fps = 1000.0 / widget.last_frame_ms if widget.last_frame_ms > 0 else 0.0
			self.overlay.update_gl_metrics(f"GL Frame: {widget.last_frame_ms:.2f} ms ({fps:.0f} FPS)")

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		margin = 10
		x = self.opengl_widget.size().width() - self.overlay.width()
		self.overlay.move(x, 0)

	def update_scene_entities(self) -> None:
		if self._has_reset:
			for entity in self.scene_entities:
				entity.apply_one_shot()
			self._has_reset = False

		for entity in self.scene_entities:
			entity.apply_continuous()

	def tick(self) -> None:
		elapsed = self.timer.nsecsElapsed() / 1e9
		self.timer.restart()

		now = self._tick_duration_window_timer.nsecsElapsed() / 1e9
		self._tick_duration_samples.append((now, elapsed))
		while self._tick_duration_samples and now - self._tick_duration_samples[0][0] > 1.0:
			self._tick_duration_samples.popleft()

		average_elapsed = sum(sample_elapsed for _, sample_elapsed in self._tick_duration_samples) / len(self._tick_duration_samples)
		self.overlay.update_metrics(f"Tick Duration: {average_elapsed * 1000:.2f} ms")

		if not self.paused:
			self.accumulator += elapsed

			while self.accumulator >= self.dt:
				self.update_scene_entities()
				self.accumulator -= self.dt

		self.opengl_widget.update()

	def pause_button_clicked(self, state: bool) -> None:
		self.paused = state
		self.physics_step_action.setEnabled(state)

	def physics_step_clicked(self) -> None:
		self.update_scene_entities()

	def toggle_debug(self, checked: bool) -> None:
		self._debug_mode = checked
		self.overlay.setVisible(checked)


class DebugOverlay(QWidget):
	def __init__(self, parent=None) -> None:
		super().__init__(parent)

		# Translucent background
		self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
		self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

		# Layout
		layout = QVBoxLayout(self)
		layout.setContentsMargins(6, 4, 6, 4)
		layout.setSpacing(2)

		# Metrics
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


@dataclass
class Transformation:
	matrix: NDArray
	linear: bool
	column_major: bool = False
	enabled: bool = True
	continuous: bool = True
	name: str = ""

	def transposed(self) -> Transformation:
		return Transformation(matrix=self.matrix.T, linear=self.linear, column_major=not self.column_major, enabled=self.enabled, continuous=self.continuous, name=self.name)


class FixedPoint:
	def __init__(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.my_index: int = my_index
		self.other_entity: SceneEntity = other_entity
		self.other_index: int = other_index


class SceneEntity:
	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False, fixed_point: FixedPoint | None = None) -> None:
		self.scene: QGraphicsScene = scene
		points_view: NDArray = points.T if column_major else points
		self.original_points: NDArray = np.empty((points_view.shape[0], points_view.shape[1] + 1), dtype=points_view.dtype)
		self.original_points[:, :-1] = points_view
		self.original_points[:, -1] = 1
		self.points: NDArray = self.original_points.copy()
		self.column_major: bool = column_major
		self.fixed_point: FixedPoint | None = fixed_point
		self.transformations: list[Transformation] = []
		self.combined_one_shot: list[Transformation] = []
		self.combined_continuous: list[Transformation] = []

	def _polygon_from_points(self) -> QPolygonF:
		return QPolygonF([QPointF(x, y) for x, y, _ in self.points])

	def _create_graphics_item(self) -> QGraphicsPolygonItem:
		return self.scene.addPolygon(self._polygon_from_points(), self.pen)

	def add_to_scene(self, render_order: float = 1, color: QColor = QColor("#00AFFF")) -> None:
		self.pen = QPen(color, 2)
		self.pen.setCosmetic(True)
		self.poly_item: QGraphicsPolygonItem = self._create_graphics_item()
		self.poly_item.setZValue(render_order)

	def fix_point_to(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.fixed_point = FixedPoint(my_index, other_entity, other_index)

	def add_transformation(self, t: Transformation | Iterable[Transformation]) -> None:
		if isinstance(t, Iterable) and not isinstance(t, (str, bytes)):
			items = cast(tuple[Transformation], tuple(t))
		else:
			items = cast(tuple[Transformation], (t,))
		self.transformations.extend(items)

	def compute_transformations(self) -> None:
		def append_or_combine(target_list: list[Transformation], tr: Transformation):
			# Normalize matrix layout to row_major form.
			tr = tr.transposed() if tr.column_major else tr

			if not target_list or tr.linear != target_list[-1].linear:
				target_list.append(tr)
				return

			# Combine with previous
			prev = target_list[-1]
			if tr.linear:
				prev.matrix = prev.matrix @ tr.matrix
			else:
				prev.matrix = prev.matrix + tr.matrix

		def to_homogeneous_list(transformations: list[Transformation]) -> list[NDArray]:
			mats = []
			for tr in transformations:
				if tr.linear:
					mats.append(linear_to_homogeneous(tr.matrix))
				else:
					mats.append(vec_to_homogeneous(tr.matrix))
			return mats

		self.combined_one_shot: list[Transformation] = []
		self.combined_continuous: list[Transformation] = []

		for tr in self.transformations:
			if not tr.enabled:
				continue
			if tr.continuous:
				append_or_combine(self.combined_continuous, tr)
			else:
				append_or_combine(self.combined_one_shot, tr)

		# Convert to homogeneous
		one_shot_h: list[NDArray] = to_homogeneous_list(self.combined_one_shot)
		continuous_h: list[NDArray] = to_homogeneous_list(self.combined_continuous)

		# Compute final matrices
		self.combined_one_shot_homogenous: NDArray | None = np.linalg.multi_dot(one_shot_h) if len(one_shot_h) > 1 else one_shot_h[0] if one_shot_h else None
		self.combined_continuous_homogenous: NDArray | None = np.linalg.multi_dot(continuous_h) if len(continuous_h) > 1 else continuous_h[0] if continuous_h else None

	def apply_one_shot(self) -> None:
		if self.combined_one_shot_homogenous is None:
			return
		self.points[:] = self.original_points @ self.combined_one_shot_homogenous
		self.poly_item.setPolygon(self._polygon_from_points())

	def _apply_fixed_point(self) -> None:
		if self.fixed_point is None:
			return

		fp = self.fixed_point
		delta = fp.other_entity.points[fp.other_index] - self.points[fp.my_index]

		self.points[:, :-1] += delta[:-1]

	def apply_continuous(self) -> None:
		if self.combined_continuous_homogenous is None:
			return
		self.points[:] = self.points @ self.combined_continuous_homogenous
		self._apply_fixed_point()
		self.poly_item.setPolygon(self._polygon_from_points())


class Polygon(SceneEntity):
	"""Nice name for default polygon SceneEntity"""


class PointSet(SceneEntity):
	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False) -> None:
		super().__init__(scene, points, column_major)

	def _create_graphics_item(self):
		"""!WIP"""
		...


def vec_to_homogeneous(arr: NDArray) -> NDArray:
	# --- Case 1: single vector (n,) ---
	if arr.ndim == 1:
		n = arr.shape[0]
		h = np.eye(n + 1, dtype=arr.dtype)
		h[-1, :-1] = arr
		return h

	# --- Case 2: batch of vectors (m, n) ---
	elif arr.ndim == 2:
		m, n = arr.shape
		h = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)

		# Fill identity blocks
		idx = np.arange(n + 1)
		h[:, idx, idx] = 1

		# Insert translation vectors
		h[:, -1, :-1] = arr

		return h

	else:
		raise ValueError("Input must be shape (n,) or (m,n)")


def linear_to_homogeneous(arr: NDArray) -> NDArray:
	# --- Case 1: single matrix (n, n) ---
	if arr.ndim == 2:
		n = arr.shape[0]
		hom = np.eye(n + 1, dtype=arr.dtype)
		hom[:n, :n] = arr
		return hom

	# --- Case 2: batch of matrices (m, n, n) ---
	elif arr.ndim == 3:
		m, n, n2 = arr.shape
		if n != n2:
			raise ValueError("Batch matrices must be square")

		hom = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)
		idx = np.arange(n + 1)
		hom[:, idx, idx] = 1
		hom[:, :n, :n] = arr
		return hom

	else:
		raise ValueError("Input must be (n,n) or (m,n,n)")


if __name__ == "__main__":
	# fmt: off
	sq1: NDArray = np.array([
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	], dtype=float)
	# fmt: on

	theta = pi / 180
	# fmt: off
	t0: Transformation = Transformation(np.array([
		[cos(theta), -sin(theta)],
		[sin(theta), cos(theta)],
	], dtype=float), True, column_major=True)
	# fmt: on

	fmt: QSurfaceFormat = QSurfaceFormat()
	fmt.setSamples(4)  # 4x multisampling for anti-aliasing
	fmt.setVersion(4, 6)
	fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
	fmt.setDepthBufferSize(24)  # 24-bit depth buffer for self.ctx.enable(mgl.DEPTH_TEST) sorting
	QSurfaceFormat.setDefaultFormat(fmt)

	app = QApplication(sys.argv)
	window = NDimLabWindow(begin_paused=False)

	# squares: list[SceneEntity] = []
	# squares.append(SceneEntity(window.scene, sq1))
	# squares[0].add_to_scene()
	# squares[0].add_transformation(t0)
	# squares[0].compute_transformations()
	# for i in range(1, 10):
	# 	squares.append(SceneEntity(window.scene, sq1, fixed_point=FixedPoint(0, squares[i - 1], 2)))
	# 	squares[i].add_to_scene(i + 1, QColor(randint(0, 255), randint(0, 255), randint(0, 255)))
	# 	# fmt: off
	# 	squares[i].add_transformation(Transformation(np.array([
	# 		[cos((i + 1) * theta), -sin((i + 1) * theta)],
	# 		[sin((i + 1) * theta), cos((i + 1) * theta)],
	# 	], dtype=float), True, column_major=True))
	# 	# fmt: on
	# 	squares[i].compute_transformations()

	# window.scene_entities.extend(squares)
	window.show()
	# window.requestActivate()
	sys.exit(app.exec())
