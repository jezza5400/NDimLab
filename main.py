from random import randint
from typing import cast
from collections.abc import Iterable
from collections import deque
from dataclasses import dataclass
import sys
import numpy as np
from numpy.typing import NDArray
from math import cos, sin, pi, log10, floor, isclose, ceil
from PySide6.QtCore import Qt, QPointF, QLineF, QTimer, QElapsedTimer, QRectF, QRect
from PySide6.QtGui import QAction, QPen, QPolygonF, QPainter, QColor, QWheelEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem, QLabel


class ViewAxisMarkers:
	def __init__(self, visible: bool, major: bool, minor: bool, size_pct: int | float): ...


class InfiniteAxesView(QGraphicsView):
	def __init__(self, axis_markers: bool = False, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.axis_markers: bool = axis_markers

		self.major_pen = QPen(QColor("#666666"), 1, Qt.PenStyle.SolidLine)
		self.major_pen.setCosmetic(True)

		self.minor_pen = QPen(QColor("#1F1F1F"), 1, Qt.PenStyle.SolidLine)
		self.minor_pen.setCosmetic(True)

		self.axis_pen = QPen(QColor("#E6E6E6"), 2, Qt.PenStyle.SolidLine)
		self.axis_pen.setCosmetic(True)

		self.major_step: float = 5.0
		self.minor_step: float = self.major_step / 5.0

	def _zoom(self, factor: float, anchor_scene_pos: QPointF | None = None) -> None:
		if anchor_scene_pos is None:
			anchor_scene_pos = self.mapToScene(self.viewport().rect().center())

		self.translate(anchor_scene_pos.x(), anchor_scene_pos.y())
		self.scale(factor, factor)
		self.translate(-anchor_scene_pos.x(), -anchor_scene_pos.y())

		visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
		approx_major_step = visible_rect.width() / 5.0
		if approx_major_step > 0:
			pow10 = 10 ** floor(log10(approx_major_step))
			ratio = approx_major_step / pow10

			if ratio < 2:
				self.major_step = 1.0 * pow10
			elif ratio < 5:
				self.major_step = 2.0 * pow10
			else:
				self.major_step = 5.0 * pow10

			self.minor_step = self.major_step / 5.0

	def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
		super().drawBackground(painter, rect)
		painter.save()
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

		left_start = floor(rect.left() / self.minor_step) * self.minor_step
		right_end = ceil(rect.right() / self.minor_step) * self.minor_step
		top_start = floor(rect.top() / self.minor_step) * self.minor_step
		bottom_end = ceil(rect.bottom() / self.minor_step) * self.minor_step

		x = left_start
		while x <= right_end:
			if isclose(x, 0.0, abs_tol=self.minor_step * 0.1):
				x += self.minor_step
				continue
			is_major = isclose(x % self.major_step, 0, abs_tol=self.minor_step * 0.1) or isclose(x % self.major_step, self.major_step, abs_tol=self.minor_step * 0.1)
			painter.setPen(self.major_pen if is_major else self.minor_pen)
			painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
			x += self.minor_step

		y = top_start
		while y <= bottom_end:
			if isclose(y, 0.0, abs_tol=self.minor_step * 0.1):
				y += self.minor_step
				continue
			is_major = isclose(y % self.major_step, 0, abs_tol=self.minor_step * 0.1) or isclose(y % self.major_step, self.major_step, abs_tol=self.minor_step * 0.1)
			painter.setPen(self.major_pen if is_major else self.minor_pen)
			painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
			y += self.minor_step

		painter.setPen(self.axis_pen)
		if rect.left() <= 0 <= rect.right():
			painter.drawLine(QLineF(0, rect.top(), 0, rect.bottom()))
		if rect.top() <= 0 <= rect.bottom():
			painter.drawLine(QLineF(rect.left(), 0, rect.right(), 0))

		painter.restore()

	def keyPressEvent(self, event) -> None:
		zoom_in_factor = 1.1
		zoom_out_factor = 1 / zoom_in_factor

		if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
			self._zoom(zoom_in_factor)
			return
		if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Minus:
			self._zoom(zoom_out_factor)
			return

		super().keyPressEvent(event)

	def wheelEvent(self, event: QWheelEvent) -> None:
		zoom_in_factor = 1.02
		zoom_out_factor = 1 / zoom_in_factor

		mouse_scene_pos = self.mapToScene(event.position().toPoint())

		if event.angleDelta().y() > 0:
			self._zoom(zoom_in_factor, mouse_scene_pos)
		else:
			self._zoom(zoom_out_factor, mouse_scene_pos)

		event.accept()


class NDimLabWindow(QMainWindow):
	def __init__(self, begin_paused: bool = False, scale: int = 1) -> None:
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

		# --- Scene + View ---
		self.scene = QGraphicsScene()
		self.scene.setBackgroundBrush(Qt.GlobalColor.black)
		self.scene.setSceneRect(-1e12, -1e12, 2e12, 2e12)
		self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
		self.view = InfiniteAxesView(axis_markers=True, scene=self.scene)
		self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
		self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)  # AnchorViewCenter is default
		self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)  # NoAnchor is default
		# Initial scaling and y-axis flip
		anchor_scene_pos = self.view.mapToScene(self.view.viewport().rect().center())
		self.view.translate(anchor_scene_pos.x(), anchor_scene_pos.y())
		self.view.scale(scale, -scale)
		self.view.translate(-anchor_scene_pos.x(), -anchor_scene_pos.y())

		# --- Central widget + layout ---
		central = QWidget()
		layout = QVBoxLayout(central)
		self.setCentralWidget(central)
		layout.addWidget(self.view)

		# --- Debug Overlay ---
		self.overlay = DebugOverlay(self.view)
		self.overlay.hide()

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()
		self._tick_duration_samples: deque[tuple[float, float]] = deque()
		self._tick_duration_window_timer = QElapsedTimer()
		self._tick_duration_window_timer.start()

		self.accumulator = 0.0
		self.dt = 1.0 / 60.0

		self.frame_timer = QTimer()
		self.frame_timer.timeout.connect(self.tick)
		self.frame_timer.start(0)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		# margin = 10
		x = self.view.viewport().width() - self.overlay.width()
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

		self.view.viewport().update()

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
		layout.setContentsMargins(0, 0, 0, 0)  # No layout padding
		self.metrics_label = QLabel("Tick Duration: _")
		self.metrics_label.setStyleSheet("color: #00FF00; background-color: rgba(20, 20, 20, 150); padding: 5px;")

		layout.addWidget(self.metrics_label)

		self.setFixedSize(170, 25)

	def update_metrics(self, text) -> None:
		self.metrics_label.setText(text)


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

	app = QApplication(sys.argv)
	window = NDimLabWindow(begin_paused=False, scale=30)

	squares: list[SceneEntity] = []
	squares.append(SceneEntity(window.scene, sq1))
	squares[0].add_to_scene()
	squares[0].add_transformation(t0)
	squares[0].compute_transformations()
	for i in range(1, 10):
		squares.append(SceneEntity(window.scene, sq1, fixed_point=FixedPoint(0, squares[i - 1], 2)))
		squares[i].add_to_scene(i + 1, QColor(randint(0, 255), randint(0, 255), randint(0, 255)))
		# fmt: off
		squares[i].add_transformation(Transformation(np.array([
			[cos((i + 1) * theta), -sin((i + 1) * theta)],
			[sin((i + 1) * theta), cos((i + 1) * theta)],
		], dtype=float), True, column_major=True))
		# fmt: on
		squares[i].compute_transformations()

	window.scene_entities.extend(squares)
	window.show()
	sys.exit(app.exec())
