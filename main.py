from random import randint
from typing import cast
from collections.abc import Iterable
from dataclasses import dataclass
import sys
import numpy as np
from numpy.typing import NDArray
from math import cos, sin, pi
from PySide6.QtCore import Qt, QPointF, QLineF, QTimer, QElapsedTimer, QRectF, QRect
from PySide6.QtGui import QAction, QPen, QPolygonF, QPainter, QColor, QWheelEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem


class ViewAxisMarkers:
	def __init__(self, visible: bool, major: bool, minor: bool, size_pct: int | float): ...


class InfiniteAxesView(QGraphicsView):
	def __init__(self, axis_markers: bool = False, *args, **kwargs):
		self.axis_markers: bool = axis_markers
		super().__init__(*args, **kwargs)

	def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
		super().drawBackground(painter, rect)

		painter_pen = QPen(QColor("#E6E6E6"), 2, Qt.PenStyle.SolidLine)
		painter_pen.setCosmetic(True)
		painter.setPen(painter_pen)

		# Draw axes clipped to visible region
		painter.drawLine(QLineF(rect.left(), 0, rect.right(), 0))
		painter.drawLine(QLineF(0, rect.top(), 0, rect.bottom()))

		if self.axis_markers:
			size_pct: float = 0.015
			size_pct_x: float = size_pct * (rect.right() - rect.left())
			size_pct_y: float = size_pct * (rect.bottom() - rect.top())
			for i in range(int(rect.left()), int(rect.right()) + 1):
				painter.drawLine(QLineF(i, -size_pct_x, i, size_pct_x))
			for i in range(int(rect.top()), int(rect.bottom()) + 1):
				painter.drawLine(QLineF(-size_pct_y, i, size_pct_y, i))

	def _zoom(self, factor: float, anchor_scene_pos: QPointF | None = None) -> None:
		if anchor_scene_pos is None:
			anchor_scene_pos = self.mapToScene(self.viewport().rect().center())

		self.translate(anchor_scene_pos.x(), anchor_scene_pos.y())
		self.scale(factor, factor)
		self.translate(-anchor_scene_pos.x(), -anchor_scene_pos.y())

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
		self.has_reset = True

		# --- MenuBar Actions ---
		pause_action = QAction("&Pause", self)
		pause_action.toggled.connect(self.pause_button_clicked)
		pause_action.setCheckable(True)
		pause_action.setChecked(self.paused)

		self.physics_step_action = QAction("&Step", self)
		self.physics_step_action.triggered.connect(self.physics_step_clicked)
		self.physics_step_action.setEnabled(False)

		# --- MenuBar ---
		menuBar = self.menuBar()
		pause_menu = menuBar.addMenu("&Menu")
		pause_menu.addAction(pause_action)
		pause_menu.addAction(self.physics_step_action)

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

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()

		self.accumulator = 0.0
		self.dt = 1.0 / 60.0

		self.frame_timer = QTimer()
		self.frame_timer.timeout.connect(self.tick)
		self.frame_timer.start(0)

	def update_scene_entities(self) -> None:
		if self.has_reset:
			for entity in self.scene_entities:
				entity.apply_one_shot()
			self.has_reset = False

		for entity in self.scene_entities:
			entity.apply_continuous()

	def tick(self) -> None:
		elapsed = self.timer.nsecsElapsed() / 1e9
		self.timer.restart()

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
		H = np.eye(n + 1, dtype=arr.dtype)
		H[-1, :-1] = arr
		return H

	# --- Case 2: batch of vectors (m, n) ---
	elif arr.ndim == 2:
		m, n = arr.shape
		H = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)

		# Fill identity blocks
		idx = np.arange(n + 1)
		H[:, idx, idx] = 1

		# Insert translation vectors
		H[:, -1, :-1] = arr

		return H

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
	sq1: NDArray = np.array(
		[
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		],
		dtype=float,
	)

	theta = pi / 180
	t0: Transformation = Transformation(
		np.array(
			[
				[cos(theta), -sin(theta)],
				[sin(theta), cos(theta)],
			],
			dtype=float,
		),
		True,
		column_major=True,
	)

	app = QApplication(sys.argv)
	window = NDimLabWindow(scale=30)

	squares: list[SceneEntity] = []
	squares.append(SceneEntity(window.scene, sq1))
	squares[0].add_to_scene()
	squares[0].add_transformation(t0)
	squares[0].compute_transformations()
	for i in range(1, 10):
		squares.append(SceneEntity(window.scene, sq1, fixed_point=FixedPoint(0, squares[i - 1], 2)))
		squares[i].add_to_scene(i + 1, QColor(randint(0, 255), randint(0, 255), randint(0, 255)))
		squares[i].add_transformation(
			Transformation(
				np.array(
					[
						[cos((i + 1) * theta), -sin((i + 1) * theta)],
						[sin((i + 1) * theta), cos((i + 1) * theta)],
					],
					dtype=float,
				),
				True,
				column_major=True,
			)
		)
		squares[i].compute_transformations()

	window.scene_entities.extend(squares)
	window.show()
	app.exec()
