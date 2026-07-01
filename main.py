import random
import sys
import numpy as np
from numpy.typing import NDArray
from math import cos, sin, pi
from PySide6.QtCore import Qt, QPointF, QLineF, QTimer, QElapsedTimer, QRectF, QRect
from PySide6.QtGui import QAction, QPen, QPolygonF, QPainter, QColor, QWheelEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem


class InfiniteAxesView(QGraphicsView):
	def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
		super().drawBackground(painter, rect)

		painter_pen = QPen(QColor("#E6E6E6"), 2, Qt.PenStyle.SolidLine)
		painter_pen.setCosmetic(True)
		painter.setPen(painter_pen)

		# Draw axes clipped to visible region
		painter.drawLine(QLineF(rect.left(), 0, rect.right(), 0))
		painter.drawLine(QLineF(0, rect.top(), 0, rect.bottom()))

	def _zoom(self, factor: float, anchor_scene_pos: QPointF | None = None) -> None:
		# If no custom anchor is provided (Keyboard), fall back to screen center
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

		# --- MenuBar ---
		menuBar = self.menuBar()
		pause_menu = menuBar.addMenu("&Menu")
		pause_menu.addAction(pause_action)

		# --- Central widget + layout ---
		central = QWidget()
		layout = QVBoxLayout(central)
		self.setCentralWidget(central)

		# --- Scene + View ---
		self.scene = QGraphicsScene()
		self.scene.setBackgroundBrush(Qt.GlobalColor.black)
		self.scene.setSceneRect(-1e12, -1e12, 2e12, 2e12)
		self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
		self.view = InfiniteAxesView(self.scene)
		self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
		self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
		self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
		self.view.scale(scale, -scale)

		layout.addWidget(self.view)

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()

		self.accumulator = 0.0
		self.dt = 1.0 / 60.0

		self.frame_timer = QTimer()
		self.frame_timer.timeout.connect(self.tick)
		self.frame_timer.start(0)

	def tick(self) -> None:
		elapsed = self.timer.nsecsElapsed() / 1e9
		self.timer.restart()

		if not self.paused:
			self.accumulator += elapsed

			while self.accumulator >= self.dt:
				for entity in self.scene_entities:
					if self.has_reset:
						entity.apply_one_shot()
						self.has_reset = False
					entity.apply_continuous()
				self.accumulator -= self.dt

		self.view.viewport().update()

	def pause_button_clicked(self, state: bool) -> None:
		self.paused = state


class Transformation:
	def __init__(self, matrix: NDArray, linear: bool, column_major: bool = False, enabled: bool = True, continuous: bool = True, name: str = "") -> None:
		self.matrix: NDArray = matrix
		self.linear: bool = linear
		self.column_major: bool = column_major
		self.enabled: bool = enabled
		self.continuous: bool = continuous
		self.name: str = name


class FixedPoint:
	def __init__(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.my_index: int = my_index
		self.other_entity: SceneEntity = other_entity
		self.other_index: int = other_index


class SceneEntity:
	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False, fixed_point: FixedPoint | None = None) -> None:
		self.scene: QGraphicsScene = scene
		self.original_points: NDArray = points.T.copy() if column_major else points.copy()
		self.points: NDArray = points.T.copy() if column_major else points.copy()
		self.column_major: bool = column_major
		self.fixed_point: FixedPoint | None = fixed_point
		self.transformations: list[Transformation] = []
		self.combined_one_shot: list[Transformation] = []
		self.combined_continuous: list[Transformation] = []

	def _create_graphics_item(self) -> QGraphicsPolygonItem:
		return self.scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in self.points]), self.pen)

	def add_to_scene(self, render_order: float = 1, color: QColor = QColor("#00AFFF")) -> None:
		self.pen = QPen(color, 2)
		self.pen.setCosmetic(True)
		self.poly_item: QGraphicsPolygonItem = self._create_graphics_item()
		self.poly_item.setZValue(render_order)

	def fix_point_to(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.fixed_point = FixedPoint(my_index, other_entity, other_index)

	def _apply_fixed_point(self) -> None:
		if self.fixed_point is None:
			return
		
		fp = self.fixed_point
		delta = fp.other_entity.points[fp.other_index] - self.points[fp.my_index]

		self.points[:] += delta

	def add_transformation(self, transformation: Transformation) -> None:
		self.transformations.append(transformation)

	def compute_transformations(self) -> None:
		self.combined_one_shot: list[Transformation] = []
		self.combined_continuous: list[Transformation] = []

		for tr in self.transformations:
			if tr.enabled and not tr.continuous:
				if not self.combined_one_shot or tr.linear != self.combined_one_shot[-1].linear:
					self.combined_one_shot.append(tr)
				elif self.combined_one_shot[-1].linear == tr.linear:
					if tr.linear:
						self.combined_one_shot[-1].matrix = self.combined_one_shot[-1].matrix @ tr.matrix
					else:
						self.combined_one_shot[-1].matrix = self.combined_one_shot[-1].matrix + tr.matrix

			elif tr.enabled and tr.continuous:
				if not self.combined_continuous or tr.linear != self.combined_continuous[-1].linear:
					self.combined_continuous.append(tr)
				elif self.combined_continuous[-1].linear == tr.linear:
					if tr.linear:
						self.combined_continuous[-1].matrix = self.combined_continuous[-1].matrix @ tr.matrix
					else:
						self.combined_continuous[-1].matrix = self.combined_continuous[-1].matrix + tr.matrix

	def apply_one_shot(self) -> None:
		self.points[:] = self.original_points.copy()
		for tr in self.combined_one_shot:
			if tr.linear:
				self.points[:] = self.points @ tr.matrix.T
			else:
				self.points[:] = self.points + tr.matrix
		self.poly_item.setPolygon(QPolygonF([QPointF(x, y) for x, y in self.points]))

	def apply_continuous(self) -> None:
		for tr in self.combined_continuous:
			if tr.linear:
				self.points[:] = self.points @ tr.matrix.T
			else:
				self.points[:] = self.points + tr.matrix
		self._apply_fixed_point()
		self.poly_item.setPolygon(QPolygonF([QPointF(x, y) for x, y in self.points]))


class Polygon(SceneEntity):
	"""Nice name for default polygon SceneEntity"""


class PointSet(SceneEntity):
	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False) -> None:
		super().__init__(scene, points, column_major)

	def _create_graphics_item(self):
		"""!WIP"""
		...


if __name__ == "__main__":
	sq1: NDArray = np.array([
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	], dtype=float)

	theta = pi / 180
	t0: Transformation = Transformation(np.array([
		[cos(theta), -sin(theta)],
		[sin(theta), cos(theta)],
	], dtype=float), True, True)

	app = QApplication(sys.argv)
	window = NDimLabWindow(scale=30)

	squares: list[SceneEntity] = []
	squares.append(SceneEntity(window.scene, sq1))
	squares[0].add_to_scene()
	squares[0].add_transformation(t0)
	squares[0].compute_transformations()
	for i in range(1, 10):
		squares.append(SceneEntity(window.scene, sq1, fixed_point=FixedPoint(0, squares[i-1], 2)))
		squares[i].add_to_scene(i + 1, QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
		squares[i].add_transformation(Transformation(np.array([
			[cos((i + 1) * theta), -sin((i + 1) * theta)],
			[sin((i + 1) * theta), cos((i + 1) * theta)],
		], dtype=float), True, True))
		squares[i].compute_transformations()

	window.scene_entities.extend(squares)
	window.show()
	app.exec()
