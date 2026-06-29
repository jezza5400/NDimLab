import sys
import numpy as np
from numpy.typing import NDArray
from math import cos, sin, pi
from PySide6.QtCore import Qt, QPointF, QTimer, QElapsedTimer
from PySide6.QtGui import QAction, QPen, QPolygonF, QPainter, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem


class NDimLabWindow(QMainWindow):
	def __init__(self, begin_paused: bool = False) -> None:
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
		self.scene = QGraphicsScene(-5, -5, 10, 10)
		self.scene.setBackgroundBrush(Qt.GlobalColor.black)
		self.view = QGraphicsView(self.scene)
		self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		self.view.scale(100, -100)  # Flip Y-axis

		layout.addWidget(self.view)

		# --- Axis ---
		axis_pen = QPen(QColor("#E6E6E6"), 1)
		axis_pen.setCosmetic(True)

		self.scene.addLine(-5, 0, 5, 0, axis_pen)
		self.scene.addLine(0, -5, 0, 5, axis_pen)

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()

		self.accumulator = 0.0
		self.dt = 1.0 / 30.0

		self.frame_timer = QTimer()
		self.frame_timer.timeout.connect(self.tick)
		self.frame_timer.start(0)

	def tick(self) -> None:
		"""Main loop"""
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
		...
		print("Toggled:", state)


class Transformation:
	def __init__(self, matrix: NDArray, linear: bool, column_major: bool = False, enabled: bool = True, continuous: bool = True, name: str = "") -> None:
		self.matrix: NDArray = matrix
		self.linear: bool = linear
		self.column_major: bool = column_major
		self.enabled: bool = enabled
		self.continuous: bool = continuous
		self.name: str = name

class SceneEntity:
	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False) -> None:
		self.scene: QGraphicsScene = scene
		self.original_points: NDArray = points.T.copy() if column_major else points.copy()
		self.points: NDArray = points.T.copy() if column_major else points.copy()
		self.column_major: bool = column_major
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
	window = NDimLabWindow()
	
	sq1_entity = SceneEntity(window.scene, sq1)
	sq1_entity.add_to_scene()
	sq1_entity.add_transformation(t0)
	sq1_entity.compute_transformations()
	window.scene_entities.append(sq1_entity)
	# window.resize(800, 600)
	window.show()
	app.exec()
