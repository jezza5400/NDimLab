from math import cos, sin, pi
import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
from PySide6.QtGui import QPen, QPolygonF, QPainter, QColor
from PySide6.QtCore import Qt, QPointF, QTimer, QElapsedTimer

app = QApplication(sys.argv)
scene = QGraphicsScene(-5, -5, 10, 10)
scene.setBackgroundBrush(Qt.GlobalColor.black)

# A simple 2‑point line
unit_square = np.array(
	[
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	dtype=float,
)
poly_pen = QPen(QColor("#ffa13c"), 2)
poly_pen.setCosmetic(True)
poly_item = scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square]), poly_pen)
poly_item.setZValue(1)

unit_square_child = np.array(
	[
		[unit_square[2, 0], unit_square[2, 1]],
		[unit_square[2, 0] + 1, unit_square[2, 1]],
		[unit_square[2, 0] + 1, unit_square[2, 1] + 1],
		[unit_square[2, 0], unit_square[2, 1] + 1],
	]
)
poly_pen = QPen(QColor("#00baff"), 2)
poly_pen.setCosmetic(True)
poly_item_2 = scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square_child]), poly_pen)
poly_item_2.setZValue(2)

unit_square_child_child = np.array(
	[
		[unit_square_child[2, 0], unit_square_child[2, 1]],
		[unit_square_child[2, 0] + 1, unit_square_child[2, 1]],
		[unit_square_child[2, 0] + 1, unit_square_child[2, 1] + 1],
		[unit_square_child[2, 0], unit_square_child[2, 1] + 1],
	]
)
poly_pen = QPen(QColor("#00ff82"), 2)
poly_pen.setCosmetic(True)
poly_item_3 = scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square_child_child]), poly_pen)
poly_item_3.setZValue(3)

unit_square_child_child_child = np.array(
	[
		[unit_square_child_child[2, 0], unit_square_child_child[2, 1]],
		[unit_square_child_child[2, 0] + 1, unit_square_child_child[2, 1]],
		[unit_square_child_child[2, 0] + 1, unit_square_child_child[2, 1] + 1],
		[unit_square_child_child[2, 0], unit_square_child_child[2, 1] + 1],
	]
)
poly_pen = QPen(QColor("#ff69b1"), 2)
poly_pen.setCosmetic(True)
poly_item_4 = scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square_child_child_child]), poly_pen)
poly_item_4.setZValue(4)

axis_pen = QPen(QColor("#E6E6E6"), 1)
axis_pen.setCosmetic(True)
x_axis = scene.addPolygon(QPolygonF([QPointF(-5, 0), QPointF(5, 0)]), axis_pen)
x_axis.setZValue(0)
y_axis = scene.addPolygon(QPolygonF([QPointF(0, -5), QPointF(0, 5)]), axis_pen)
y_axis.setZValue(0)

# Rotation matrix for 1 degree per frame
theta = pi / 180
rotation = np.array(
	[
		[cos(theta), -sin(theta)],
		[sin(theta), cos(theta)],
	]
)
rotation_T = rotation.T


def update_polygon() -> None:
	# Rotate parent
	unit_square[:] = unit_square @ rotation_T

	# Rotate child around parent anchor
	unit_square_child[:] = (unit_square_child - unit_square_child[0]) @ np.array([[cos(2 * theta), sin(2 * theta)], [-sin(2 * theta), cos(2 * theta)]]) + unit_square[2]

	# Rotate child_child around child anchor
	unit_square_child_child[:] = (unit_square_child_child - unit_square_child_child[0]) @ np.array([[cos(3 * theta), sin(3 * theta)], [-sin(3 * theta), cos(3 * theta)]]) + unit_square_child[2]

	# Rotate child_child_child around child_child anchor
	unit_square_child_child_child[:] = (unit_square_child_child_child - unit_square_child_child_child[0]) @ np.array(
		[[cos(4 * theta), sin(4 * theta)], [-sin(4 * theta), cos(4 * theta)]]
	) + unit_square_child_child[2]

	# Update the polygon[s]
	poly_item.setPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square]))
	poly_item_2.setPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square_child]))
	poly_item_3.setPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square_child_child]))
	poly_item_4.setPolygon(QPolygonF([QPointF(x, y) for x, y in unit_square_child_child_child]))


timer = QElapsedTimer()
timer.start()
accumulator = 0.0
dt = 1.0 / 30.0  # fixed physics timestep


def tick() -> None:
	global accumulator  # noqa: PLW0603

	elapsed = timer.nsecsElapsed() / 1e9
	timer.restart()

	accumulator += elapsed

	while accumulator >= dt:
		update_polygon()
		accumulator -= dt

	view.viewport().update()


frame_timer = QTimer()
frame_timer.timeout.connect(tick)
frame_timer.start(0)

view = QGraphicsView(scene)
view.setRenderHint(QPainter.RenderHint.Antialiasing, True)

view.scale(100, -100)  # Flip y-axis to use mathematical directions
view.show()

app.exec()
