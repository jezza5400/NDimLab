from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import (
	QPointF,
)
from PySide6.QtGui import (
	QColor,
	QPen,
	QPolygonF,
)
from PySide6.QtWidgets import (
	QGraphicsItem,
	QGraphicsPolygonItem,
	QGraphicsScene,
)

from lib import DEFAULT_QCOLOR
from lib.scene.math_utils import linear_to_homogeneous, vec_to_homogeneous


@dataclass(eq=False)
class Transformation:
	"""eq=False: removes dataclass-generated __eq__"""

	matrix: NDArray
	linear: bool
	homogeneous: bool = False
	column_major: bool = False
	enabled: bool = True
	continuous: bool = True
	name: str = ""
	matrix_text: NDArray | None = None

	def transposed(self) -> Transformation:
		return Transformation(
			matrix=self.matrix.T,
			linear=self.linear,
			homogeneous=self.homogeneous,
			column_major=not self.column_major,
			enabled=self.enabled,
			continuous=self.continuous,
			name=self.name,
			matrix_text=self.matrix_text.T if self.matrix_text is not None else None,
		)


class FixedPoint:
	def __init__(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.my_index: int = my_index
		self.other_entity: SceneEntity = other_entity
		self.other_index: int = other_index


class SceneEntity(ABC):
	IS_POLYGON: ClassVar[bool] = True

	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False, fixed_point: FixedPoint | None = None) -> None:
		self.scene: QGraphicsScene = scene
		points_view: NDArray = points.T if column_major else points
		self.original_points: NDArray = np.empty((points_view.shape[0], points_view.shape[1] + 1), dtype=points_view.dtype)
		self.original_points[:, :-1] = points_view
		self.original_points[:, -1] = 1
		self.points: NDArray = self.original_points.copy()
		self.column_major: bool = column_major
		self.oneshot_applied: bool = False
		self.fixed_point: FixedPoint | None = fixed_point
		self.transformations: list[Transformation] = []
		self.combined_one_shot: list[Transformation] = []
		self.combined_continuous: list[Transformation] = []
		self.graphics_item: QGraphicsItem | None = None  # set by add_to_scene(); stays None for entities that never call it (e.g. PointSet today)
		self.combined_one_shot_homogenous: NDArray | None = None
		self.combined_continuous_homogenous: NDArray | None = None
		self.dimension: int = self.original_points.shape[1] - 1  # Exclude homogeneous column
		self.projection_matrix: NDArray = self._make_projection_matrix()
		self.color: QColor = QColor(DEFAULT_QCOLOR)
		self.is_polygon: bool = self.IS_POLYGON  # drives GPU primitive; toggleable independently of Python type via set_is_polygon()

	def set_is_polygon(self, value: bool) -> None:
		self.is_polygon = value

	def _make_projection_matrix(self) -> NDArray:
		"""Maps this entity's (dim+1)-homogeneous points down to 4-component clip space.
		Default: dims 0,1,2 -> x,y,z linearly; higher dims discarded; w always 1."""
		proj = np.zeros((self.dimension + 1, 4), dtype=np.float32)
		for i in range(min(self.dimension, 3)):
			proj[i, i] = 1.0
		proj[self.dimension, 3] = 1.0
		return proj

	def get_render_matrix(self) -> NDArray:
		return self.projection_matrix.astype(np.float32)

	def get_color_array(self) -> NDArray:
		c = self.color
		return np.array([c.redF(), c.greenF(), c.blueF(), c.alphaF()], dtype=np.float32)

	@abstractmethod
	def _create_graphics_item(self) -> QGraphicsItem:
		"""Create and return the scene graphics item representing this entity."""
		raise NotImplementedError

	def update_graphics_item(self) -> None:
		"""Sync the graphics item with self.points. Override in subclasses that render something."""

	def add_to_scene(self, render_order: float = 1, color: QColor = DEFAULT_QCOLOR) -> None:
		self.pen = QPen(color, 2)
		self.pen.setCosmetic(True)
		self.graphics_item = self._create_graphics_item()
		self.graphics_item.setZValue(render_order)

	def add_transformation(self, t: Transformation | Iterable[Transformation]) -> None:
		if isinstance(t, Iterable) and not isinstance(t, (str, bytes)):
			items = cast(tuple[Transformation], tuple(t))
		else:
			items = cast(tuple[Transformation], (t,))
		self.transformations.extend(items)
		self.compute_transformations()

	def remove_transformation(self, t: Transformation) -> None:
		"""Remove a specific transformation. Raises ValueError if not present."""
		self.transformations.remove(t)
		self.compute_transformations()

	def clear_transformations(self) -> None:
		self.transformations.clear()
		self.compute_transformations()

	def set_transformation_enabled(self, t: Transformation, enabled: bool) -> None:
		t.enabled = enabled
		self.compute_transformations()

	def set_transformation_continuous(self, t: Transformation, continuous: bool) -> None:
		t.continuous = continuous
		self.compute_transformations()

	def update_transformation_matrix(self, t: Transformation, matrix: NDArray, matrix_text: NDArray | None = None) -> None:
		t.matrix = matrix
		t.matrix_text = matrix_text
		self.compute_transformations()

	def move_transformation(self, t: Transformation, delta: int) -> None:
		idx = self.transformations.index(t)
		new_idx = idx + delta
		if 0 <= new_idx < len(self.transformations):
			self.transformations[idx], self.transformations[new_idx] = self.transformations[new_idx], self.transformations[idx]
			self.compute_transformations()

	def set_column_major(self, column_major: bool) -> None:
		for t in self.transformations:
			if t.column_major != column_major:
				t.matrix = t.matrix.T
				if t.matrix_text is not None:
					t.matrix_text = t.matrix_text.T
				t.column_major = column_major
		self.compute_transformations()

	def fix_point_to(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.fixed_point = FixedPoint(my_index, other_entity, other_index)

	def compute_transformations(self) -> None:
		def append_or_combine(target_list: list[Transformation], tr: Transformation):
			if tr.column_major:
				tr = tr.transposed()
			else:
				tr = Transformation(matrix=tr.matrix.copy(), linear=tr.linear, homogeneous=tr.homogeneous, column_major=tr.column_major, enabled=tr.enabled, continuous=tr.continuous, name=tr.name)

			same_kind = bool(target_list) and target_list[-1].homogeneous == tr.homogeneous and (tr.homogeneous or target_list[-1].linear == tr.linear)
			if not same_kind:
				target_list.append(tr)
				return

			prev = target_list[-1]
			if tr.homogeneous or tr.linear:
				prev.matrix = prev.matrix @ tr.matrix
			else:
				prev.matrix = prev.matrix + tr.matrix

		def to_homogeneous_list(transformations: list[Transformation]) -> list[NDArray]:
			mats = []
			for tr in transformations:
				if tr.homogeneous:
					mats.append(tr.matrix)
				elif tr.linear:
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

		one_shot_h: list[NDArray] = to_homogeneous_list(self.combined_one_shot)
		continuous_h: list[NDArray] = to_homogeneous_list(self.combined_continuous)

		self.combined_one_shot_homogenous = np.linalg.multi_dot(one_shot_h) if len(one_shot_h) > 1 else one_shot_h[0] if one_shot_h else None
		self.combined_continuous_homogenous = np.linalg.multi_dot(continuous_h) if len(continuous_h) > 1 else continuous_h[0] if continuous_h else None

	def _apply_fixed_point(self) -> None:
		if self.fixed_point is None:
			return

		fp = self.fixed_point
		delta = fp.other_entity.points[fp.other_index] - self.points[fp.my_index]

		self.points[:, :-1] += delta[:-1]

	def apply_transformations(self) -> None:
		if self.oneshot_applied:
			if self.combined_continuous_homogenous is None:
				return
			self.points[:] = self.points @ self.combined_continuous_homogenous
			self._apply_fixed_point()
			self.update_graphics_item()
		else:
			self.oneshot_applied = True
			if self.combined_one_shot_homogenous is None:
				return
			self.points[:] = self.original_points @ self.combined_one_shot_homogenous
			self.update_graphics_item()


class Polygon(SceneEntity):
	"""Nice name for default polygon SceneEntity"""

	def _polygon_from_points(self) -> QPolygonF:
		pts = self.points
		xs = pts[:, 0] if pts.shape[1] > 0 else np.zeros(pts.shape[0])
		ys = pts[:, 1] if pts.shape[1] > 1 else np.zeros(pts.shape[0])
		return QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])

	def _create_graphics_item(self) -> QGraphicsPolygonItem:
		return self.scene.addPolygon(self._polygon_from_points(), self.pen)

	def update_graphics_item(self) -> None:
		cast(QGraphicsPolygonItem, self.graphics_item).setPolygon(self._polygon_from_points())


class PointSet(SceneEntity):
	IS_POLYGON: ClassVar[bool] = False

	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False) -> None:
		super().__init__(scene, points, column_major)

	def _create_graphics_item(self) -> QGraphicsItem:
		"""!WIP"""
		raise NotImplementedError
