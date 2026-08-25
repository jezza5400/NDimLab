from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem, QGraphicsScene

from lib import DEFAULT_QCOLOR
from lib.scene.math_utils import linear_to_homogeneous, vec_to_homogeneous


@dataclass(eq=False)
class Transformation:
	"""
	Represents a linear, homogeneous, or translation transformation applied to a SceneEntity.

	Args:
		matrix (NDArray): The numeric matrix or vector defining the transformation.
		linear (bool): Whether the transformation is linear.
		homogeneous (bool): Whether the transformation is homogeneous.
		column_major (bool): Whether the matrix is stored in column-major order.
		enabled (bool): Whether the transformation is active.
		continuous (bool): Whether the transformation applies continuously each tick.
		name (str): Optional user-visible name for the transformation.
		matrix_text (NDArray | None): Optional text-based matrix representation.

	Returns:
		**instance:** `Transformation`
		A transformation object that can be applied to SceneEntity geometry.
	"""

	matrix: NDArray
	linear: bool
	homogeneous: bool = False
	column_major: bool = False
	enabled: bool = True
	continuous: bool = True
	name: str = ""
	matrix_text: NDArray | None = None

	def transposed(self) -> Transformation:
		"""
		Return a new Transformation with its matrix transposed.

		Returns:
			**transform:** `Transformation`
			A transformation identical to this one except with transposed matrix
			and flipped column-major flag.
		"""
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
		"""
		Represents a constraint linking one entity's point to another entity's point.

		Args:
			my_index (int): Index of the constrained point on this entity.
			other_entity (SceneEntity): The entity whose point acts as the anchor.
			other_index (int): Index of the anchor point on the other entity.
		"""
		self.my_index: int = my_index
		self.other_entity: SceneEntity = other_entity
		self.other_index: int = other_index


class SceneEntity(ABC):
	IS_POLYGON: ClassVar[bool] = True

	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False, fixed_point: FixedPoint | None = None) -> None:
		"""
		Abstract base class representing a geometric entity in the scene.

		Args:
			scene (QGraphicsScene): The Qt scene the entity belongs to.
			points (NDArray): The entity's point coordinates.
			column_major (bool): Whether the input points are column-major.
			fixed_point (FixedPoint | None): Optional fixed-point constraint.
		"""
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
		self.graphics_item: QGraphicsItem | None = None
		self.combined_one_shot_homogenous: NDArray | None = None
		self.combined_continuous_homogenous: NDArray | None = None
		self.dimension: int = self.original_points.shape[1] - 1
		self.projection_matrix: NDArray = self._make_projection_matrix()
		self.color: QColor = QColor(DEFAULT_QCOLOR)
		self.is_polygon: bool = self.IS_POLYGON

	def set_is_polygon(self, value: bool) -> None:
		"""
		Set whether this entity should be treated as a polygon.

		Args:
			value (bool): True if polygon, False if point set.
		"""
		self.is_polygon = value

	def _make_projection_matrix(self) -> NDArray:
		"""
		Create a projection matrix mapping homogeneous coordinates to 4D clip space.

		Returns:
			**matrix:** `NDArray`
			A projection matrix that preserves x, y, z (if present) and sets w = 1.
		"""
		proj = np.zeros((self.dimension + 1, 4), dtype=np.float32)
		for i in range(min(self.dimension, 3)):
			proj[i, i] = 1.0
		proj[self.dimension, 3] = 1.0
		return proj

	def get_render_matrix(self) -> NDArray:
		"""
		Get the projection matrix used for GPU rendering.

		Returns:
			**matrix:** `NDArray`
		"""
		return self.projection_matrix.astype(np.float32)

	def get_color_array(self) -> NDArray:
		"""
		Return the entity color as a float RGBA array.

		Returns:
			**rgba:** `NDArray`
			A 4-element array containing red, green, blue, alpha in float form.
		"""
		c = self.color
		return np.array([c.redF(), c.greenF(), c.blueF(), c.alphaF()], dtype=np.float32)

	@abstractmethod
	def _create_graphics_item(self) -> QGraphicsItem:
		"""
		Create and return the QGraphicsItem representing this entity.

		Returns:
			**item:** `QGraphicsItem`
		"""
		raise NotImplementedError

	def update_graphics_item(self) -> None:
		"""
		Update the graphics item to reflect the current point positions.
		"""
		# Subclasses override if they render something.

	def add_to_scene(self, render_order: float = 1, color: QColor = DEFAULT_QCOLOR) -> None:
		"""
		Create the graphics item and add it to the scene.

		Args:
			render_order (float): Z-value for rendering order.
			color (QColor): Pen color for drawing.
		"""
		self.pen = QPen(color, 2)
		self.pen.setCosmetic(True)
		self.graphics_item = self._create_graphics_item()
		self.graphics_item.setZValue(render_order)

	def add_transformation(self, t: Transformation | Iterable[Transformation]) -> None:
		"""
		Add one or more transformations to the entity.

		Args:
			t (Transformation | Iterable[Transformation]): Transformations to add.
		"""
		if isinstance(t, Iterable) and not isinstance(t, (str, bytes)):
			items = cast(tuple[Transformation], tuple(t))
		else:
			items = cast(tuple[Transformation], (t,))
		self.transformations.extend(items)
		self.compute_transformations()

	def remove_transformation(self, t: Transformation) -> None:
		"""
		Remove a transformation from the entity.

		Args:
			t (Transformation): The transformation to remove.

		Raises:
			ValueError: If the transformation is not present.
		"""
		self.transformations.remove(t)
		self.compute_transformations()

	def clear_transformations(self) -> None:
		"""
		Remove all transformations from the entity.
		"""
		self.transformations.clear()
		self.compute_transformations()

	def set_transformation_enabled(self, t: Transformation, enabled: bool) -> None:
		"""
		Enable or disable a transformation.

		Args:
			t (Transformation): The transformation to modify.
			enabled (bool): Whether the transformation is active.
		"""
		t.enabled = enabled
		self.compute_transformations()

	def set_transformation_continuous(self, t: Transformation, continuous: bool) -> None:
		"""
		Set whether a transformation applies continuously.

		Args:
			t (Transformation): The transformation to modify.
			continuous (bool): Whether it applies continuously.
		"""
		t.continuous = continuous
		self.compute_transformations()

	def update_transformation_matrix(self, t: Transformation, matrix: NDArray, matrix_text: NDArray | None = None) -> None:
		"""
		Update the numeric and optional text matrix of a transformation.

		Args:
			t (Transformation): The transformation to update.
			matrix (NDArray): New numeric matrix.
			matrix_text (NDArray | None): Optional text representation.
		"""
		t.matrix = matrix
		t.matrix_text = matrix_text
		self.compute_transformations()

	def move_transformation(self, t: Transformation, delta: int) -> None:
		"""
		Move a transformation up or down in the transformation list.

		Args:
			t (Transformation): The transformation to reorder.
			delta (int): +1 to move down, -1 to move up.
		"""
		idx = self.transformations.index(t)
		new_idx = idx + delta
		if 0 <= new_idx < len(self.transformations):
			self.transformations[idx], self.transformations[new_idx] = self.transformations[new_idx], self.transformations[idx]
			self.compute_transformations()

	def set_column_major(self, column_major: bool) -> None:
		"""
		Convert all transformations to the specified matrix orientation.

		Args:
			column_major (bool): Whether matrices should be column-major.
		"""
		for t in self.transformations:
			if t.column_major != column_major:
				t.matrix = t.matrix.T
				if t.matrix_text is not None:
					t.matrix_text = t.matrix_text.T
				t.column_major = column_major
		self.compute_transformations()

	def fix_point_to(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		"""
		Constrain one of this entity's points to another entity's point.

		Args:
			my_index (int): Index of this entity's point.
			other_entity (SceneEntity): Anchor entity.
			other_index (int): Anchor point index.
		"""
		self.fixed_point = FixedPoint(my_index, other_entity, other_index)

	def compute_transformations(self) -> None:
		"""
		Combine and categorize transformations into one-shot and continuous groups.
		"""

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

		self.combined_one_shot = []
		self.combined_continuous = []

		for tr in self.transformations:
			if not tr.enabled:
				continue
			if tr.continuous:
				append_or_combine(self.combined_continuous, tr)
			else:
				append_or_combine(self.combined_one_shot, tr)

		one_shot_h = to_homogeneous_list(self.combined_one_shot)
		continuous_h = to_homogeneous_list(self.combined_continuous)

		self.combined_one_shot_homogenous = np.linalg.multi_dot(one_shot_h) if len(one_shot_h) > 1 else one_shot_h[0] if one_shot_h else None
		self.combined_continuous_homogenous = np.linalg.multi_dot(continuous_h) if len(continuous_h) > 1 else continuous_h[0] if continuous_h else None

	def _apply_fixed_point(self) -> None:
		"""
		Apply fixed-point constraints after transformations.
		"""
		if self.fixed_point is None:
			return

		fp = self.fixed_point
		delta = fp.other_entity.points[fp.other_index] - self.points[fp.my_index]
		self.points[:, :-1] += delta[:-1]

	def apply_transformations(self) -> None:
		"""
		Apply one-shot or continuous transformations to the entity's points.
		"""
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
	def _polygon_from_points(self) -> QPolygonF:
		"""
		Convert the entity's points into a QPolygonF.

		Returns:
			**polygon:** `QPolygonF`
		"""
		pts = self.points
		xs = pts[:, 0] if pts.shape[1] > 0 else np.zeros(pts.shape[0])
		ys = pts[:, 1] if pts.shape[1] > 1 else np.zeros(pts.shape[0])
		return QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])

	def _create_graphics_item(self) -> QGraphicsPolygonItem:
		"""
		Create the QGraphicsPolygonItem used to render this polygon.

		Returns:
			**item:** `QGraphicsPolygonItem`
		"""
		return self.scene.addPolygon(self._polygon_from_points(), self.pen)

	def update_graphics_item(self) -> None:
		"""
		Update the polygon graphics item to match current point positions.
		"""
		cast(QGraphicsPolygonItem, self.graphics_item).setPolygon(self._polygon_from_points())


class PointSet(SceneEntity):
	"""
	A set of points rendered individually (WIP).

	Returns:
		**instance:** `PointSet`
	"""

	IS_POLYGON: ClassVar[bool] = False

	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False) -> None:
		"""
		Initialize a PointSet entity.

		Args:
			scene (QGraphicsScene): The Qt scene.
			points (NDArray): Point coordinates.
			column_major (bool): Whether points are column-major.
		"""
		super().__init__(scene, points, column_major)

	def _create_graphics_item(self) -> QGraphicsItem:
		"""
		Create the graphics item for the point set.

		Returns:
			**item:** `QGraphicsItem`

		Raises:
			NotImplementedError: PointSet rendering is not yet implemented.
		"""
		raise NotImplementedError
