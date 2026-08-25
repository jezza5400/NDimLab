from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import (
	QColor,
)
from PySide6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QFrame,
	QGridLayout,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from lib import DEFAULT_QCOLOR
from lib.scene.entity import SceneEntity, Transformation
from lib.widgets.color_swatch import ColorSwatchButton
from lib.widgets.matrix import LiveMatrixWidget

if TYPE_CHECKING:
	from main import NDimLabWindow


class TransformationRow(QFrame):
	"""UI row for a single Transformation: matrix editor + controls."""

	def __init__(
		self,
		entity: SceneEntity,
		transformation: Transformation,
		on_removed: Callable[[TransformationRow], None],
		on_changed: Callable[[], None],
		on_move: Callable[[TransformationRow, int], None],
		parent: QWidget | None = None,
	) -> None:
		"""
		UI row representing a single transformation applied to a SceneEntity.

		Args:
			entity (SceneEntity): The entity this transformation belongs to.
			transformation (Transformation): The transformation being edited.
			on_removed (Callable[[TransformationRow], None]): Callback when row is removed.
			on_changed (Callable[[], None]): Callback when transformation changes.
			on_move (Callable[[TransformationRow, int], None]): Callback to reorder rows.
			parent (QWidget | None): Optional parent widget.
		"""
		super().__init__(parent)
		self.entity = entity
		self.transformation = transformation
		self.on_removed = on_removed
		self.on_changed = on_changed
		self.on_move = on_move

		self.setFrameShape(QFrame.Shape.StyledPanel)
		layout = QVBoxLayout(self)
		layout.setContentsMargins(4, 4, 4, 4)
		layout.setSpacing(4)

		header = QHBoxLayout()
		header.addWidget(QLabel(transformation.name or "Transformation"))
		header.addStretch()

		self.continuous_check = QCheckBox("Continuous")
		self.continuous_check.setChecked(transformation.continuous)
		self.continuous_check.toggled.connect(self._continuous_toggled)
		header.addWidget(self.continuous_check)

		up_btn = QPushButton("▲")
		up_btn.setFixedWidth(20)
		up_btn.clicked.connect(lambda: self.on_move(self, -1))
		header.addWidget(up_btn)

		down_btn = QPushButton("▼")
		down_btn.setFixedWidth(20)
		down_btn.clicked.connect(lambda: self.on_move(self, 1))
		header.addWidget(down_btn)

		remove_btn = QPushButton("✕")
		remove_btn.setFixedWidth(24)
		remove_btn.clicked.connect(self._remove_clicked)
		header.addWidget(remove_btn)

		layout.addLayout(header)

		if transformation.homogeneous:
			rows = cols = transformation.matrix.shape[0]
			initial = transformation.matrix
			initial_text = transformation.matrix_text
		elif transformation.linear:
			rows, cols = transformation.matrix.shape
			initial = transformation.matrix
			initial_text = transformation.matrix_text
		else:
			dim = transformation.matrix.shape[0]
			if transformation.column_major:
				rows, cols = dim, 1
			else:
				rows, cols = 1, dim
			initial = transformation.matrix.reshape(rows, cols)
			initial_text = transformation.matrix_text.reshape(rows, cols) if transformation.matrix_text is not None else None

		self.matrix_widget = LiveMatrixWidget(rows, cols, on_change=self._matrix_changed)
		if initial_text is not None:
			self.matrix_widget.set_text_values(initial_text)
		else:
			self.matrix_widget.set_values(initial)
		layout.addWidget(self.matrix_widget)

	def _matrix_changed(self, data: NDArray) -> None:
		"""
		Handle updates from the matrix editor and apply them to the transformation.

		Args:
			data (NDArray): The updated matrix data from the editor.
		"""
		text_grid = np.array(self.matrix_widget.get_matrix_text(), dtype=object)
		if self.transformation.homogeneous or self.transformation.linear:
			matrix = data
		else:
			matrix = data.reshape(-1)
			text_grid = text_grid.reshape(-1)
		self.entity.update_transformation_matrix(self.transformation, matrix, text_grid)
		self.on_changed()

	def _continuous_toggled(self, checked: bool) -> None:
		"""
		Toggle whether the transformation is applied continuously.

		Args:
			checked (bool): New continuous state.
		"""
		self.entity.set_transformation_continuous(self.transformation, checked)
		self.on_changed()

	def _remove_clicked(self) -> None:
		"""
		Remove this transformation from the entity and notify the parent UI.
		"""
		self.entity.remove_transformation(self.transformation)
		self.on_removed(self)


class EntityRow(QFrame):
	"""UI card for one SceneEntity: points editor + transformations list."""

	def __init__(self, window: NDimLabWindow, entity: SceneEntity, on_removed: Callable[[EntityRow], None], parent: QWidget | None = None) -> None:
		"""
		UI card representing a SceneEntity. Contains:

		Args:
			window (NDimLabWindow): Reference to the main application window.
			entity (SceneEntity): The entity represented by this row.
			on_removed (Callable[[EntityRow], None]): Callback when the entity is removed.
			parent (QWidget | None): Optional parent widget.

		Returns:
			**instance:** `EntityRow`
			A UI card for editing a SceneEntity.
		"""
		super().__init__(parent)
		self.window_ref = window
		self.entity = entity
		self.on_removed = on_removed
		self.transformation_rows: list[TransformationRow] = []

		self.setFrameShape(QFrame.Shape.Box)
		outer = QVBoxLayout(self)
		outer.setContentsMargins(6, 6, 6, 6)
		outer.setSpacing(4)

		header = QHBoxLayout()

		up_btn = QPushButton("▲")
		up_btn.setFixedWidth(20)
		up_btn.clicked.connect(lambda: self.window_ref.move_entity_row(self, -1))
		header.addWidget(up_btn)

		down_btn = QPushButton("▼")
		down_btn.setFixedWidth(20)
		down_btn.clicked.connect(lambda: self.window_ref.move_entity_row(self, 1))
		header.addWidget(down_btn)

		self.title_label = QLabel(self._title_text())
		header.addWidget(self.title_label)
		header.addStretch()

		self.kind_button = QPushButton("Polygon" if entity.is_polygon else "Points")
		self.kind_button.setCheckable(True)
		self.kind_button.setChecked(entity.is_polygon)
		self.kind_button.setFixedWidth(64)
		self.kind_button.toggled.connect(self._kind_toggled)
		header.addWidget(self.kind_button)

		self.color_button = ColorSwatchButton(entity.color)
		self.color_button.color_changed = self._color_changed
		header.addWidget(self.color_button)

		self.expand_button = QPushButton("▼")
		self.expand_button.setFixedWidth(24)
		self.expand_button.clicked.connect(self._toggle_expanded)
		header.addWidget(self.expand_button)

		remove_btn = QPushButton("✕")
		remove_btn.setFixedWidth(24)
		remove_btn.clicked.connect(self._remove_clicked)
		header.addWidget(remove_btn)

		outer.addLayout(header)

		self.body = QWidget()
		body_layout = QVBoxLayout(self.body)
		body_layout.setContentsMargins(0, 0, 0, 0)
		body_layout.setSpacing(6)

		body_layout.addWidget(QLabel("Points:"))
		self.points_row_layout = QHBoxLayout()
		self.points_widget = self._make_points_widget()
		self.points_row_layout.addWidget(self.points_widget)
		body_layout.addLayout(self.points_row_layout)

		trans_header = QHBoxLayout()
		trans_header.addWidget(QLabel("Transformations:"))
		self.trans_kind_combo = QComboBox()
		self.trans_kind_combo.addItems(["Linear", "Translation", "Homogeneous"])
		trans_header.addWidget(self.trans_kind_combo)
		add_trans_btn = QPushButton("+")
		add_trans_btn.setFixedWidth(24)
		add_trans_btn.clicked.connect(self._add_transformation)
		trans_header.addWidget(add_trans_btn)
		trans_header.addStretch()
		body_layout.addLayout(trans_header)

		self.transformations_container = QVBoxLayout()
		self.transformations_container.setSpacing(4)
		body_layout.addLayout(self.transformations_container)

		outer.addWidget(self.body)

	def _make_points_widget(self) -> LiveMatrixWidget:
		"""
		Build the points editor widget, respecting global column-major settings.

		Returns:
			**widget:** `LiveMatrixWidget`
			A matrix editor for the entity's point data.
		"""
		dim = self.entity.dimension
		count = self.entity.points.shape[0]
		values = self.entity.original_points[:, :dim]
		if self.window_ref.column_major_global:
			rows, cols = dim, count
			values = values.T
		else:
			rows, cols = count, dim

		widget = LiveMatrixWidget(rows, cols, on_change=self._points_changed)
		widget.set_values(values)
		return widget

	def rebuild_points_widget(self) -> None:
		"""
		Recreate the points editor to match the current column-major configuration.
		"""
		old_widget = self.points_widget
		self.points_row_layout.removeWidget(old_widget)
		old_widget.deleteLater()

		self.points_widget = self._make_points_widget()
		self.points_row_layout.addWidget(self.points_widget)

	def _title_text(self) -> str:
		"""
		Build the title text describing the entity (type, dimension, point count).

		Returns:
			**title:** `str`
			A formatted title string for the entity row.
		"""
		kind = "Polygon" if self.entity.is_polygon else "Points"
		return f"{kind} ({self.entity.dimension}D, {self.entity.points.shape[0]} pts)"

	def _kind_toggled(self, checked: bool) -> None:
		"""
		Toggle whether the entity is treated as a polygon or a point set.

		Args:
			checked (bool): New polygon state.
		"""
		self.entity.set_is_polygon(checked)
		self.kind_button.setText("Polygon" if checked else "Points")
		self.title_label.setText(self._title_text())
		self._repaint()

	def _toggle_expanded(self) -> None:
		"""
		Expand or collapse the entity row body.
		"""
		visible = not self.body.isVisible()
		self.body.setVisible(visible)
		self.expand_button.setText("▼" if visible else "▶")

	def _color_changed(self, color: QColor) -> None:
		"""
		Update the entity's color and repaint the OpenGL widget.

		Args:
			color (QColor): The newly selected color.
		"""
		self.entity.color = color
		self._repaint()

	def _points_changed(self, data: NDArray) -> None:
		"""
		Update the entity's point data based on the matrix editor contents.

		Args:
			data (NDArray): Updated point matrix.
		"""
		cols = self.entity.dimension
		points_data = data.T if self.window_ref.column_major_global else data
		self.entity.original_points[:, :cols] = points_data
		self.entity.points[:, :cols] = points_data
		self._apply_and_repaint()

	def _add_transformation(self) -> None:
		"""
		Create a new transformation of the selected type and add it to the entity.
		"""
		dim = self.entity.dimension
		kind = self.trans_kind_combo.currentText()
		col_major = self.window_ref.column_major_global

		def kind_label(t: Transformation) -> str:
			if t.homogeneous:
				return "Homogeneous"
			return "Linear" if t.linear else "Translation"

		existing_of_kind = sum(1 for t in self.entity.transformations if kind_label(t) == kind)
		name = f"{kind} {existing_of_kind + 1}"

		if kind == "Homogeneous":
			matrix = np.eye(dim + 1, dtype=np.float32)
			t = Transformation(matrix=matrix, linear=True, homogeneous=True, column_major=col_major, continuous=True, name=name)
		elif kind == "Translation":
			matrix = np.zeros(dim, dtype=np.float32)
			t = Transformation(matrix=matrix, linear=False, homogeneous=False, column_major=col_major, continuous=True, name=name)
		else:
			matrix = np.eye(dim, dtype=np.float32)
			t = Transformation(matrix=matrix, linear=True, homogeneous=False, column_major=col_major, continuous=True, name=name)

		self.entity.add_transformation(t)
		row = TransformationRow(self.entity, t, on_removed=self._remove_transformation_row, on_changed=self._apply_and_repaint, on_move=self._move_transformation_row, parent=self)
		self.transformation_rows.append(row)
		self.transformations_container.addWidget(row)
		self._apply_and_repaint()

	def _remove_transformation_row(self, row: TransformationRow) -> None:
		"""
		Remove a transformation row from the UI and entity.

		Args:
			row (TransformationRow): The row to remove.
		"""
		self.transformation_rows.remove(row)
		self.transformations_container.removeWidget(row)
		row.deleteLater()
		self._apply_and_repaint()

	def _move_transformation_row(self, row: TransformationRow, delta: int) -> None:
		"""
		Move a transformation row up or down in the list.

		Args:
			row (TransformationRow): The row to move.
			delta (int): +1 to move down, -1 to move up.
		"""
		idx = self.transformation_rows.index(row)
		new_idx = idx + delta
		if not (0 <= new_idx < len(self.transformation_rows)):
			return
		self.entity.move_transformation(row.transformation, delta)
		self.transformation_rows[idx], self.transformation_rows[new_idx] = self.transformation_rows[new_idx], self.transformation_rows[idx]
		self.transformations_container.removeWidget(row)
		self.transformations_container.insertWidget(new_idx, row)
		self._apply_and_repaint()

	def rebuild_transformation_rows(self) -> None:
		"""
		Rebuild all transformation rows from the entity's transformation list.
		"""
		for row in self.transformation_rows:
			self.transformations_container.removeWidget(row)
			row.deleteLater()
		self.transformation_rows.clear()

		for t in self.entity.transformations:
			row = TransformationRow(self.entity, t, on_removed=self._remove_transformation_row, on_changed=self._apply_and_repaint, on_move=self._move_transformation_row, parent=self)
			self.transformation_rows.append(row)
			self.transformations_container.addWidget(row)

	def _apply_and_repaint(self) -> None:
		"""
		Apply pending changes and repaint the OpenGL widget.
		"""
		self._repaint()

	def _repaint(self) -> None:
		"""
		Request an OpenGL redraw from the main window.
		"""
		self.window_ref.opengl_widget.update()

	def _remove_clicked(self) -> None:
		"""
		Remove this entity from the scene and notify the parent UI.
		"""
		if self.entity in self.window_ref.scene_entities:
			self.window_ref.scene_entities.remove(self.entity)
		self._repaint()
		self.on_removed(self)


class EntityCreatePanel(QFrame):
	"""Collapsible '+ New Entity' form for creating scene entities."""

	def __init__(self, on_create: Callable[[str, int, int, QColor], None], parent: QWidget | None = None) -> None:
		"""
		Collapsible panel for creating new scene entities. Provides fields for:

		Args:
			on_create (Callable[[str, int, int, QColor], None]): Callback invoked when the user submits the form.
			parent (QWidget | None): Optional parent widget.

		Returns:
			**instance:** `EntityCreatePanel`
			A UI panel for creating new SceneEntity objects.
		"""
		super().__init__(parent)
		self.on_create = on_create
		self.setFrameShape(QFrame.Shape.StyledPanel)

		outer = QVBoxLayout(self)
		outer.setContentsMargins(6, 6, 6, 6)
		outer.setSpacing(4)

		self.toggle_button = QPushButton("+ New Entity")
		self.toggle_button.clicked.connect(self._toggle_form)
		outer.addWidget(self.toggle_button)

		self.form = QWidget()
		form_layout = QGridLayout(self.form)

		form_layout.addWidget(QLabel("Type:"), 0, 0)
		self.type_combo = QComboBox()
		self.type_combo.addItems(["Points", "Polygon"])
		form_layout.addWidget(self.type_combo, 0, 1)

		form_layout.addWidget(QLabel("Dimension:"), 1, 0)
		self.dimension_spin = QSpinBox()
		self.dimension_spin.setRange(1, 16)
		self.dimension_spin.setValue(2)
		form_layout.addWidget(self.dimension_spin, 1, 1)

		form_layout.addWidget(QLabel("Point Count:"), 2, 0)
		self.count_spin = QSpinBox()
		self.count_spin.setRange(1, 64)
		self.count_spin.setValue(4)
		form_layout.addWidget(self.count_spin, 2, 1)

		form_layout.addWidget(QLabel("Color:"), 4, 0)
		self.color_button = ColorSwatchButton(QColor(DEFAULT_QCOLOR))
		form_layout.addWidget(self.color_button, 4, 1)

		create_btn = QPushButton("Create")
		create_btn.clicked.connect(self._create_clicked)
		form_layout.addWidget(create_btn, 5, 0, 1, 2)

		outer.addWidget(self.form)
		self.form.setVisible(False)

	def _toggle_form(self) -> None:
		"""
		Expand or collapse the creation form.
		"""
		self.form.setVisible(not self.form.isVisible())

	def _create_clicked(self) -> None:
		"""
		Collect form values and invoke the on_create callback to create a new entity.
		"""
		kind = self.type_combo.currentText()
		dim = self.dimension_spin.value()
		count = self.count_spin.value()
		color = QColor(self.color_button.color)
		self.on_create(kind, dim, count, color)
