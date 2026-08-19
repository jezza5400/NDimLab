import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtGui import QColor

from lib.math_eval import Evaluator
from lib.scene.entity import PointSet, Polygon, SceneEntity, Transformation

if TYPE_CHECKING:
	from main import NDimLabWindow

DEFAULT_SAVE_PATH = Path(__file__).parent.parent.parent / "save_data.json"


def _kind_label(t: Transformation) -> str:
	if t.homogeneous:
		return "homogeneous"
	return "linear" if t.linear else "translation"


def _transformation_to_dict(t: Transformation) -> dict:
	if t.matrix_text is not None:
		values = [str(v) for v in np.asarray(t.matrix_text).flatten().tolist()]
	else:
		values = [f"{float(v):g}" for v in np.asarray(t.matrix).flatten().tolist()]

	return {
		"type": _kind_label(t),
		"values": values,
		"continuous": t.continuous,
		"enabled": t.enabled,
		"name": t.name,
	}


def _entity_to_dict(entity: SceneEntity, entity_id: str) -> dict:
	dim = entity.dimension
	points = entity.original_points[:, :dim]

	return {
		"id": entity_id,
		"type": "polygon" if entity.is_polygon else "points",
		"color": entity.color.name(),
		"dimensions": dim,
		"point_count": int(points.shape[0]),
		"points": [float(v) for v in points.flatten().tolist()],
		"transformations": [_transformation_to_dict(t) for t in entity.transformations],
	}


def scene_to_dict(window: NDimLabWindow) -> dict:
	return {
		"settings": {
			"column_major": window.column_major_global,
			"z_order_draw": window.z_order_enabled,
			"ticks_per_second": window.ticks_per_second,
		},
		"entities": [_entity_to_dict(entity, f"entity-{i}") for i, entity in enumerate(window.scene_entities)],
	}


def save_scene(window: NDimLabWindow, path: Path = DEFAULT_SAVE_PATH, minify: bool = True) -> None:
	data = scene_to_dict(window)
	with open(path, "w", encoding="utf-8") as file:
		if minify:
			json.dump(data, file, separators=(",", ":"))
		else:
			json.dump(data, file, indent="\t")


def _build_transformation(
	kind: str,
	values: list[str],
	dim: int,
	column_major: bool,
	continuous: bool,
	enabled: bool,
	name: str,
) -> Transformation:
	if kind == "homogeneous":
		size = dim + 1
		rows, cols = size, size
	elif kind == "linear":
		rows, cols = dim, dim
	else:  # translation
		rows, cols = (dim, 1) if column_major else (1, dim)

	text_grid = np.array([str(v) for v in values], dtype=object).reshape(rows, cols)
	numeric = np.array(
		[[Evaluator.evaluate_expression(str(cell).strip() or "0") for cell in row] for row in text_grid],
		dtype=np.float32,
	)

	if kind == "homogeneous":
		matrix, matrix_text = numeric, text_grid
		linear, homogeneous = True, True
	elif kind == "linear":
		matrix, matrix_text = numeric, text_grid
		linear, homogeneous = True, False
	else:
		matrix, matrix_text = numeric.reshape(-1), text_grid.reshape(-1)
		linear, homogeneous = False, False

	return Transformation(
		matrix=matrix,
		linear=linear,
		homogeneous=homogeneous,
		column_major=column_major,
		enabled=enabled,
		continuous=continuous,
		name=name,
		matrix_text=matrix_text,
	)


def load_scene(window: NDimLabWindow, path: Path = DEFAULT_SAVE_PATH) -> None:
	with open(path, encoding="utf-8") as file:
		data = json.load(file)

	window.clear_scene()

	settings = data.get("settings", {})
	column_major = bool(settings.get("column_major", False))

	window.z_order_enabled = bool(settings.get("z_order_draw", False))
	window.opengl_widget.z_order_enabled = window.z_order_enabled
	window.z_order_checkbox.blockSignals(True)
	window.z_order_checkbox.setChecked(window.z_order_enabled)
	window.z_order_checkbox.blockSignals(False)

	window.ticks_per_second = int(settings.get("ticks_per_second", 60))
	window.tick_rate_spin.blockSignals(True)
	window.tick_rate_spin.setValue(window.ticks_per_second)
	window.tick_rate_spin.blockSignals(False)
	window.tick_timer.setInterval(window._tick_interval_ms())

	window.column_major_global = column_major
	window.column_major_checkbox.blockSignals(True)
	window.column_major_checkbox.setChecked(column_major)
	window.column_major_checkbox.blockSignals(False)

	for entity_data in data.get("entities", []):
		dim = int(entity_data["dimensions"])
		point_count = int(entity_data["point_count"])
		flat_points = entity_data.get("points", [])
		points = np.array(flat_points, dtype=np.float32).reshape(point_count, dim)

		color = QColor(entity_data.get("color", "#ffffff"))
		is_polygon = entity_data.get("type") == "polygon"

		entity: SceneEntity
		if is_polygon:
			entity = Polygon(window._dummy_scene, points)
			entity.color = color
			entity.add_to_scene(color=color)
		else:
			entity = PointSet(window._dummy_scene, points)
			entity.color = color
		entity.set_is_polygon(is_polygon)

		for t_data in entity_data.get("transformations", []):
			t = _build_transformation(
				kind=t_data["type"],
				values=t_data["values"],
				dim=dim,
				column_major=column_major,
				continuous=bool(t_data.get("continuous", True)),
				enabled=bool(t_data.get("enabled", True)),
				name=t_data.get("name", ""),
			)
			entity.add_transformation(t)

		window.scene_entities.append(entity)
		window.add_entity_row(entity)

	window.opengl_widget.update()
