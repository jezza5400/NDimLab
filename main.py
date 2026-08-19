import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
	QElapsedTimer,
	Qt,
	QTimer,
)
from PySide6.QtGui import (
	QAction,
	QColor,
	QIcon,
	QSurfaceFormat,
)
from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QFileDialog,
	QGraphicsScene,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QMessageBox,
	QScrollArea,
	QSpinBox,
	QSplitter,
	QVBoxLayout,
	QWidget,
)

from lib import ZOOM_IN_FACTOR_KEY
from lib.gl.opengl_widget import OpenGLWidget
from lib.scene.entity import PointSet, Polygon, SceneEntity
from lib.scene.serialization import DEFAULT_SAVE_PATH, load_scene, save_scene
from lib.ui.debug_overlay import DebugOverlay
from lib.ui.rows import EntityCreatePanel, EntityRow

ICON_PATH = Path(__file__).parent / "icons" / "app_icon.svg"


class NDimLabWindow(QMainWindow):
	def __init__(self, begin_paused: bool = False) -> None:
		super().__init__()
		self.setWindowTitle("NDimLab")
		self.setWindowIcon(QIcon(str(ICON_PATH)))

		self.paused: bool = begin_paused
		self.scene_entities: list[SceneEntity] = []
		self._debug_mode = False
		self._dummy_scene = QGraphicsScene()  # required by SceneEntity.__init__; unused for GPU rendering
		self.entity_rows: list[EntityRow] = []
		self.column_major_global: bool = False
		self.z_order_enabled: bool = False
		self.ticks_per_second: int = 60

		# --- Sidebar ---
		sidebar = QWidget()
		sidebar.setMinimumWidth(450)
		sidebar_layout = QVBoxLayout(sidebar)

		title_row = QHBoxLayout()
		title_row.addWidget(QLabel("<h2>Scene Entities</h2>"))
		self.pause_indicator = QLabel()
		self.pause_indicator.setStyleSheet("font-weight: bold; font-size: 13px;")
		title_row.addWidget(self.pause_indicator)
		title_row.addStretch()
		sidebar_layout.addLayout(title_row)
		self._update_pause_indicator()

		col_major_row = QHBoxLayout()
		col_major_row.addWidget(QLabel("Column-major input:"))
		self.column_major_checkbox = QCheckBox()
		self.column_major_checkbox.toggled.connect(self._set_column_major_global)
		col_major_row.addWidget(self.column_major_checkbox)
		col_major_row.addSpacing(12)
		col_major_row.addWidget(QLabel("Z-order draw:"))
		self.z_order_checkbox = QCheckBox()
		self.z_order_checkbox.toggled.connect(self._set_z_order_enabled)
		col_major_row.addWidget(self.z_order_checkbox)
		col_major_row.addSpacing(12)
		col_major_row.addWidget(QLabel("Ticks/sec:"))
		self.tick_rate_spin = QSpinBox()
		self.tick_rate_spin.setRange(1, 999)
		self.tick_rate_spin.setValue(self.ticks_per_second)
		self.tick_rate_spin.valueChanged.connect(self._set_tick_rate)
		col_major_row.addWidget(self.tick_rate_spin)
		col_major_row.addStretch()
		sidebar_layout.addLayout(col_major_row)

		self.create_panel = EntityCreatePanel(on_create=self._create_entity)
		sidebar_layout.addWidget(self.create_panel)

		self.entity_list_container = QVBoxLayout()
		self.entity_list_container.setSpacing(6)
		self.entity_list_container.addStretch()
		entity_list_widget = QWidget()
		entity_list_widget.setLayout(self.entity_list_container)

		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setWidget(entity_list_widget)
		sidebar_layout.addWidget(scroll, stretch=1)

		# --- OpenGL ---
		self.opengl_widget = OpenGLWidget(self)
		self.opengl_widget.scene_entities = self.scene_entities  # shared reference: append/remove stays in sync automatically
		self.opengl_widget.z_order_enabled = self.z_order_enabled

		# --- Central QSplitter widget ---
		splitter = QSplitter(Qt.Orientation.Horizontal)
		splitter.addWidget(sidebar)
		splitter.addWidget(self.opengl_widget)
		splitter.setSizes([360, 640])
		self.setCentralWidget(splitter)

		# --- Debug Overlay ---
		self.overlay = DebugOverlay(self.opengl_widget)
		self.overlay.hide()

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()
		self._tick_duration_samples: deque[tuple[float, float]] = deque()
		self._tick_duration_window_timer = QElapsedTimer()
		self._tick_duration_window_timer.start()

		self.tick_timer = QTimer()
		self.tick_timer.timeout.connect(self.tick)
		self.tick_timer.start(self._tick_interval_ms())

		self.gl_interval_timer = QTimer()
		self.gl_interval_timer.timeout.connect(self.update_gl_overlay)
		self.gl_interval_timer.start(250)

		# --- MenuBar Actions ---
		save_action = QAction("&Save Scene...", self)
		save_action.setShortcut("Ctrl+S")
		save_action.triggered.connect(self.save_scene_clicked)

		load_action = QAction("&Load Scene...", self)
		load_action.setShortcut("Ctrl+O")
		load_action.triggered.connect(self.load_scene_clicked)

		pause_action = QAction("&Pause", self)
		pause_action.setCheckable(True)
		pause_action.setShortcut("P")
		pause_action.toggled.connect(self.pause_button_clicked)

		self.physics_step_action = QAction("&Step", self)
		self.physics_step_action.setShortcut("S")
		self.physics_step_action.triggered.connect(self.physics_step_clicked)
		self.physics_step_action.setEnabled(False)
		pause_action.setChecked(self.paused)

		debug_action = QAction("&Debug", self)
		debug_action.setCheckable(True)
		debug_action.toggled.connect(self.toggle_debug)
		debug_action.setChecked(self._debug_mode)
		debug_action.setShortcut("F3")

		reset_camera_action = QAction("&Reset Camera", self)
		reset_camera_action.setShortcut("Ctrl+Shift+0")
		reset_camera_action.triggered.connect(lambda: self.opengl_widget.zoom(reset_camera_pos=True))

		reset_trans_action = QAction("Reset &Transformations", self)
		reset_trans_action.setShortcut("Ctrl+R")
		reset_trans_action.triggered.connect(self.reset_transformations)

		reset_zoom_action = QAction("Reset Zoom", self)
		reset_zoom_action.setShortcut("Ctrl+0")
		reset_zoom_action.triggered.connect(lambda: self.opengl_widget.zoom())

		zoom_in_action = QAction("Zoom In", self)
		zoom_in_action.setShortcut("Ctrl+=")
		zoom_in_action.triggered.connect(lambda: self.opengl_widget.zoom(ZOOM_IN_FACTOR_KEY, reset_zoom=False))

		zoom_out_action = QAction("Zoom Out", self)
		zoom_out_action.setShortcut("Ctrl+-")
		zoom_out_action.triggered.connect(lambda: self.opengl_widget.zoom(1 / ZOOM_IN_FACTOR_KEY, reset_zoom=False))

		menu_bar = self.menuBar()

		menu_bar_menu = menu_bar.addMenu("&Menu")
		menu_bar_view = menu_bar.addMenu("&View")

		menu_bar_menu.addAction(save_action)
		menu_bar_menu.addAction(load_action)
		menu_bar_menu.addSeparator()
		menu_bar_menu.addAction(pause_action)
		menu_bar_menu.addAction(self.physics_step_action)
		menu_bar_menu.addAction(reset_trans_action)
		menu_bar_menu.addAction(debug_action)

		menu_bar_view.addAction(reset_camera_action)
		menu_bar_view.addAction(reset_zoom_action)
		menu_bar_view.addAction(zoom_in_action)
		menu_bar_view.addAction(zoom_out_action)

		if self.paused:
			self.tick_timer.stop()

	def _tick_interval_ms(self) -> int:
		return max(1, int(1000 / self.ticks_per_second))

	def _set_tick_rate(self, value: int) -> None:
		self.ticks_per_second = value
		self.tick_timer.setInterval(self._tick_interval_ms())

	def _set_z_order_enabled(self, checked: bool) -> None:
		self.z_order_enabled = checked
		self.opengl_widget.z_order_enabled = checked
		self.opengl_widget.update()

	def _set_column_major_global(self, checked: bool) -> None:
		self.column_major_global = checked

		for entity in self.scene_entities:
			entity.set_column_major(checked)

		for row in self.entity_rows:
			row.rebuild_transformation_rows()
			row.rebuild_points_widget()

		self.opengl_widget.update()

	def add_entity_row(self, entity: SceneEntity) -> EntityRow:
		row = EntityRow(self, entity, on_removed=self._remove_entity_row)
		row.rebuild_transformation_rows()
		self.entity_rows.append(row)
		self.entity_list_container.insertWidget(self.entity_list_container.count() - 1, row)
		return row

	def clear_scene(self) -> None:
		for row in list(self.entity_rows):
			self.entity_list_container.removeWidget(row)
			row.deleteLater()
		self.entity_rows.clear()
		self.scene_entities.clear()
		self.opengl_widget.update()

	def _create_entity(self, kind: str, dim: int, count: int, color: QColor) -> None:
		points = np.zeros((count, dim), dtype=np.float32)

		entity: SceneEntity
		if kind == "Polygon":
			entity = Polygon(self._dummy_scene, points)
			entity.color = color
			entity.add_to_scene(color=color)
		else:
			entity = PointSet(self._dummy_scene, points)
			entity.color = color

		self.scene_entities.append(entity)
		self.add_entity_row(entity)
		self.opengl_widget.update()

	def _remove_entity_row(self, row: EntityRow) -> None:
		self.entity_rows.remove(row)
		self.entity_list_container.removeWidget(row)
		row.deleteLater()

	def move_entity_row(self, row: EntityRow, delta: int) -> None:
		idx = self.entity_rows.index(row)
		new_idx = idx + delta
		if not (0 <= new_idx < len(self.entity_rows)):
			return
		self.entity_rows[idx], self.entity_rows[new_idx] = self.entity_rows[new_idx], self.entity_rows[idx]
		self.scene_entities[idx], self.scene_entities[new_idx] = self.scene_entities[new_idx], self.scene_entities[idx]
		self.entity_list_container.removeWidget(row)
		self.entity_list_container.insertWidget(new_idx, row)
		self.opengl_widget.update()

	def update_gl_overlay(self) -> None:
		widget = self.opengl_widget
		since_last = widget.time_since_last_paint()

		IDLE_THRESHOLD_MS = 250

		if widget.last_frame_ms is None or since_last > IDLE_THRESHOLD_MS:
			self.overlay.update_gl_metrics(f"GL: idle ({since_last} ms)")
		else:
			fps = 1000.0 / widget.last_frame_ms if widget.last_frame_ms > 0 else 0.0
			self.overlay.update_gl_metrics(f"GL Frame: {widget.last_frame_ms:.2f} ms ({fps:.0f} FPS)")

		self.overlay.move(widget.width() - self.overlay.width(), 0)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		x = self.opengl_widget.size().width() - self.overlay.width()
		self.overlay.move(x, 0)

	def update_scene_entities(self) -> None:
		for entity in self.scene_entities:
			entity.apply_transformations()

	def reset_transformations(self) -> None:
		for entity in self.scene_entities:
			entity.points[:] = entity.original_points
			entity.oneshot_applied = False
			entity.update_graphics_item()
		self.opengl_widget.update()

	def save_scene_clicked(self) -> None:
		path_str, _ = QFileDialog.getSaveFileName(self, "Save Scene", str(DEFAULT_SAVE_PATH), "JSON Files (*.json)")
		if not path_str:
			return
		try:
			save_scene(self, Path(path_str), minify=False)
		except OSError as exc:
			QMessageBox.critical(self, "Save Failed", str(exc))

	def load_scene_clicked(self) -> None:
		path_str, _ = QFileDialog.getOpenFileName(self, "Load Scene", str(DEFAULT_SAVE_PATH.parent), "JSON Files (*.json)")
		if not path_str:
			return
		try:
			load_scene(self, Path(path_str))
		except (OSError, ValueError, KeyError) as exc:
			QMessageBox.critical(self, "Load Failed", str(exc))

	def tick(self) -> None:
		elapsed = self.timer.nsecsElapsed() / 1e9
		self.timer.restart()

		now = self._tick_duration_window_timer.nsecsElapsed() / 1e9
		self._tick_duration_samples.append((now, elapsed))
		while self._tick_duration_samples and now - self._tick_duration_samples[0][0] > 1.0:
			self._tick_duration_samples.popleft()

		average_elapsed = sum(sample_elapsed for _, sample_elapsed in self._tick_duration_samples) / len(self._tick_duration_samples)
		self.overlay.update_metrics(f"Tick Duration: {average_elapsed * 1000:.2f} ms")

		if self.paused:
			return

		self.update_scene_entities()

		# Only redraw when something could actually have changed.
		if any(entity.combined_continuous_homogenous is not None for entity in self.scene_entities):
			self.opengl_widget.update()

	def _update_pause_indicator(self) -> None:
		if self.paused:
			self.pause_indicator.setText("● Paused")
			self.pause_indicator.setStyleSheet("font-weight: bold; font-size: 13px; color: #ef4444;")
		else:
			self.pause_indicator.setText("● Running")
			self.pause_indicator.setStyleSheet("font-weight: bold; font-size: 13px; color: #22c55e;")

	def pause_button_clicked(self, state: bool) -> None:
		self.paused = state
		self.physics_step_action.setEnabled(state)
		if state:
			self.tick_timer.stop()
		else:
			self.tick_timer.start(self._tick_interval_ms())
		self._update_pause_indicator()

	def physics_step_clicked(self) -> None:
		self.update_scene_entities()
		self.opengl_widget.update()

	def toggle_debug(self, checked: bool) -> None:
		self._debug_mode = checked
		self.overlay.setVisible(checked)


if __name__ == "__main__":
	fmt: QSurfaceFormat = QSurfaceFormat()
	fmt.setSamples(4)
	fmt.setVersion(4, 6)
	fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
	fmt.setDepthBufferSize(24)
	QSurfaceFormat.setDefaultFormat(fmt)

	app = QApplication(sys.argv)

	if os.name == "nt":  # Only runs on Windows
		import ctypes

		myappid = "jeremy.ndimlab.main.1.0"
		ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

	window = NDimLabWindow(begin_paused=True)
	window.show()
	window.resize(500, 500)

	sys.exit(app.exec())
