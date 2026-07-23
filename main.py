import ast
import math
import operator
import sys
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type, cast
import moderngl as mgl
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import (
	Qt,
	QPointF,
	QTimer,
	QElapsedTimer,
	Signal,
)
from PySide6.QtGui import (
	QAction,
	QColor,
	QKeyEvent,
	QMouseEvent,
	QPainter,
	QPen,
	QPolygonF,
	QSurfaceFormat,
	QWheelEvent,
	QResizeEvent,
	QCloseEvent,
)
from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QComboBox,
	QFrame,
	QGraphicsItem,
	QGraphicsPolygonItem,
	QGraphicsScene,
	QGridLayout,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QPushButton,
	QColorDialog,
	QScrollArea,
	QSpinBox,
	QSplitter,
	QVBoxLayout,
	QWidget,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget


def load_shader(path: Path) -> str:
	return Path(path).read_text(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
GRID_VERT_SHADER = load_shader(SCRIPT_DIR / "shaders" / "grid.vert")
GRID_FRAG_SHADER = load_shader(SCRIPT_DIR / "shaders" / "grid.frag")
TEXTURE_VERT = load_shader(SCRIPT_DIR / "shaders" / "texture.vert")
TEXTURE_FRAG = load_shader(SCRIPT_DIR / "shaders" / "texture.frag")
POLY_VERT = load_shader(SCRIPT_DIR / "shaders" / "poly.vert")
POLY_FRAG = load_shader(SCRIPT_DIR / "shaders" / "poly.frag")

# fmt: off
BG_VERTS = np.array([
	-1.0, -1.0, 1.0, -1.0, 1.0, 1.0,
	-1.0, -1.0, 1.0, 1.0, -1.0, 1.0
], dtype=np.float32)
DEFAULT_COLOR = np.array([
	0.22, 0.73, 0.75, 1.0
], dtype=np.float32)
# fmt: on

DEFAULT_QCOLOR = QColor.fromRgbF(float(DEFAULT_COLOR[0]), float(DEFAULT_COLOR[1]), float(DEFAULT_COLOR[2]), float(DEFAULT_COLOR[3]))

ZOOM_IN_FACTOR_KEY = 1.1
ZOOM_IN_FACTOR_WHEEL = 0.1
ZOOM_IN_FACTOR_TRACKPAD = 0.001


class Evaluator:
	BINARY_OPERATORS: dict[Type[ast.operator], Callable[[float, float], float]] = {
		ast.Add: operator.add,
		ast.Sub: operator.sub,
		ast.Mult: operator.mul,
		ast.Div: operator.truediv,
	}

	UNARY_OPERATORS: dict[Type[ast.unaryop], Callable[[float], float]] = {
		ast.USub: operator.neg,
		ast.UAdd: operator.pos,
	}

	# Trig functions take/return degrees
	FUNCTIONS: dict[str, Callable[[float], float]] = {
		"sin": lambda x: math.sin(math.radians(x)),
		"cos": lambda x: math.cos(math.radians(x)),
		"tan": lambda x: math.tan(math.radians(x)),
		"asin": lambda x: math.degrees(math.asin(x)),
		"acos": lambda x: math.degrees(math.acos(x)),
		"atan": lambda x: math.degrees(math.atan(x)),
		"sqrt": math.sqrt,
		"abs": abs,
		"exp": math.exp,
		"log": math.log,
	}

	@classmethod
	def evaluate_expression(cls, expr: str, caller: "MatrixLineEdit") -> float:
		"""Evaluates a math string safely adhering to BODMAS rules."""
		if not expr:
			return 0.0

		expr = expr.replace("·", "*")

		try:
			node = ast.parse(expr, mode="eval").body
			return cls._eval_node(node)
		except Exception as e:
			raise ValueError(f"Invalid math expression: '{expr}'") from e

	@classmethod
	def _eval_node(cls, node: ast.AST) -> float:
		if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
			return float(node.value)

		elif isinstance(node, ast.BinOp) and type(node.op) in cls.BINARY_OPERATORS:
			left = cls._eval_node(node.left)
			right = cls._eval_node(node.right)
			op_func = cls.BINARY_OPERATORS[type(node.op)]
			return op_func(left, right)

		elif isinstance(node, ast.UnaryOp) and type(node.op) in cls.UNARY_OPERATORS:
			operand = cls._eval_node(node.operand)
			op_func = cls.UNARY_OPERATORS[type(node.op)]
			return op_func(operand)

		elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() in cls.FUNCTIONS and len(node.args) == 1 and not node.keywords:
			arg = cls._eval_node(node.args[0])
			return cls.FUNCTIONS[node.func.id.lower()](arg)

		else:
			raise ValueError("Unsupported operation or expression structure")


class MatrixLineEdit(QLineEdit):
	"""Custom QLineEdit that catches symbols like '*' and converts them to math ones like '·',
	and dynamically outlines red on invalid mathematical input.
	"""

	VALID_CHARACTERS = frozenset({"+", "-", "*", "/", ".", "(", ")"})

	has_error_changed = Signal(bool)

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self._has_error: bool = False

		self.setStyleSheet("""
			MatrixLineEdit {
				border: 2px solid #a0a0a0;
				border-radius: 4px;
				padding: 4px;
				background-color: white;
			}
			MatrixLineEdit[has_error="true"] {
				border: 2px solid #ef4444;
				background-color: #fef2f2;
			}
			MatrixLineEdit:focus {
				border: 2px solid #3b82f6;
			}
			MatrixLineEdit[has_error="true"]:focus {
				border: 2px solid #dc2626;
			}
		""")

		self.textChanged.connect(self.validate_current_text)

	def set_has_error(self, value: bool) -> None:
		"""Updates the error state, forces a layout paint cycle, and emits a signal."""
		if self._has_error != value:
			self._has_error = value

			self.setProperty("has_error", value)

			self.style().unpolish(self)
			self.style().polish(self)

			self.has_error_changed.emit(value)

	def validate_current_text(self, text: str) -> None:
		"""Runs input through the evaluator to update the error state."""
		if not text.strip():
			self.set_has_error(False)
			return

		try:
			Evaluator.evaluate_expression(text, self)
			self.set_has_error(False)
		except ValueError:
			self.set_has_error(True)

	def keyPressEvent(self, arg__1: QKeyEvent) -> None:
		key = arg__1.key()
		text = arg__1.text()

		if key in (
			Qt.Key.Key_Backspace,
			Qt.Key.Key_Delete,
			Qt.Key.Key_Left,
			Qt.Key.Key_Right,
			Qt.Key.Key_Up,
			Qt.Key.Key_Down,
			Qt.Key.Key_Home,
			Qt.Key.Key_End,
			Qt.Key.Key_Return,
			Qt.Key.Key_Enter,
			Qt.Key.Key_Tab,
		):
			super().keyPressEvent(arg__1)

		elif arg__1.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
			super().keyPressEvent(arg__1)

		elif text == "*":
			self.insert("·")

		elif text and (text.isdigit() or text.isalpha() or text in self.VALID_CHARACTERS):
			super().keyPressEvent(arg__1)


class MatrixWidget(QWidget):
	def __init__(self, rows=3, cols=None, parent=None):
		"""If cols=None, cols will be set to rows"""
		super().__init__(parent)
		self.rows = rows
		self.cols = rows if cols is None else cols

		self.grid = QGridLayout(self)
		self.grid.setSpacing(6)
		self.grid.setContentsMargins(14, 8, 14, 8)

		self.cells = []

		cell_style = """
			QLineEdit {
				border: 1px solid palette(mid);
				background-color: palette(base);
				color: palette(text);
				font-family: 'Courier New', monospace;
				font-size: 14px;
				font-weight: bold;
				min-width: 45px;
				max-width: 90px;
				height: 26px;
			}
			QLineEdit:focus {
				border: 1px solid palette(highlight);
				background-color: palette(window);
			}
		"""

		for r in range(self.rows):
			row_cells = []
			for c in range(self.cols):
				cell = MatrixLineEdit()
				cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
				cell.setStyleSheet(cell_style)
				self.grid.addWidget(cell, r, c)
				row_cells.append(cell)
			self.cells.append(row_cells)

	def paintEvent(self, event):
		super().paintEvent(event)
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)

		text_color = self.palette().color(self.foregroundRole())

		pen = QPen(text_color, 2)
		painter.setPen(pen)

		w = self.width()
		h = self.height()
		bracket_depth = 8
		offset = 2

		painter.drawLine(offset + bracket_depth, offset, offset, offset)
		painter.drawLine(offset, offset, offset, h - offset)
		painter.drawLine(offset, h - offset, offset + bracket_depth, h - offset)

		painter.drawLine(w - offset - bracket_depth, offset, w - offset, offset)
		painter.drawLine(w - offset, offset, w - offset, h - offset)
		painter.drawLine(w - offset, h - offset, w - offset - bracket_depth, h - offset)

	def get_matrix_data(self, default_val: int = 0) -> NDArray:
		"""Extracts matrix contents into float32 NumPy array.

		If a cell is empty or contains an invalid string, it falls back to `default_val`.
		"""
		data = []
		for row in self.cells:
			row_data = []
			for cell in row:
				evaluated_val = Evaluator.evaluate_expression(cell.text().strip(), cell)
				try:
					row_data.append(evaluated_val)
				except ValueError:
					row_data.append(default_val)
			data.append(row_data)

		return np.array(data, dtype=np.float32)


class LiveMatrixWidget(MatrixWidget):
	"""MatrixWidget that reports its numeric contents on every edit via on_change."""

	def __init__(self, rows: int, cols: int, on_change: Callable[[NDArray], None] | None = None, parent: QWidget | None = None) -> None:
		super().__init__(rows, cols, parent)
		self.on_change = on_change
		for row in self.cells:
			for cell in row:
				cell.editingFinished.connect(self._emit_change)

	def _emit_change(self) -> None:
		if self.on_change:
			self.on_change(self.get_matrix_data())

	def set_values(self, data: NDArray) -> None:
		"""Populate cells from an (rows, cols) array without triggering on_change."""
		for r, row in enumerate(self.cells):
			for c, cell in enumerate(row):
				cell.blockSignals(True)
				cell.setText(f"{float(data[r, c]):g}")
				cell.blockSignals(False)


class ColorSwatchButton(QPushButton):
	"""Small button showing a color swatch; opens a color picker on click."""

	def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.setFixedSize(28, 22)
		self.color: QColor = color
		self.color_changed: Callable[[QColor], None] | None = None
		self._refresh_style()
		self.clicked.connect(self._pick_color)

	def _refresh_style(self) -> None:
		self.setStyleSheet(f"background-color: {self.color.name()}; border: 1px solid palette(mid);")

	def _pick_color(self) -> None:
		new_color = QColorDialog.getColor(self.color, self, "Pick Color")
		if new_color.isValid():
			self.color = new_color
			self._refresh_style()
			if self.color_changed:
				self.color_changed(self.color)


class OpenGLWidget(QOpenGLWidget):
	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent=parent)

		self.zoom_level = 30
		self.OG_ZOOM = self.zoom_level
		self.camera_pos: tuple[int | float, int | float] = (0, 0)
		self.mouse_pos: QPointF | None = None
		self.is_panning = False
		self.is_zooming = False
		self.pressed_keys = set()

		self._paint_clock = QElapsedTimer()
		self._paint_clock.start()
		self.last_frame_ms: float | None = None

		self.ctx: mgl.Context | None = None

		self.bg_prog: mgl.Program | None = None
		self.bg_vbo: mgl.Buffer | None = None
		self.bg_vao: mgl.VertexArray | None = None

		self.fbo: mgl.Framebuffer | None = None
		self.grid_texture: mgl.Texture | None = None

		self.tex_prog: mgl.Program | None = None
		self.tex_vao: mgl.VertexArray | None = None

		self.poly_prog: mgl.Program | None = None
		self.poly_ssbo: mgl.Buffer | None = None
		self.poly_trans_ssbo: mgl.Buffer | None = None
		self.poly_vao: mgl.VertexArray | None = None

		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.setMinimumSize(200, 200)

	def initializeGL(self) -> None:
		self.makeCurrent()
		self.ctx = mgl.create_context()
		self.ctx.enable(mgl.PROGRAM_POINT_SIZE)

		self.screen_fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())

		self.bg_prog = self.ctx.program(vertex_shader=GRID_VERT_SHADER, fragment_shader=GRID_FRAG_SHADER)
		self.u_bg_resolution = cast(mgl.Uniform, self.bg_prog["u_resolution"])
		self.u_bg_zoom = cast(mgl.Uniform, self.bg_prog["u_zoom"])
		self.u_bg_camera_pos = cast(mgl.Uniform, self.bg_prog["u_camera_pos"])

		self.bg_vbo = self.ctx.buffer(BG_VERTS.tobytes())
		self.bg_vao = self.ctx.vertex_array(self.bg_prog, [(self.bg_vbo, "2f", "inPosition")])

		self.tex_prog = self.ctx.program(vertex_shader=TEXTURE_VERT, fragment_shader=TEXTURE_FRAG)
		self.tex_vao = self.ctx.vertex_array(self.tex_prog, [(self.bg_vbo, "2f", "inPosition")])

		self.poly_prog = self.ctx.program(vertex_shader=POLY_VERT, fragment_shader=POLY_FRAG)
		self.poly_vao = self.ctx.vertex_array(self.poly_prog, [])
		self.poly_ssbo = self.ctx.buffer(reserve=16)
		self.poly_ssbo.bind_to_storage_buffer(binding=0)
		self.poly_trans_ssbo = self.ctx.buffer(reserve=16)
		self.poly_trans_ssbo.bind_to_storage_buffer(binding=1)
		self.u_poly_dimension = cast(mgl.Uniform, self.poly_prog["u_dimension"])
		self.u_poly_pointColor = cast(mgl.Uniform, self.poly_prog["u_pointColor"])

		if self.width() > 0 and self.height() > 0:
			self.bake_grid()

	def paintGL(self) -> None:
		if self.ctx is None:
			return
		assert self.bg_vao is not None
		assert self.tex_vao is not None
		assert self.poly_vao is not None

		self.last_frame_ms = self._paint_clock.restart()

		self.makeCurrent()

		self.screen_fbo.use()
		self.ctx.clear(0.0, 0.0, 0.0, 1.0)

		if self.is_panning or self.is_zooming or self.grid_texture is None:
			ratio = self.devicePixelRatioF()
			w, h = int(self.width() * ratio), int(self.height() * ratio)

			self.u_bg_resolution.value = (w, h)
			self.u_bg_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
			self.u_bg_zoom.value = self.zoom_level

			self.bg_vao.render(mgl.TRIANGLES)
		else:
			self.grid_texture.use(location=0)
			self.tex_vao.render(mgl.TRIANGLES)

		top_level = self.window()
		if isinstance(top_level, NDimLabWindow):
			for entity in top_level.scene_entities:
				points_data = np.ascontiguousarray(entity.points[:, : entity.dimension], dtype=np.float32).tobytes()

				if self.poly_ssbo is None or self.poly_ssbo.size < len(points_data):
					self.poly_ssbo = self.ctx.buffer(points_data)
					self.poly_ssbo.bind_to_storage_buffer(binding=0)
				else:
					self.poly_ssbo.write(points_data)

				render_matrix = entity.get_render_matrix()
				trans_data = render_matrix.tobytes()

				if self.poly_trans_ssbo is None or self.poly_trans_ssbo.size < len(trans_data):
					self.poly_trans_ssbo = self.ctx.buffer(trans_data)
					self.poly_trans_ssbo.bind_to_storage_buffer(binding=1)
				else:
					self.poly_trans_ssbo.write(trans_data)

				self.u_poly_dimension.write(np.uint32(entity.dimension).tobytes())
				self.u_poly_pointColor.write(entity.get_color_array().tobytes())

				primitive = mgl.LINE_LOOP if isinstance(entity, Polygon) else mgl.POINTS
				self.poly_vao.render(primitive, vertices=entity.points.shape[0])

	def resizeGL(self, w: int, h: int) -> None:
		if self.ctx is None:
			return

		ratio = self.devicePixelRatioF()
		phys_w, phys_h = int(w * ratio), int(h * ratio)
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		self.screen_fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
		self.ctx.viewport = (0, 0, phys_w, phys_h)

		self.bake_grid(phys_w, phys_h)

	def resizeEvent(self, e: QResizeEvent) -> None:
		super().resizeEvent(e)

		main_window = self.window()

		update_func = getattr(main_window, "update_gl_overlay", None)
		overlay = getattr(main_window, "overlay", None)

		if update_func and overlay and overlay.isVisible():
			update_func()

	def closeEvent(self, event: QCloseEvent) -> None:
		if self.grid_texture is not None:
			self.grid_texture.release()
			self.grid_texture = None
		self.ctx = None

		super().closeEvent(event)

	def zoom(self, multiplier: float | None = None, reset_zoom: bool = True, reset_camera_pos: bool = False, bake: bool = True) -> None:
		"""When bake is True grid will be baked and paintGL will be called"""
		if reset_zoom:
			self.zoom_level = self.OG_ZOOM
		if multiplier:
			self.zoom_level *= multiplier
		if reset_camera_pos:
			self.camera_pos = (0, 0)
		if bake:
			self.bake_grid()
			self.update()

	def keyPressEvent(self, event: QKeyEvent) -> None:
		moved = False
		is_zoom_key = False

		if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
			self.zoom_level *= ZOOM_IN_FACTOR_KEY
			moved = True
			is_zoom_key = True
		elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Minus:
			self.zoom_level *= 1 / ZOOM_IN_FACTOR_KEY
			moved = True
			is_zoom_key = True
		elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_0:
			self.zoom_level = self.OG_ZOOM
			moved = True

		if moved:
			self.pressed_keys.add(event.key())
			if is_zoom_key:
				self.is_zooming = True
				self.update()
			else:
				self.bake_grid()
				self.update()
			event.accept()
			return

		super().keyPressEvent(event)

	def keyReleaseEvent(self, event: QKeyEvent) -> None:
		self.pressed_keys.discard(event.key())
		if self.is_zooming and not (Qt.Key.Key_Minus in self.pressed_keys or Qt.Key.Key_Plus in self.pressed_keys or Qt.Key.Key_Equal in self.pressed_keys):
			self.is_zooming = False
			self.bake_grid()
			self.update()

		super().keyReleaseEvent(event)

	def mousePressEvent(self, event: QMouseEvent) -> None:
		self.mouse_pos = event.position()

		super().mousePressEvent(event)

	def mouseReleaseEvent(self, event: QMouseEvent) -> None:
		if event.button() == Qt.MouseButton.LeftButton and self.is_panning:
			self.is_panning = False
			self.bake_grid()
			self.update()

		super().mouseReleaseEvent(event)

	def wheelEvent(self, event: QWheelEvent) -> None:
		pixel = event.pixelDelta()
		angle = event.angleDelta()

		px, py = pixel.x(), pixel.y()
		ax, ay = angle.x(), angle.y()

		is_trackpad = px != 0 or py != 0 or ax != 0 or (ay % 120 != 0)

		if is_trackpad:
			dy = 1 + (ay * ZOOM_IN_FACTOR_TRACKPAD)
		else:
			dy = 1 + (ay / 120.0) * ZOOM_IN_FACTOR_WHEEL

		self.zoom_level *= dy
		self.is_zooming = True
		self.update()

		event.accept()

	def mouseMoveEvent(self, event: QMouseEvent) -> None:
		if self.mouse_pos is None:
			return

		if event.buttons() == Qt.MouseButton.LeftButton:
			self.is_panning = True
			mouse_mov = [event.position().x() - self.mouse_pos.x(), event.position().y() - self.mouse_pos.y()]
			mouse_mov = [x / self.zoom_level / self.devicePixelRatioF() for x in mouse_mov]
			self.mouse_pos = event.position()
			self.camera_pos = (self.camera_pos[0] - mouse_mov[0], self.camera_pos[1] + mouse_mov[1])
			self.update()

		super().mouseMoveEvent(event)

	def bake_grid(self, w: int | None = None, h: int | None = None) -> None:
		if self.ctx is None:
			print("Cannot bake grid when context is None, Returning.", file=sys.stderr)
			return
		assert self.bg_vao is not None

		if w is None or h is None:
			ratio = self.devicePixelRatioF()
			w, h = int(self.width() * ratio), int(self.height() * ratio)

		if self.grid_texture is not None:
			self.grid_texture.release()
		if self.fbo is not None:
			self.fbo.release()

		self.grid_texture = self.ctx.texture((w, h), components=4)
		self.grid_texture.filter = (mgl.NEAREST, mgl.NEAREST)

		self.fbo = self.ctx.framebuffer(color_attachments=[self.grid_texture])

		self.fbo.use()
		self.fbo.clear(0.0, 0.0, 0.0, 1.0)

		self.u_bg_resolution.value = (w, h)
		self.u_bg_camera_pos.value = (self.camera_pos[0], self.camera_pos[1])
		self.u_bg_zoom.value = self.zoom_level

		self.bg_vao.render(mgl.TRIANGLES)

	def fixed_update(self) -> None:
		self.update()

	def time_since_last_paint(self) -> int:
		"""Milliseconds elapsed since paintGL() last ran."""
		return self._paint_clock.elapsed()


class TransformationRow(QFrame):
	"""UI row for a single Transformation: matrix editor + controls."""

	def __init__(
		self,
		entity: "SceneEntity",
		transformation: "Transformation",
		on_removed: Callable[["TransformationRow"], None],
		on_changed: Callable[[], None],
		on_move: Callable[["TransformationRow", int], None],
		parent: QWidget | None = None,
	) -> None:
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

		self.column_major_check = QCheckBox("Col-major")
		self.column_major_check.setChecked(transformation.column_major)
		self.column_major_check.toggled.connect(self._column_major_toggled)
		header.addWidget(self.column_major_check)

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
		elif transformation.linear:
			rows, cols = transformation.matrix.shape
			initial = transformation.matrix
		else:
			rows, cols = 1, transformation.matrix.shape[0]
			initial = transformation.matrix.reshape(1, -1)

		self.matrix_widget = LiveMatrixWidget(rows, cols, on_change=self._matrix_changed)
		self.matrix_widget.set_values(initial)
		layout.addWidget(self.matrix_widget)

	def _matrix_changed(self, data: NDArray) -> None:
		matrix = data if (self.transformation.homogeneous or self.transformation.linear) else data.reshape(-1)
		self.entity.update_transformation_matrix(self.transformation, matrix)
		self.on_changed()

	def _continuous_toggled(self, checked: bool) -> None:
		self.entity.set_transformation_continuous(self.transformation, checked)
		self.on_changed()

	def _column_major_toggled(self, checked: bool) -> None:
		self.entity.set_transformation_column_major(self.transformation, checked)
		self.on_changed()

	def _remove_clicked(self) -> None:
		self.entity.remove_transformation(self.transformation)
		self.on_removed(self)


class EntityRow(QFrame):
	"""UI card for one SceneEntity: points editor + transformations list."""

	def __init__(self, window: "NDimLabWindow", entity: "SceneEntity", on_removed: Callable[["EntityRow"], None], parent: QWidget | None = None) -> None:
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

		kind = "Polygon" if isinstance(entity, Polygon) else "Points"
		self.title_label = QLabel(f"{kind} ({entity.dimension}D, {entity.points.shape[0]} pts)")
		header.addWidget(self.title_label)
		header.addStretch()

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
		rows, cols = entity.points.shape[0], entity.dimension
		self.points_widget = LiveMatrixWidget(rows, cols, on_change=self._points_changed)
		self.points_widget.set_values(entity.original_points[:, :cols])

		points_row_layout = QHBoxLayout()
		points_row_layout.addWidget(self.points_widget)
		body_layout.addLayout(points_row_layout)

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

	def _toggle_expanded(self) -> None:
		visible = not self.body.isVisible()
		self.body.setVisible(visible)
		self.expand_button.setText("▼" if visible else "▶")

	def _color_changed(self, color: QColor) -> None:
		self.entity.color = color
		self._repaint()

	def _points_changed(self, data: NDArray) -> None:
		cols = self.entity.dimension
		self.entity.original_points[:, :cols] = data
		self.entity.points[:, :cols] = data
		self._apply_and_repaint()

	def _add_transformation(self) -> None:
		dim = self.entity.dimension
		kind = self.trans_kind_combo.currentText()
		col_major = self.window_ref.column_major_global
		name = f"Transform {len(self.entity.transformations) + 1}"

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
		self.transformation_rows.remove(row)
		self.transformations_container.removeWidget(row)
		row.deleteLater()
		self._apply_and_repaint()

	def _move_transformation_row(self, row: TransformationRow, delta: int) -> None:
		idx = self.transformation_rows.index(row)
		new_idx = idx + delta
		if not (0 <= new_idx < len(self.transformation_rows)):
			return
		self.entity.move_transformation(row.transformation, delta)
		self.transformation_rows[idx], self.transformation_rows[new_idx] = self.transformation_rows[new_idx], self.transformation_rows[idx]
		self.transformations_container.removeWidget(row)
		self.transformations_container.insertWidget(new_idx, row)
		self._apply_and_repaint()

	def _apply_and_repaint(self) -> None:
		self.entity.apply_one_shot()
		self.window_ref.update_scene_entities()
		self._repaint()

	def _repaint(self) -> None:
		self.window_ref.opengl_widget.update()

	def _remove_clicked(self) -> None:
		if self.entity in self.window_ref.scene_entities:
			self.window_ref.scene_entities.remove(self.entity)
		self._repaint()
		self.on_removed(self)


class EntityCreatePanel(QFrame):
	"""Collapsible '+ New Entity' form for creating scene entities."""

	def __init__(self, on_create: Callable[[str, int, int, QColor], None], parent: QWidget | None = None) -> None:
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
		self.form.setVisible(not self.form.isVisible())

	def _create_clicked(self) -> None:
		kind = self.type_combo.currentText()
		dim = self.dimension_spin.value()
		count = self.count_spin.value()
		color = QColor(self.color_button.color)
		self.on_create(kind, dim, count, color)


class NDimLabWindow(QMainWindow):
	def __init__(self, begin_paused: bool = False) -> None:
		super().__init__()
		self.setWindowTitle("NDimLab")
		self.paused: bool = begin_paused
		self.scene_entities: list[SceneEntity] = []
		self._has_reset = True
		self._debug_mode = False
		self._dummy_scene = QGraphicsScene()  # required by SceneEntity.__init__; unused for GPU rendering
		self.entity_rows: list[EntityRow] = []
		self.column_major_global: bool = False

		# --- Sidebar ---
		sidebar = QWidget()
		sidebar_layout = QVBoxLayout(sidebar)
		sidebar_layout.addWidget(QLabel("<h2>Scene Entities</h2>"))

		col_major_row = QHBoxLayout()
		col_major_row.addWidget(QLabel("Column-major input:"))
		self.column_major_checkbox = QCheckBox()
		self.column_major_checkbox.toggled.connect(self._set_column_major_global)
		col_major_row.addWidget(self.column_major_checkbox)
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

		# --- Central QSplitter widget ---
		splitter = QSplitter(Qt.Orientation.Horizontal)
		splitter.addWidget(sidebar)
		splitter.addWidget(self.opengl_widget)
		splitter.setSizes([200, 800])
		self.setCentralWidget(splitter)

		# --- Debug Overlay ---
		self.overlay = DebugOverlay(self.opengl_widget)
		self.overlay.hide()

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

		reset_camera_action = QAction("&Reset Camera", self)
		reset_camera_action.setShortcut("Ctrl+Shift+0")
		reset_camera_action.triggered.connect(lambda: self.opengl_widget.zoom(reset_camera_pos=True))

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

		menu_bar_menu.addAction(pause_action)
		menu_bar_menu.addAction(self.physics_step_action)
		menu_bar_menu.addAction(debug_action)

		menu_bar_view.addAction(reset_camera_action)
		menu_bar_view.addAction(reset_zoom_action)
		menu_bar_view.addAction(zoom_in_action)
		menu_bar_view.addAction(zoom_out_action)

		# --- Timers ---
		self.timer = QElapsedTimer()
		self.timer.start()
		self._tick_duration_samples: deque[tuple[float, float]] = deque()
		self._tick_duration_window_timer = QElapsedTimer()
		self._tick_duration_window_timer.start()

		self.accumulator = 0.0
		self.dt = 1.0 / 60.0

		# self.frame_timer = QTimer()
		# self.frame_timer.timeout.connect(self.tick)
		# self.frame_timer.start(0)

		self.gl_interval_timer = QTimer()
		self.gl_interval_timer.timeout.connect(self.update_gl_overlay)
		self.gl_interval_timer.start(250)

	def _set_column_major_global(self, checked: bool) -> None:
		self.column_major_global = checked

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

		row = EntityRow(self, entity, on_removed=self._remove_entity_row)
		self.entity_rows.append(row)
		self.entity_list_container.insertWidget(self.entity_list_container.count() - 1, row)
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

		self.opengl_widget.update()

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

		self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
		self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(6, 4, 6, 4)
		layout.setSpacing(2)

		label_style = "color: #00FF00; background: transparent;"
		self.tick_duration = QLabel("Tick Duration: _")
		self.tick_duration.setStyleSheet(label_style)
		self.gl_paint_interval = QLabel("GL Frame Interval: _")
		self.gl_paint_interval.setStyleSheet(label_style)

		layout.addWidget(self.tick_duration)
		layout.addWidget(self.gl_paint_interval)

		self.setFixedSize(170, 50)

	def paintEvent(self, event) -> None:
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)
		painter.setPen(Qt.PenStyle.NoPen)
		painter.setBrush(QColor(20, 20, 20, 150))
		painter.drawRoundedRect(self.rect(), 4, 4)
		super().paintEvent(event)

	def update_metrics(self, text) -> None:
		self.tick_duration.setText(text)

	def update_gl_metrics(self, text) -> None:
		self.gl_paint_interval.setText(text)


@dataclass
class Transformation:
	matrix: NDArray
	linear: bool
	homogeneous: bool = False
	column_major: bool = False
	enabled: bool = True
	continuous: bool = True
	name: str = ""

	def transposed(self) -> Transformation:
		return Transformation(
			matrix=self.matrix.T,
			linear=self.linear,
			homogeneous=self.homogeneous,
			column_major=not self.column_major,
			enabled=self.enabled,
			continuous=self.continuous,
			name=self.name,
		)


class FixedPoint:
	def __init__(self, my_index: int, other_entity: SceneEntity, other_index: int) -> None:
		self.my_index: int = my_index
		self.other_entity: SceneEntity = other_entity
		self.other_index: int = other_index


class SceneEntity(ABC):
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
		self.graphics_item: QGraphicsItem  # set in add_to_scene()
		self.combined_one_shot_homogenous: NDArray | None = None
		self.combined_continuous_homogenous: NDArray | None = None
		self.dimension: int = self.original_points.shape[1] - 1  # Exclude homogeneous column
		self.projection_matrix: NDArray = self._make_projection_matrix()
		self.color: QColor = QColor(DEFAULT_QCOLOR)

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

	def _update_graphics_item(self) -> None:
		"""Sync the graphics item with self.points. Override in subclasses that render something."""

	def add_to_scene(self, render_order: float = 1, color: QColor = QColor("#00AFFF")) -> None:
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

	def set_transformation_column_major(self, t: Transformation, column_major: bool) -> None:
		t.column_major = column_major
		self.compute_transformations()

	def update_transformation_matrix(self, t: Transformation, matrix: NDArray) -> None:
		t.matrix = matrix
		self.compute_transformations()

	def move_transformation(self, t: Transformation, delta: int) -> None:
		idx = self.transformations.index(t)
		new_idx = idx + delta
		if 0 <= new_idx < len(self.transformations):
			self.transformations[idx], self.transformations[new_idx] = self.transformations[new_idx], self.transformations[idx]
			self.compute_transformations()

	def fix_point_to(self, my_index: int, other_entity: "SceneEntity", other_index: int) -> None:
		self.fixed_point = FixedPoint(my_index, other_entity, other_index)

	def compute_transformations(self) -> None:
		def append_or_combine(target_list: list[Transformation], tr: Transformation):
			tr = tr.transposed() if tr.column_major else tr

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

	def apply_one_shot(self) -> None:
		if self.combined_one_shot_homogenous is None:
			return
		self.points[:] = self.original_points @ self.combined_one_shot_homogenous
		self._update_graphics_item()

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
		self._update_graphics_item()


class Polygon(SceneEntity):
	"""Nice name for default polygon SceneEntity"""

	def _polygon_from_points(self) -> QPolygonF:
		return QPolygonF([QPointF(x, y) for x, y, _ in self.points])

	def _create_graphics_item(self) -> QGraphicsPolygonItem:
		return self.scene.addPolygon(self._polygon_from_points(), self.pen)

	def _update_graphics_item(self) -> None:
		cast(QGraphicsPolygonItem, self.graphics_item).setPolygon(self._polygon_from_points())


class PointSet(SceneEntity):
	def __init__(self, scene: QGraphicsScene, points: NDArray, column_major: bool = False) -> None:
		super().__init__(scene, points, column_major)

	def _create_graphics_item(self) -> QGraphicsItem:
		"""!WIP"""
		raise NotImplementedError


def vec_to_homogeneous(arr: NDArray) -> NDArray:
	if arr.ndim == 1:
		n = arr.shape[0]
		h = np.eye(n + 1, dtype=arr.dtype)
		h[-1, :-1] = arr
		return h

	elif arr.ndim == 2:
		m, n = arr.shape
		h = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)

		idx = np.arange(n + 1)
		h[:, idx, idx] = 1

		h[:, -1, :-1] = arr

		return h

	else:
		raise ValueError("Input must be shape (n,) or (m,n)")


def linear_to_homogeneous(arr: NDArray) -> NDArray:
	if arr.ndim == 2:
		n = arr.shape[0]
		hom = np.eye(n + 1, dtype=arr.dtype)
		hom[:n, :n] = arr
		return hom

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
	fmt: QSurfaceFormat = QSurfaceFormat()
	fmt.setSamples(4)
	fmt.setVersion(4, 6)
	fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
	fmt.setDepthBufferSize(24)
	QSurfaceFormat.setDefaultFormat(fmt)

	app = QApplication(sys.argv)

	window = NDimLabWindow(begin_paused=False)
	window.show()
	window.resize(500, 500)

	sys.exit(app.exec())
