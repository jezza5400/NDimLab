from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor


def load_shader(path: Path) -> str:
	"""
	Load a shader file from disk and return its text contents.

	Args:
		path (Path): Filesystem path to the shader file.

	Returns:
		**shader_source:** `str`
		The UTF-8 decoded shader source code.
	"""
	return Path(path).read_text(encoding="utf-8")


DEFAULT_COLOR = np.array([0.22, 0.73, 0.75, 1.0], dtype=np.float32)
DEFAULT_QCOLOR = QColor.fromRgbF(float(DEFAULT_COLOR[0]), float(DEFAULT_COLOR[1]), float(DEFAULT_COLOR[2]), float(DEFAULT_COLOR[3]))

SCRIPT_DIR = Path(__file__).resolve().parent.parent

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
# fmt: on

ZOOM_IN_FACTOR_KEY = 1.1
ZOOM_IN_FACTOR_WHEEL = 0.1
ZOOM_IN_FACTOR_TRACKPAD = 0.001
