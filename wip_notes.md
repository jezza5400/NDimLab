# TODO

Matrix & NDC Optimization (Homogeneous 'w' Scaling):

- Keep original 2D/3D mesh points static in VRAM (VBO). Do NOT update vertex arrays on CPU during window/viewport resize.
- Update the single Uniform Projection Matrix on resize instead:
  - Scale terms in matrix (like bottom-right 'w' or matrix scale factors) handle aspect ratio and global unit scaling.
  - Recall: NDC (Normalized Device Coordinates) expects coordinates mapped from -1 to +1.
      The GPU auto-performs Perspective Divide (x/w, y/w, z/w) post-vertex shader.
- Note for Vulkan setup:
  - Ensure GLSL/HLSL column vs. row-major layout matches host memory layout.
  - Flip Y-axis scale in Projection Matrix (Vulkan NDC Y is inverted compared to OpenGL).

Evaluate Migration to Slang Shading Language:

- Language & Features:
  - Slang uses modern C#-like syntax (attributes `[...]`, generics, interfaces, modules).
  - FOSS & Vendor-Neutral: Hosted by Khronos Group (multi-vendor governance).
  - Code Once, Target Anything: Compiles to SPIR-V (Vulkan), HLSL (DX12), MSL (Metal), CUDA, or GLSL.

- ModernGL / Python Integration:
  - Use `slangc` CLI or Slang API to emit GLSL target strings (`-target glsl`).
  - Pass compiled GLSL output into `ctx.program(...)` in ModernGL.
  - Check out `slangpy` if integrating GPU compute shaders with PyTorch/NumPy workflows.

Add a per-entity polygon/points toggle.

Add toggle to make Z-Value effect draw order.

Add more obvious paused indicator.

# demos/opengl.py

## Critical

### Context Management Bug

You are calling `self.ctx = mgl.create_context()`. In moderngl, this creates a new OpenGL context. However, `QOpenGLWindow` already manages its own context. When you call `self.makeCurrent()`, you make the window's context current, but `self.ctx` still points to the new context you created. Consequently, calls like `self.ctx.clear()`, `self.ctx.enable()`, and `self.ctx.viewport()` will be operating on a "ghost" context that isn't attached to the window, meaning nothing will appear on the screen or it will crash.

**Fix:** Since moderngl doesn't easily "wrap" an existing `QOpenGLContext`, you should ensure `self.ctx` is the context associated with the window. However, since moderngl prefers to own the context, the most common fix in this specific setup is to ensure `mgl.create_context()` is called and then you make that context current (which is difficult with `QOpenGLWindow`'s `makeCurrent`). Alternatively, use a `QOpenGLContext` and tell moderngl to use it if possible, or stick to the standard `mgl.create_context()` but ensure it's the only context being used.

## Warning

### Viewport Assignment Bug

`self.ctx.viewport = (0, 0, phys_w, phys_h)` is an assignment, not a function call. In moderngl, `viewport` is a method. This line will set an attribute on the ctx object but will not actually update the OpenGL viewport.

**Fix:** Change  
`self.ctx.viewport = (0, 0, phys_w, phys_h)`  
→  
`self.ctx.viewport(0, 0, phys_w, phys_h)`.

## There-is-a-much-better-way

### Manual Context State Management

You are manually calling `self.ctx.disable(mgl.DEPTH_TEST)` and `self.ctx.enable(mgl.DEPTH_TEST)` inside `paintGL`. While correct, this can be simplified. If you use a Framebuffer Object (FBO) for your background, you can set the depth test state once during the FBO's creation or during the `bake_grid` phase.

### Redundant `makeCurrent()`

You call `self.makeCurrent()` at the start of `paintGL` and `initializeGL`. While safe, `paintGL` is guaranteed to have the context current by the Qt event loop. You only need it in `initializeGL`.

---

# demos/locked_points_demo.py

## Warning

### Global Variable Usage

You are using `global accumulator` in the `tick` function. While not a "bug" for a small project, it's better practice to encapsulate this in a class or a state dictionary to avoid side effects.

### Hardcoded Rotation Matrices

You are manually constructing  
`np.array([[cos(2 * theta), sin(2 * theta)], [-sin(2 * theta), cos(2 * theta)]])`  
multiple times. This is prone to typos (e.g., swapping a sign).

**Fix:** Create a helper function `get_rotation_matrix(angle_degrees)` or use a proper rotation matrix formula.

### Coordinate Flipping

`view.scale(100, -100)` is used to flip the Y-axis. While common in OpenGL, be aware that this also flips your coordinate system for any mouse interaction (like clicking points).

## Note

### Matrix Multiplication Order

You are using `unit_square[:] = unit_square @ rotation_T`. This is correct for row-vector transformation (`v·M`), but ensure you stay consistent with this throughout the project, as many OpenGL tutorials use column-vector notation (`M·v`).

---

# demos/numpy_promotion.py

## Note

### Implicit Axis Logic

In `vec_to_homogeneous`, you use `h = np.eye(n + 1, dtype=arr.dtype)`.  
If the input `arr` is a 1D array (a single vector), this produces a 2D matrix.  
If the input is a 2D array (batch of vectors), you create a 3D array.

### Efficiency

`h[:, idx, idx] = 1` is very efficient. However, for the translation part `h[:, -1, :-1] = arr`, ensure that the input `arr` is always the correct shape, otherwise NumPy will broadcast incorrectly or throw a shape mismatch.

---

# demos/matrix_gui.py

## There-is-a-much-better-way

### Manual Bracket Drawing

You are using `painter.drawLine` to draw the `[` and `]` brackets. This doesn't account for the high-DPI scaling of the widget perfectly because it's using fixed pixel offsets.

**Fix:** Use `painter.drawText` or calculate the bracket position relative to `self.contentsRect()`.

## Note

### Style Sheet Complexity

The `cell_style` string is quite large. If you plan to expand this, consider moving styles to an external `.qss` file or a dedicated constant.

---

# demos/pyside6_widgets.py

## Note

### Widget Instantiation

The loop  
`for widget in widgets: layout.addWidget(widget())`  
is very clean and clever for a demo script.

---

# shaders/ (General Observations)

## Warning

### Precision

In your fragment shaders (e.g., `grid.frag`), ensure you are using  
`precision highp float;`  
if you ever plan to run these on mobile (OpenGL ES), though for desktop it's usually the default.

### Uniform Naming

Ensure the names in `opengl.py` (like `u_bg_resolution`) exactly match the uniform declarations in your `.frag` and `.vert` files.

---

# pyproject.toml

## Note

### Python Version

You have `requires-python = ">=3.14"`. As of now, Python 3.14 is in development. If you want to support the current stable releases, you might want 3.12 or 3.13.

### Ruff Linting

You have  
`select = ["PLW0602", "PLW0603", "PLW0604"]`.  
These are specifically for "unused" variables/imports. Since you have `typeCheckingMode = "strict"` in pyright, you might find you're getting a lot of redundant errors. This is fine for a personal project, but `reportUnknown` types might be noisier than you expect.
