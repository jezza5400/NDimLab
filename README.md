# NDimLab

[![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads)

NDimLab is a Python library for working with and visualizing n-dimensional arrays and tensors.

## Notes

- One-shot transformations must be applied before continuous transformations who subsequently get repeatedly applied once each physics update.
- When combining transformations (and internally promoting to homogeneous): first combine all adjacent matrices of the same type (linear or vector), then promote all to homogeneous, and finally combine into one.

## Qt Dark Mode in Virtual Environments

**Problem:**  
PySide6 inside a Python virtual environment uses its own plugin directory, which does not include system Qt theme plugins (like `qt6ct`). Because of this, Qt cannot detect the system’s dark mode and falls back to the **Fusion light theme**.

**Fix:**  
Manually point Qt to the system plugin paths by adding these lines to the bottom of your venv’s `bin/activate` script:

```bash
export QT_PLUGIN_PATH=/usr/lib/qt6/plugins
export QT_QPA_PLATFORMTHEME=qt6ct
export QT_QPA_PLATFORMTHEME_PATH=/usr/lib/qt6/plugins/platformthemes
```

This restores proper dark‑mode support when running inside the venv.

## Plans: PySide6 + OpenGL Matrix Visualizer

### 1. Architectural Overview (The Pipeline Split)

Instead of relying on the CPU to calculate and redraw shapes every frame, the application splits responsibilities between processors to bypass Python's Single Thread limitations (the GIL) and maximize performance.

- **CPU (Python + NumPy + PySide6):** Acts as the **Director**. Handles the user interface (sliders, text boxes), manages the application state, calculates high-level transformation matrices, and tells the GPU when to draw.
- **GPU (OpenGL Core Profile 4.6):** Acts as the **Factory**. Automatically distributes workloads across thousands of tiny cores. It applies matrix math to all polygon points simultaneously and renders pixels to the screen.

### 2. Data Flow Strategy

#### A. At Startup / Initialization (`initializeGL`)

1. Create your base shape vertices (e.g., thousands of $X, Y$ coordinate pairs) as a **flat NumPy array**.
2. Upload this array **once** over the PCIe lane to the GPU's Video Memory (VRAM) using a **Vertex Buffer Object (VBO)**.
3. *Note:* Leave these original points parked statically in VRAM. They should never be modified or re-uploaded during normal animations.

#### B. Every Physics Tick / Frame Update (`paintGL`)

1. **CPU Side:** Rather than continuously overriding the points array (`points[:] = points @ T`), track state variables (like an accumulating rotation angle or offset).
2. **CPU Side:** Multiply all active operations into a **single, final transformation matrix** (only 36–64 bytes of data) using PySide6's built-in math classes.
3. **Transfer:** Stream just that tiny matrix over to the GPU as a shader **Uniform**.
4. **GPU Side:** The GPU reads the static original points from VRAM, multiplies them by the uniform matrix "in-flight" inside the Vertex Shader, and draws them instantly.

### 3. PySide6 Built-in Math Utilities (Optimization)

Instead of installing external 3D math libraries (like `pyrr` or `numpy-stl`), leverage the highly optimized, C++ backed math primitives included directly in `PySide6.QtGui`. They are designed to map perfectly to OpenGL concepts with zero Python overhead.

#### Key Classes to Use

- **`QMatrix4x4`**: Handles all projection, translation, rotation, and scaling matrices.
- **`QVector3D` / `QVector4D`**: Handles 3D coordinates, direction vectors, and color vectors.

#### Core Advantages & Integration

- **Native C++ Performance:** All matrix multiplications are performed in compiled C++ code inside the Qt framework, completely avoiding slow Python loops.
- **Zero-Copy Shader Uploads:** You do not need to convert a `QMatrix4x4` back into a NumPy array to pass it to OpenGL. Use the `.constData()` method to expose a direct C-pointer to the matrix data, which can be fed straight into `glUniformMatrix4fv`.

```python
# Example: Uploading a PySide6 Matrix straight to a Shader Uniform
matrix = QMatrix4x4()
matrix.perspective(45.0, width / height, 0.1, 100.0)
matrix.translate(0.0, 0.0, -5.0)
matrix.rotate(angle, 0.0, 1.0, 0.0)

# Pass directly to PyOpenGL using .constData()
glUniformMatrix4fv(matrix_uniform_location, 1, GL_FALSE, matrix.constData())

```

- **Fluent API:** PySide6 matrices have built-in semantic methods like `.lookAt()`, `.perspective()`, `.rotate()`, and `.translate()`, saving you from manually writing complex trigonometry matrices.

### 4. Handling UI & Interactive Features (Lazy Evaluation)

To keep performance perfectly smooth, separate **Rendering Math** from **UI Logic**:

#### Hovering / Showing Point Coordinates

- **Don't** transform all 40,000 points on the CPU every frame just in case the user wants to see a coordinate.
- **Do** calculate coordinates lazily. When a user hovers over a specific vertex, grab that *one* point from your Python-side copy of the original array, multiply it by the current matrix on the CPU, and push it to the UI label.

#### Live Global Coordinate Spreadsheet/List

- If the user toggles a view showing *all* current coordinates changing in real time, execute the full NumPy multiplication (`live_points = original_points @ combined_matrix`) **only while that UI panel is open**.
- *Optimization Tip:* To prevent Qt UI lag, use virtual scrolling or pagination to only render text for rows currently visible on the screen.

### 5. Key Advantages of this Architecture

- **Zero Precision Drift:** Because you always multiply the *original* base points by a freshly calculated total matrix, you avoid the cumulative floating-point rounding errors that warp shapes over time.
- **Minimal PCIe Traffic:** You stream 64 bytes (the $4\times4$ matrix) per frame instead of hundreds of kilobytes of modified point coordinates.
- **Infinite Scalability:** OpenGL handles the hardware core distribution automatically. The exact same code will instantly scale from an Intel integrated GPU up to a dedicated desktop graphics card without modifications.

## QOpenGLWidget execution order

- **`initializeGL()`** is guaranteed to run once before the first time `resizeGL()` or `paintGL()` is called.
- **`initializeGL()`** can technically be called again if the underlying GL context is destroyed and recreated (e.g. the widget is reparented into a different top-level window, or the driver resets the context).
- Unlike `QOpenGLWindow`, a `QOpenGLWidget` renders into an off-screen **framebuffer object (FBO)**, not directly to the native window surface. Qt then composites that FBO's texture into the normal widget-painting pipeline alongside sibling widgets.
- **`resizeGL()`** fires whenever the widget is resized, and also on first show, since new widgets get an automatic resize event. It's also where the backing FBO gets reallocated at the new size.
- **`update()`** is the correct way to *request* a repaint from outside `paintGL()`, but it's *asynchronous*: it schedules a `paintEvent()` on the event loop rather than calling `paintGL()` directly.
- `paintGL()` isn't called by Qt's window system directly - it's invoked from inside `QOpenGLWidget::paintEvent()`, which itself is driven by the same widget-update mechanism as any other `QWidget` (so it can also be triggered indirectly by parent-widget repaints, not just `update()` calls).

```mermaid
---
config:
  layout: dagre
---
flowchart TB
    subgraph SETUP["Startup (runs once)"]
        direction TB
        A(["Program Start"]) --> B["__init__()"]
        B --> C["Python sets up instance variables\n(no GL context yet)"]
        C --> D["widget.show()"]
        D --> E["OS creates native window;\nQt creates GL context + FBO"]
        E --> F["initializeGL()\ncompile shaders, upload VBOs/textures"]
        F --> G["resizeGL(w, h)\nallocate FBO, set viewport & projection"]
        G --> H["paintEvent() → paintGL()\nfirst frame rendered into FBO,\nthen composited to screen"]
    end

    H --> LOOP{"Event Loop\n(Qt waits for next event)"}

    subgraph RUNTIME["Runtime Loop (repeats)"]
        direction TB
        LOOP -->|Window/layout resized| RZ["resizeGL(w, h)\nFBO reallocated"]
        RZ --> SCHED["Qt auto-schedules a repaint"]
        LOOP -->|Key / Mouse event| EV["keyPressEvent() / mouseMoveEvent() / etc."]
        EV --> UPD["self.update()"]
        LOOP -->|Timer / animation tick| TMR["QTimer callback"]
        TMR --> UPD
        LOOP -->|Parent widget repaints| PARENT["Parent/sibling widget update"]
        PARENT --> SCHED
        LOOP -->|Nothing pending| IDLE["Idle — CPU free, no draw"]
        IDLE --> LOOP
        SCHED --> PE["paintEvent()"]
        UPD --> PE
        PE --> PG["paintGL() runs again\n(renders into FBO)"]
        PG --> COMP["Qt composites FBO texture\nwith rest of widget tree"]
        COMP --> LOOP
    end

    classDef setupNode fill:#cfe8ff,stroke:#4a90d9,stroke-width:1px,color:#1a1a1a;
    classDef loopNode fill:#ffe3b3,stroke:#d98e2b,stroke-width:1px,color:#1a1a1a;
    classDef decision fill:#e0c3fc,stroke:#8e44ad,stroke-width:1px,color:#1a1a1a;
    classDef idleNode fill:#e0e0e0,stroke:#888,stroke-width:1px,color:#1a1a1a;

    class A,B,C,D,E,F,G,H setupNode;
    class RZ,SCHED,EV,UPD,TMR,PE,PG,PARENT,COMP loopNode;
    class LOOP decision;
    class IDLE idleNode;
```

<span style="color:#ff9100">Note:</span> with `QOpenGLWindow` the `paintGL()` method draws essentially straight to the screen surface, but with `QOpenGLWidget` it draws into an FBO first, and that FBO is then blended into the rest of the widget hierarchy like any other widget's content. That's what makes `QOpenGLWidget` composable with normal Qt widgets (possible to layer a `QPushButton` on top of it, put it in a layout, etc.) at the cost of an extra copy/blit per frame.

## QOpenGLWindow execution order

- **`initializeGL()`** is guaranteed to run once before the first time resizeGL() or paintGL() is called.
- **`initializeGL()`** can technically be called again if the underlying GL context is destroyed and recreated. (e.g. some GPU driver resets or screen/adapter changes)
- **`resizeGL()`** fires whenever the widget is resized, and also on first show, since new widgets get an automatic resize event.
- **`update()`** is the correct way to *request* a repaint from outside `paintGL()` but as it's *asynchronous*: it schedules a repaint on the event loop rather than calling `paintGL()` directly.

```mermaid
---
config:
  layout: dagre
---
flowchart TB
    subgraph SETUP["Startup (runs once)"]
        direction TB
        A(["Program Start"]) --> B["__init__()"]
        B --> C["Python sets up instance variables\n(no GL context yet)"]
        C --> D["window.show()"]
        D --> E["OS creates native window + GL context"]
        E --> F["initializeGL()\ncompile shaders, upload VBOs/textures"]
        F --> G["resizeGL(w, h)\nset viewport & projection"]
        G --> H["paintGL()\nfirst frame rendered"]
    end

    H --> LOOP{"Event Loop\n(Qt waits for next event)"}

    subgraph RUNTIME["Runtime Loop (repeats)"]
        direction TB
        LOOP -->|Window resized| RZ["resizeGL(w, h)"]
        RZ --> SCHED["Qt auto-schedules a repaint"]
        LOOP -->|Key / Mouse event| EV["keyPressEvent() / mouseMoveEvent() / etc."]
        EV --> UPD["self.update()"]
        LOOP -->|Timer / animation tick| TMR["QTimer callback"]
        TMR --> UPD
        LOOP -->|Nothing pending| IDLE["Idle — CPU free, no draw"]
        IDLE --> LOOP
        SCHED --> PG["paintGL() runs again"]
        UPD --> PG
        PG --> LOOP
    end

    classDef setupNode fill:#cfe8ff,stroke:#4a90d9,stroke-width:1px,color:#1a1a1a;
    classDef loopNode fill:#ffe3b3,stroke:#d98e2b,stroke-width:1px,color:#1a1a1a;
    classDef decision fill:#e0c3fc,stroke:#8e44ad,stroke-width:1px,color:#1a1a1a;
    classDef idleNode fill:#e0e0e0,stroke:#888,stroke-width:1px,color:#1a1a1a;

    class A,B,C,D,E,F,G,H setupNode;
    class RZ,SCHED,EV,UPD,TMR,PG loopNode;
    class LOOP decision;
    class IDLE idleNode;
```
