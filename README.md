# NDimLab

[![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads)

NDimLab is a Python library for working with and visualizing n-dimensional arrays and tensors.

Notes:

- One-shot transformations must be applied before continuous transformations who subsequently get repeatedly applied once each physics update.

## Dev Note: Qt Dark Mode in Virtual Environments

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
