# togpri/gui/widgets/radargram_canvas.py
"""
Canvas Matplotlib embebido con vista doble (original / procesado).
"""
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QSlider, QSplitter, QSizePolicy
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


COLORMAPS = ["gray", "seismic", "RdBu_r", "viridis", "plasma", "bwr", "Greys"]


class _SingleCanvas(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._data: np.ndarray | None = None
        self._extent = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Figura y eje inicial
        self.fig = Figure(figsize=(8, 4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

    def plot(self, data: np.ndarray, extent=None, cmap: str = "gray",
             clip_pct: float = 1.0):
        # Guardar referencia
        self._data = data
        self._extent = extent

        # Limpiar completamente la figura (incluidas colorbars)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if data is None:
            self.canvas.draw()
            return

        # Calcular límites de visualización
        vmin = np.percentile(data, clip_pct)
        vmax = np.percentile(data, 100 - clip_pct)

        kwargs = dict(
            cmap=cmap,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        if extent is not None:
            kwargs["extent"] = extent

        im = self.ax.imshow(data, **kwargs)

        # Nueva barra de color (la anterior ya desapareció con fig.clear())
        cbar = self.fig.colorbar(
            im, ax=self.ax, fraction=0.03, pad=0.02, label="Amplitud"
        )

        # Etiquetas y título
        self.ax.set_title(self._title, fontsize=10, fontweight="bold")
        if extent is not None:
            self.ax.set_xlabel("Distancia (m)")
            self.ax.set_ylabel("Profundidad (m)")
        else:
            self.ax.set_xlabel("Traza")
            self.ax.set_ylabel("Sample")

        self.canvas.draw()


class RadargramCanvas(QWidget):
    """
    Widget con dos canvases (original arriba, procesado abajo) +
    controles de colormap y clipping.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cmap = "gray"
        self._clip_pct = 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Barra de controles ───────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        ctrl.addWidget(self.cmap_combo)

        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("Clip %:"))
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(0, 10)   # 0.0 % … 5.0 %
        self.clip_slider.setValue(2)       # default 1 %
        self.clip_slider.setFixedWidth(120)
        self.clip_label = QLabel("1.0%")
        self.clip_slider.valueChanged.connect(self._on_clip_changed)
        ctrl.addWidget(self.clip_slider)
        ctrl.addWidget(self.clip_label)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── Canvases ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        self._canvas_orig = _SingleCanvas("Original")
        self._canvas_proc = _SingleCanvas("Procesado")
        splitter.addWidget(self._canvas_orig)
        splitter.addWidget(self._canvas_proc)
        splitter.setSizes([400, 400])
        root.addWidget(splitter)

    # ── Slots ────────────────────────────────────────────────────────────
    def _on_cmap_changed(self, cmap: str):
        self._cmap = cmap
        self._redraw()

    def _on_clip_changed(self, val: int):
        self._clip_pct = val * 0.5   # slider 0-10 → 0.0–5.0 %
        self.clip_label.setText(f"{self._clip_pct:.1f}%")
        self._redraw()

    def _redraw(self):
        self._canvas_orig.plot(
            self._canvas_orig._data,
            self._canvas_orig._extent,
            self._cmap,
            self._clip_pct,
        )
        self._canvas_proc.plot(
            self._canvas_proc._data,
            self._canvas_proc._extent,
            self._cmap,
            self._clip_pct,
        )

    # ── API pública ──────────────────────────────────────────────────────
    def show_pair(self, original: np.ndarray, processed: np.ndarray,
                  extent=None):
        self._canvas_orig.plot(original, extent, self._cmap, self._clip_pct)
        self._canvas_proc.plot(processed, extent, self._cmap, self._clip_pct)

    def clear(self):
        self._canvas_orig.plot(None)
        self._canvas_proc.plot(None)