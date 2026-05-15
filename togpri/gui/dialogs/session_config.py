# togpri/gui/dialogs/session_config.py
"""
Diálogo de configuración de sesión GPR.
Define la geometría de la malla de perfiles paralelos.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QRadioButton, QButtonGroup, QDoubleSpinBox,
    QDialogButtonBox, QLabel, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── Resultado del diálogo ──────────────────────────────────────────────────
@dataclass
class SessionGeometry:
    """
    Geometría completa de la sesión.

    Attributes
    ----------
    line_spacing_m   : separación entre perfiles (m)
    alternating      : True si impares→ pares←
    start_corner     : 'SW' | 'SE' | 'NW' | 'NE'
    main_axis        : 'X' (perfiles horizontales) | 'Y' (verticales)
    origin_xy        : (x0, y0) coordenadas UTM del punto de inicio (puede ser 0,0)
    profiles         : lista de dicts con la geometría de cada perfil
        [{
            'filename': str,
            'index': int,           # 0-based
            'origin_xy': (x, y),
            'direction_xy': (dx, dy),  # vector unitario
        }]
    """
    line_spacing_m: float
    alternating: bool
    start_corner: str
    main_axis: str
    origin_xy: tuple
    profiles: list


def compute_geometry(
    filenames: List[str],
    line_spacing_m: float,
    alternating: bool,
    start_corner: str,
    main_axis: str,
    origin_xy: tuple = (0.0, 0.0),
) -> SessionGeometry:
    """
    Calcula la geometría de cada perfil a partir de los parámetros de la malla.

    Parameters
    ----------
    filenames      : lista de nombres de archivo (ya ordenados)
    line_spacing_m : separación entre perfiles (m)
    alternating    : True = impares directo, pares invertido
    start_corner   : 'SW' | 'SE' | 'NW' | 'NE'
    main_axis      : 'X' | 'Y'
    origin_xy      : coordenadas del punto de inicio del primer perfil
    """
    n = len(filenames)
    x0, y0 = origin_xy

    profiles = []
    for i, fname in enumerate(filenames):
        # ── Dirección del perfil ────────────────────────────────────────
        if main_axis == "X":
            # Perfiles van en dirección X
            base_dir = np.array([1.0, 0.0])
            perp_dir = np.array([0.0, 1.0])
        else:
            # Perfiles van en dirección Y
            base_dir = np.array([0.0, 1.0])
            perp_dir = np.array([1.0, 0.0])

        # Invertir según esquina de inicio
        if main_axis == "X":
            if start_corner in ("SE", "NE"):
                base_dir = -base_dir
            if start_corner in ("NW", "NE"):
                perp_dir = -perp_dir
        else:
            if start_corner in ("NW", "NE"):
                base_dir = -base_dir
            if start_corner in ("SE", "NE"):
                perp_dir = -perp_dir

        # Alternar dirección en perfiles pares
        if alternating and (i % 2 == 1):
            direction = -base_dir
        else:
            direction = base_dir

        # ── Origen del perfil ───────────────────────────────────────────
        # El perfil i empieza desplazado i * line_spacing_m en la dirección perpendicular
        offset = i * line_spacing_m * perp_dir
        origin = np.array([x0, y0]) + offset

        # Si el perfil va invertido, el origen es el extremo opuesto
        # (lo calculamos cuando conozcamos la longitud real del perfil,
        #  por ahora dejamos el mismo origen y lo ajustamos en grid3d)

        profiles.append({
            "filename": fname,
            "index": i,
            "origin_xy": (float(origin[0]), float(origin[1])),
            "direction_xy": (float(direction[0]), float(direction[1])),
        })

    return SessionGeometry(
        line_spacing_m=line_spacing_m,
        alternating=alternating,
        start_corner=start_corner,
        main_axis=main_axis,
        origin_xy=origin_xy,
        profiles=profiles,
    )


# ── Diálogo Qt ─────────────────────────────────────────────────────────────
class SessionConfigDialog(QDialog):
    """
    Diálogo de configuración de sesión.
    Se abre al cargar una carpeta con archivos .rd3.
    """

    def __init__(self, filenames: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de la sesión GPR")
        self.setMinimumWidth(600)
        self._filenames = filenames
        self._geometry: SessionGeometry | None = None

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Separación entre perfiles ─────────────────────────────────────
        form = QFormLayout()
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(0.01, 100.0)
        self.spin_spacing.setSingleStep(0.25)
        self.spin_spacing.setValue(0.5)
        self.spin_spacing.setDecimals(2)
        self.spin_spacing.setSuffix(" m")
        form.addRow("Separación entre perfiles:", self.spin_spacing)
        root.addLayout(form)

        # ── Dirección de los perfiles ─────────────────────────────────────
        dir_box = QGroupBox("Sentido de los perfiles")
        dir_layout = QVBoxLayout(dir_box)
        self._bg_dir = QButtonGroup(self)
        self.rb_same = QRadioButton("Todos en el mismo sentido")
        self.rb_alt  = QRadioButton("Alternados (impares →, pares ←)")
        self.rb_same.setChecked(True)
        self._bg_dir.addButton(self.rb_same, 0)
        self._bg_dir.addButton(self.rb_alt,  1)
        dir_layout.addWidget(self.rb_same)
        dir_layout.addWidget(self.rb_alt)
        root.addWidget(dir_box)

        # ── Esquina de inicio ─────────────────────────────────────────────
        corner_box = QGroupBox("Esquina de inicio del primer perfil")
        corner_grid = QHBoxLayout(corner_box)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        self._bg_corner = QButtonGroup(self)
        self.rb_sw = QRadioButton("SW (inferior izquierda)")
        self.rb_nw = QRadioButton("NW (superior izquierda)")
        self.rb_se = QRadioButton("SE (inferior derecha)")
        self.rb_ne = QRadioButton("NE (superior derecha)")
        self.rb_sw.setChecked(True)
        for i, rb in enumerate([self.rb_sw, self.rb_nw, self.rb_se, self.rb_ne]):
            self._bg_corner.addButton(rb, i)
        left_col.addWidget(self.rb_nw)
        left_col.addWidget(self.rb_sw)
        right_col.addWidget(self.rb_ne)
        right_col.addWidget(self.rb_se)
        corner_grid.addLayout(left_col)
        corner_grid.addLayout(right_col)
        root.addWidget(corner_box)

        # ── Eje principal ─────────────────────────────────────────────────
        axis_box = QGroupBox("Dirección principal de los perfiles")
        axis_layout = QHBoxLayout(axis_box)
        self._bg_axis = QButtonGroup(self)
        self.rb_x = QRadioButton("Horizontal (eje X →)")
        self.rb_y = QRadioButton("Vertical (eje Y ↑)")
        self.rb_x.setChecked(True)
        self._bg_axis.addButton(self.rb_x, 0)
        self._bg_axis.addButton(self.rb_y, 1)
        axis_layout.addWidget(self.rb_x)
        axis_layout.addWidget(self.rb_y)
        root.addWidget(axis_box)

        # ── Coordenadas de origen (opcionales) ───────────────────────────
        origin_box = QGroupBox("Punto de origen del primer perfil (opcional)")
        origin_form = QFormLayout(origin_box)
        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(-1e7, 1e7)
        self.spin_x0.setDecimals(3)
        self.spin_x0.setValue(0.0)
        self.spin_x0.setSuffix(" m")
        self.spin_y0 = QDoubleSpinBox()
        self.spin_y0.setRange(-1e7, 1e7)
        self.spin_y0.setDecimals(3)
        self.spin_y0.setValue(0.0)
        self.spin_y0.setSuffix(" m")
        origin_form.addRow("X (Easting):",  self.spin_x0)
        origin_form.addRow("Y (Northing):", self.spin_y0)
        root.addWidget(origin_box)

        # ── Vista previa de la malla ──────────────────────────────────────
        btn_preview = QPushButton("🗺  Vista previa de la malla")
        btn_preview.clicked.connect(self._update_preview)
        root.addWidget(btn_preview)

        self.fig_preview = Figure(figsize=(5, 3), tight_layout=True)
        self.ax_preview = self.fig_preview.add_subplot(111)
        self.canvas_preview = FigureCanvasQTAgg(self.fig_preview)
        self.canvas_preview.setFixedHeight(200)
        root.addWidget(self.canvas_preview)

        # ── Botones OK / Cancelar ─────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # Mostrar preview inicial
        self._update_preview()

    # ── Preview de la malla ─────────────────────────────────────────────────
    def _update_preview(self):
        geom = self._build_geometry()
        self.ax_preview.clear()

        for p in geom.profiles:
            x0, y0 = p["origin_xy"]
            dx, dy = p["direction_xy"]
            # Dibujar una línea corta representando cada perfil
            x1 = x0 + dx * geom.line_spacing_m * len(geom.profiles) * 0.5
            y1 = y0 + dy * geom.line_spacing_m * len(geom.profiles) * 0.5
            color = "steelblue" if p["index"] % 2 == 0 else "tomato"
            self.ax_preview.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )
            self.ax_preview.text(
                x0, y0, str(p["index"] + 1),
                fontsize=7, ha="center", va="center",
                color=color
            )

        self.ax_preview.set_title(
            f"{len(geom.profiles)} perfiles — spacing {geom.line_spacing_m:.2f} m",
            fontsize=9
        )
        self.ax_preview.set_aspect("equal")
        self.ax_preview.grid(True, alpha=0.3)
        self.canvas_preview.draw()

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _build_geometry(self) -> SessionGeometry:
        corner_map = {0: "SW", 1: "NW", 2: "SE", 3: "NE"}
        return compute_geometry(
            filenames=self._filenames,
            line_spacing_m=self.spin_spacing.value(),
            alternating=self.rb_alt.isChecked(),
            start_corner=corner_map[self._bg_corner.checkedId()],
            main_axis="X" if self.rb_x.isChecked() else "Y",
            origin_xy=(self.spin_x0.value(), self.spin_y0.value()),
        )

    def _on_accept(self):
        self._geometry = self._build_geometry()
        self.accept()

    # ── API pública ──────────────────────────────────────────────────────────
    @property
    def session_geometry(self) -> SessionGeometry | None:
        return self._geometry