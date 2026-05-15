# togpri/gui/widgets/topo_editor.py
"""
Editor topográfico interactivo integrado en Qt.
El usuario hace clic sobre el perfil para colocar puntos de control
e introduce su cota z. La curva PCHIP se actualiza en tiempo real.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QDoubleSpinBox, QDialog, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QSizePolicy, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from togpri.processing.topography import interpolate_surface, save_surface_csv


# ── Diálogo para introducir la cota z ──────────────────────────────────────
class ZInputDialog(QDialog):
    def __init__(self, distance: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cota topográfica")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Distancia en el perfil: <b>{distance:.2f} m</b>"))
        layout.addWidget(QLabel("Cota z (m sobre el nivel de referencia):"))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(-9999.0, 9999.0)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.1)
        self.spin.setValue(0.0)
        layout.addWidget(self.spin)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @property
    def z_value(self) -> float:
        return self.spin.value()


# ── Widget principal ────────────────────────────────────────────────────────
class TopoEditorWidget(QWidget):
    """
    Editor topográfico 1D embebido en la ventana principal.

    Señales
    -------
    surface_ready(np.ndarray, np.ndarray)
        Emitida cuando se interpola la superficie.
        Parámetros: (distance_m, surface_z)
    """
    surface_ready = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._distance_m: np.ndarray | None = None
        self._control_d: list[float] = []
        self._control_z: list[float] = []
        self._surface_z: np.ndarray | None = None
        self._cid = None   # connection id del evento matplotlib

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # Canvas
        self.fig = Figure(figsize=(10, 3), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        toolbar = NavigationToolbar2QT(self.canvas, self)
        root.addWidget(toolbar)
        root.addWidget(self.canvas)

        # Botones
        btn_row = QHBoxLayout()

        self.btn_add = QPushButton("➕ Modo añadir punto (clic en gráfico)")
        self.btn_add.setCheckable(True)
        self.btn_add.toggled.connect(self._toggle_add_mode)
        btn_row.addWidget(self.btn_add)

        btn_interpolate = QPushButton("〰 Interpolar")
        btn_interpolate.clicked.connect(self._interpolate)
        btn_row.addWidget(btn_interpolate)

        btn_undo = QPushButton("↩ Deshacer último")
        btn_undo.clicked.connect(self._undo)
        btn_row.addWidget(btn_undo)

        btn_clear = QPushButton("🗑 Limpiar todo")
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_clear)

        btn_save = QPushButton("💾 Guardar CSV")
        btn_save.clicked.connect(self._save_csv)
        btn_row.addWidget(btn_save)

        root.addLayout(btn_row)

        # Tabla de puntos de control
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Distancia (m)", "Cota z (m)"])
        self.table.setFixedHeight(120)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.itemChanged.connect(self._on_table_edited)
        root.addWidget(self.table)

    # ── Carga de perfil ──────────────────────────────────────────────────────
    def load_profile(self, distance_m: np.ndarray, data: np.ndarray | None = None):
        """
        Carga un perfil nuevo. Opcionalmente muestra el radargrama de fondo
        (data: ndarray procesado) para orientar el posicionamiento de cotas.
        """
        self._distance_m = distance_m
        self._control_d.clear()
        self._control_z.clear()
        self._surface_z = None
        self.table.setRowCount(0)

        self.ax.clear()

        if data is not None:
            # Mostrar el radargrama como fondo (comprimido verticalmente)
            vmin = np.percentile(data, 1)
            vmax = np.percentile(data, 99)
            self.ax.imshow(
                data, cmap="gray", aspect="auto", origin="upper",
                extent=[distance_m[0], distance_m[-1], data.shape[0], 0],
                vmin=vmin, vmax=vmax, alpha=0.4,
            )
            self.ax.set_ylabel("Sample")
        else:
            self.ax.set_ylabel("Cota z (m)")

        self.ax.set_xlabel("Distancia (m)")
        self.ax.set_title("Editor topográfico — haz clic para añadir puntos de control")
        self.canvas.draw()

    # ── Modo añadir punto ───────────────────────────────────────────────────
    def _toggle_add_mode(self, active: bool):
        if active:
            self._cid = self.canvas.mpl_connect("button_press_event", self._on_click)
            self.btn_add.setText("🔴 Modo activo — clic para colocar punto")
        else:
            if self._cid is not None:
                self.canvas.mpl_disconnect(self._cid)
                self._cid = None
            self.btn_add.setText("➕ Modo añadir punto (clic en gráfico)")

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        distance = float(event.xdata)
        dlg = ZInputDialog(distance, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        z = dlg.z_value
        self._add_control_point(distance, z)

    def _add_control_point(self, distance: float, z: float):
        self._control_d.append(distance)
        self._control_z.append(z)
        self._update_table()
        self._draw_control_points()

    # ── Interpolación ────────────────────────────────────────────────────────
    def _interpolate(self):
        if self._distance_m is None:
            QMessageBox.warning(self, "Sin perfil", "Carga primero un perfil GPR.")
            return
        if len(self._control_d) < 2:
            QMessageBox.warning(self, "Pocos puntos",
                                "Necesitas al menos 2 puntos de control.")
            return

        try:
            self._surface_z = interpolate_surface(
                self._control_d, self._control_z, self._distance_m
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de interpolación", str(e))
            return

        self._draw_surface()
        self.surface_ready.emit(self._distance_m.copy(), self._surface_z.copy())

    # ── Dibujo ───────────────────────────────────────────────────────────────
    def _draw_control_points(self):
        # Eliminar scatter anterior si existe
        for artist in self.ax.collections:
            artist.remove()
        for line in self.ax.lines:
            line.remove()

        if self._control_d:
            self.ax.scatter(
                self._control_d, self._control_z,
                color="red", s=60, zorder=5, label="Puntos de control"
            )

        if self._surface_z is not None:
            self.ax.plot(
                self._distance_m, self._surface_z,
                "b-", linewidth=2, label="Superficie interpolada"
            )

        self.ax.legend(fontsize=8)
        self.canvas.draw()

    def _draw_surface(self):
        self._draw_control_points()   # redibuja todo incluyendo la curva

    # ── Tabla ────────────────────────────────────────────────────────────────
    def _update_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._control_d))
        for i, (d, z) in enumerate(zip(self._control_d, self._control_z)):
            self.table.setItem(i, 0, QTableWidgetItem(f"{d:.3f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{z:.3f}"))
        self.table.blockSignals(False)

    def _on_table_edited(self, item: QTableWidgetItem):
        """Permite editar los puntos directamente en la tabla."""
        row = item.row()
        try:
            val = float(item.text())
        except ValueError:
            return
        if item.column() == 0:
            self._control_d[row] = val
        else:
            self._control_z[row] = val
        self._draw_control_points()

    # ── Acciones ─────────────────────────────────────────────────────────────
    def _undo(self):
        if self._control_d:
            self._control_d.pop()
            self._control_z.pop()
            self._surface_z = None
            self._update_table()
            self._draw_control_points()

    def _clear(self):
        self._control_d.clear()
        self._control_z.clear()
        self._surface_z = None
        self.table.setRowCount(0)
        self._draw_control_points()

    def _save_csv(self):
        if self._surface_z is None:
            QMessageBox.warning(self, "Sin superficie",
                                "Interpola primero la superficie.")
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar superficie", "surface.csv", "CSV (*.csv)"
        )
        if path:
            save_surface_csv(self._distance_m, self._surface_z, Path(path))

    # ── API pública ──────────────────────────────────────────────────────────
    @property
    def surface_z(self) -> np.ndarray | None:
        return self._surface_z

    @property
    def control_points(self) -> tuple[list[float], list[float]]:
        return self._control_d.copy(), self._control_z.copy()