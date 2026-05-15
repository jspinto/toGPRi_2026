# togpri/gui/main_window.py
"""
Ventana principal de toGPRi v2.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QLabel, QAbstractItemView, QStatusBar, QSplitter,
    QTabWidget, QDialog
)
from PyQt6.QtCore import Qt

from togpri.io.ramac import read_ramac
from togpri.core.gprdata import GPRData
from togpri.gui.widgets.filter_panel import FilterPanel
from togpri.gui.widgets.radargram_canvas import RadargramCanvas
from togpri.gui.widgets.topo_editor import TopoEditorWidget
from togpri.gui.widgets.grid3d_panel import Grid3DPanel
from togpri.gui.dialogs.session_config import SessionConfigDialog, SessionGeometry


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("toGPRi v2")
        self.resize(1600, 900)

        self._loaded: dict[str, GPRData] = {}
        self._current_path: Path | None = None
        self._session_geometry: SessionGeometry | None = None

        self._build_ui()

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    # ── Construcción de UI ─────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Columna izquierda: lista de archivos
        left = QVBoxLayout()
        left.setSpacing(4)

        btn_open = QPushButton("📂  Abrir carpeta…")
        btn_open.clicked.connect(self._open_folder)
        left.addWidget(btn_open)

        left.addWidget(QLabel("Archivos .rd3:"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_list.currentItemChanged.connect(self._on_current_file_changed)
        left.addWidget(self.file_list)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(220)

        # Panel de filtros
        self.filter_panel = FilterPanel()
        self.filter_panel.preview_requested.connect(self._run_preview)
        self.filter_panel.apply_all_requested.connect(self._apply_all)
        self.filter_panel.filters_changed.connect(self._run_preview)

        # Pestañas centrales
        self.tabs = QTabWidget()

        self.canvas = RadargramCanvas()
        self.tabs.addTab(self.canvas, "📡 Radargramas")

        self.topo_editor = TopoEditorWidget()
        self.topo_editor.surface_ready.connect(self._on_surface_ready)
        self.tabs.addTab(self.topo_editor, "🏔 Topografía")

        from togpri.gui.widgets.local_cube_panel import LocalCubePanel
        self.local_cube_panel = LocalCubePanel()
        self.local_cube_panel.set_data_source(lambda: self._loaded)
        self.tabs.addTab(self.local_cube_panel, "🗺 Cortes locales")

        self.grid3d_panel = Grid3DPanel()
        self.grid3d_panel.set_data_source(lambda: self._loaded)
        self.tabs.addTab(self.grid3d_panel, "🧊 Malla 3D")

        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.filter_panel)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        root.addWidget(splitter)

    # ── Slots: archivos ────────────────────────────────────────────────────
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta GPR")
        if not folder:
            return
        folder = Path(folder)
        files = sorted(folder.glob("*.rd3")) + sorted(folder.glob("*.RD3"))
        if not files:
            self._status.showMessage("No se encontraron archivos .rd3 en esa carpeta.")
            return

        filenames = [f.name for f in files]
        dlg = SessionConfigDialog(filenames, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._session_geometry = dlg.session_geometry
        self._loaded.clear()

        self.file_list.clear()
        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            self.file_list.addItem(item)

        self._status.showMessage(
            f"{len(files)} archivos cargados — "
            f"spacing {self._session_geometry.line_spacing_m:.2f} m"
        )

    def _on_current_file_changed(self, current, _previous):
        if current is None:
            return
        self._current_path = Path(current.data(Qt.ItemDataRole.UserRole))
        self._run_preview()

    def _get_selected_paths(self) -> list[Path]:
        return [
            Path(item.data(Qt.ItemDataRole.UserRole))
            for item in self.file_list.selectedItems()
        ]

    # ── Slots: procesado ───────────────────────────────────────────────────
    def _run_preview(self):
        if self._current_path is None:
            return

        pipeline = self.filter_panel.get_filter_pipeline()
        self._status.showMessage(f"Procesando {self._current_path.name}…")

        try:
            gpr = read_ramac(self._current_path)
            original = gpr.data.copy()
            for step in pipeline:
                gpr.apply_filter(step["spec"].func, **step["kwargs"])
            self._on_preview_ready(gpr, original)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._status.showMessage(f"❌ Error: {e}")

    def _on_preview_ready(self, gpr: GPRData, original: np.ndarray):
        depth_m = gpr.get_depth_axis()
        dist_m  = gpr.get_distance_axis()
        extent  = [0, dist_m[-1], depth_m[-1], depth_m[0]]

        self.canvas.show_pair(original, gpr.processed_data, extent=extent)
        self.topo_editor.load_profile(dist_m, gpr.processed_data)

        # Aplicar geometría de la sesión a este perfil
        if self._session_geometry is not None:
            for prof in self._session_geometry.profiles:
                if prof["filename"] == self._current_path.name:
                    gpr.geometry["origin_xy"]    = prof["origin_xy"]
                    gpr.geometry["direction_xy"] = prof["direction_xy"]
                    break

        self._loaded[str(self._current_path)] = gpr

        self._status.showMessage(
            f"{self._current_path.name} — "
            f"{gpr.ntraces} trazas × {gpr.nsamples} samples  |  "
            f"prof. máx. ≈ {depth_m[-1]:.2f} m"
        )

    def _apply_all(self):
        paths = self._get_selected_paths()
        if not paths:
            self._status.showMessage("⚠ No hay archivos seleccionados.")
            return

        pipeline = self.filter_panel.get_filter_pipeline()
        for path in paths:
            try:
                gpr = read_ramac(path)
                for step in pipeline:
                    gpr.apply_filter(step["spec"].func, **step["kwargs"])
                if self._session_geometry is not None:
                    for prof in self._session_geometry.profiles:
                        if prof["filename"] == path.name:
                            gpr.geometry["origin_xy"]    = prof["origin_xy"]
                            gpr.geometry["direction_xy"] = prof["direction_xy"]
                            break
                self._loaded[str(path)] = gpr
            except Exception as e:
                self._status.showMessage(f"❌ Error en {path.name}: {e}")
                return

        self._status.showMessage(
            f"✅ {len(paths)} archivos procesados correctamente."
        )

    # ── Slots: topografía ──────────────────────────────────────────────────
    def _on_surface_ready(self, distance_m: np.ndarray, surface_z: np.ndarray):
        if self._current_path is None:
            return
        key = str(self._current_path)
        if key not in self._loaded:
            return

        gpr = self._loaded[key]
        gpr.geometry["surface_z"]  = surface_z.tolist()
        gpr.geometry["distance_m"] = distance_m.tolist()

        self._status.showMessage(
            f"✅ Superficie topográfica guardada para {self._current_path.name}"
        )