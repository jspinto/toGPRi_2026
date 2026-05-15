# togpri/gui/widgets/grid3d_panel.py
"""
Panel de construcción del volumen 3D y exportación de cortes.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QDoubleSpinBox, QLabel, QFileDialog,
    QGroupBox, QSizePolicy, QMessageBox, QLineEdit, QSpinBox,
    QComboBox, QCheckBox,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from togpri.processing.grid3d import Grid3DBuilder, GPRVolume
from togpri.processing.ply_export import volume_to_ply


class Grid3DPanel(QWidget):
    """
    Pestaña '🧊 Malla 3D'.
    Permite construir el volumen, exportar cortes y nubes PLY.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._volume: GPRVolume | None = None
        self._get_loaded_fn = None   # callback → dict[str, GPRData]

        # diccionario material -> velocidad media en m/ns
        self._material_velocities: dict[str, float] = {
            "Aire": 0.30,
            "Asfalto": 0.16,
            "Hielo": 0.17,
            "Arena seca": 0.15,
            "Sal seca": 0.13,
            "Granito": 0.13,
            "Caliza": 0.12,
            "Lutitas / shale": 0.09,
            "Limos": 0.08,
            "Arcillas": 0.06,
            "Arena saturada": 0.055,
            "Agua dulce": 0.033,
            "Agua de mar": 0.02,
        }

        self._build_ui()

    def set_data_source(self, fn):
        """fn() debe devolver el dict _loaded de MainWindow."""
        self._get_loaded_fn = fn

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Parámetros de rejilla ────────────────────────────────────────────
        params_box = QGroupBox("Parámetros de la rejilla")
        params_form = QFormLayout(params_box)

        self.spin_cell_xy = QDoubleSpinBox()
        self.spin_cell_xy.setRange(0.01, 10.0)
        self.spin_cell_xy.setValue(0.05)
        self.spin_cell_xy.setSingleStep(0.01)
        self.spin_cell_xy.setSuffix(" m")
        params_form.addRow("Celda XY:", self.spin_cell_xy)

        self.spin_cell_z = QDoubleSpinBox()
        self.spin_cell_z.setRange(0.001, 1.0)
        self.spin_cell_z.setValue(0.02)
        self.spin_cell_z.setSingleStep(0.005)
        self.spin_cell_z.setSuffix(" m")
        params_form.addRow("Celda Z:", self.spin_cell_z)

        # Velocidad EM (m/ns)
        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.01, 0.3)
        self.spin_velocity.setValue(0.1)
        self.spin_velocity.setSingleStep(0.005)
        self.spin_velocity.setDecimals(3)
        self.spin_velocity.setSuffix(" m/ns")
        params_form.addRow("Velocidad EM:", self.spin_velocity)

        # Selector de material principal
        self.combo_material = QComboBox()
        self.combo_material.addItems([
            "Personalizado",
            "Aire",
            "Asfalto",
            "Hielo",
            "Arena seca",
            "Sal seca",
            "Granito",
            "Caliza",
            "Lutitas / shale",
            "Limos",
            "Arcillas",
            "Arena saturada",
            "Agua dulce",
            "Agua de mar",
        ])
        params_form.addRow("Material principal:", self.combo_material)

        # Etiqueta con velocidad en cm/ns
        self.lbl_vel_cmns = QLabel("")
        params_form.addRow("Velocidad media:", self.lbl_vel_cmns)

        # Modo rápido: decimar puntos antes de griddata
        fast_row = QHBoxLayout()
        self.chk_fast_mode = QCheckBox("Modo rápido (decimar puntos)")
        self.chk_fast_mode.setChecked(True)
        fast_row.addWidget(self.chk_fast_mode)

        self.spin_max_points_m = QDoubleSpinBox()
        self.spin_max_points_m.setRange(0.1, 50.0)
        self.spin_max_points_m.setValue(5.0)  # 5 millones
        self.spin_max_points_m.setSingleStep(1.0)
        self.spin_max_points_m.setSuffix(" M puntos")
        fast_row.addWidget(self.spin_max_points_m)

        params_form.addRow("Decimado previo:", fast_row)

        root.addWidget(params_box)

        # Conexiones para actualizar velocidad según material y etiqueta
        self.combo_material.currentTextChanged.connect(self._on_material_changed)
        self.spin_velocity.valueChanged.connect(self._update_vel_label)
        self._update_vel_label(self.spin_velocity.value())

        # ── Botón construir ──────────────────────────────────────────────────
        btn_build = QPushButton("🧊  Construir volumen 3D")
        btn_build.clicked.connect(self._build_volume)
        root.addWidget(btn_build)

        self.lbl_status = QLabel("Sin volumen.")
        root.addWidget(self.lbl_status)

        # ── Exportar cortes ──────────────────────────────────────────────────
        export_box = QGroupBox("Exportar cortes horizontales (GeoTIFF)")
        export_layout = QVBoxLayout(export_box)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Profundidades (m), separadas por comas:"))
        self.edit_depths = QLineEdit("0.5, 1.0, 1.5, 2.0")
        depth_row.addWidget(self.edit_depths)
        export_layout.addLayout(depth_row)

        epsg_row = QHBoxLayout()
        epsg_row.addWidget(QLabel("EPSG del CRS:"))
        self.spin_epsg = QSpinBox()
        self.spin_epsg.setRange(1024, 99999)
        self.spin_epsg.setValue(25830)
        epsg_row.addWidget(self.spin_epsg)
        epsg_row.addStretch()
        export_layout.addLayout(epsg_row)

        btn_export = QPushButton("💾  Exportar GeoTIFFs")
        btn_export.clicked.connect(self._export_geotiffs)
        export_layout.addWidget(btn_export)

        root.addWidget(export_box)

        # ── Exportar nube de puntos PLY ──────────────────────────────────────
        ply_box = QGroupBox("Exportar nube de puntos (PLY)")
        ply_form = QFormLayout(ply_box)

        self.spin_stride_xy = QSpinBox()
        self.spin_stride_xy.setRange(1, 20)
        self.spin_stride_xy.setValue(2)
        self.spin_stride_xy.setSingleStep(1)
        ply_form.addRow("Stride XY (submuestreo):", self.spin_stride_xy)

        self.spin_stride_z = QSpinBox()
        self.spin_stride_z.setRange(1, 20)
        self.spin_stride_z.setValue(4)
        self.spin_stride_z.setSingleStep(1)
        ply_form.addRow("Stride Z:", self.spin_stride_z)

        btn_export_ply = QPushButton("💾  Exportar nube PLY")
        btn_export_ply.clicked.connect(self._export_ply)
        ply_form.addRow(btn_export_ply)

        root.addWidget(ply_box)

        # ── Vista previa del corte ───────────────────────────────────────────
        preview_box = QGroupBox("Vista previa de un corte")
        preview_layout = QVBoxLayout(preview_box)

        depth_preview_row = QHBoxLayout()
        depth_preview_row.addWidget(QLabel("Profundidad de vista previa (m):"))
        self.spin_preview_depth = QDoubleSpinBox()
        self.spin_preview_depth.setRange(0.0, 50.0)
        self.spin_preview_depth.setValue(1.0)
        self.spin_preview_depth.setSingleStep(0.1)
        self.spin_preview_depth.setSuffix(" m")
        depth_preview_row.addWidget(self.spin_preview_depth)
        btn_preview = QPushButton("👁  Ver corte")
        btn_preview.clicked.connect(self._show_slice_preview)
        depth_preview_row.addWidget(btn_preview)
        preview_layout.addLayout(depth_preview_row)

        self.fig_preview = Figure(figsize=(6, 5), tight_layout=True)
        self.ax_preview = self.fig_preview.add_subplot(111)
        self.canvas_preview = FigureCanvasQTAgg(self.fig_preview)
        self.canvas_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self.canvas_preview)
        root.addWidget(preview_box)

    # ── Callbacks de velocidad / material ───────────────────────────────────
    def _on_material_changed(self, text: str):
        """Cuando el usuario cambia el material, actualiza la velocidad."""
        v = self._material_velocities.get(text)
        if v is None:
            # "Personalizado": no tocamos el spin
            return
        self.spin_velocity.setValue(v)

    def _update_vel_label(self, v_m_ns: float):
        """Actualiza la etiqueta con la velocidad en cm/ns."""
        v_cm_ns = v_m_ns * 100.0
        self.lbl_vel_cmns.setText(f"{v_cm_ns:.1f} cm/ns")

    # ── Lógica ───────────────────────────────────────────────────────────────
    def _build_volume(self):
        if self._get_loaded_fn is None:
            QMessageBox.warning(self, "Sin datos", "No hay datos cargados.")
            return

        loaded = self._get_loaded_fn()
        if not loaded:
            QMessageBox.warning(self, "Sin datos", "Carga y procesa perfiles primero.")
            return

        # max_points: None si no queremos decimar
        if self.chk_fast_mode.isChecked():
            max_points = int(self.spin_max_points_m.value() * 1_000_000)
        else:
            max_points = None

        builder = Grid3DBuilder(
            cell_size_xy=self.spin_cell_xy.value(),
            cell_size_z=self.spin_cell_z.value(),
            max_points=max_points,
        )
        velocity = self.spin_velocity.value()
        n_added = 0

        for path_str, gpr in loaded.items():
            gpr.geometry["filename"] = Path(path_str).name
            prof = Grid3DBuilder.profile_from_gprdata(
                gpr, velocity_m_ns=velocity
            )
            if prof is not None:
                builder.add_profile(prof)
                n_added += 1

        if n_added == 0:
            QMessageBox.warning(
                self, "Sin perfiles válidos",
                "Ningún perfil tiene topografía y geometría completas.\n"
                "Edita primero la topografía en la pestaña 🏔 Topografía."
            )
            return

        self.lbl_status.setText(f"Construyendo volumen con {n_added} perfiles…")
        try:
            self._volume = builder.build()
            nx, ny, nz = self._volume.shape
            self.lbl_status.setText(
                f"✅ Volumen: {nx}×{ny}×{nz} celdas — "
                f"XY={self.spin_cell_xy.value():.3f} m  Z={self.spin_cell_z.value():.3f} m"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._volume = None
            self.lbl_status.setText(f"❌ Error: {e}")

    def _show_slice_preview(self):
        if self._volume is None:
            QMessageBox.warning(self, "Sin volumen", "Construye primero el volumen.")
            return

        depth = self.spin_preview_depth.value()
        z_mean = float(np.nanmean(self._volume.zi))
        slice_data = self._volume.get_depth_slice(depth, z_mean)

        self.fig_preview.clear()
        ax = self.fig_preview.add_subplot(111)
        if np.all(~np.isfinite(slice_data)):
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        else:
            vmin = np.nanpercentile(slice_data, 1)
            vmax = np.nanpercentile(slice_data, 99)
            im = ax.imshow(
                slice_data.T,
                origin="lower",
                aspect="equal",
                cmap="gray",
                vmin=vmin, vmax=vmax,
                extent=[
                    self._volume.xi[0], self._volume.xi[-1],
                    self._volume.yi[0], self._volume.yi[-1],
                ],
            )
            self.fig_preview.colorbar(im, ax=ax, label="Amplitud", fraction=0.03)
            ax.set_title(f"Corte horizontal a {depth:.2f} m de profundidad")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
        self.canvas_preview.draw()

    def _export_geotiffs(self):
        if self._volume is None:
            QMessageBox.warning(self, "Sin volumen", "Construye primero el volumen.")
            return

        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
        except ImportError:
            QMessageBox.critical(
                self, "Falta rasterio",
                "Instala rasterio:\n  pip install rasterio"
            )
            return

        # Parsear profundidades
        try:
            depths = [float(d.strip()) for d in self.edit_depths.text().split(",")]
        except ValueError:
            QMessageBox.warning(self, "Error", "Formato de profundidades incorrecto.")
            return

        if not depths:
            QMessageBox.warning(self, "Error", "No se especificaron profundidades.")
            return

        # Elegir carpeta de salida
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de exportación")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        crs = CRS.from_epsg(self.spin_epsg.value())

        z_mean = float(np.nanmean(self._volume.zi))
        exported = []

        for depth in depths:
            slice_data = self._volume.get_depth_slice(depth, z_mean)
            # slice_data: (nx, ny) = (X, Y)
            # Rasterio espera (rows, cols) = (Y, X) → transponer
            slice_xy = slice_data.T  # (ny, nx)

            height, width = slice_xy.shape  # rows, cols

            transform = from_bounds(
                self._volume.xi[0], self._volume.yi[0],
                self._volume.xi[-1], self._volume.yi[-1],
                width,   # número de columnas (X)
                height,  # número de filas (Y)
            )

            # 1) Export científico (float32, amplitud real)
            fname_float = out_dir / f"slice_{depth:.2f}m.tif"
            with rasterio.open(
                fname_float, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(slice_xy.astype(np.float32), 1)
            exported.append(fname_float.name)

            # 2) Export de vista (uint8, 0–255 con stretch 2–98 %)
            finite = np.isfinite(slice_xy)
            if np.any(finite):
                vmin = np.nanpercentile(slice_xy, 2)
                vmax = np.nanpercentile(slice_xy, 98)
                if vmax > vmin:
                    scaled = (slice_xy - vmin) / (vmax - vmin)
                else:
                    scaled = np.zeros_like(slice_xy)
                scaled = np.clip(scaled, 0, 1)
                scaled[~finite] = 0
                scaled = (scaled * 255).astype(np.uint8)

                fname_view = out_dir / f"slice_{depth:.2f}m_view.tif"
                with rasterio.open(
                    fname_view, "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype="uint8",
                    crs=crs,
                    transform=transform,
                ) as dst:
                    dst.write(scaled, 1)
                exported.append(fname_view.name)

        QMessageBox.information(
            self, "Exportación completada",
            "Exportados GeoTIFFs en:\n"
            f"{out_dir}\n\n" +
            "\n".join(exported)
        )

    def _export_ply(self):
        if self._volume is None:
            QMessageBox.warning(self, "Sin volumen", "Construye primero el volumen.")
            return

        # Elegir archivo de salida
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar nube de puntos PLY",
            "volume_pointcloud.ply",
            "PLY (*.ply)",
        )
        if not fname:
            return
        out_path = Path(fname)

        stride_xy = self.spin_stride_xy.value()
        stride_z  = self.spin_stride_z.value()

        try:
            n_points = volume_to_ply(
                volume=self._volume,
                out_path=out_path,
                stride_xy=stride_xy,
                stride_z=stride_z,
                normalize_intensity=True,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error al exportar PLY",
                f"Ocurrió un error exportando la nube PLY:\n{e}"
            )
            return

        QMessageBox.information(
            self,
            "PLY exportado",
            f"Nube de puntos guardada en:\n{out_path}\n\n"
            f"Puntos exportados: {n_points}"
        )