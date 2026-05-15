# togpri/gui/widgets/local_cube_panel.py
"""
Pestaña '🗺 Cortes locales' — modo toGPRi clásico.
Construye un LocalCube y exporta cortes horizontales (TIFF con/sin georef).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QDoubleSpinBox, QLabel, QFileDialog,
    QGroupBox, QSizePolicy, QMessageBox, QLineEdit,
    QComboBox, QCheckBox, QSpinBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from togpri.processing.local_cube import LocalCubeBuilder, LocalCube
from togpri.processing.local_tiff_export import export_local_slice_tiff


class LocalCubePanel(QWidget):
    """Pestaña de cortes locales tipo toGPRi."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cube: LocalCube | None = None
        self._get_loaded_fn = None
        self._build_ui()

    def set_data_source(self, fn):
        self._get_loaded_fn = fn

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Parámetros del cubo ──────────────────────────────────────────────
        cube_box = QGroupBox("Parámetros del cubo local")
        form = QFormLayout(cube_box)

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.01, 0.3)
        self.spin_velocity.setValue(0.1)
        self.spin_velocity.setSingleStep(0.005)
        self.spin_velocity.setDecimals(3)
        self.spin_velocity.setSuffix(" m/ns")
        form.addRow("Velocidad EM:", self.spin_velocity)

        self.edit_spacings = QLineEdit("0.5")
        self.edit_spacings.setToolTip(
            "Separación entre perfiles en metros.\n"
            "Un único valor = uniforme.\n"
            "Lista separada por comas (n_perfiles-1 valores) para separación variable."
        )
        form.addRow("Separación entre perfiles (m):", self.edit_spacings)

        self.combo_fliplr = QComboBox()
        self.combo_fliplr.addItems(["none", "odd", "even", "all"])
        form.addRow("Fliplr:", self.combo_fliplr)

        root.addWidget(cube_box)

        btn_build = QPushButton("🗂  Construir cubo local")
        btn_build.clicked.connect(self._build_cube)
        root.addWidget(btn_build)

        self.lbl_status = QLabel("Sin cubo.")
        root.addWidget(self.lbl_status)

        # ── Exportar cortes ──────────────────────────────────────────────────
        exp_box = QGroupBox("Exportar cortes horizontales")
        exp_form = QFormLayout(exp_box)

        self.edit_depths = QLineEdit("0.25, 0.5, 0.75, 1.0, 1.25, 1.5")
        exp_form.addRow("Profundidades (m), separadas por comas:", self.edit_depths)

        self.spin_band = QDoubleSpinBox()
        self.spin_band.setRange(0.0, 2.0)
        self.spin_band.setValue(0.1)
        self.spin_band.setSingleStep(0.05)
        self.spin_band.setSuffix(" m")
        self.spin_band.setToolTip(
            "Semiancho de la banda a promediar (0 = corte exacto).\n"
            "0.1 m → promedia entre depth-0.1 y depth+0.1 m."
        )
        exp_form.addRow("Semiancho de banda:", self.spin_band)

        self.combo_band_mode = QComboBox()
        self.combo_band_mode.addItems(["max", "mean", "min"])
        exp_form.addRow("Modo de banda:", self.combo_band_mode)

        # Resolución de exportación (píxeles cuadrados)
        self.spin_resolution = QDoubleSpinBox()
        self.spin_resolution.setRange(0.0, 10.0)
        self.spin_resolution.setValue(0.05)  # 5 cm por píxel
        self.spin_resolution.setSingleStep(0.01)
        self.spin_resolution.setDecimals(3)
        self.spin_resolution.setSuffix(" m/píx")
        self.spin_resolution.setToolTip(
            "Resolución de salida en metros por píxel.\n"
            "0 = sin remuestreo (usa la resolución nativa del cubo)."
        )
        exp_form.addRow("Resolución salida:", self.spin_resolution)

        # Canal alpha opcional (como en toGPRi v1)
        self.chk_alpha = QCheckBox("Usar canal alpha")
        exp_form.addRow(self.chk_alpha)

        alpha_row = QHBoxLayout()
        self.spin_alpha_min = QSpinBox()
        self.spin_alpha_min.setRange(0, 255)
        self.spin_alpha_min.setValue(0)
        alpha_row.addWidget(QLabel("Alpha min:"))
        alpha_row.addWidget(self.spin_alpha_min)

        self.spin_alpha_max = QSpinBox()
        self.spin_alpha_max.setRange(0, 255)
        self.spin_alpha_max.setValue(30)
        alpha_row.addWidget(QLabel("  Alpha max:"))
        alpha_row.addWidget(self.spin_alpha_max)

        exp_form.addRow(alpha_row)

        # Georref opcional
        self.chk_georef = QCheckBox("Georreferenciar TIFFs")
        exp_form.addRow(self.chk_georef)

        epsg_row = QHBoxLayout()
        self.spin_epsg = QSpinBox()
        self.spin_epsg.setRange(1024, 99999)
        self.spin_epsg.setValue(25830)
        epsg_row.addWidget(QLabel("EPSG:"))
        epsg_row.addWidget(self.spin_epsg)

        self.edit_origin = QLineEdit("0.0, 0.0")
        self.edit_origin.setToolTip("X,Y de la esquina superior izquierda en metros.")
        epsg_row.addWidget(QLabel("  Origen X,Y:"))
        epsg_row.addWidget(self.edit_origin)

        self.spin_rotation = QDoubleSpinBox()
        self.spin_rotation.setRange(-360.0, 360.0)
        self.spin_rotation.setValue(0.0)
        self.spin_rotation.setSuffix("°")
        self.spin_rotation.setToolTip("Rotación horaria en grados (como en toGPRi v1).")
        epsg_row.addWidget(QLabel("  Rot.:"))
        epsg_row.addWidget(self.spin_rotation)
        exp_form.addRow(epsg_row)

        btn_export = QPushButton("💾  Exportar TIFFs locales")
        btn_export.clicked.connect(self._export_tiffs)
        exp_form.addRow(btn_export)

        btn_export_ply = QPushButton("🧩  Exportar nube de puntos PLY")
        btn_export_ply.clicked.connect(self._export_ply)
        exp_form.addRow(btn_export_ply)

        root.addWidget(exp_box)

        # ── Vista previa ─────────────────────────────────────────────────────
        preview_box = QGroupBox("Vista previa")
        pv_layout = QVBoxLayout(preview_box)
        pv_row = QHBoxLayout()

        pv_row.addWidget(QLabel("Profundidad:"))
        self.spin_preview_depth = QDoubleSpinBox()
        self.spin_preview_depth.setRange(0.0, 50.0)
        self.spin_preview_depth.setValue(0.5)
        self.spin_preview_depth.setSingleStep(0.05)
        self.spin_preview_depth.setSuffix(" m")
        pv_row.addWidget(self.spin_preview_depth)

        btn_preview = QPushButton("👁  Ver corte")
        btn_preview.clicked.connect(self._show_preview)
        pv_row.addWidget(btn_preview)
        pv_layout.addLayout(pv_row)

        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        pv_layout.addWidget(self.canvas)
        root.addWidget(preview_box)

    # ── Lógica ───────────────────────────────────────────────────────────────

    def _parse_spacings(self):
        txt = self.edit_spacings.text().strip()
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        if len(parts) == 1:
            return float(parts[0])
        return [float(p) for p in parts]

    def _build_cube(self):
        if self._get_loaded_fn is None:
            QMessageBox.warning(self, "Sin datos", "No hay datos cargados.")
            return
        loaded = self._get_loaded_fn()
        if not loaded:
            QMessageBox.warning(self, "Sin datos", "Carga y procesa perfiles primero.")
            return

        try:
            spacings = self._parse_spacings()
        except ValueError:
            QMessageBox.warning(self, "Error", "Formato de separaciones incorrecto.")
            return

        builder = LocalCubeBuilder(
            velocity_m_ns=self.spin_velocity.value(),
            profile_spacings=spacings,
            fliplr=self.combo_fliplr.currentText(),
        )
        for gpr in loaded.values():
            builder.add_profile(gpr)

        self.lbl_status.setText("Construyendo cubo local…")
        try:
            self._cube = builder.build()
            ns, nt, np_ = self._cube.shape
            self.lbl_status.setText(
                f"✅ Cubo: {ns} muestras × {nt} trazas × {np_} perfiles  —  "
                f"along={self._cube.along_m[-1]:.2f} m  "
                f"across={self._cube.across_m[-1]:.2f} m  "
                f"depth={self._cube.depth_m[-1]:.2f} m"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._cube = None
            self.lbl_status.setText(f"❌ Error: {e}")

    def _show_preview(self):
        if self._cube is None:
            QMessageBox.warning(self, "Sin cubo", "Construye primero el cubo local.")
            return

        depth = self.spin_preview_depth.value()
        band = self.spin_band.value()
        mode = self.combo_band_mode.currentText()

        if band > 0:
            try:
                slice_data = self._cube.get_depth_band(
                    depth - band, depth + band, mode=mode
                )
            except ValueError:
                slice_data = self._cube.get_depth_slice(depth)
        else:
            slice_data = self._cube.get_depth_slice(depth)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        vmin = np.nanpercentile(slice_data, 2)
        vmax = np.nanpercentile(slice_data, 98)
        im = ax.imshow(
            slice_data.T,
            origin="upper",
            aspect="auto",
            cmap="gray",
            vmin=vmin, vmax=vmax,
            extent=[
                self._cube.along_m[0], self._cube.along_m[-1],
                self._cube.across_m[-1], self._cube.across_m[0],
            ],
        )
        self.fig.colorbar(im, ax=ax, label="Amplitud", fraction=0.03)
        band_str = f" ±{band:.2f} m ({mode})" if band > 0 else ""
        ax.set_title(f"Corte local a {depth:.2f} m{band_str}")
        ax.set_xlabel("Distancia (m)")
        ax.set_ylabel("Perfil (m)")
        self.canvas.draw()

    def _export_tiffs(self):
        if self._cube is None:
            QMessageBox.warning(self, "Sin cubo", "Construye primero el cubo local.")
            return

        try:
            depths = [float(d.strip()) for d in self.edit_depths.text().split(",")]
        except ValueError:
            QMessageBox.warning(self, "Error", "Formato de profundidades incorrecto.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de exportación")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        band = self.spin_band.value()
        mode = self.combo_band_mode.currentText()
        target_res = self.spin_resolution.value()  # 0.0 → sin remuestreo

        georef = self.chk_georef.isChecked()
        epsg = self.spin_epsg.value() if georef else None
        rotation = self.spin_rotation.value() if georef else 0.0
        origin_xy = None
        if georef:
            try:
                ox, oy = [float(v.strip()) for v in self.edit_origin.text().split(",")]
                origin_xy = (ox, oy)
            except ValueError:
                QMessageBox.warning(self, "Error", "Formato de origen X,Y incorrecto.")
                return

        # Alpha opcional
        use_alpha = self.chk_alpha.isChecked()
        alpha_range = None
        if use_alpha:
            a_min = self.spin_alpha_min.value()
            a_max = self.spin_alpha_max.value()
            if a_max < a_min:
                a_min, a_max = a_max, a_min
            alpha_range = (a_min, a_max)

        # Tamaño de celda nativo en metros
        n_along = len(self._cube.along_m)
        n_across = len(self._cube.across_m)
        cell_along = (
            (self._cube.along_m[-1] - self._cube.along_m[0]) / max(n_along - 1, 1)
        )
        cell_across = (
            (self._cube.across_m[-1] - self._cube.across_m[0]) / max(n_across - 1, 1)
            if n_across > 1 else 1.0
        )

        exported = []

        # DEBUG rápido de geometría del cubo
        print("along_m[0], along_m[-1], len =", 
              self._cube.along_m[0], self._cube.along_m[-1], len(self._cube.along_m))
        print("across_m[0], across_m[-1], len =", 
              self._cube.across_m[0], self._cube.across_m[-1], len(self._cube.across_m))
        print("target_res =", target_res)

        for depth in depths:
            if band > 0:
                try:
                    slice_data = self._cube.get_depth_band(
                        depth - band, depth + band, mode=mode
                    )
                except ValueError:
                    slice_data = self._cube.get_depth_slice(depth)
            else:
                slice_data = self._cube.get_depth_slice(depth)

            # Remuestreo opcional a resolución uniforme (píxeles cuadrados)
            if target_res > 0.0:
                slice_data_resampled = self._resample_slice_to_resolution(
                    slice_data, target_res=target_res
                )
                out_cell_along = target_res
                out_cell_across = target_res
            else:
                slice_data_resampled = slice_data
                out_cell_along = cell_along
                out_cell_across = cell_across

            # Igual que en la preview: filas=across, columnas=along
            slice_for_tiff = slice_data_resampled.T

            fname = out_dir / f"local_slice_{depth:.2f}m.tif"
            export_local_slice_tiff(
                slice_data=slice_for_tiff,
                out_path=fname,
                epsg=epsg,
                origin_xy=origin_xy,
                rotation_deg=rotation,
                cell_size_along=out_cell_along,
                cell_size_across=out_cell_across,
                alpha_range=alpha_range,
            )
            exported.append(fname.name)

        QMessageBox.information(
            self, "Exportación completada",
            f"Exportados en:\n{out_dir}\n\n" + "\n".join(exported)
        )

    def _resample_slice_to_resolution(self, slice_data: np.ndarray, target_res: float) -> np.ndarray:
        """
        Remuestrea slice_data (n_along, n_across) a una rejilla con píxeles cuadrados
        de tamaño target_res (m), manteniendo la geometría física del cubo.

        Usa un zoom bilineal sobre los ejes along y across.
        """
        from scipy.ndimage import zoom  # necesitas scipy instalado

        cube = self._cube
        if cube is None or target_res <= 0.0:
            return slice_data

        n_along = len(cube.along_m)
        n_across = len(cube.across_m)

        if n_along < 2 or n_across < 2:
            return slice_data

        # Resolución nativa del cubo en cada eje (m/píxel)
        native_along = (cube.along_m[-1] - cube.along_m[0]) / (n_along - 1)
        native_across = (cube.across_m[-1] - cube.across_m[0]) / (n_across - 1)

        # Factores de zoom: cuánto tengo que escalar para pasar de la resolución
        # nativa a la resolución objetivo (píxeles cuadrados target_res)
        zoom_along = native_along / target_res
        zoom_across = native_across / target_res

        # Limitamos un poco los factores para evitar zooms absurdos
        zoom_along = max(0.1, min(zoom_along, 10.0))
        zoom_across = max(0.1, min(zoom_across, 10.0))

        # slice_data está como (n_along, n_across)
        out = zoom(slice_data, (zoom_along, zoom_across), order=1)
        return out
  

    def _export_ply(self):
        if self._cube is None:
            QMessageBox.warning(self, "Sin cubo", "Construye primero el cubo local.")
            return

        # Umbral: solo puntos con amplitud por encima del percentil X
        threshold, ok = __import__("PyQt6.QtWidgets", fromlist=["QInputDialog"]).QInputDialog.getDouble(
            self, "Umbral de amplitud",
            "Percentil mínimo de amplitud (0 = todos los puntos):",
            20.0, 0.0, 99.0, 1
        )
        if not ok:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar PLY", "local_cube.ply", "PLY (*.ply)"
        )
        if not out_path:
            return

        cube = self._cube
        along = cube.along_m      # eje X  (distancia a lo largo del perfil)
        across = cube.across_m    # eje Y  (distancia entre perfiles)
        depth = cube.depth_m      # eje Z  (profundidad, positivo hacia abajo)

        # Construir rejilla de coordenadas
        # cube.data tiene forma (n_samples, n_traces, n_profiles)
        data = cube.data
        amp_abs = np.abs(data)
        cutoff = np.nanpercentile(amp_abs, threshold)

        points = []
        for iz, z in enumerate(depth):
            for it, x in enumerate(along):
                for ip, y in enumerate(across):
                    val = amp_abs[iz, it, ip]
                    if np.isnan(val) or val < cutoff:
                        continue
                    # Normalizar amplitud a 0-255 para color gris
                    points.append((x, y, -z, val))  # Z negativo = hacia abajo

        if not points:
            QMessageBox.warning(self, "Sin puntos", "No hay puntos con ese umbral.")
            return

        pts = np.array(points)
        vmin, vmax = pts[:, 3].min(), pts[:, 3].max()
        colors = ((pts[:, 3] - vmin) / max(vmax - vmin, 1e-9) * 255).astype(np.uint8)

        out_path = Path(out_path)
        with open(out_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i, (x, y, z, _) in enumerate(pts):
                c = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {c} {c} {c}\n")

        QMessageBox.information(
            self, "PLY exportado",
            f"✅ {len(pts):,} puntos exportados en:\n{out_path}"
        )