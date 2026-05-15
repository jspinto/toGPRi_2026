# togpri/gui/widgets/filter_panel.py
"""
Panel de filtros auto-generado desde FILTER_REGISTRY.
Emite la señal `filters_changed` cuando el usuario modifica algo.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QScrollArea, QGroupBox, QDoubleSpinBox,
    QSpinBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from togpri.processing.filters import FILTER_REGISTRY, FilterSpec, ParamSpec


class FilterPanel(QWidget):
    """
    Panel lateral con los filtros activables y sus parámetros.
    Señales:
        filters_changed()  — el usuario cambió algo (activa preview automático si está habilitado)
        preview_requested()    — botón "Preview" pulsado
        apply_all_requested()  — botón "Aplicar a todos" pulsado
    """
    filters_changed   = pyqtSignal()
    preview_requested = pyqtSignal()
    apply_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(280)

        self._checks: dict[str, QCheckBox] = {}
        self._param_widgets: dict[str, dict[str, QDoubleSpinBox | QSpinBox]] = {}

        self._build_ui()

    # ── Build ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        root.addWidget(QLabel("<b>Filtros de procesamiento</b>"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(6)

        for spec in FILTER_REGISTRY:
            group = self._make_filter_group(spec)
            vbox.addWidget(group)

        vbox.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)

        # Botones de acción
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        btn_preview = QPushButton("▶  Preview")
        btn_preview.setToolTip("Aplica los filtros al archivo seleccionado y actualiza la vista")
        btn_preview.clicked.connect(self.preview_requested.emit)
        root.addWidget(btn_preview)

        btn_all = QPushButton("⚡  Aplicar a todos los seleccionados")
        btn_all.clicked.connect(self.apply_all_requested.emit)
        root.addWidget(btn_all)

    def _make_filter_group(self, spec: FilterSpec) -> QGroupBox:
        box = QGroupBox()
        box.setCheckable(False)
        vbox = QVBoxLayout(box)
        vbox.setSpacing(2)

        # Checkbox principal
        chk = QCheckBox(spec.label)
        chk.setChecked(True)
        chk.stateChanged.connect(lambda _: self.filters_changed.emit())
        self._checks[spec.name] = chk
        vbox.addWidget(chk)

        # Parámetros
        param_widgets: dict[str, QDoubleSpinBox | QSpinBox] = {}
        for p in spec.params:
            row = QHBoxLayout()
            lbl = QLabel(f"  {p.label}:")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl.setMinimumWidth(130)
            row.addWidget(lbl)

            if p.type == int:
                spin = QSpinBox()
                spin.setRange(int(p.min_val or 0), int(p.max_val or 9999))
                spin.setSingleStep(int(p.step or 1))
                spin.setValue(int(p.default))
            else:
                spin = QDoubleSpinBox()
                spin.setRange(float(p.min_val or 0), float(p.max_val or 9999))
                spin.setSingleStep(float(p.step or 0.01))
                spin.setValue(float(p.default))
                spin.setDecimals(4)

            if p.tooltip:
                spin.setToolTip(p.tooltip)

            spin.valueChanged.connect(lambda _: self.filters_changed.emit())
            row.addWidget(spin)
            param_widgets[p.name] = spin
            vbox.addLayout(row)

        self._param_widgets[spec.name] = param_widgets
        return box

    # ── API pública ──────────────────────────────────────────────────────────
    def get_filter_pipeline(self) -> list[dict]:
        """
        Devuelve la lista de filtros activos con sus parámetros actuales.
        Formato: [{"spec": FilterSpec, "kwargs": {param: value, ...}}, ...]
        """
        pipeline = []
        for spec in FILTER_REGISTRY:
            if not self._checks[spec.name].isChecked():
                continue
            kwargs = {
                p_name: widget.value()
                for p_name, widget in self._param_widgets[spec.name].items()
            }
            pipeline.append({"spec": spec, "kwargs": kwargs})
        return pipeline

    def set_filter_active(self, name: str, active: bool):
        if name in self._checks:
            self._checks[name].setChecked(active)