# togpri/core/gprdata.py
"""
Contenedor central de datos GPR.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable


class GPRData:
    """
    Almacena datos GPR crudos y procesados junto con metadatos.

    Attributes
    ----------
    data : ndarray (samples × traces)  — datos originales (read-only semántico)
    header : dict                      — metadatos del archivo fuente
    source_format : str                — 'RAMAC', 'GSSI', etc.
    processing_history : list          — registro de filtros aplicados
    _processed : ndarray               — copia de trabajo (se modifica con apply_filter)
    geometry : dict                    — trace_spacing_m, origin_xy, direction_xy…
    """

    def __init__(self, data: np.ndarray, header: dict, source_format: str = "unknown"):
        self.data = data.astype(np.float64)
        self.header = header
        self.source_format = source_format
        self.processing_history: list[dict] = []
        self._processed: np.ndarray = self.data.copy()
        self.geometry: dict[str, Any] = {}

    # ── Propiedades ────────────────────────────────────────────────────────
    @property
    def processed_data(self) -> np.ndarray:
        return self._processed

    @property
    def nsamples(self) -> int:
        return self._processed.shape[0]

    @property
    def ntraces(self) -> int:
        return self._processed.shape[1]

    # ── Filtros ────────────────────────────────────────────────────────────
    def apply_filter(self, func: Callable, **kwargs) -> "GPRData":
        """
        Aplica un filtro in-place sobre processed_data.
        Registra nombre + kwargs en el historial.
        """
        self._processed = func(self._processed, **kwargs)
        self.processing_history.append({"filter": func.__name__, "kwargs": kwargs})
        return self

    def reset_processing(self) -> "GPRData":
        """Descarta todo el procesado y vuelve a los datos originales."""
        self._processed = self.data.copy()
        self.processing_history.clear()
        return self

    # ── Ejes ───────────────────────────────────────────────────────────────
    def get_time_axis(self) -> np.ndarray:
        """Devuelve eje de tiempo en ns."""
        tw = float(self.header.get("timewindow", 100.0))
        return np.linspace(0.0, tw, self.nsamples)

    def get_depth_axis(self, velocity_m_ns: float = 0.1) -> np.ndarray:
        """Devuelve eje de profundidad en metros."""
        return (velocity_m_ns * self.get_time_axis()) / 2.0

    def get_distance_axis(self) -> np.ndarray:
        spacing = float(self.geometry.get(
            "trace_spacing_m",
            self.header.get("trace_interval", 0.05)
        ))
        return np.arange(self.ntraces) * spacing

    # ── Información ────────────────────────────────────────────────────────
    def info(self):
        print(f"GPRData [{self.source_format}]")
        print(f"  Shape original : {self.data.shape}")
        print(f"  Shape procesado: {self._processed.shape}")
        print(f"  Timewindow     : {self.header.get('timewindow')} ns")
        print(f"  Frecuencia     : {self.header.get('frequency')} MHz")
        print(f"  Pasos de filtro: {len(self.processing_history)}")

    # ── Persistencia ───────────────────────────────────────────────────────
    def save(self, path: Path, fmt: str = "numpy"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "numpy":
            np.savez_compressed(
                path,
                original=self.data,
                processed=self._processed,
                geometry=json.dumps(self.geometry),
                header=json.dumps(self.header, default=str),
            )
        elif fmt == "json_metadata":
            meta = {
                "source_format": self.source_format,
                "shape": list(self.data.shape),
                "header": self.header,
                "geometry": self.geometry,
                "processing_history": self.processing_history,
            }
            path.write_text(json.dumps(meta, indent=2, default=str))
        else:
            raise ValueError(f"Formato desconocido: {fmt}")

    @classmethod
    def load(cls, path: Path) -> "GPRData":
        """Carga desde .npz guardado con save()."""
        path = Path(path)
        npz = np.load(path, allow_pickle=False)
        header = json.loads(str(npz["header"]))
        geometry = json.loads(str(npz["geometry"]))
        obj = cls(npz["original"], header)
        obj._processed = npz["processed"].copy()
        obj.geometry = geometry
        return obj