# togpri/io/ramac.py
"""
Lector de archivos RAMAC (.rd3 / .rad) — Måla Geoscience
"""
from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from togpri.core.gprdata import GPRData


class RAMACReader:

    _KEY_MAP = {
        "SAMPLES": "samples",
        "Samples": "samples",
        "LAST TRACE": "traces",
        "Last trace": "traces",
        "FREQUENCY": "frequency",
        "Frequency": "frequency",
        "ANTENNA": "antenna",
        "Antenna": "antenna",
        "TIMEWINDOW": "timewindow",
        "Timewindow": "timewindow",
        "Time window": "timewindow",
        "DISTANCE INTERVAL": "trace_interval",
        "Distance interval": "trace_interval",
    }

    def __init__(self, filepath: str | Path):
        p = Path(filepath)
        if p.suffix.lower() == ".rad":
            self.rad_file = p
            self.rd3_file = p.with_suffix(".rd3")
        elif p.suffix.lower() == ".rd3":
            self.rd3_file = p
            self.rad_file = p.with_suffix(".rad")
        else:
            raise ValueError(f"Extensión no reconocida: {p.suffix}")

        if not self.rad_file.exists():
            raise FileNotFoundError(f"Header no encontrado: {self.rad_file}")
        if not self.rd3_file.exists():
            raise FileNotFoundError(f"Datos no encontrados: {self.rd3_file}")

    # ── Header ─────────────────────────────────────────────────────────────
    def read_header(self) -> dict:
        header: dict = {}
        with open(self.rad_file, encoding="latin-1", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value or "e" in value.lower():
                        header[key] = float(value)
                    else:
                        header[key] = int(value)
                except ValueError:
                    header[key] = value

        # Normalizar claves
        for orig, norm in self._KEY_MAP.items():
            if orig in header and norm not in header:
                header[norm] = header[orig]

        return header

    # ── Datos ──────────────────────────────────────────────────────────────
    def read_data(self, dtype=np.int16, flip_horizontal: bool = False) -> GPRData:
        header = self.read_header()

        samples = self._extract_int(header, ["samples", "SAMPLES", "Samples"])
        traces  = self._extract_traces(header)

        if samples is None:
            raise ValueError("No se puede determinar el número de samples desde el header.")
        if traces is None:
            raise ValueError("No se puede determinar el número de trazas desde el header.")

        raw = np.fromfile(self.rd3_file, dtype=dtype)
        expected = samples * traces
        if raw.size != expected:
            traces = raw.size // samples
            raw = raw[: traces * samples]

        data = raw.reshape(traces, samples).T.astype(np.float64)

        if flip_horizontal:
            data = np.fliplr(data)
            header["orientation_corrected"] = "fliplr"

        header["samples"] = samples
        header["traces"]  = traces

        gpr = GPRData(data, header, source_format="RAMAC")

        if "trace_interval" in header:
            gpr.geometry["trace_spacing_m"] = float(header["trace_interval"])

        return gpr

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_int(header: dict, keys: list[str]):
        for k in keys:
            if k in header:
                return int(header[k])
        return None

    @staticmethod
    def _extract_traces(header: dict):
        for k in ["traces", "LAST TRACE", "Last trace", "TRACES"]:
            if k in header:
                v = header[k]
                if isinstance(v, str):
                    m = re.search(r"\d+", v)
                    return int(m.group()) if m else None
                return int(v)
        return None

    @staticmethod
    def read(filepath, dtype=np.int16, flip_horizontal=False) -> GPRData:
        return RAMACReader(filepath).read_data(
            dtype=dtype, flip_horizontal=flip_horizontal
        )


# ── Función de acceso rápido ───────────────────────────────────────────────
def read_ramac(filepath, dtype=np.int16, flip_horizontal=False) -> GPRData:
    """Función conveniente para leer archivos RAMAC."""
    return RAMACReader.read(filepath, dtype=dtype, flip_horizontal=flip_horizontal)