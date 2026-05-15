# togpri/processing/filters.py
"""
Filtros de procesamiento GPR — todos los parámetros son kwargs explícitos
para que la GUI pueda exponerlos con sliders/spinboxes.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import detrend, butter, filtfilt
from dataclasses import dataclass, field
from typing import Any


# ── Descriptor de parámetro (para construir la GUI dinámicamente) ──────────
@dataclass
class ParamSpec:
    name: str
    label: str
    type: type          # float | int | bool
    default: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    tooltip: str = ""


@dataclass
class FilterSpec:
    name: str           # identificador interno
    label: str          # texto en la GUI
    func: object        # callable
    params: list[ParamSpec] = field(default_factory=list)


# ── Filtros ────────────────────────────────────────────────────────────────
class GPRFilters:

    # --- Time zero -----------------------------------------------------------
    @staticmethod
    def time_zero_cut(data: np.ndarray, cut_samples: int = 50) -> np.ndarray:
        """Elimina las primeras N muestras."""
        if cut_samples <= 0:
            return data
        if cut_samples >= data.shape[0]:
            raise ValueError("cut_samples >= número de muestras")
        return data[cut_samples:, :]

    @staticmethod
    def detect_time_zero(data: np.ndarray, max_search_samples: int | None = None) -> int:
        ns = data.shape[0]
        search = min(max_search_samples or ns // 4, ns)
        energy = np.mean(np.abs(data[:search, :]), axis=1)
        return int(np.argmax(energy))

    @staticmethod
    def auto_time_zero_cut(data: np.ndarray, margin: int = 5,
                           max_search_samples: int | None = None) -> np.ndarray:
        """Detecta la onda directa y corta antes de ella (con margen)."""
        ns = data.shape[0]
        search = min(max_search_samples or ns // 4, ns)
        energy = np.mean(np.abs(data[:search, :]), axis=1)
        zero = int(np.argmax(energy))
        cut = max(0, zero - margin)
        return data[cut:, :]

    # --- Background removal --------------------------------------------------
    @staticmethod
    def background_removal(data: np.ndarray) -> np.ndarray:
        """Resta la traza media a todas las trazas (suprime el clutter horizontal)."""
        return data - np.mean(data, axis=1, keepdims=True)

    # --- Ganancias ----------------------------------------------------------
    @staticmethod
    def exponential_gain(data: np.ndarray, alpha: float = 0.03) -> np.ndarray:
        """Ganancia exponencial para compensar la atenuación."""
        t = np.arange(data.shape[0])
        return data * np.exp(alpha * t)[:, None]

    @staticmethod
    def agc(data: np.ndarray, window: int = 50) -> np.ndarray:
        """
        Automatic Gain Control: normaliza la amplitud por ventana deslizante.
        Evita la saturación de eventos profundos.
        """
        nsamples, ntraces = data.shape
        out = np.zeros_like(data)
        half = window // 2
        for i in range(nsamples):
            i0 = max(0, i - half)
            i1 = min(nsamples, i + half)
            rms = np.sqrt(np.mean(data[i0:i1, :] ** 2, axis=0))
            rms[rms < 1e-10] = 1e-10
            out[i, :] = data[i, :] / rms
        return out

    # --- Dewow --------------------------------------------------------------
    @staticmethod
    def dewow(data: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Elimina la componente de muy baja frecuencia (wow) por traza.
        Usa media móvil suavizada y la resta.
        """
        from scipy.ndimage import uniform_filter1d
        low = uniform_filter1d(data, size=window, axis=0, mode="nearest")
        return data - low

    # --- Filtros de frecuencia ----------------------------------------------
    @staticmethod
    def bandpass(data: np.ndarray, low_mhz: float, high_mhz: float,
                 fs_mhz: float = 1000.0, order: int = 4) -> np.ndarray:
        """
        Filtro paso banda Butterworth.

        Parameters
        ----------
        low_mhz, high_mhz : frecuencias de corte en MHz
        fs_mhz            : frecuencia de muestreo (MHz) = samples / timewindow_ns
        order             : orden del filtro
        """
        nyq = fs_mhz / 2.0
        low = np.clip(low_mhz / nyq, 1e-4, 0.9999)
        high = np.clip(high_mhz / nyq, 1e-4, 0.9999)
        if low >= high:
            raise ValueError("low_mhz debe ser < high_mhz")
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, data, axis=0)

    @staticmethod
    def lowpass(data: np.ndarray, cutoff_mhz: float,
                fs_mhz: float = 1000.0, order: int = 4) -> np.ndarray:
        nyq = fs_mhz / 2.0
        cut = np.clip(cutoff_mhz / nyq, 1e-4, 0.9999)
        b, a = butter(order, cut, btype="low")
        return filtfilt(b, a, data, axis=0)

    # --- Stacking -----------------------------------------------------------
    @staticmethod
    def trace_stacking(data: np.ndarray, n: int = 3) -> np.ndarray:
        """
        Apilado de N trazas contiguas (mejora SNR en datos ruidosos).
        El número de trazas resultante = floor(ntraces / n).
        """
        nsamples, ntraces = data.shape
        n_out = ntraces // n
        stacked = np.zeros((nsamples, n_out), dtype=data.dtype)
        for i in range(n_out):
            stacked[:, i] = np.mean(data[:, i * n:(i + 1) * n], axis=1)
        return stacked

    # --- Migración F-K (Kirchhoff simple) -----------------------------------
    @staticmethod
    def fk_filter(data: np.ndarray, mask_fn=None) -> np.ndarray:
        """
        Filtro en el dominio F-K (frecuencia-número de onda).
        mask_fn: callable(F, K) → bool array. Si None, aplica filtro identidad.
        """
        spectrum = np.fft.fft2(data)
        if mask_fn is not None:
            nf, nk = spectrum.shape
            F = np.fft.fftfreq(nf)
            K = np.fft.fftfreq(nk)
            FF, KK = np.meshgrid(F, K, indexing="ij")
            mask = mask_fn(FF, KK).astype(float)
            spectrum *= mask
        return np.real(np.fft.ifft2(spectrum))


# ── Registro de filtros con metadatos para la GUI ─────────────────────────
FILTER_REGISTRY: list[FilterSpec] = [
    FilterSpec(
        name="auto_time_zero_cut",
        label="Auto Time-Zero Cut",
        func=GPRFilters.auto_time_zero_cut,
        params=[
            ParamSpec("margin", "Margen (samples)", int, 5, 0, 50, 1,
                      "Samples que se conservan antes del máximo de energía"),
        ],
    ),
    FilterSpec(
        name="background_removal",
        label="Background Removal",
        func=GPRFilters.background_removal,
        params=[],
    ),
    FilterSpec(
        name="exponential_gain",
        label="Exponential Gain",
        func=GPRFilters.exponential_gain,
        params=[
            ParamSpec("alpha", "Alpha", float, 0.003, 0.0001, 0.5, 0.001,
                      "Tasa de crecimiento exponencial"),
        ],
    ),
    FilterSpec(
        name="agc",
        label="AGC (Auto Gain Control)",
        func=GPRFilters.agc,
        params=[
            ParamSpec("window", "Ventana (samples)", int, 50, 5, 200, 5,
                      "Ventana deslizante para normalización de amplitud"),
        ],
    ),
    FilterSpec(
        name="dewow",
        label="Dewow",
        func=GPRFilters.dewow,
        params=[
            ParamSpec("window", "Ventana (samples)", int, 10, 3, 100, 1,
                      "Ventana de la media móvil para eliminar el wow"),
        ],
    ),
    FilterSpec(
        name="bandpass",
        label="Bandpass Filter",
        func=GPRFilters.bandpass,
        params=[
            ParamSpec("low_mhz", "Frecuencia baja (MHz)", float, 50.0, 1.0, 2000.0, 5.0),
            ParamSpec("high_mhz", "Frecuencia alta (MHz)", float, 400.0, 1.0, 2000.0, 5.0),
            ParamSpec("order", "Orden", int, 4, 1, 8, 1, "Orden del filtro Butterworth"),
        ],
    ),
    FilterSpec(
        name="trace_stacking",
        label="Trace Stacking",
        func=GPRFilters.trace_stacking,
        params=[
            ParamSpec("n", "N trazas a apilar", int, 3, 2, 20, 1,
                      "Promedia N trazas contiguas"),
        ],
    ),
]