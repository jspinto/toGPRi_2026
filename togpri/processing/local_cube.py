# togpri/processing/local_cube.py
"""
Cubo 3D local tipo AY3D del toGPRi original.

Sistema de referencia LOCAL (no coordenadas reales):
  - eje 0 (depth):  profundidad en metros
  - eje 1 (along):  distancia a lo largo de cada perfil en metros
  - eje 2 (across): distancia acumulada entre perfiles en metros

No se usa griddata: cada perfil ocupa su "slice" across directamente,
interpolando en along si hace falta para igualar el número de trazas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np


@dataclass
class LocalCube:
    """
    Cubo 3D en sistema local alineado con los perfiles.

    data: np.ndarray (n_depth, n_along, n_across)
    depth_m: vector (n_depth,) — profundidades en metros
    along_m: vector (n_along,) — distancias a lo largo del perfil en metros
    across_m: vector (n_across,) — distancias acumuladas entre perfiles en metros
    """
    data: np.ndarray
    depth_m: np.ndarray
    along_m: np.ndarray
    across_m: np.ndarray

    @property
    def shape(self):
        return self.data.shape

    def rotated_plan(self) -> "LocalCube":
        """
        Devuelve un nuevo cubo con los ejes along/across intercambiados.

        Útil cuando la geometría física del terreno está girada respecto
        a cómo se ha construido internamente el cubo.

        Entrada:
          data.shape = (n_depth, n_along, n_across)

        Salida:
          data.shape = (n_depth, n_across, n_along)
          along_m <- across_m
          across_m <- along_m
        """
        return LocalCube(
            data=self.data.transpose(0, 2, 1).copy(),
            depth_m=self.depth_m.copy(),
            along_m=self.across_m.copy(),
            across_m=self.along_m.copy(),
        )

    def get_depth_slice(
        self,
        depth_m: float,
    ) -> np.ndarray:
        """
        Corte horizontal exacto a una profundidad.
        Devuelve array (n_along, n_across).
        """
        idx = int(np.argmin(np.abs(self.depth_m - depth_m)))
        return self.data[idx, :, :]

    def get_depth_band(
        self,
        z_min: float,
        z_max: float,
        mode: Literal["max", "mean", "min"] = "max",
    ) -> np.ndarray:
        """
        Corte horizontal que promedia/maximiza entre dos profundidades.
        Devuelve array (n_along, n_across).
        Equivalente al 'overlaid raster' del toGPRi original.
        """
        mask = (self.depth_m >= z_min) & (self.depth_m <= z_max)
        if not np.any(mask):
            raise ValueError(
                f"No hay muestras entre {z_min} m y {z_max} m."
            )
        band = self.data[mask, :, :]  # (ns_band, n_along, n_across)
        if mode == "max":
            return np.nanmax(np.abs(band), axis=0)
        elif mode == "min":
            return np.nanmin(np.abs(band), axis=0)
        else:  # mean
            return np.nanmean(np.abs(band), axis=0)


class LocalCubeBuilder:
    """
    Construye un LocalCube a partir de una lista de GPRData procesados.

    Parámetros:
    - velocity_m_ns: velocidad EM media en m/ns.
    - profile_spacings: lista de separaciones en metros entre perfiles
      consecutivos (len = n_profiles - 1). Si se pasa un único float,
      se usa como separación uniforme.
    - n_traces_target: número de trazas al que se remuestrea cada perfil
      (None = usar el máximo de todos los perfiles).
    - fliplr: 'none', 'odd', 'even' o 'all'.
    """

    FLIPLR_OPTIONS = ("none", "odd", "even", "all")

    def __init__(
        self,
        velocity_m_ns: float = 0.1,
        profile_spacings: float | List[float] = 0.5,
        n_traces_target: int | None = None,
        fliplr: Literal["none", "odd", "even", "all"] = "none",
    ) -> None:
        self.velocity_m_ns = float(velocity_m_ns)
        self._spacings_input = profile_spacings
        self.n_traces_target = n_traces_target
        self.fliplr = fliplr
        self._profiles: list = []   # lista de GPRData

    # ── API pública ──────────────────────────────────────────────────────────

    def add_profile(self, gpr) -> None:
        """Añade un GPRData ya procesado."""
        self._profiles.append(gpr)

    def clear(self) -> None:
        self._profiles.clear()

    @property
    def n_profiles(self) -> int:
        return len(self._profiles)

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> LocalCube:
        if not self._profiles:
            raise ValueError("No hay perfiles añadidos.")

        n_profiles = len(self._profiles)

        # Separaciones entre perfiles (n_profiles - 1 valores)
        spacings = self._resolve_spacings(n_profiles)

        # Profundidades (tomamos el primer perfil como referencia)
        depth_m = np.asarray(
            self._profiles[0].get_depth_axis(self.velocity_m_ns)
        )
        n_depth = len(depth_m)

        # Número de trazas objetivo
        max_traces = max(p.processed_data.shape[1] for p in self._profiles)
        n_traces = self.n_traces_target if self.n_traces_target else max_traces

        # Eje along: de 0 a la longitud máxima del perfil más largo
        max_dist = max(
            float(p.get_distance_axis()[-1]) for p in self._profiles
        )
        along_m = np.linspace(0.0, max_dist, n_traces)

        # Eje across: distancias acumuladas entre perfiles
        across_m = np.concatenate([[0.0], np.cumsum(spacings)])

        # Rellenar el cubo
        cube = np.full((n_depth, n_traces, n_profiles), np.nan, dtype=np.float32)

        for pi, gpr in enumerate(self._profiles):
            data = np.asarray(gpr.processed_data, dtype=np.float32)
            # data: (n_samples, n_traces_orig)

            # Ajustar n_samples si difiere de n_depth
            if data.shape[0] != n_depth:
                data = _resample_depth(data, data.shape[0], n_depth)

            # Flip según configuración
            if self._should_flip(pi):
                data = data[:, ::-1]

            # Remuestrear trazas al número objetivo
            n_traces_orig = data.shape[1]
            if n_traces_orig != n_traces:
                data = _resample_traces(data, n_traces_orig, n_traces)

            cube[:, :, pi] = data

        return LocalCube(
            data=cube,
            depth_m=depth_m,
            along_m=along_m,
            across_m=across_m,
        )

    # ── Privados ─────────────────────────────────────────────────────────────

    def _resolve_spacings(self, n_profiles: int) -> np.ndarray:
        """Devuelve vector de n_profiles-1 separaciones."""
        s = self._spacings_input
        if n_profiles == 1:
            return np.array([], dtype=float)
        if isinstance(s, (int, float)):
            return np.full(n_profiles - 1, float(s))
        s_arr = np.asarray(s, dtype=float)
        if s_arr.size == n_profiles - 1:
            return s_arr
        if s_arr.size == 1:
            return np.full(n_profiles - 1, s_arr[0])
        raise ValueError(
            f"profile_spacings debe tener {n_profiles - 1} valores "
            f"(o un único float), pero tiene {s_arr.size}."
        )

    def _should_flip(self, pi: int) -> bool:
        """Decide si el perfil pi (0-based) debe ser reflejado."""
        if self.fliplr == "all":
            return True
        if self.fliplr == "odd":
            return (pi + 1) % 2 == 1   # perfiles 1, 3, 5, ... (1-based odd)
        if self.fliplr == "even":
            return (pi + 1) % 2 == 0   # perfiles 2, 4, 6, ... (1-based even)
        return False


# ── Utilidades de remuestreo ─────────────────────────────────────────────────

def _resample_traces(data: np.ndarray, n_orig: int, n_target: int) -> np.ndarray:
    """Remuestrea el eje de trazas (columnas) de n_orig a n_target."""
    if n_orig == n_target:
        return data
    from scipy.interpolate import interp1d
    x_orig = np.linspace(0.0, 1.0, n_orig)
    x_new = np.linspace(0.0, 1.0, n_target)
    f = interp1d(x_orig, data, axis=1, kind="linear", fill_value="extrapolate")
    return f(x_new).astype(np.float32)


def _resample_depth(data: np.ndarray, n_orig: int, n_target: int) -> np.ndarray:
    """Remuestrea el eje de profundidad (filas) de n_orig a n_target."""
    if n_orig == n_target:
        return data
    from scipy.interpolate import interp1d
    x_orig = np.linspace(0.0, 1.0, n_orig)
    x_new = np.linspace(0.0, 1.0, n_target)
    f = interp1d(x_orig, data, axis=0, kind="linear", fill_value="extrapolate")
    return f(x_new).astype(np.float32)