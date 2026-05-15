# togpri/processing/grid3d.py
"""
Construcción de un volumen 3D regular (voxeles) a partir de perfiles GPR.

- Cada perfil se describe mediante ProfileData3D.
- Grid3DBuilder toma varios perfiles y los interpola en una rejilla regular.
- GPRVolume contiene el volumen resultante y utilidades para extraer cortes.

La interpolación se hace desde datos dispersos (x, y, z, amplitud) a una
rejilla (xi, yi, zi) usando scipy.interpolate.griddata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ProfileData3D:
    """
    Datos de un perfil GPR con geometría 3D.

    data: matriz (n_samples, n_traces) de amplitudes procesadas.
    distance_m: vector (n_traces,) con distancia acumulada a lo largo del perfil.
    depth_m: vector (n_samples,) con profundidad de cada muestra.
    surface_z: vector (n_traces,) con cota z de la superficie en cada traza.
    origin_xy: (x, y) del origen del perfil.
    direction_xy: vector director (dx, dy) del perfil en planta.
    """

    filename: str
    data: np.ndarray
    distance_m: np.ndarray
    depth_m: np.ndarray
    surface_z: np.ndarray
    origin_xy: Tuple[float, float]
    direction_xy: Tuple[float, float]


class GPRVolume:
    """
    Volumen 3D regular de amplitudes GPR.

    data tiene forma (nx, ny, nz), con ejes xi, yi, zi.
    """

    def __init__(
        self,
        xi: np.ndarray,
        yi: np.ndarray,
        zi: np.ndarray,
        data: np.ndarray,
    ) -> None:
        self.xi = np.asarray(xi)
        self.yi = np.asarray(yi)
        self.zi = np.asarray(zi)
        self.data = np.asarray(data)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def get_depth_slice(
        self,
        depth_m: float,
        reference_z: float | None = None,
    ) -> np.ndarray:
        """
        Devuelve un corte horizontal para una profundidad dada.

        Si reference_z es None, se interpreta depth_m directamente como cota z.
        Si reference_z se pasa (por ejemplo la media de surface_z),
        se interpreta depth_m como profundidad bajo la superficie media y
        se calcula z_target = reference_z - depth_m.
        """
        if reference_z is None:
            z_target = depth_m
        else:
            z_target = reference_z - depth_m

        # Índice de zi más cercano a z_target
        idx = int(np.argmin(np.abs(self.zi - z_target)))
        slice_data = self.data[:, :, idx]
        return slice_data


class Grid3DBuilder:
    """
    Builder para construir un GPRVolume a partir de varios perfiles GPR.

    Parámetros:
    - cell_size_xy: tamaño de celda en planta (m).
    - cell_size_z: tamaño de celda en profundidad (m).
    - max_points: máximo de puntos de entrada a griddata (None = sin límite).
    """

    def __init__(
        self,
        cell_size_xy: float = 0.05,
        cell_size_z: float = 0.02,
        max_points: int | None = None,
    ) -> None:
        self.cell_size_xy = float(cell_size_xy)
        self.cell_size_z = float(cell_size_z)
        self.max_points = max_points
        self._profiles: List[ProfileData3D] = []

    # ── API pública ─────────────────────────────────────────────────────────

    def add_profile(self, profile: ProfileData3D) -> None:
        """Añade un perfil al builder."""
        self._profiles.append(profile)

    def clear(self) -> None:
        """Elimina todos los perfiles acumulados."""
        self._profiles.clear()

    @property
    def n_profiles(self) -> int:
        return len(self._profiles)

    # ── Conversión desde GPRData ────────────────────────────────────────────

    @staticmethod
    def profile_from_gprdata(
        gpr,
        velocity_m_ns: float = 0.1,
    ) -> ProfileData3D | None:
        """
        Convierte un GPRData (con geometry completa) en ProfileData3D.

        Si falta 'surface_z' en geometry, asume superficie plana z=0
        (topografía por defecto).

        Devuelve None solo si faltan datos de geometría en planta.
        """
        geom = gpr.geometry

        # Geometría en planta obligatoria
        if "origin_xy" not in geom or "direction_xy" not in geom:
            print(
                f"  ⚠ Sin geometría (origin_xy/direction_xy): "
                f"{geom.get('filename', '?')}"
            )
            return None

        # Eje distancia
        distance_m = np.asarray(
            geom.get("distance_m", gpr.get_distance_axis())
        )

        # Topografía: si no hay, superficie plana z=0
        if "surface_z" in geom:
            surface_z = np.asarray(geom["surface_z"])
        else:
            n_traces = gpr.processed_data.shape[1]
            surface_z = np.zeros(n_traces, dtype=float)

        depth_m = np.asarray(gpr.get_depth_axis(velocity_m_ns))

        # Ajustar longitudes si hay discrepancias (p.ej. por trace_stacking)
        n_traces = gpr.processed_data.shape[1]
        if distance_m.size != n_traces:
            distance_m = np.linspace(distance_m[0], distance_m[-1], n_traces)
        if surface_z.size != n_traces:
            surface_z = np.interp(
                np.linspace(0, 1, n_traces),
                np.linspace(0, 1, surface_z.size),
                surface_z,
            )

        return ProfileData3D(
            filename=geom.get("filename", ""),
            data=np.asarray(gpr.processed_data),
            distance_m=distance_m,
            depth_m=depth_m,
            surface_z=surface_z,
            origin_xy=tuple(geom["origin_xy"]),
            direction_xy=tuple(geom["direction_xy"]),
        )

    # ── Construcción del volumen ────────────────────────────────────────────

    def build(self) -> GPRVolume:
        """
        Interpola todos los perfiles en una rejilla 3D regular y devuelve un GPRVolume.

        Requiere al menos un perfil añadido.
        """
        if not self._profiles:
            raise ValueError("No hay perfiles añadidos al Grid3DBuilder.")

        # Recolectar puntos dispersos (x, y, z, amplitud)
        all_points: List[np.ndarray] = []
        all_values: List[np.ndarray] = []

        for prof in self._profiles:
            data = np.asarray(prof.data)  # (n_samples, n_traces)
            n_samples, n_traces = data.shape

            depth = prof.depth_m.reshape(-1, 1)           # (n_samples, 1)
            distance = prof.distance_m.reshape(1, -1)     # (1, n_traces)

            origin = np.asarray(prof.origin_xy, dtype=float)
            direction = np.asarray(prof.direction_xy, dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm == 0:
                print(f"  ⚠ Dirección nula en perfil {prof.filename}, se ignora.")
                continue
            direction /= norm

            # Posición XY de cada traza
            trace_param = distance.ravel()[:, None]  # (n_traces, 1)
            trace_xy = origin[None, :] + trace_param * direction[None, :]

            # Superficie z en cada traza
            surface_z = prof.surface_z.reshape(1, -1)    # (1, n_traces)

            # Profundidad positiva hacia abajo: z = surface_z - depth
            z = surface_z - depth                        # (n_samples, n_traces)

            # Expandir x, y a la malla de muestras
            x = np.broadcast_to(trace_xy[:, 0], (n_samples, n_traces))
            y = np.broadcast_to(trace_xy[:, 1], (n_samples, n_traces))

            all_points.append(
                np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            )
            all_values.append(data.ravel())

        if not all_points:
            raise ValueError(
                "No se pudo usar ningún perfil (todos tenían problemas de geometría)."
            )

        points = np.concatenate(all_points, axis=0)
        values = np.concatenate(all_values, axis=0)

        n_points = points.shape[0]
        print(f"Puntos originales para griddata: {n_points/1e6:.1f} M")

        # ── Decimado opcional de puntos antes de griddata ───────────────────
        if self.max_points is not None and n_points > self.max_points:
            rng = np.random.default_rng()
            idx = rng.choice(n_points, size=self.max_points, replace=False)
            points = points[idx]
            values = values[idx]
            print(f"Decimado a: {self.max_points/1e6:.1f} M puntos")

        # Definir dominios de la rejilla
        xmin, ymin, zmin = points.min(axis=0)
        xmax, ymax, zmax = points.max(axis=0)

        xi = np.arange(xmin, xmax + self.cell_size_xy * 0.5, self.cell_size_xy)
        yi = np.arange(ymin, ymax + self.cell_size_xy * 0.5, self.cell_size_xy)
        zi = np.arange(zmin, zmax + self.cell_size_z * 0.5, self.cell_size_z)

        nx, ny, nz = len(xi), len(yi), len(zi)
        n_voxels = nx * ny * nz
        print(f"Rejilla 3D: {nx} x {ny} x {nz} = {n_voxels/1e6:.1f} M vóxeles")

        # Límite de seguridad de tamaño de volumen (ajustable)
        max_voxels = 20_000_000
        if n_voxels > max_voxels:
            raise ValueError(
                f"Rejilla 3D demasiado grande: {nx}×{ny}×{nz} "
                f"({n_voxels/1e6:.1f} M). "
                "Aumenta el tamaño de celda XY/Z o usa menos perfiles."
            )

        # Crear rejilla de destino
        X, Y, Z = np.meshgrid(xi, yi, zi, indexing="xy")
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Interpolación: lineal + relleno nearest
        try:
            from scipy.interpolate import griddata  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para construir la malla 3D necesitas instalar scipy:\n"
                "    pip install scipy"
            ) from exc

        grid_vals = griddata(points, values, grid_points, method="linear")
        nan_mask = ~np.isfinite(grid_vals)
        if np.any(nan_mask):
            grid_vals[nan_mask] = griddata(
                points, values, grid_points[nan_mask], method="nearest"
            )

        volume_data = grid_vals.astype("float32").reshape(nx, ny, nz)

        return GPRVolume(xi=xi, yi=yi, zi=zi, data=volume_data)