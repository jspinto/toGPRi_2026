# togpri/processing/topography.py
"""
Corrección topográfica 1D por perfil.
Separa la lógica de interpolación de la GUI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator


def build_profile_layout(n_traces: int, trace_spacing_m: float) -> pd.DataFrame:
    """
    Construye un DataFrame con la geometría básica del perfil.
    Columnas: trace_idx, distance_m
    """
    return pd.DataFrame({
        "trace_idx": np.arange(n_traces),
        "distance_m": np.arange(n_traces) * trace_spacing_m,
    })


def interpolate_surface(
    control_distances: list[float],
    control_z: list[float],
    distance_m: np.ndarray,
) -> np.ndarray:
    """
    Interpola la superficie topográfica en todas las trazas usando PCHIP
    (monotone cubic — no produce oscilaciones entre puntos de control).

    Parameters
    ----------
    control_distances : distancias (m) de los puntos de control
    control_z         : cotas z (m) en esos puntos
    distance_m        : array de distancias de todas las trazas

    Returns
    -------
    ndarray (n_traces,) con la cota z interpolada en cada traza
    """
    if len(control_distances) < 2:
        raise ValueError("Se necesitan al menos 2 puntos de control para interpolar.")

    # Ordenar por distancia
    order = np.argsort(control_distances)
    xp = np.array(control_distances)[order]
    zp = np.array(control_z)[order]

    interp = PchipInterpolator(xp, zp, extrapolate=True)
    return interp(distance_m)


def compute_absolute_z(
    surface_z_by_trace: np.ndarray,
    depth_m: np.ndarray,
) -> np.ndarray:
    """
    Calcula la matriz de cotas absolutas Z para cada celda del radargrama.

    Parameters
    ----------
    surface_z_by_trace : (n_traces,)  — cota z de la superficie en cada traza
    depth_m            : (n_samples,) — profundidad de cada sample

    Returns
    -------
    ndarray (n_samples, n_traces) con la cota absoluta de cada celda
    """
    return surface_z_by_trace[None, :] - depth_m[:, None]


def apply_topo_correction(
    data: np.ndarray,
    surface_z: np.ndarray,
    depth_m: np.ndarray,
    output_dz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reinterpolación del radargrama en una rejilla Z regular con corrección
    topográfica (cada traza se desplaza verticalmente según su cota).

    Parameters
    ----------
    data       : (n_samples, n_traces)
    surface_z  : (n_traces,) cota de superficie
    depth_m    : (n_samples,) eje de profundidad original
    output_dz  : paso vertical de la rejilla de salida (m). Si None, usa
                 la resolución original.

    Returns
    -------
    corrected  : (n_z_out, n_traces) radargrama en rejilla Z regular
    z_axis     : (n_z_out,) eje Z absoluto de salida
    """
    z_abs = compute_absolute_z(surface_z, depth_m)   # (n_samples, n_traces)

    z_top = z_abs.max()
    z_bot = z_abs.min()

    if output_dz is None:
        output_dz = (depth_m[1] - depth_m[0]) if len(depth_m) > 1 else 0.01

    z_axis = np.arange(z_top, z_bot - output_dz, -output_dz)
    n_z_out = len(z_axis)
    n_traces = data.shape[1]
    corrected = np.full((n_z_out, n_traces), np.nan)

    for ti in range(n_traces):
        z_col = z_abs[:, ti]          # cotas absolutas de esta traza
        amp_col = data[:, ti]
        # Reinterpolamos sobre z_axis usando los valores de esta traza
        # (z_col es descendente, invertimos para que scipy funcione)
        idx_sorted = np.argsort(z_col)
        zs = z_col[idx_sorted]
        amps = amp_col[idx_sorted]
        corrected[:, ti] = np.interp(z_axis, zs, amps,
                                     left=np.nan, right=np.nan)

    return corrected, z_axis


def save_surface_csv(
    distance_m: np.ndarray,
    surface_z: np.ndarray,
    path: Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"distance_m": distance_m, "surface_z": surface_z})
    df.to_csv(path, index=False)


def load_surface_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["distance_m"].to_numpy(), df["surface_z"].to_numpy()