# togpri/processing/ply_export.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from .grid3d import GPRVolume


def volume_to_ply(
    volume: GPRVolume,
    out_path: Path | str,
    stride_xy: int = 1,
    stride_z: int = 1,
    normalize_intensity: bool = True,
) -> int:
    """
    Exporta un GPRVolume como nube de puntos PLY (ASCII).

    - stride_xy: submuestreo en X e Y (1 = sin submuestreo).
    - stride_z: submuestreo en Z.
    - normalize_intensity: si True, escala amplitudes a [0, 1].

    Devuelve el número de puntos exportados.
    """
    out_path = Path(out_path)

    # Volumen y ejes
    vol = np.asarray(volume.data)       # (nx, ny, nz)
    xi = np.asarray(volume.xi)
    yi = np.asarray(volume.yi)
    zi = np.asarray(volume.zi)

    nx, ny, nz = vol.shape

    # Submuestreo por stride
    vol_sub = vol[::stride_xy, ::stride_xy, ::stride_z]
    x_sub = xi[::stride_xy]
    y_sub = yi[::stride_xy]
    z_sub = zi[::stride_z]

    # Rejilla de coordenadas
    X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub, indexing="xy")  # (nx', ny', nz')

    # Aplanar
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    If = vol_sub.ravel()

    # Filtrar valores no finitos
    mask = np.isfinite(If)
    Xf = Xf[mask]
    Yf = Yf[mask]
    Zf = Zf[mask]
    If = If[mask]

    if Xf.size == 0:
        raise ValueError("El volumen no contiene valores finitos para exportar.")

    # Normalizar intensidad si se desea
    if normalize_intensity:
        a = np.abs(If)
        vmin = np.percentile(a, 2)
        vmax = np.percentile(a, 98)
        if vmax > vmin:
            Inorm = (a - vmin) / (vmax - vmin)
        else:
            Inorm = np.zeros_like(a)
        Inorm = np.clip(Inorm, 0.0, 1.0)
    else:
        Inorm = If.astype(float)

    n_points = Xf.size

    # Escribir PLY ASCII sencillo con intensidad [0,255] como "red"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Escalar a 0–255 para color
        col = (Inorm * 255).astype(np.uint8)
        for x, y, z, c in zip(Xf, Yf, Zf, col):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(c)} {int(c)} {int(c)}\n")

    return n_points