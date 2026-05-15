# togpri/processing/local_tiff_export.py
"""
Exporta cortes del LocalCube a TIFF (con o sin georreferenciación).

Dos modos:
  - Sin georef: TIFF simple indexado (como el toGPRi v1 original sin georef).
  - Con georef: GeoTIFF con EPSG + esquina superior izquierda + rotación
    (como el worldfile/GeoTIFF de toGPRi v1).

Convención interna:
  - LocalCube.data tiene shape (n_depth, n_along, n_across)
  - Un corte horizontal slice_data tiene shape (n_along, n_across)

Convención raster:
  - rasterio escribe arrays 2D como (rows, cols) = (height, width)
  - por tanto, para exportar un corte horizontal:
      rows   <- across
      cols   <- along
    y hay que transponer slice_data antes de escribirlo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .local_cube import LocalCube


def export_local_cube_depth_tiff(
    cube: LocalCube,
    out_path: Path | str,
    *,
    depth_m: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    band_mode: Literal["max", "mean", "min"] = "max",
    rotate_plan: bool = False,
    mode: Literal["indexed", "bw"] = "indexed",
    alpha_range: tuple[int, int] | None = None,
    epsg: int | None = None,
    origin_xy: tuple[float, float] | None = None,
    rotation_deg: float = 0.0,
) -> None:
    """
    Exporta a TIFF un corte horizontal de un LocalCube.

    Opciones de corte:
      - depth_m: corte exacto
      - z_min + z_max: banda de profundidades agregada

    Si rotate_plan=True, intercambia along/across antes de extraer el corte.
    Esto sirve para corregir una planta girada (por ejemplo pasar de ~9x20 a ~20x9).

    La exportación final usa:
      - columnas = along
      - filas    = across
    """
    cube_to_export = cube.rotated_plan() if rotate_plan else cube

    if depth_m is not None:
        slice_data = cube_to_export.get_depth_slice(depth_m)
    elif z_min is not None and z_max is not None:
        slice_data = cube_to_export.get_depth_band(z_min, z_max, mode=band_mode)
    else:
        raise ValueError(
            "Debes indicar depth_m o bien z_min y z_max."
        )

    if len(cube_to_export.along_m) > 1:
        cell_size_along = float(cube_to_export.along_m[1] - cube_to_export.along_m[0])
    else:
        cell_size_along = 1.0

    if len(cube_to_export.across_m) > 1:
        cell_size_across = float(cube_to_export.across_m[1] - cube_to_export.across_m[0])
    else:
        cell_size_across = 1.0

    export_local_slice_tiff(
        slice_data=slice_data,
        out_path=out_path,
        mode=mode,
        alpha_range=alpha_range,
        epsg=epsg,
        origin_xy=origin_xy,
        rotation_deg=rotation_deg,
        cell_size_along=cell_size_along,
        cell_size_across=cell_size_across,
    )


def export_local_slice_tiff(
    slice_data: np.ndarray,
    out_path: Path | str,
    mode: Literal["indexed", "bw"] = "indexed",
    alpha_range: tuple[int, int] | None = None,
    epsg: int | None = None,
    origin_xy: tuple[float, float] | None = None,
    rotation_deg: float = 0.0,
    cell_size_along: float = 1.0,
    cell_size_across: float = 1.0,
) -> None:
    """
    Exporta un corte 2D (n_along, n_across) como TIFF.

    - slice_data: array (n_along, n_across) de amplitudes.
    - mode: 'indexed' o 'bw'. De momento ambos se guardan como uint8 de una banda;
      'indexed' queda reservado para una futura paleta explícita.
    - alpha_range: si no es None, añade canal alpha y hace transparente
      el rango [a, b] de valores 0-255.
    - epsg, origin_xy, rotation_deg: si se proporcionan, genera GeoTIFF
      con rotación.
    - cell_size_along, cell_size_across: tamaño de celda en metros para el
      transform afín.

    Convención espacial:
      - columnas raster = eje along
      - filas raster    = eje across

    Por tanto:
      slice_data.shape        = (n_along, n_across)
      img8_rc.shape exportado = (n_across, n_along)
    """
    out_path = Path(out_path)

    # Validación básica
    data = np.asarray(slice_data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"slice_data debe ser 2D con shape (n_along, n_across), no {data.shape}."
        )

    # Normalizar a 0-255
    finite = np.isfinite(data)
    if np.any(finite):
        vmin = np.percentile(data[finite], 2)
        vmax = np.percentile(data[finite], 98)
        if vmax > vmin:
            img = (data - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(data)
    else:
        img = np.zeros_like(data)

    img = np.clip(img, 0, 1)
    img[~finite] = 0
    img8 = (img * 255).astype(np.uint8)

    # Pasar de (along, across) a (rows=across, cols=along)
    img8_rc = img8.T
    height, width = img8_rc.shape

    # Construir transform
    transform = None
    if epsg is not None and origin_xy is not None:
        from affine import Affine
        import math

        angle_rad = math.radians(-rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        ox, oy = origin_xy

        sx = float(cell_size_along)   # metros por columna (along -> X local)
        sy = float(cell_size_across)  # metros por fila    (across -> Y local)

        # Coordenadas de esquina superior izquierda del pixel (col,row):
        # x = ox + sx*cos(col) + sy*sin(row)
        # y = oy + sx*sin(col) - sy*cos(row)
        transform = Affine(
            sx * cos_a,  sy * sin_a, ox,
            sx * sin_a, -sy * cos_a, oy,
        )

    try:
        import rasterio
        from rasterio.crs import CRS

        rasterio_kwargs = dict(
            driver="GTiff",
            height=height,
            width=width,
            dtype="uint8",
        )

        if transform is not None:
            rasterio_kwargs["transform"] = transform
        if epsg is not None:
            rasterio_kwargs["crs"] = CRS.from_epsg(epsg)

        if alpha_range is not None:
            a_low, a_high = alpha_range
            alpha = np.where(
                (img8_rc >= a_low) & (img8_rc <= a_high),
                0, 255
            ).astype(np.uint8)

            rasterio_kwargs["count"] = 2
            with rasterio.open(out_path, "w", **rasterio_kwargs) as dst:
                dst.write(img8_rc, 1)
                dst.write(alpha, 2)
        else:
            rasterio_kwargs["count"] = 1
            with rasterio.open(out_path, "w", **rasterio_kwargs) as dst:
                dst.write(img8_rc, 1)

    except ImportError:
        # Fallback sin rasterio: solo TIFF con PIL (sin georef)
        try:
            from PIL import Image

            if alpha_range is not None:
                a_low, a_high = alpha_range
                alpha = np.where(
                    (img8_rc >= a_low) & (img8_rc <= a_high),
                    0, 255
                ).astype(np.uint8)
                rgba = np.dstack([img8_rc, img8_rc, img8_rc, alpha])
                im = Image.fromarray(rgba, mode="RGBA")
            else:
                im = Image.fromarray(img8_rc, mode="L")

            im.save(str(out_path))
        except ImportError as exc:
            raise ImportError(
                "Necesitas rasterio o Pillow para exportar TIFFs.\n"
                "  pip install rasterio"
            ) from exc