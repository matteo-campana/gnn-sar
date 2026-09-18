"""Georeferenced raster I/O (proposal §5.1, "GDAL/``rasterio``").

The rest of the package is deliberately array-only: everything from
``(T, B, H, W)`` onwards works on plain NumPy and needs no geospatial stack.
This module is the single place where a coordinate reference system exists. It
reads a set of already-preprocessed GeoTIFFs into that array, checks they all
sit on the **same grid**, and writes node-level predictions back out as a
georeferenced map.

Alignment is checked rather than fixed. Reprojecting and resampling inside the
ingestion would hide exactly the kind of mistake that quietly ruins a time
series — an orbit reprocessed on a shifted grid, a Sentinel-2 band left at 20 m
— so the contract is that the caller aligns the stack once, with
``gdalwarp``/SNAP, and this module refuses anything that does not match.

Requires the optional ``rasterio`` dependency
(``poetry install --extras geo``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "RasterGrid",
    "list_rasters",
    "read_raster",
    "read_aligned_stack",
    "write_raster",
]

_TRANSFORM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RasterGrid:
    """The georeferencing every raster of a study area has to share.

    Attributes:
        height: rows.
        width: columns.
        transform: the six affine coefficients in **GDAL order** —
            ``(origin x, pixel width, row rotation, origin y, column rotation,
            pixel height)``, the tuple ``rasterio``'s ``Affine.to_gdal()``
            returns and ``Affine.from_gdal()`` consumes. Pixel height is
            normally negative: north is up.
        crs: the coordinate reference system as a string (e.g. ``"EPSG:32633"``),
            or ``None`` when the source carried none.
    """

    height: int
    width: int
    transform: tuple[float, float, float, float, float, float]
    crs: str | None = None

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("a raster grid must have at least one pixel")
        if len(self.transform) != 6:
            raise ValueError(
                f"transform must hold 6 affine coefficients, got {len(self.transform)}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """``(x, y)`` pixel size in CRS units, both positive."""
        return (abs(self.transform[1]), abs(self.transform[5]))

    @property
    def origin(self) -> tuple[float, float]:
        """Map coordinates of the upper-left corner of the upper-left pixel."""
        return (self.transform[0], self.transform[3])

    def matches(self, other: "RasterGrid", tolerance: float = _TRANSFORM_TOLERANCE) -> bool:
        """Whether two rasters describe exactly the same pixels on the ground."""
        return (
            self.shape == other.shape
            and self.crs == other.crs
            and np.allclose(self.transform, other.transform, atol=tolerance, rtol=0.0)
        )

    def describe(self) -> str:
        return (
            f"{self.height}x{self.width} px, {self.pixel_size[0]:g} x "
            f"{self.pixel_size[1]:g} units, crs={self.crs}, "
            f"origin=({self.origin[0]:g}, {self.origin[1]:g})"
        )


def _require_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "reading and writing GeoTIFFs requires rasterio; install it with "
            "`poetry install --extras geo` (or `pip install rasterio`). The "
            "array-level API of gnn_sar does not need it."
        ) from exc
    return rasterio


def list_rasters(directory: str | Path, pattern: str = "*.tif") -> list[Path]:
    """Every raster of ``directory`` matching ``pattern``, sorted by name.

    Sentinel product names start with the mission and carry the sensing time in
    a fixed position, so a plain name sort is also a chronological sort. The
    ingestion still reads the dates explicitly — it does not rely on this.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory")
    return sorted(directory.glob(pattern))


def read_raster(
    path: str | Path,
    bands: Sequence[int] | None = None,
    masked: bool = True,
) -> tuple[np.ndarray, RasterGrid]:
    """Read one raster as ``(B, H, W)`` float64 together with its grid.

    Args:
        path: the GeoTIFF (or anything else GDAL opens).
        bands: 1-based band indices to read, in the order wanted. ``None``
            reads every band in file order.
        masked: turn the file's nodata value into ``NaN``. Keeping nodata as a
            number would let it through compositing as if it were backscatter.

    Returns:
        ``(values, grid)``.
    """
    rasterio = _require_rasterio()
    path = Path(path)
    with rasterio.open(path) as source:
        indexes = list(range(1, source.count + 1)) if bands is None else list(bands)
        unknown = [i for i in indexes if i < 1 or i > source.count]
        if unknown:
            raise ValueError(
                f"{path.name} has {source.count} bands; requested {unknown}"
            )
        values = source.read(indexes, masked=masked).astype(np.float64)
        if masked:
            values = np.ma.filled(values, np.nan)
        transform = tuple(float(c) for c in source.transform.to_gdal())
        grid = RasterGrid(
            height=int(source.height),
            width=int(source.width),
            transform=transform,  # type: ignore[arg-type]
            crs=None if source.crs is None else str(source.crs),
        )
    return np.asarray(values, dtype=np.float64), grid


def read_aligned_stack(
    paths: Sequence[str | Path],
    bands: Sequence[int] | None = None,
    grid: RasterGrid | None = None,
    masked: bool = True,
) -> tuple[np.ndarray, RasterGrid]:
    """Read several rasters into one ``(K, B, H, W)`` stack on a common grid.

    Args:
        paths: the acquisitions, in the order they should be stacked.
        bands: 1-based band indices, identical for every file.
        grid: the grid every file must sit on. ``None`` takes the first file's
            grid as the reference — which is why the error message below names
            both files.
        masked: see :func:`read_raster`.

    Raises:
        ValueError: when a raster does not sit exactly on the reference grid.
            Resample it beforehand rather than letting the series drift.
    """
    if not paths:
        raise ValueError("no rasters to read")

    reference = grid
    reference_name = "the reference grid"
    layers = []
    for path in paths:
        values, found = read_raster(path, bands=bands, masked=masked)
        if reference is None:
            reference, reference_name = found, Path(path).name
        elif not reference.matches(found):
            raise ValueError(
                f"{Path(path).name} ({found.describe()}) is not on the same grid "
                f"as {reference_name} ({reference.describe()}). Align the stack "
                f"first, e.g. `gdalwarp -t_srs {reference.crs} -tr "
                f"{reference.pixel_size[0]:g} {reference.pixel_size[1]:g} "
                f"-te <xmin ymin xmax ymax> -r bilinear in.tif out.tif`."
            )
        layers.append(values)

    assert reference is not None  # for type checkers; `paths` is non-empty
    return np.stack(layers), reference


def write_raster(
    path: str | Path,
    values: np.ndarray,
    grid: RasterGrid,
    nodata: float | None = np.nan,
    dtype: str = "float32",
    compress: str = "deflate",
) -> Path:
    """Write ``(H, W)`` or ``(B, H, W)`` values as a GeoTIFF.

    This is the last step of the pipeline: a sub-annual sealing probability, an
    onset date or an early-warning alert painted back onto the raster by
    :meth:`gnn_sar.data.segmentation.NodeGeometries.rasterise` becomes a map a
    GIS can open next to the ISPRA/SNPA layer.
    """
    rasterio = _require_rasterio()
    from rasterio.transform import Affine

    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"values must be (H, W) or (B, H, W), got {values.shape}")
    if values.shape[-2:] != grid.shape:
        raise ValueError(
            f"values raster {values.shape[-2:]} does not match the grid {grid.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=values.shape[0],
        dtype=dtype,
        crs=grid.crs,
        transform=Affine.from_gdal(*grid.transform),
        nodata=nodata,
        compress=compress,
    ) as destination:
        destination.write(values.astype(dtype))
    return path
