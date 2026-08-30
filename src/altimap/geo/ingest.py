"""Strict ingest for the production GeoTIFF-only pipeline.

The model sees an 8-bit display stretch, but all outputs retain the exact
source grid.  This separation prevents normalisation for ML from changing
coordinates, pixel centres, or metric measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


class RasterValidationError(ValueError):
    """Input cannot safely enter the metric GeoTIFF pipeline."""


@dataclass(frozen=True)
class GeoTiffInput:
    path: Path
    width: int
    height: int
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    pixel_size_x_m: float
    pixel_size_y_m: float
    source_dtype: str
    band_count: int

    @property
    def gsd_m(self) -> float:
        return float((self.pixel_size_x_m + self.pixel_size_y_m) / 2.0)


def _pixel_size_metres(src: rasterio.DatasetReader) -> tuple[float, float]:
    if src.crs is None:
        raise RasterValidationError("GeoTIFF has no CRS; PNG/JPG and ungeoreferenced TIFF are out of scope")
    if src.transform.is_identity:
        raise RasterValidationError("GeoTIFF has no usable affine transform")
    if not np.isclose(src.transform.b, 0.0) or not np.isclose(src.transform.d, 0.0):
        raise RasterValidationError("rotated GeoTIFF grids are not supported by the metric mesh exporter yet")
    if src.crs.is_geographic:
        # Degree pixels are not a stable metric mesh unit. Reprojection is an
        # explicit future stage, rather than a hidden and potentially wrong one.
        raise RasterValidationError("input CRS is geographic; reproject to a projected metre CRS before processing")
    x, y = abs(src.transform.a), abs(src.transform.e)
    if x <= 0 or y <= 0:
        raise RasterValidationError("invalid GeoTIFF pixel size")
    unit = (src.crs.linear_units or "").lower()
    if unit not in {"metre", "meter", "m", "metres", "meters"}:
        raise RasterValidationError(f"projected CRS must use metres, found {src.crs.linear_units!r}")
    return float(x), float(y)


def inspect_geotiff(path: str | Path) -> GeoTiffInput:
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise RasterValidationError("only georeferenced .tif/.tiff optical imagery is accepted")
    with rasterio.open(path) as src:
        if src.count < 3:
            raise RasterValidationError("GeoTIFF must contain at least three optical RGB bands")
        if src.width < 2 or src.height < 2:
            raise RasterValidationError("GeoTIFF is too small to form terrain")
        px, py = _pixel_size_metres(src)
        bounds = src.bounds
        return GeoTiffInput(path, src.width, src.height, src.transform, src.crs,
                            (bounds.left, bounds.bottom, bounds.right, bounds.top),
                            px, py, src.dtypes[0], src.count)


def _stretch_to_uint8(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.empty(rgb.shape, dtype=np.uint8)
    for i in range(3):
        channel = rgb[i].astype(np.float32, copy=False)
        values = channel[valid & np.isfinite(channel)]
        if values.size == 0:
            raise RasterValidationError("RGB bands contain no finite pixels")
        lo, hi = np.percentile(values, (2.0, 98.0))
        out[i] = np.clip((channel - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255)
    return np.moveaxis(out, 0, -1)


def read_geotiff_rgb(path: str | Path) -> tuple[GeoTiffInput, np.ndarray, np.ndarray]:
    """Return metadata, ML/display RGB (H,W,3 uint8), and valid-pixel mask."""
    meta = inspect_geotiff(path)
    with rasterio.open(meta.path) as src:
        rgb = src.read((1, 2, 3), masked=True)
        mask = ~np.any(np.ma.getmaskarray(rgb), axis=0)
        rgb_values = np.ma.getdata(rgb)
    return meta, _stretch_to_uint8(rgb_values, mask), mask
