"""The interface every AltiMap subsystem communicates through.

A produced elevation result is always three things: a Cloud-Optimized GeoTIFF
holding float32 metres, the source RGB on the identical grid, and a JSON
sidecar describing both. Nothing else crosses a subsystem boundary.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# "unknown" is intentional: an unverified DEM vertical reference must never be
# silently advertised as ellipsoidal height.
VALID_DATUMS = ("ellipsoidal", "orthometric", "relative", "unknown")


@dataclasses.dataclass(frozen=True)
class Sidecar:
    """Metadata accompanying an elevation raster.

    `height_range_m` exists so the viewer can configure its colour ramp and
    camera bounds without scanning the raster.
    """

    gsd_m: float
    source_gsd_m: float
    datum: str
    vertical_unit: str
    model_version: str
    height_range_m: tuple[float, float]
    tile_overlap_px: int
    dtm_source: str | None

    def __post_init__(self) -> None:
        if self.datum not in VALID_DATUMS:
            raise ValueError(
                f"datum must be one of {VALID_DATUMS}, got {self.datum!r}"
            )
        if self.gsd_m <= 0:
            raise ValueError(f"gsd_m must be positive, got {self.gsd_m}")
        if self.vertical_unit != "m":
            raise ValueError(
                f"vertical_unit must be 'm' — metres are the project-wide unit, "
                f"got {self.vertical_unit!r}"
            )

    def to_json(self, path: Path) -> None:
        payload = dataclasses.asdict(self)
        payload["height_range_m"] = list(self.height_range_m)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> Sidecar:
        payload: dict[str, Any] = json.loads(path.read_text())
        payload["height_range_m"] = tuple(payload["height_range_m"])
        return cls(**payload)


def sidecar_path_for(raster_path: Path) -> Path:
    """The sidecar lives beside the raster with a .json suffix."""
    return raster_path.with_suffix(".json")


def write_elevation_cog(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: CRS,
    sidecar: Sidecar,
) -> None:
    """Write a float32 elevation COG plus its sidecar.

    NaN is the nodata value throughout the project — sentinel values like
    -9999 silently corrupt statistics when a mask is forgotten.
    """
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")

    data = array.astype(np.float32, copy=False)
    with rasterio.open(
        path,
        "w",
        driver="COG",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        blocksize=256,
        compress="deflate",
        predictor=3,
        overview_resampling="average",
    ) as dst:
        dst.write(data, 1)

    sidecar.to_json(sidecar_path_for(path))


def read_elevation(path: Path) -> tuple[np.ndarray, Affine, CRS]:
    """Read an elevation raster as float32 with nodata represented as NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        return data, src.transform, src.crs
