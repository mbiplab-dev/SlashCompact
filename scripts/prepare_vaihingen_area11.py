"""Prepare a real ISPRS Vaihingen Area 11 integration fixture.

The public semantic-label archive's area tiles deliberately lack CRS metadata.
The accompanying ``Vaihingen_dsm_tiles_geoinfo.zip`` defines the Area 11
world-file grid, and its VRT defines EPSG:32633.  This script restores that
metadata explicitly and never derives input ground elevation from the reference
DSM.  Instead it makes a provisional lower-envelope DTM from independent ALS
returns; it is suitable for an integration demo, not an accuracy claim.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window, transform
from scipy.ndimage import distance_transform_edt, gaussian_filter


AREA11_RGB = "top/top_mosaic_09cm_area11.tif"


def _world_file(contents: str) -> Affine:
    values = [float(line) for line in contents.splitlines()]
    if len(values) != 6:
        raise ValueError("invalid world file")
    a, d, b, e, c, f = values
    return Affine(a, b, c, d, e, f)


def _write(path: Path, data: np.ndarray, transform_: Affine, crs: CRS) -> None:
    count = 1 if data.ndim == 2 else data.shape[0]
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[-2], width=data.shape[-1], count=count,
                       dtype=data.dtype, crs=crs, transform=transform_, compress="deflate", tiled=True) as dst:
        dst.write(data if data.ndim == 3 else data[None])


def als_lower_envelope(als_dir: Path, bounds: rasterio.coords.BoundingBox, transform_: Affine,
                       shape: tuple[int, int]) -> np.ndarray:
    """Rasterize minimum ALS returns and fill/smooth gaps into a provisional DTM."""
    h, w = shape
    grid = np.full((h, w), np.inf, dtype=np.float32)
    used = 0
    for source in als_dir.glob("*.LAS"):
        header = laspy.open(source).header
        if header.maxs[0] < bounds.left or header.mins[0] > bounds.right or header.maxs[1] < bounds.bottom or header.mins[1] > bounds.top:
            continue
        with laspy.open(source) as reader:
            for points in reader.chunk_iterator(1_000_000):
                x, y, z = np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)
                keep = (x >= bounds.left) & (x < bounds.right) & (y > bounds.bottom) & (y <= bounds.top)
                if not np.any(keep):
                    continue
                cols = ((x[keep] - transform_.c) / transform_.a).astype(np.int64)
                rows = ((y[keep] - transform_.f) / transform_.e).astype(np.int64)
                valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                np.minimum.at(grid, (rows[valid], cols[valid]), z[keep][valid])
                used += int(valid.sum())
    valid = np.isfinite(grid)
    if used == 0 or not valid.any():
        raise RuntimeError("no ALS points intersect requested Area 11 crop")
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = grid[tuple(nearest)]
    # 1 m smoothing removes point-scale noise while preserving the local ground
    # trend. This DTM is intentionally labelled provisional in its filename.
    return gaussian_filter(filled, sigma=max(1, round(1.0 / abs(transform_.a)))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create georeferenced Vaihingen Area 11 RGB/DTM/reference DSM files")
    parser.add_argument("--root", type=Path, default=Path(r"C:\Vaihingen\Vaihingen"))
    parser.add_argument("--output", type=Path, default=Path("data/vaihingen-area11"))
    parser.add_argument("--row", type=int, default=1024, help="top row of a safe real integration crop")
    parser.add_argument("--col", type=int, default=640)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    archive = args.root / "ISPRS_semantic_labeling_Vaihingen.zip"
    dsm_path = args.root / "dsm_09cm_matching_area11.tif"
    info_zip = args.root / "Vaihingen_dsm_tiles_geoinfo.zip"
    if not archive.is_file() or not dsm_path.is_file() or not info_zip.is_file():
        raise FileNotFoundError("expected ISPRS archive, Area 11 DSM, and DSM geoinfo zip beneath --root")
    with zipfile.ZipFile(info_zip) as z:
        raw = z.read("dsm_09cm_matching_area11.tfw").decode("ascii")
    args.output.mkdir(parents=True, exist_ok=True)
    grid = _world_file(raw)
    crs = CRS.from_epsg(32633)
    window = Window(args.col, args.row, args.size, args.size)
    with rasterio.open(f"zip://{archive}!{AREA11_RGB}") as rgb_src:
        rgb = rgb_src.read((1, 2, 3), window=window)
    with rasterio.open(dsm_path) as reference_src:
        reference = reference_src.read(1, window=window)
    crop_transform = transform(window, grid)
    bounds = rasterio.transform.array_bounds(args.size, args.size, crop_transform)
    bbox = rasterio.coords.BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])
    dtm = als_lower_envelope(args.root / "ALS", bbox, crop_transform, (args.size, args.size))
    _write(args.output / "rgb.tif", rgb, crop_transform, crs)
    _write(args.output / "dtm_provisional_als.tif", dtm, crop_transform, crs)
    _write(args.output / "reference_dsm.tif", reference.astype(np.float32), crop_transform, crs)
    (args.output / "README.md").write_text(
        "Real ISPRS Vaihingen Area 11 crop. rgb.tif and reference_dsm.tif are matching 9 cm grids. "
        "dtm_provisional_als.tif is a lower-envelope, 1 m smoothed surface from unclassified ALS strips; "
        "it is an independent integration input, not a surveyed ground-truth DTM.\n"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
