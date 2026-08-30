"""Build 3D reference scenes for every georeferenced Vaihingen camera TIFF.

This is deliberately a zero-ML visual/reference path.  Each 12-bit image is
downsampled for the browser, then the supplied 9 cm matching DSM is sampled on
that image's own (possibly rotated) affine grid.  The orthophoto is draped over
the reference DSM.  It shows what a high-quality geospatial reconstruction
looks like; it must not be reported as monocular model output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject


def stretch_rgb(bands: np.ndarray) -> np.ndarray:
    output = np.empty(bands.shape, dtype=np.uint8)
    for channel in range(3):
        values = bands[channel].astype(np.float32)
        lo, hi = np.percentile(values[np.isfinite(values)], (1, 99))
        output[channel] = np.clip((values - lo) * 255 / max(hi - lo, 1e-6), 0, 255)
    return np.moveaxis(output, 0, -1)


def pack_rg16(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("reference DSM produced no finite pixels")
    lo, hi = float(finite.min()), float(finite.max())
    normalized = np.nan_to_num((values - lo) / max(hi - lo, 1e-6), nan=0.0)
    encoded = np.rint(np.clip(normalized, 0, 1) * 65535).astype(np.uint16)
    high, low = (encoded >> 8).astype(np.uint8), (encoded & 255).astype(np.uint8)
    return np.dstack((high, low, np.zeros_like(high))), lo, hi


def export_scene(image_path: Path, dsm: rasterio.DatasetReader, output: Path,
                 max_dimension: int) -> dict:
    with rasterio.open(image_path) as source:
        if source.crs is None or source.count < 3:
            raise ValueError(f"{image_path.name} is not a georeferenced RGB TIFF")
        scale = max(source.width, source.height) / max_dimension
        width = max(2, round(source.width / scale))
        height = max(2, round(source.height / scale))
        rgb = source.read((1, 2, 3), out_shape=(3, height, width), resampling=Resampling.bilinear)
        transform = source.transform * Affine.scale(source.width / width, source.height / height)
        crs = source.crs

    # The downloaded Vaihingen DSM VRT says zone 33, while every original image
    # says zone 32 and both use the same 497k/5.42M coordinate grid.  Vaihingen
    # is geographically in UTM 32N.  Treat the DSM's horizontal numbers as zone
    # 32 so reproject does not introduce a fictitious 450 km shift.
    corners = [transform * point for point in ((0, 0), (width, 0), (0, height), (width, height))]
    xs, ys = [point[0] for point in corners], [point[1] for point in corners]
    source_window = from_bounds(min(xs), min(ys), max(xs), max(ys), dsm.transform)
    source_window = source_window.round_offsets().round_lengths().intersection(Window(0, 0, dsm.width, dsm.height))
    source_dsm = dsm.read(1, window=source_window)
    # The delivered mosaic uses -9999 without declaring it in the GeoTIFF
    # nodata tag.  Treat it explicitly as missing before interpolation.
    source_dsm = np.where(source_dsm <= -9000, np.nan, source_dsm).astype(np.float32)
    source_transform = dsm.window_transform(source_window)
    terrain = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=source_dsm, destination=terrain,
        src_transform=source_transform, src_crs=CRS.from_epsg(32632),
        dst_transform=transform, dst_crs=crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    valid = np.isfinite(terrain)
    if valid.mean() < 0.2:
        raise ValueError(f"only {valid.mean():.1%} of {image_path.name} overlaps the reference DSM")
    if not valid.all():
        terrain[~valid] = float(np.nanmedian(terrain))

    scene_dir = output / "scenes" / image_path.stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb8 = stretch_rgb(rgb)
    Image.fromarray(rgb8, "RGB").save(scene_dir / "rgb.jpg", quality=91)
    packed, elev_min, elev_max = pack_rg16(terrain)
    Image.fromarray(packed, "RGB").save(scene_dir / "dsm-rg16.png", compress_level=6)

    col_m = float(np.hypot(transform.a, transform.d))
    row_m = float(np.hypot(transform.b, transform.e))
    return {
        "id": image_path.stem,
        "source": str(image_path),
        "width": width, "height": height,
        "source_width": 7680, "source_height": 13824,
        "crs": crs.to_string(), "transform": list(transform)[:6],
        "pixel_size_m": [col_m, row_m],
        "ground_size_m": [col_m * (width - 1), row_m * (height - 1)],
        "elevation_range_m": [elev_min, elev_max],
        "valid_reference_fraction": float(valid.mean()),
        "surface_source": "ISPRS Vaihingen 09 cm matching reference DSM (zero ML)",
        "warning": "Reference-backed visualization; not monocular-model output.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"C:\Vaihingen\Vaihingen"))
    parser.add_argument("--output", type=Path, default=Path("viewer/web/production/reference-data"))
    parser.add_argument("--max-dimension", type=int, default=768)
    args = parser.parse_args()
    images = sorted((args.root / "Images").glob("*.tif"))
    if not images:
        raise FileNotFoundError(f"no TIFF files under {args.root / 'Images'}")
    records = []
    with rasterio.open(args.root / "DSM" / "DSM_09cm_matching.tif") as dsm:
        for index, image in enumerate(images, 1):
            print(f"[{index}/{len(images)}] {image.name}", flush=True)
            records.append(export_scene(image, dsm, args.output, args.max_dimension))
    args.output.mkdir(parents=True, exist_ok=True)
    index = {
        "title": "Vaihingen original TIFFs over reference DSM",
        "pipeline": "zero-ML reference visualization",
        "scene_count": len(records),
        "warning": "These scenes use the supplied reference DSM and demonstrate rendering quality, not model prediction accuracy.",
        "scenes": records,
    }
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(args.output / "index.json")


if __name__ == "__main__":
    main()
