"""Export a real-metre terrain mesh instead of a normalized display plane."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from altimap.geo.ingest import GeoTiffInput


def _indices(height: int, width: int) -> np.ndarray:
    grid = np.arange(height * width).reshape(height, width)
    a, b, c, d = grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel(), grid[1:, :-1].ravel(), grid[1:, 1:].ravel()
    return np.concatenate((np.c_[a, c, b], np.c_[b, c, d])).astype(np.int64)


def _sample(arr: np.ndarray, max_cells: int) -> np.ndarray:
    stride = max(1, int(np.ceil(max(arr.shape) / max_cells)))
    sampled = arr[::stride, ::stride]
    return sampled if sampled.shape[0] > 1 and sampled.shape[1] > 1 else arr


def export_metric_terrain(dsm: np.ndarray, rgb: np.ndarray, meta: GeoTiffInput, output: str | Path,
                          max_cells: int = 512) -> None:
    """Write GLB whose X/Z axes are projected metres and Y is DSM metres.

    The mesh stores an origin in its extras so a viewer can use small local
    coordinates while preserving full projected coordinates in the manifest.
    """
    import trimesh

    sampled_dsm = _sample(np.asarray(dsm, dtype=np.float32), max_cells)
    sampled_rgb = _sample(np.asarray(rgb), max_cells)
    h, w = sampled_dsm.shape
    sx = (meta.width - 1) / max(w - 1, 1) * meta.pixel_size_x_m
    sy = (meta.height - 1) / max(h - 1, 1) * meta.pixel_size_y_m
    xx, zz = np.meshgrid(np.arange(w, dtype=np.float32) * sx, -np.arange(h, dtype=np.float32) * sy)
    base = float(np.nanmin(sampled_dsm)) if np.isfinite(sampled_dsm).any() else 0.0
    yy = np.nan_to_num(sampled_dsm, nan=base) - base
    uvx, uvy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    image = Image.fromarray(sampled_rgb.astype(np.uint8), "RGB")
    mesh = trimesh.Trimesh(vertices=np.c_[xx.ravel(), yy.ravel(), zz.ravel()], faces=_indices(h, w),
                           visual=trimesh.visual.TextureVisuals(uv=np.c_[uvx.ravel(), uvy.ravel()],
                           material=trimesh.visual.material.SimpleMaterial(image=image)), process=False)
    mesh.metadata.update({"altimap_origin_projected_m": [meta.transform.c, meta.transform.f, base],
                          "altimap_crs": meta.crs.to_string(), "altimap_horizontal_units": "m",
                          "altimap_vertical_units": "m"})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)
    output.with_suffix(".mesh.json").write_text(json.dumps(mesh.metadata, indent=2) + "\n", encoding="utf-8")
