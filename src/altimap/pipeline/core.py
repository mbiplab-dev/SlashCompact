"""The deterministic GeoTIFF -> nDSM/DTM/DSM processing contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from altimap.contract import Sidecar, write_elevation_cog
from altimap.geo.ingest import GeoTiffInput, read_geotiff_rgb
from altimap.mesh.terrain import export_metric_terrain
from altimap.pipeline.tiling import TileBlender, tiles


class HeightModel(Protocol):
    name: str
    def predict_ndsm(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...


@dataclass(frozen=True)
class PipelineConfig:
    tile_size: int = 512
    tile_overlap_px: int = 128
    mesh_cells: int = 512
    vertical_datum: str = "unknown"


@dataclass(frozen=True)
class PipelineResult:
    job_dir: Path
    manifest_path: Path
    ndsm_path: Path
    dtm_path: Path
    dsm_path: Path
    confidence_path: Path
    mesh_path: Path


class GeoTiffPipeline:
    """Produces metric DSM by composing metric ground with predicted nDSM.

    The model predicts *height above local ground*, not sensor depth.  This
    explicit composition is what makes the output a DSM in metres.
    """
    def __init__(self, model: HeightModel, config: PipelineConfig = PipelineConfig()) -> None:
        self.model, self.config = model, config

    def _predict(self, rgb: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blend = TileBlender(rgb.shape[:2])
        confidence = TileBlender(rgb.shape[:2])
        for tile in tiles(rgb.shape[:2], self.config.tile_size, self.config.tile_overlap_px):
            predicted, conf = self.model.predict_ndsm(rgb[tile.y0:tile.y1, tile.x0:tile.x1])
            predicted = np.asarray(predicted, dtype=np.float32)
            if predicted.shape != (tile.y1 - tile.y0, tile.x1 - tile.x0):
                raise ValueError("height model returned a shape different from the image tile")
            blend.add(tile, np.maximum(predicted, 0.0))
            confidence.add(tile, np.ones_like(predicted) if conf is None else np.asarray(conf, dtype=np.float32))
        ndsm, conf = blend.finish(), confidence.finish()
        ndsm[~valid], conf[~valid] = np.nan, np.nan
        return ndsm, np.clip(conf, 0.0, 1.0)

    def run(self, image_path: str | Path, dtm: np.ndarray, output_dir: str | Path,
            dtm_source: str, dtm_vertical_datum: str = "unknown") -> PipelineResult:
        meta, rgb, valid = read_geotiff_rgb(image_path)
        if dtm.shape != (meta.height, meta.width):
            raise ValueError("DTM must be resampled onto the exact RGB GeoTIFF grid before composition")
        dtm = np.asarray(dtm, dtype=np.float32).copy()
        dtm[~valid] = np.nan
        ndsm, confidence = self._predict(rgb, valid)
        dsm = dtm + ndsm
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {name: out / f"{name}.tif" for name in ("ndsm", "dtm", "dsm", "confidence")}
        ranges = {name: _range(arr) for name, arr in {"ndsm": ndsm, "dtm": dtm, "dsm": dsm, "confidence": confidence}.items()}
        for name, array in (("ndsm", ndsm), ("dtm", dtm), ("dsm", dsm), ("confidence", confidence)):
            datum = "relative" if name == "ndsm" else (
                dtm_vertical_datum if dtm_vertical_datum in {"ellipsoidal", "orthometric"} else "unknown")
            write_elevation_cog(paths[name], array, meta.transform, meta.crs, Sidecar(
                gsd_m=meta.gsd_m, source_gsd_m=meta.gsd_m, datum=datum, vertical_unit="m",
                model_version=self.model.name, height_range_m=ranges[name],
                tile_overlap_px=self.config.tile_overlap_px, dtm_source=dtm_source,
            ))
        mesh_path = out / "terrain.glb"
        export_metric_terrain(dsm, rgb, meta, mesh_path, max_cells=self.config.mesh_cells)
        _write_viewer_previews(out, dsm, rgb)
        manifest_path = out / "manifest.json"
        manifest = _manifest(meta, self, paths, mesh_path, dtm_source, dtm_vertical_datum, ranges)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return PipelineResult(out, manifest_path, paths["ndsm"], paths["dtm"], paths["dsm"], paths["confidence"], mesh_path)


def _range(array: np.ndarray) -> tuple[float, float]:
    values = array[np.isfinite(array)]
    return (float(values.min()), float(values.max())) if values.size else (0.0, 0.0)


def _manifest(meta: GeoTiffInput, pipeline: GeoTiffPipeline, paths: dict[str, Path], mesh_path: Path,
              dtm_source: str, vertical_datum: str, ranges: dict[str, tuple[float, float]]) -> dict:
    return {"schema_version": "1.0", "input": {"path": meta.path.name, "crs": meta.crs.to_string(),
            "transform": list(meta.transform)[:6], "width": meta.width, "height": meta.height,
            "bounds": meta.bounds, "gsd_m": meta.gsd_m}, "model": {"name": pipeline.model.name,
            "tile_size": pipeline.config.tile_size, "tile_overlap_px": pipeline.config.tile_overlap_px},
            "vertical_reference": {"dtm_source": dtm_source, "datum": vertical_datum,
            "warning": "Vertical datum is source-defined; do not treat it as ellipsoidal unless verified."},
            "artifacts": {name: {"path": path.name, "range_m": ranges[name]} for name, path in paths.items()} |
                         {"terrain_glb": {"path": mesh_path.name, "horizontal_units": "m", "vertical_units": "m"},
                          "viewer_depth": {"path": "dsm-rg16.png", "encoding": "rg16-linear", "range_m": ranges["dsm"]},
                          "viewer_rgb": {"path": "rgb.jpg"}},
            "invariant": "dsm.tif = dtm.tif + ndsm.tif on the same pixel grid"}


def _write_viewer_previews(out: Path, dsm: np.ndarray, rgb: np.ndarray) -> None:
    """Small, lossless browser assets; COGs remain the authoritative outputs."""
    values = dsm[np.isfinite(dsm)]
    lo, hi = (float(values.min()), float(values.max())) if values.size else (0.0, 0.0)
    normalized = np.nan_to_num((dsm - lo) / max(hi - lo, 1e-6), nan=0.0)
    encoded = np.rint(np.clip(normalized, 0, 1) * 65535).astype(np.uint16)
    high = (encoded >> 8).astype(np.uint8)
    low = (encoded & 255).astype(np.uint8)
    # Store the 16-bit value in the PNG's R/G colour channels.  Using Pillow
    # mode "LA" puts the second byte in alpha; a browser canvas then expands
    # luminance into R/G/B and the viewer reads two copies of the high byte,
    # creating false stair-step spikes despite a correct source DSM.
    Image.fromarray(np.dstack((high, low, np.zeros_like(high))), "RGB").save(out / "dsm-rg16.png")
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(out / "rgb.jpg", quality=92)
