from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from altimap.contract import read_elevation
from altimap.geo.ingest import RasterValidationError, read_geotiff_rgb
from altimap.pipeline.core import GeoTiffPipeline, PipelineConfig
from altimap.pipeline.tiling import TileBlender, Tile, tiles


class ConstantModel:
    name = "test-constant-ndsm"

    def predict_ndsm(self, rgb: np.ndarray):
        return np.full(rgb.shape[:2], 4.0, np.float32), np.full(rgb.shape[:2], 0.75, np.float32)


def _rgb_tif(path: Path, transform, crs) -> None:
    arr = np.zeros((3, 9, 11), dtype=np.uint16)
    arr[0] = 100; arr[1] = 1000; arr[2] = 5000
    with rasterio.open(path, "w", driver="GTiff", height=9, width=11, count=3, dtype="uint16",
                       crs=crs, transform=transform, nodata=0) as dst:
        dst.write(arr)


def test_pipeline_preserves_grid_and_composes_metric_dsm(tmp_path, test_transform, test_crs) -> None:
    image = tmp_path / "rgb.tif"; _rgb_tif(image, test_transform, test_crs)
    dtm = np.arange(99, dtype=np.float32).reshape(9, 11)
    result = GeoTiffPipeline(ConstantModel(), PipelineConfig(tile_size=5, tile_overlap_px=2, mesh_cells=8)).run(
        image, dtm, tmp_path / "job", "test-dtm", "orthometric")
    ndsm, tx, crs = read_elevation(result.ndsm_path)
    dsm, _, _ = read_elevation(result.dsm_path)
    np.testing.assert_allclose(ndsm, 4.0)
    np.testing.assert_allclose(dsm, dtm + ndsm)
    assert tx == test_transform and crs == test_crs
    assert result.mesh_path.exists()
    assert (result.job_dir / "dsm-rg16.png").exists()
    assert (result.job_dir / "rgb.jpg").exists()
    packed = np.asarray(Image.open(result.job_dir / "dsm-rg16.png").convert("RGB"))
    assert not np.array_equal(packed[..., 0], packed[..., 1])
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["invariant"] == "dsm.tif = dtm.tif + ndsm.tif on the same pixel grid"
    assert manifest["vertical_reference"]["datum"] == "orthometric"


def test_ingest_rejects_ungeoreferenced_or_non_rgb(tmp_path, test_transform, test_crs) -> None:
    wrong = tmp_path / "wrong.tif"
    with rasterio.open(wrong, "w", driver="GTiff", height=5, width=5, count=1, dtype="uint8",
                       crs=test_crs, transform=test_transform) as dst:
        dst.write(np.ones((1, 5, 5), dtype=np.uint8))
    try:
        read_geotiff_rgb(wrong)
    except RasterValidationError as exc:
        assert "three" in str(exc)
    else:
        raise AssertionError("single-band TIFF should fail validation")


def test_tiling_blends_every_pixel_and_covers_edge() -> None:
    planned = tiles((17, 21), size=8, overlap=3)
    assert planned[-1].x1 == 21
    blender = TileBlender((17, 21))
    for tile in planned:
        blender.add(tile, np.full((tile.y1 - tile.y0, tile.x1 - tile.x0), 3, np.float32))
    np.testing.assert_allclose(blender.finish(), 3.0)
