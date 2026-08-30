import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from altimap.contract import Sidecar, read_elevation, write_elevation_cog


def test_sidecar_roundtrip(tmp_path: Path) -> None:
    sidecar = Sidecar(
        gsd_m=2.0,
        source_gsd_m=0.6,
        datum="ellipsoidal",
        vertical_unit="m",
        model_version="test-v1",
        height_range_m=(0.0, 84.3),
        tile_overlap_px=130,
        dtm_source="3dep-seamless",
    )
    path = tmp_path / "dsm.json"
    sidecar.to_json(path)
    assert Sidecar.from_json(path) == sidecar


def test_sidecar_rejects_bad_datum() -> None:
    with pytest.raises(ValueError, match="datum"):
        Sidecar(
            gsd_m=2.0,
            source_gsd_m=0.6,
            datum="nonsense",
            vertical_unit="m",
            model_version="test-v1",
            height_range_m=(0.0, 1.0),
            tile_overlap_px=0,
            dtm_source=None,
        )


def test_elevation_cog_roundtrip_preserves_georeferencing(tmp_path: Path) -> None:
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(500000.0, 4500000.0, 2.0, 2.0)
    crs = CRS.from_epsg(32612)
    sidecar = Sidecar(
        gsd_m=2.0,
        source_gsd_m=0.6,
        datum="ellipsoidal",
        vertical_unit="m",
        model_version="test-v1",
        height_range_m=(1.0, 4.0),
        tile_overlap_px=0,
        dtm_source=None,
    )
    path = tmp_path / "dsm.tif"
    write_elevation_cog(path, array, transform, crs, sidecar)

    out, out_transform, out_crs = read_elevation(path)
    np.testing.assert_allclose(out, array)
    assert out_transform == transform
    assert out_crs == crs
    assert (tmp_path / "dsm.json").exists()
    with rasterio.open(path) as src:
        assert src.driver == "GTiff"  # COG is serialized as TIFF with COG layout.
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"


def test_nodata_becomes_nan(tmp_path: Path) -> None:
    array = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 0.0, 2.0, 2.0)
    sidecar = Sidecar(
        gsd_m=2.0,
        source_gsd_m=0.6,
        datum="relative",
        vertical_unit="m",
        model_version="test-v1",
        height_range_m=(1.0, 4.0),
        tile_overlap_px=0,
        dtm_source=None,
    )
    path = tmp_path / "dsm.tif"
    write_elevation_cog(path, array, transform, CRS.from_epsg(32612), sidecar)

    out, _, _ = read_elevation(path)
    assert np.isnan(out[0, 1])
    assert out[0, 0] == 1.0
