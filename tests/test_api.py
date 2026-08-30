from __future__ import annotations

import time

import numpy as np
import rasterio
from fastapi.testclient import TestClient

from altimap.api import create_app


class TinyModel:
    name = "tiny-test-model"
    def predict_ndsm(self, rgb):
        return np.ones(rgb.shape[:2], np.float32), None


def _write_inputs(tmp_path, transform, crs):
    rgb, dtm = tmp_path / "rgb.tif", tmp_path / "dtm.tif"
    with rasterio.open(rgb, "w", driver="GTiff", width=6, height=6, count=3, dtype="uint8", crs=crs, transform=transform) as ds:
        ds.write(np.full((3, 6, 6), 80, np.uint8))
    with rasterio.open(dtm, "w", driver="GTiff", width=6, height=6, count=1, dtype="float32", crs=crs, transform=transform) as ds:
        ds.write(np.full((1, 6, 6), 100, np.float32))
    return rgb, dtm


def test_job_api_creates_artifacts(tmp_path, test_transform, test_crs):
    rgb, dtm = _write_inputs(tmp_path, test_transform, test_crs)
    app = create_app(tmp_path / "jobs", TinyModel)
    with TestClient(app) as client:
        with rgb.open("rb") as image_file, dtm.open("rb") as dtm_file:
            response = client.post("/api/jobs", files={"image": ("rgb.tif", image_file, "image/tiff"),
                "dtm": ("dtm.tif", dtm_file, "image/tiff")}, data={"dtm_source": "test", "vertical_datum": "orthometric"})
        assert response.status_code == 202
        job = response.json()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] != "queued" and job["status"] != "running": break
            time.sleep(.02)
        assert job["status"] == "complete", job
        assert client.get(job["manifest"]).status_code == 200
        assert client.get(job["artifacts"]["dsm.tif"]).status_code == 200
