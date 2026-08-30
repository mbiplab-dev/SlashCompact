"""FastAPI façade around the deterministic GeoTIFF pipeline.

It intentionally accepts a DTM on the exact input grid for the first release.
That makes datum provenance inspectable and avoids pretending that a remote DEM
download is accurate or even available for every scene.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from altimap.geo.ingest import RasterValidationError, inspect_geotiff
from altimap.pipeline.core import GeoTiffPipeline, HeightModel, PipelineResult


@dataclass
class Job:
    id: str
    directory: Path
    status: str = "queued"
    error: str | None = None
    result: PipelineResult | None = None
    future: Future | None = field(default=None, repr=False)

    def public(self) -> dict:
        data = {"id": self.id, "status": self.status, "error": self.error}
        if self.result:
            data["manifest"] = f"/api/jobs/{self.id}/manifest"
            data["artifacts"] = {name: f"/api/jobs/{self.id}/artifacts/{name}" for name in
                ("ndsm.tif", "dtm.tif", "dsm.tif", "confidence.tif", "terrain.glb", "dsm-rg16.png", "rgb.jpg")}
        return data


def _same_grid(image: Path, dtm: Path) -> np.ndarray:
    source = inspect_geotiff(image)
    with rasterio.open(dtm) as ref:
        if ref.count != 1:
            raise RasterValidationError("DTM must be a single-band float elevation GeoTIFF")
        if ref.width != source.width or ref.height != source.height or ref.crs != source.crs or ref.transform != source.transform:
            raise RasterValidationError("DTM must already match RGB GeoTIFF CRS, transform, width, and height")
        array = ref.read(1).astype(np.float32)
        if ref.nodata is not None and not np.isnan(ref.nodata):
            array[array == ref.nodata] = np.nan
        return array


def create_app(root: str | Path, model_factory: Callable[[], HeightModel], static_dir: str | Path | None = None) -> FastAPI:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, Job] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="altimap-gpu")
    lock = threading.Lock()
    app = FastAPI(title="AltiMap GeoTIFF-to-3D", version="1.0")

    def process(job: Job, image: Path, dtm_path: Path, dtm_source: str, datum: str) -> None:
        job.status = "running"
        try:
            dtm = _same_grid(image, dtm_path)
            job.result = GeoTiffPipeline(model_factory()).run(image, dtm, job.directory, dtm_source, datum)
            job.status = "complete"
        except Exception as exc:  # errors must become visible to a browser client
            job.status, job.error = "failed", str(exc)

    @app.get("/api/health")
    def health() -> dict:
        with lock:
            return {"status": "ok", "queue_depth": sum(j.status == "queued" for j in jobs.values()),
                    "active": sum(j.status == "running" for j in jobs.values()), "max_concurrent_gpu_jobs": 1}

    @app.post("/api/jobs", status_code=202)
    async def create_job(image: UploadFile = File(...), dtm: UploadFile = File(...),
                         dtm_source: str = Form(...), vertical_datum: str = Form("unknown")) -> dict:
        if Path(image.filename or "").suffix.lower() not in {".tif", ".tiff"}:
            raise HTTPException(415, "image must be a georeferenced RGB GeoTIFF (.tif/.tiff)")
        if Path(dtm.filename or "").suffix.lower() not in {".tif", ".tiff"}:
            raise HTTPException(415, "dtm must be a GeoTIFF (.tif/.tiff)")
        if vertical_datum not in {"ellipsoidal", "orthometric", "unknown"}:
            raise HTTPException(422, "vertical_datum must be ellipsoidal, orthometric, or unknown")
        job = Job(uuid.uuid4().hex, root / uuid.uuid4().hex)
        # Avoid unrelated user-controlled filename paths.
        job.directory.mkdir(parents=True)
        rgb_path, dtm_path = job.directory / "source_rgb.tif", job.directory / "source_dtm.tif"
        rgb_path.write_bytes(await image.read()); dtm_path.write_bytes(await dtm.read())
        try:
            inspect_geotiff(rgb_path)
        except (RasterValidationError, rasterio.errors.RasterioError) as exc:
            shutil.rmtree(job.directory)
            raise HTTPException(422, str(exc)) from exc
        with lock:
            jobs[job.id] = job
            job.future = executor.submit(process, job, rgb_path, dtm_path, dtm_source, vertical_datum)
        return job.public()

    def get_job(job_id: str) -> Job:
        with lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        return get_job(job_id).public()

    @app.get("/api/jobs/{job_id}/manifest")
    def manifest(job_id: str) -> dict:
        job = get_job(job_id)
        if not job.result:
            raise HTTPException(409, "job has not completed")
        return json.loads(job.result.manifest_path.read_text(encoding="utf-8"))

    @app.get("/api/jobs/{job_id}/artifacts/{artifact}")
    def artifact(job_id: str, artifact: str):
        job = get_job(job_id)
        if not job.result or artifact not in {"ndsm.tif", "dtm.tif", "dsm.tif", "confidence.tif", "terrain.glb", "dsm-rg16.png", "rgb.jpg", "manifest.json"}:
            raise HTTPException(404, "artifact not available")
        path = job.directory / artifact
        if not path.is_file():
            raise HTTPException(404, "artifact not available")
        return FileResponse(path)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        job = get_job(job_id)
        if job.status == "running":
            raise HTTPException(409, "cannot delete a running job")
        with lock:
            del jobs[job_id]
        shutil.rmtree(job.directory, ignore_errors=True)

    if static_dir is not None:
        static = Path(static_dir)
        if static.is_dir():
            vendor = static.parent / "vendor"
            if vendor.is_dir():
                app.mount("/vendor", StaticFiles(directory=vendor), name="viewer-vendor")
            app.mount("/", StaticFiles(directory=static, html=True), name="viewer")
    return app
