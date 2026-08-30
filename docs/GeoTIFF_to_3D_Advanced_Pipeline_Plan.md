# AltiMap / DepthWizard - Advanced GeoTIFF-to-3D Implementation Plan

## 1. Objective and final technical decision

Build one production path only:

`georeferenced optical RGB GeoTIFF -> metric nDSM -> terrain-calibrated absolute DSM -> textured, navigable 3D model`

PNG/JPG and non-georeferenced TIFF support are outside this version. The main upload API must reject inputs without a valid CRS and affine transform instead of silently returning relative depth.

The final model stack is:

1. **Depth Anything V2 Small**, frozen, to generate a relative-depth prior.
2. **RDAH-Net**, using the official DFC2019/Track1 checkpoint, to convert RGB plus the relative-depth prior into a metric nDSM (height above local ground).
3. **A terrain/DTM source**, aligned to the GeoTIFF grid, to provide absolute ground elevation.
4. **Absolute DSM composition:** `DSM = DTM + nDSM`.
5. **A metric Three.js/GLB renderer** using the source RGB as the texture and the DSM as elevation.

No monocular paper solves all five steps. RDAH-Net solves the learned nDSM step; geospatial alignment and terrain composition solve absolute elevation; the mesh/viewer solves the final 3D deliverable.

## 2. Audit of the current SlashCompact repository

### What is already useful and should be retained

- GeoTIFF CRS/transform inspection in `viewer/geo.py`.
- Windowed DEM access and source-handle reuse logic in `viewer/dem.py`.
- Float elevation COG/sidecar concept in `src/altimap/contract.py`.
- FastAPI upload and static-app serving pattern in `viewer/server.py`.
- GPU height-texture displacement, orbit mode, first-person controls, probe logic, and local vendoring of Three.js.
- Texture orientation tests, depth-sign tests, DEM-handle regression tests, and pure NumPy refinement utilities.
- The local ISPRS Vaihingen data as a direct GeoTIFF/DSM/3D fixture.

### What the current repo actually does

- The upload path runs **DA3-SMALL raw relative depth**, detrends it, and attempts a linear fit to a coarse DEM.
- Repository diagnostics already show that DA3 often predicts a scene-wide ramp on nadir imagery and that the building-height correlation is effectively absent in tested scenes.
- The latest `export_dem_direct.py` path bypasses AI and drapes the optical image over a coarse external DEM, optionally inserting Overture buildings with known heights.
- This proves rendering and DEM access, but it does **not estimate high-resolution building/tree height from the uploaded optical GeoTIFF**.

### Main gaps to replace

- No RDAH-Net or other trained remote-sensing height model is integrated.
- No production nDSM output exists.
- No reliable `DSM = DTM + nDSM` composition exists in the upload path.
- The documented `eval/`, training pipeline, and installed CLI do not exist in the source tree.
- `pyproject.toml` omits runtime packages used by the server/viewer and points to a nonexistent `altimap.eval.cli`.
- The browser mesh is a fixed square with 3072 segments per axis (about 18.9 million triangles), ignores rectangular ground aspect, and uses normalized rather than metric horizontal coordinates.
- GLB export also normalizes every scene to a 1x1 square and therefore loses true horizontal dimensions and geospatial origin.
- Current `datum="ellipsoidal"` metadata is unsafe: the horizontal CRS does not establish a vertical datum, and DEM sources may be orthometric or unspecified.
- Tests cover helper functions but not model loading, checkpoint preprocessing, tiled inference, API jobs, artifact generation, end-to-end GeoTIFF preservation, or browser performance.

## 3. Research-paper decisions

### 3.1 RDAH-Net - production model for v1

RDAH-Net is the implementable primary model because it directly uses the problem statement's required pretrained monocular backbone pattern:

- Input 1: the original optical RGB image.
- Input 2: a frozen Depth Anything V2 relative-depth prior.
- Separate RGB and depth feature branches.
- CBAM feature refinement.
- Bidirectional cross-modal attention with positional encoding.
- PixelShuffle decoder for sharp, full-resolution prediction.
- Output: metric **nDSM**, not geodetic absolute DSM.

The paper reports 5.37M trainable parameters, 1.54 m in-domain MAE on DFC2019-Track1, and 2.56/2.80 m on its HK/Swiss in-domain tests. Its ablation shows that removing the depth prior roughly doubles DFC2019 MAE (1.54 to 3.04 m), validating the two-input design.

The official MIT-licensed repository and three approximately 65 MB checkpoints are now public. Use Figshare file `63637257` (Track1 checkpoint) for the first integration because DFC2019's 0.35 m WorldView-3 imagery is the closest official checkpoint domain to high-resolution orthophotos.

Important integration constraint: the official repository does not include the script that generated its Depth Anything V2 TIFF priors, and its preprocessing code is inconsistent with parts of the paper's fixed-point description. Therefore the first model milestone is a reproduction gate, not blind checkpoint import.

### 3.2 Depth Anything V2 Small - prior generator only

Use the official 24.8M-parameter Small model because its weights are Apache-2.0 and it is the exact family used by RDAH-Net. Use official raw float32 relative-depth inference with no per-tile min-max normalization unless reproduction testing proves the checkpoint expects another transform. Preserve the prior on the RGB tile's exact grid.

Do not treat its output as metres, terrain elevation, nDSM, or final mesh height.

### 3.3 Depth2Elevation - benchmark and phase-2 architecture ideas

Depth2Elevation fine-tunes a Depth Anything ViT-B using scale adapters, a multi-scale resolution-agnostic decoder, and MSE + scale-aware + gradient losses. It reports strong boundaries and results on GAMUS, DFC2019, and Vaihingen, including better DFC2019 MAE than its reproduced baselines.

It is not the v1 implementation because:

- no official code/checkpoint was located;
- it has about 99.5M parameters and was trained using four RTX 4090 GPUs;
- reproducing it faithfully would add model-research risk before the GeoTIFF pipeline works.

Adopt two ideas later: multi-scale supervision and gradient/boundary loss. Treat a reimplementation as an experiment behind the same `HeightEstimator` interface.

### 3.4 HTC-DC Net - long-tail correction ideas, not a production dependency

HTC-DC addresses the real failure where vast near-zero ground pixels dominate training and tall structures are underestimated. Its Head-Tail Cut, adaptive bins, and foreground/background distribution constraints are useful.

Do not embed the full legacy repository in v1 because it depends on an old, fragile PyTorch/PyTorch3D environment and does not use the required pretrained depth prior. For fine-tuning, adapt its foreground/background weighting and height-bin evaluation into the RDAH training pipeline.

## 4. Final end-to-end data flow

### Step 1 - strict GeoTIFF ingestion

Accept `.tif`/`.tiff` only. Validate:

- at least three usable optical bands;
- non-identity affine transform;
- valid CRS;
- finite bounds and pixel sizes;
- supported numeric dtype;
- nodata/alpha mask;
- maximum pixel count and upload size;
- no unexpected rotation/shear without explicit normalization.

Classify the file as `rgb_geotiff`. A single-band elevation GeoTIFF may be supported only through an explicit `direct_dsm` endpoint; it must never be mistaken for RGB imagery.

### Step 2 - establish a metric processing grid

- Preserve the original source metadata unchanged in the manifest.
- If the source CRS is projected in metres with near-square pixels, process on that grid.
- If it is geographic (degrees) or uses non-metric units, derive a local UTM CRS from the footprint centroid and warp RGB/nodata into it.
- Keep the output GSD near the source GSD; do not invent finer resolution.
- Record both source and processing transforms so output can be reprojected back if requested.

### Step 3 - radiometric preprocessing

- Select RGB bands through color interpretation metadata first, user override second, first-three-bands only as a final fallback.
- Convert 8/16-bit imagery to model RGB using the checkpoint-compatible stretch.
- For the initial checkpoint path: use the official linear range conversion and ImageNet normalization.
- Store stretch percentiles/min-max in the manifest.
- Set nodata pixels to zero only after preserving a boolean validity mask.

### Step 4 - overlapping tiled inference

- Default tile: 512x512 pixels.
- Overlap: 128 pixels (25%).
- Reflect-pad edge tiles to a valid size; never resize them independently.
- Run Depth Anything V2 Small in evaluation/inference mode to produce a float32 relative-depth prior on the tile grid.
- Run RDAH-Net with `[relative_depth, normalized_rgb]` to produce metric nDSM.
- Use mixed precision on CUDA; concurrency is one GPU job on the RTX 3050 6 GB.
- Clamp only physically invalid negative nDSM values to zero. Do not smooth or threshold structures by default.
- Blend overlaps with a separable raised-cosine weight map; outer image borders receive nonzero weights.
- Accumulate weighted sum and weight sum in float32, then divide once after all tiles.

### Step 5 - uncertainty and quality layers

Produce a confidence raster based on:

- disagreement among overlapping tile predictions;
- test-time flip consistency for an optional high-quality mode;
- out-of-domain warnings from GSD and radiometric statistics;
- invalid/nodata coverage.

Confidence is diagnostic, not a fabricated probability. The manifest must state how it was calculated.

### Step 6 - acquire and align terrain

Terrain-source order:

1. user-supplied DTM GeoTIFF;
2. USGS 3DEP seamless bare-earth DEM for covered US regions;
3. SRTM 30 m within its coverage;
4. Copernicus GLO-30 fallback.

For SRTM/GLO-30, mark the result as an approximate terrain base because these products may retain canopy/building surface effects. Apply only low-frequency ground filtering/morphological opening; never force high-resolution nDSM structures to match the coarse DEM.

- Reproject terrain to the processing CRS/grid.
- Fill small terrain voids with constrained interpolation.
- Preserve the source vertical datum when known.
- If unknown, store `vertical_datum: unknown` rather than claiming ellipsoidal height.

### Step 7 - compose absolute surface height

Compute:

`absolute_DSM_m = aligned_DTM_m + predicted_nDSM_m`

Output all three layers because each has a different meaning:

- `ndsm.tif`: predicted height above local ground in metres;
- `dtm.tif`: aligned terrain base in source vertical datum;
- `dsm.tif`: final absolute surface elevation;
- `confidence.tif`: quality indicator.

All elevation rasters are Float32 Cloud-Optimized GeoTIFFs with CRS, affine transform, nodata, internal overviews, compression, and provenance tags.

### Step 8 - construct true-scale 3D

Use a local metric coordinate frame to avoid float precision loss from large UTM coordinates:

- origin = processing-grid centre or lower-left, stored in `manifest.json` and GLB `extras`;
- local X = pixel easting offset in metres;
- local Z = negative northing/row offset in metres;
- local Y = DSM minus a recorded base elevation;
- vertical and horizontal units are both metres at 1x exaggeration;
- retain the rectangular footprint and any source-grid rotation.

Replace the fixed 3072x3072 square with chunked geometry:

- 256x256-cell chunks;
- per-chunk skirts to hide LOD cracks;
- 3 LOD levels (1x, 2x, 4x sampling);
- browser renders visible chunks only;
- exported GLB is decimated to at most about one million vertices by default while preserving edges and aspect ratio.

The source RGB is texture-mapped by the identical grid/UV relationship. Metric normals and slope must use the X/Y pixel spacing separately, not normalized UV distance.

### Step 9 - viewer and analysis tools

The single production viewer must provide:

- orbit and pointer-lock flythrough;
- true-scale and vertical-exaggeration modes;
- RGB, elevation, slope, confidence, and optional error overlays;
- click probe returning easting, northing, DSM elevation, terrain elevation, nDSM height, confidence, and physical slope;
- two-click elevation profile/cross-section;
- output download links for nDSM, DTM, DSM, confidence, GLB, and manifest;
- reference DSM upload for signed error, MAE, RMSE, bias, and correlation.

### Step 10 - asynchronous job API

Use an in-process bounded GPU queue for the hackathon deployment; do not add Celery/Redis.

- `POST /api/jobs` - RGB GeoTIFF plus optional DTM and processing options; returns job ID.
- `GET /api/jobs/{id}` - state (`queued`, `validating`, `inference`, `terrain`, `export`, `ready`, `failed`) and progress.
- `GET /api/jobs/{id}/manifest` - complete metadata and artifact links.
- `GET /api/jobs/{id}/artifacts/{name}` - streamed artifact download.
- `DELETE /api/jobs/{id}` - remove one job and its temporary/output directory.
- `GET /api/health` - GPU, model/checkpoint readiness, disk, and queue status.

Only one inference worker uses CUDA. CPU geospatial export may run separately after GPU inference completes.

## 5. Repository restructuring

Move production code from top-level experimental scripts into the installed package:

```text
src/altimap/
  api/             FastAPI routes, jobs, schemas
  geo/             ingest, CRS/grid, terrain sources, COG writer
  ml/              DAV2 adapter, RDAH model, checkpoint manager, tiling
  pipeline/        orchestration, composition, artifact manifest
  mesh/            local-frame mesh, GLB export, LOD metadata
  eval/            alignment, metrics, report CLI
viewer/web/        one production Three.js application
scripts/           model/data preparation and Vaihingen fixtures
tests/             unit, integration, model, API, and browser smoke tests
docs/              architecture, model card, dataset card, runbook
```

Keep the existing DA3 experiments and DEM-direct route under `experiments/` or preserve them in Git history; they are useful baselines but must not remain the main product path.

Unify dependencies under Python 3.12. Use a locked base environment and a CUDA ML extra. Production dependencies must explicitly include Rasterio, pyproj, FastAPI, Uvicorn, python-multipart, Pillow, trimesh, PyTorch, torchvision, OpenCV, and the Depth Anything V2 adapter. Do not pull HTC-DC/PyTorch3D into production.

## 6. Model/checkpoint reproduction gate

Before integrating RDAH-Net into the API:

1. Download the official Track1 checkpoint from Figshare and verify its published MD5.
2. Port the MIT-licensed network into `src/altimap/ml/rdah.py`; do not invoke the research `test.py` as a subprocess.
3. Load `model_state_dict` strictly and assert every key/shape matches.
4. Reproduce official RGB normalization and raw Depth Anything V2 prior generation.
5. Evaluate the public DFC2019 Track1 test split.
6. Accept the port only if MAE is within 0.15 m of the paper's 1.54 m result, or document a verified dataset/version reason for the gap.
7. Freeze this preprocessing in golden fixtures and record checkpoint hash, DAV2 hash, input range, and package versions.

If exact reproduction is blocked by the missing prior-generation details, compare raw float32 DAV2, inverted raw depth, and documented fixed-point variants on a small validation subset and select only the transform that reproduces the published checkpoint. Do not tune preprocessing against the final held-out test set.

## 7. Vaihingen dataset usage

Use the local data in `C:\Vaihingen\Vaihingen` for two separate purposes.

### Infrastructure truth fixture

Start with area 11:

- RGB: `top/top_mosaic_09cm_area11.tif` from `ISPRS_semantic_labeling_Vaihingen.zip`;
- DSM: `dsm/dsm_09cm_matching_area11.tif`;
- georeferencing: `dsm_09cm_matching_area11.tfw` from `Vaihingen_dsm_tiles_geoinfo.zip`.

This pair proves CRS/affine handling, true-scale mesh construction, texture alignment, metric probes, and GLB export before AI is involved.

### Height-model validation/fine-tuning

The provided DSM is absolute surface elevation, not nDSM. Build a reference DTM from the supplied ALS `.LAS` strips using ground classification or a CSF/SMRF ground filter, rasterize it to the DSM grid, and derive:

`reference_nDSM = reference_DSM - reference_DTM`

Use semantic labels for building/vegetation/ground-specific metrics and masks, not as a substitute for elevation. Split by whole numbered areas so overlapping crops from one area never cross train/test boundaries.

Treat the current CRS conflict explicitly: world files/shapefiles indicate Vaihingen's UTM context while one VRT reports a conflicting zone. Resolve it against known bounds/control geometry once, record the chosen EPSG, and never silently copy contradictory metadata.

## 8. Training and phase-2 improvements

The first end-to-end release uses the official RDAH checkpoint. Fine-tune only after the reproduction and evaluation harness are working.

Fine-tuning recipe for the 6 GB RTX 3050:

- precompute frozen DAV2 priors;
- 512x512 crops, 128 overlap for validation reconstruction;
- batch size 1 or 2, AMP, gradient accumulation;
- city/area-held-out validation;
- baseline SmoothL1/L1 in metres;
- add Depth2Elevation-style gradient loss and multi-scale supervision;
- add HTC-inspired foreground/background weighting or height-bin-balanced sampling;
- report all-pixel, building, vegetation, and height-bin metrics separately.

Do not claim an improvement unless it beats the frozen official checkpoint on a held-out geographic area and does not worsen tile seams.

## 9. Evaluation and acceptance criteria

### Geospatial correctness

- Output DSM/DTM/nDSM grids have the intended CRS, transform, bounds, dimensions, and nodata.
- Source-to-output coordinate round trip is within half a processing pixel.
- DSM equals DTM + nDSM within float tolerance on every valid pixel.
- No NaN/nodata contamination enters valid statistics.
- GLB sample vertices agree with the DSM within 0.2 m after applying stored origin/base elevation.

### Model accuracy

- Reproduce RDAH DFC2019 MAE within 0.15 m of 1.54 m before adapting it.
- Report MAE, RMSE, Pearson/Spearman correlation, bias, P50/P90 absolute error, and boundary-gradient error.
- Report ground, building, vegetation, low-height, and tall-height strata.
- Compare against: DTM-only, raw DAV2 + affine DEM fit, current DA3 pipeline, official RDAH checkpoint, and any fine-tuned model.
- The production model must beat the current raw-depth affine baseline by at least 20% MAE on the selected held-out set and must preserve positive building-height correlation.

### Tiling and rendering

- Tile/stitch identity test reproduces a synthetic raster exactly when inference is replaced by identity.
- Median disagreement in overlap regions is below 0.25 m after blending.
- No visible seams in the area-11 3D fixture.
- Viewer sustains at least 45 FPS on the RTX 3050 demo machine with the standard test scene and does not allocate a fixed 18.9M-triangle mesh.
- Height/slope readouts are invariant to vertical exaggeration.

### API and deployment

- Valid area-11 GeoTIFF reaches `ready` and all artifacts reopen successfully.
- Missing CRS, identity transform, one-band imagery, empty upload, oversized raster, no DEM coverage, model failure, and disk exhaustion return explicit errors.
- Model/checkpoint download is resumable, checksum-verified, and cached outside job directories.
- A fresh documented installation can run the demo without manually editing paths.

## 10. Ordered implementation milestones

1. **Baseline freeze:** tag the current repository state; preserve DA3 and DEM-direct outputs as comparison fixtures.
2. **Packaging repair:** correct dependencies, package all runtime modules, replace the broken CLI entry point, and make tests runnable from a clean `uv` environment.
3. **GeoTIFF core:** strict ingest, metric-grid normalization, nodata/radiometry, COG writer, and expanded manifest.
4. **RDAH reproduction:** DAV2 prior adapter, official checkpoint loader, DFC2019 reproduction report, and golden tests.
5. **Large-raster inference:** overlap tiling, cosine blending, progress, uncertainty, and GPU memory guards.
6. **Absolute DSM:** terrain-source abstraction, reprojection, vertical-datum provenance, composition, and artifact validation.
7. **Metric 3D:** rectangular local-metre frame, chunked LOD, texture alignment, GLB extras, and mesh validation.
8. **Unified app:** asynchronous job API plus one GeoTIFF-only viewer with metric tools and downloads.
9. **Vaihingen validation:** area-11 direct fixture, ALS-derived DTM/nDSM, whole-area splits, and baseline report.
10. **Optional fine-tuning:** Depth2Elevation/HTC-inspired losses and domain adaptation only after the complete baseline is reproducible.
11. **Standalone delivery:** Docker/Windows run scripts, model cache bootstrap, technical documentation, model card, dataset card, and scripted judge demo.

## 11. Explicit non-goals for this version

- PNG/JPG or metadata-free TIFF processing.
- Stereo, multi-view, LiDAR-at-inference, or InSAR reconstruction.
- Claiming survey-grade accuracy from monocular imagery.
- Using coarse SRTM/GLO-30 as high-resolution DSM truth.
- Treating RDAH nDSM as absolute geodetic elevation without a terrain base.
- Reimplementing Depth2Elevation before the official RDAH baseline works.
- Depending on Overture building heights for every building; they remain an optional validation/augmentation source.

## 12. Required references and assets

- RDAH-Net paper and MIT implementation: https://github.com/Elenairene/RDAH-Net
- RDAH checkpoints: https://doi.org/10.6084/m9.figshare.31986864
- Depth Anything V2 Small: https://github.com/DepthAnything/Depth-Anything-V2
- HTC-DC Net paper/code: https://github.com/zhu-xlab/HTC-DC-Net
- Depth2Elevation paper: https://doi.org/10.1109/TGRS.2025.3564820
- ISPRS Vaihingen benchmark and local dataset under `C:\Vaihingen\Vaihingen`

