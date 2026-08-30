# AltiMap — satellite image to measurable 3D terrain

AltiMap turns a **georeferenced RGB satellite/orthophoto GeoTIFF** into a 3D
terrain that can be inspected in a browser. It is built for the SIH DepthWizard
problem: make an elevation map and a navigable 3D scene from one optical image.

## What you see in the browser

The page has two panels deliberately placed side by side:

1. **Original 2D satellite image** — this is the exact image the model receives.
2. **Interactive 3D terrain** — the same satellite image is placed over the
   calculated height surface. Drag to rotate, scroll to zoom, and click a point
   to see its map position, surface height, and local slope.

If the terrain looks too sharp or “spiky”, use **Visual vertical scale**. It
only changes the display; it never changes the real height values or downloaded
files. Spikes can come from abrupt model height estimates or a noisy ground
file, so the original 2D panel is there to judge whether a ridge is likely a
building/tree or an artefact.

## What you need to upload

You upload two GeoTIFF files that already cover the exact same place and use
the exact same pixel grid.

| Upload field | Plain meaning | Why AltiMap needs it |
| --- | --- | --- |
| **Satellite RGB GeoTIFF** | A normal colour aerial/satellite image with map coordinates embedded. | This provides roads, roofs, trees, and other visible structure. |
| **Ground-elevation GeoTIFF (DTM)** | The estimated height of bare ground at each pixel, with buildings and trees removed. | A single photo cannot know absolute height on its own. This anchors the result in metres. |
| **Ground-file source** | A short note such as `Vaihingen ALS lower-envelope provisional DTM`. | It records where the ground reference came from. |
| **Height reference** | Whether heights are relative to sea level, GPS height, or not documented. | Prevents us from falsely claiming a vertical reference that the data does not provide. |

PNG/JPG and ordinary TIFF images are rejected because they do not carry the
coordinate system and pixel-to-ground mapping required for a measurable 3D
model.

## What happens after clicking “Create 3D terrain”

1. AltiMap checks that both files have the same coordinate system, map position,
   image size, and pixel size.
2. **Depth Anything V2 Small** reads the RGB image and creates a relative-depth
   hint: it understands what appears closer/farther or taller/lower, but not
   real metres.
3. **RDAH-Net** combines that hint with the original RGB image and estimates an
   **nDSM**: height of buildings and trees above local ground.
4. AltiMap combines that estimate with the ground file:

   `Final surface height (DSM) = ground height (DTM) + above-ground height (nDSM)`

5. It uses the final surface height to make a real-metre 3D mesh and drapes the
   original satellite image across it as the texture.

The server runs one job at a time. This is intentional: it avoids exhausting
the 6 GB laptop GPU. The page reports uploading, waiting, building, completion,
or a readable failure message; it does not clear selected files after an error.

## Download results explained

| Download | What it contains | When you would use it |
| --- | --- | --- |
| **Final 3D height map (DSM)** | Ground, buildings, and trees together; each pixel stores height in metres. | Main elevation result; GIS analysis and terrain mesh. |
| **Above-ground height map (nDSM)** | Estimated building/tree height above the local ground. | Checking roofs, trees, and structures separately. |
| **Ground-height map (DTM)** | Bare terrain used as the ground base. | Understanding the absolute-height anchor. |
| **Confidence map** | Model confidence on a 0–1 scale. | Flagging areas that need caution or validation. |
| **3D model (GLB)** | Textured terrain model in real horizontal and vertical metres. | Blender, game engines, or another GLB viewer. |
| **Run record (JSON)** | Input coordinate system, model, output ranges, and ground-source note. | Reproducibility and explaining the result to judges. |

The GeoTIFF downloads are Cloud-Optimized GeoTIFFs (COGs), which remain the
authoritative GIS outputs. The on-screen preview is only for quick inspection.

## Run locally

```powershell
cd C:\Users\Rishabh\SlashCompact
uv sync --extra dev --extra ml
uv run altimap-serve --rdah-checkpoint .\models\rdah-track1-104best_model.pth
```

Open `http://127.0.0.1:8000`. The first run downloads the pretrained Depth
Anything V2 Small weights if they are not already cached.

## Real Vaihingen example

The local ISPRS Vaihingen Area 11 data is wired into the project. The prepared
test files are:

- `data\vaihingen-area11-real\rgb.tif`
- `data\vaihingen-area11-real\dtm_provisional_als.tif`
- `data\vaihingen-area11-real\reference_dsm.tif` — validation only; **do not upload it as the DTM**.

See [Vaihingen_Area11_Integration.md](docs/Vaihingen_Area11_Integration.md)
for reproduction and evaluation details.

### All 20 original TIFFs: reference-quality 3D gallery

Open `http://127.0.0.1:8000/reference.html` to inspect every TIFF under
`C:\Vaihingen\Vaihingen\Images`. These are 7680 × 13824, 12-bit, three-band
GeoTIFFs at roughly 8.3 cm/pixel. Their transforms include image rotation, so
the exporter samples elevation using the full affine transform rather than
assuming every image points north-up.

This gallery is intentionally **zero ML**:

`original GeoTIFF colour + supplied 9 cm matching DSM → textured 3D terrain`

It answers “how good can the rendering look when high-resolution elevation is
available?” It does not answer “how accurately did our monocular model predict
height?” The gallery labels every scene `ZERO ML · REFERENCE VIEW` for this
reason. Use it as a visual target and as a reference comparison for RDAH-Net.

In simple terms:

- The **20 TIFF gallery is the answer-sheet view**. We take each original
  aerial image, look up the supplied measured/reference surface height at the
  same map coordinates, and place the image over that surface. RDAH-Net is not
  used.
- The **RDAH view is the model-attempt view**. The model receives RGB plus a
  relative-depth hint, estimates above-ground height, and combines it with a
  DTM.
- A scientifically fair comparison requires running RDAH on one of the exact
  same 20 image footprints and comparing its predicted DSM pixel-by-pixel with
  that footprint's reference DSM. The current Area 11 RDAH demonstration and
  20-image gallery are useful visual examples, but they are not yet that exact
  one-to-one benchmark.

Regenerate all scenes with:

```powershell
python scripts\export_vaihingen_reference_gallery.py `
  --root C:\Vaihingen\Vaihingen `
  --max-dimension 768
```

The browser grid stays below one million triangles per loaded scene and old
GPU geometry/textures are disposed whenever another TIFF is selected.

## Honest accuracy note

The Vaihingen integration run proves that the real GeoTIFF pipeline, map-grid
preservation, 3D output, and viewer work together. It is not yet a final paper
accuracy claim: its ALS-based ground file is provisional and the RDAH paper does
not fully document its exact Depth-Anything prior-export preprocessing. Final
benchmark numbers require a surveyed independent DTM and a held-out test set.
