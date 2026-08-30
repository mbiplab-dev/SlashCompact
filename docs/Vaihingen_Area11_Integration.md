# Vaihingen Area 11: real integration workflow

This repository includes a reproducible adapter for the local dataset at
`C:\Vaihingen\Vaihingen`. It uses the matching Area 11 9 cm RGB orthophoto and
reference DSM from the ISPRS archive.

The per-area image files lack CRS metadata. The script restores the Area 11
world-file transform from `Vaihingen_dsm_tiles_geoinfo.zip` and uses the CRS
explicitly recorded by its supplied VRT: EPSG:32633. It does not use the
reference DSM as pipeline input.

The local ALS strips contain unclassified returns, so the generated DTM is a
lower-envelope, 1 m-smoothed provisional surface. It is independent of the
reference DSM but not a surveyed bare-earth DTM. Its datum remains `unknown`.

```powershell
python scripts\prepare_vaihingen_area11.py --root C:\Vaihingen\Vaihingen --output data\vaihingen-area11-real
python -m altimap.cli run data\vaihingen-area11-real\rgb.tif --dtm data\vaihingen-area11-real\dtm_provisional_als.tif --output data\vaihingen-area11-real\run-rdah-dav2 --dtm-source "Vaihingen ALS lower-envelope provisional DTM" --rdah-checkpoint models\rdah-track1-104best_model.pth
python scripts\evaluate_dsm.py data\vaihingen-area11-real\run-rdah-dav2\dsm.tif data\vaihingen-area11-real\reference_dsm.tif --label integration-only
```

On the prepared 512 × 512 test crop, this run completed end-to-end and wrote
all artifacts. The resulting evaluation is useful for catching grid, mesh,
format, and numerical regressions. It is not evidence that the model achieves
the RDAH paper's published DFC2019 accuracy: that needs the authors' exact
Depth-Anything V2 prior preprocessing and a held-out benchmark with an
independent DTM.
