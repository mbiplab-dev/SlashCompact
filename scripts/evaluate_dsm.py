"""Evaluate a predicted DSM against an independently supplied reference DSM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="integration-only")
    args = parser.parse_args()
    with rasterio.open(args.prediction) as predicted, rasterio.open(args.reference) as reference:
        if (predicted.shape, predicted.crs, predicted.transform) != (reference.shape, reference.crs, reference.transform):
            raise ValueError("prediction and reference must use exactly the same shape, CRS, and transform")
        a, b = predicted.read(1).astype(np.float32), reference.read(1).astype(np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any(): raise ValueError("no common finite pixels")
    error = a[mask] - b[mask]
    report = {"label": args.label, "pixels": int(mask.sum()), "mae_m": float(np.mean(np.abs(error))),
              "rmse_m": float(np.sqrt(np.mean(error ** 2))), "bias_m": float(np.mean(error)),
              "correlation": float(np.corrcoef(a[mask], b[mask])[0, 1]),
              "warning": "Do not treat integration-only scores as an RDAH reproduction result without an independent, surveyed DTM and documented prior preprocessing."}
    output = args.output or args.prediction.with_name("evaluation.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
