"""Command line entry point for deterministic local processing."""

from __future__ import annotations

import argparse

import numpy as np
import rasterio

from altimap.pipeline.core import GeoTiffPipeline
from altimap.ml.rdah import DepthAnythingV2Prior, RDAHNetPredictor


def main() -> None:
    parser = argparse.ArgumentParser(prog="altimap", description="GeoTIFF RGB to metric DSM artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run nDSM + DTM -> DSM pipeline")
    run.add_argument("image"); run.add_argument("--dtm", required=True); run.add_argument("--output", required=True)
    run.add_argument("--dtm-source", required=True); run.add_argument("--rdah-checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "run":
        with rasterio.open(args.dtm) as src:
            dtm = src.read(1).astype(np.float32)
        model = RDAHNetPredictor(args.rdah_checkpoint, DepthAnythingV2Prior())
        result = GeoTiffPipeline(model).run(args.image, dtm, args.output, args.dtm_source)
        print(result.manifest_path)


if __name__ == "__main__":
    main()
