"""Run the production local server with an explicit RDAH checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from altimap.api import create_app
from altimap.ml.rdah import DepthAnythingV2Prior, RDAHNetPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdah-checkpoint", required=True)
    parser.add_argument("--workspace", default="altimap-jobs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    static = Path(__file__).parents[2] / "viewer" / "web" / "production"
    # Model weights remain resident across jobs.  The API itself serializes jobs
    # so this uses one predictable GPU allocation on the 6 GB target laptop.
    cached_model = None
    def model_factory():
        nonlocal cached_model
        if cached_model is None:
            cached_model = RDAHNetPredictor(args.rdah_checkpoint, DepthAnythingV2Prior())
        return cached_model
    app = create_app(args.workspace, model_factory, static)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
