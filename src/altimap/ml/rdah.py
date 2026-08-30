"""Verified RDAH-Net inference adapter and Depth-Anything-V2 prior boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class RDAHCheckpointError(RuntimeError):
    pass


class RelativeDepthPrior(Protocol):
    """A frozen monocular model which returns its raw relative depth map."""
    name: str
    def predict_depth(self, rgb: np.ndarray) -> np.ndarray: ...


class DepthAnythingV2Prior:
    """Depth Anything V2 Small via its maintained Hugging Face checkpoint.

    The RDAH paper's released data does not document the exact depth export
    scale.  Consequently this prior is marked *candidate* until its DFC2019
    validation MAE is within the documented reproduction gate.  The raw output
    is deliberately passed through without min/max normalisation.
    """
    name = "depth-anything-v2-small-hf-candidate"

    def __init__(self, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RDAHCheckpointError("install the 'ml' extra to use the Depth Anything V2 prior") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(self.device).eval()

    def predict_depth(self, rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            predicted = self.model(**inputs).predicted_depth[:, None]
            output = self.torch.nn.functional.interpolate(predicted, rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
        return output.float().cpu().numpy()


class RDAHNetPredictor:
    name = "rdah-net-track1"

    def __init__(self, checkpoint: str | Path, depth_prior: RelativeDepthPrior, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.depth_prior = depth_prior
        if not self.checkpoint.is_file():
            raise RDAHCheckpointError(
                "RDAH-Net Track1 checkpoint is required. Download Figshare file 63637257 "
                "and provide it with --rdah-checkpoint; metric output is intentionally unavailable without it."
            )
        try:
            import torch
        except ImportError as exc:
            raise RDAHCheckpointError("install the 'ml' extra to load RDAH-Net") from exc
        from altimap.ml.rdah_architecture import build_rdah_net
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        self.model = build_rdah_net()
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RDAHCheckpointError("checkpoint is not compatible with the verified official RDAH-Net graph") from exc
        self.model.to(self.device).eval()

    def predict_ndsm(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RDAH-Net expects RGB tiles shaped (height, width, 3)")
        height, width = rgb.shape[:2]
        # The official block attention needs the four encoder scales to align.
        pad_y, pad_x = (-height) % 128, (-width) % 128
        rgb_pad = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        relative = np.asarray(self.depth_prior.predict_depth(rgb_pad), dtype=np.float32)
        if relative.shape != rgb_pad.shape[:2] or not np.isfinite(relative).all():
            raise RDAHCheckpointError("relative-depth prior must return finite raw depth on the padded input grid")
        image_tensor = self.torch.from_numpy(np.moveaxis(rgb_pad, -1, 0)[None].astype(np.float32) / 255.0).to(self.device)
        depth_tensor = self.torch.from_numpy(relative[None, None]).to(self.device)
        with self.torch.inference_mode():
            ndsm = self.model(depth_tensor, image_tensor)[0, 0].float().cpu().numpy()
        return np.maximum(ndsm[:height, :width], 0.0).astype(np.float32), None
