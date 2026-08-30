from __future__ import annotations

import numpy as np
import pytest

from altimap.ml.rdah import RDAHNetPredictor


class FlatPrior:
    name = "flat-test-prior"
    def predict_depth(self, rgb): return np.ones(rgb.shape[:2], np.float32)


def test_rdah_checkpoint_strictly_loads_when_available():
    checkpoint = __import__("pathlib").Path("models/rdah-track1-104best_model.pth")
    if not checkpoint.exists(): pytest.skip("official checkpoint is optional for a source checkout")
    model = RDAHNetPredictor(checkpoint, FlatPrior(), device="cpu")
    assert model.name == "rdah-net-track1"
    predicted, confidence = model.predict_ndsm(np.zeros((128, 128, 3), np.uint8))
    assert predicted.shape == (128, 128) and confidence is None
    assert np.isfinite(predicted).all() and (predicted >= 0).all()
