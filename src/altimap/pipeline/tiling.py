"""Overlap tiling with smooth blending for seam-free model inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


def starts(length: int, size: int, overlap: int) -> list[int]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("tile size must be positive and overlap must be smaller than tile size")
    if length <= size:
        return [0]
    stride = size - overlap
    result = list(range(0, length - size + 1, stride))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def tiles(shape: tuple[int, int], size: int = 512, overlap: int = 128) -> list[Tile]:
    h, w = shape
    return [Tile(r, c, y, min(y + size, h), x, min(x + size, w))
            for r, y in enumerate(starts(h, size, overlap))
            for c, x in enumerate(starts(w, size, overlap))]


def cosine_weight(height: int, width: int) -> np.ndarray:
    """Non-zero raised-cosine window; borders remain representable."""
    wy = np.hanning(max(height, 3))[:height] if height > 1 else np.ones(1)
    wx = np.hanning(max(width, 3))[:width] if width > 1 else np.ones(1)
    weight = np.outer(wy, wx).astype(np.float32)
    return np.maximum(weight, 1e-3)


class TileBlender:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.sum = np.zeros(shape, dtype=np.float64)
        self.weight = np.zeros(shape, dtype=np.float64)

    def add(self, tile: Tile, value: np.ndarray) -> None:
        if value.shape != (tile.y1 - tile.y0, tile.x1 - tile.x0):
            raise ValueError("model tile output does not match its input tile")
        w = cosine_weight(*value.shape)
        finite = np.isfinite(value)
        self.sum[tile.y0:tile.y1, tile.x0:tile.x1] += np.where(finite, value, 0) * w
        self.weight[tile.y0:tile.y1, tile.x0:tile.x1] += finite * w

    def finish(self) -> np.ndarray:
        result = np.full(self.sum.shape, np.nan, dtype=np.float32)
        valid = self.weight > 0
        result[valid] = (self.sum[valid] / self.weight[valid]).astype(np.float32)
        return result
