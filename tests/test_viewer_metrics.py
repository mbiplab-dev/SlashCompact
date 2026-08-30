import numpy as np
import pytest

from viewer.metrics import (
    detrend,
    fit_plane,
    luminance,
    plane_surface,
    scene_metrics,
)


def _ramp(h=32, w=48, a=2.0, b=3.0, c=5.0):
    """Exact plane z = a + b*u + c*v in normalized coords."""
    yy, xx = np.mgrid[0:h, 0:w]
    return a + b * (xx / w) + c * (yy / h)


def test_fit_plane_recovers_known_coefficients():
    (a, b, c), r2 = fit_plane(_ramp())
    assert a == pytest.approx(2.0, abs=1e-9)
    assert b == pytest.approx(3.0, abs=1e-9)
    assert c == pytest.approx(5.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_detrend_of_exact_plane_is_zero():
    depth = _ramp()
    plane, _ = fit_plane(depth)
    assert np.abs(detrend(depth, plane)).max() < 1e-9


def test_plane_surface_reproduces_the_ramp():
    depth = _ramp()
    plane, _ = fit_plane(depth)
    assert np.allclose(plane_surface(plane, depth.shape), depth, atol=1e-9)


def test_constant_depth_yields_nan_r2_not_zero_and_not_raise():
    """SS_tot is 0 here. nan must propagate as 'unmeasurable', never as 0.0."""
    (_, b, c), r2 = fit_plane(np.full((16, 16), 3.0))
    assert np.isnan(r2)
    assert b == pytest.approx(0.0, abs=1e-9)
    assert c == pytest.approx(0.0, abs=1e-9)


def test_zero_median_depth_yields_nan_residual_relief():
    depth = np.zeros((16, 16))
    depth[:8] = -1.0
    depth[8:] = 1.0  # median is 0
    m = scene_metrics(depth, np.zeros((16, 16)))
    assert np.isnan(m["residual_relief"])


def test_scene_metrics_on_pure_ramp_reports_full_r2_and_no_residual():
    depth = _ramp()
    m = scene_metrics(depth, np.zeros_like(depth))
    assert m["plane_r2"] == pytest.approx(1.0, abs=1e-12)
    assert m["residual_relief"] == pytest.approx(0.0, abs=1e-9)
    assert m["col_corr"] > 0.5   # b > 0, depth rises with x
    assert m["row_corr"] > 0.5   # c > 0, depth rises with y


def test_structure_alignment_is_high_when_relief_sits_on_image_edges():
    """A bump in depth co-located with a bump in brightness should align."""
    depth = _ramp(64, 64)
    lum = np.zeros((64, 64))
    depth[20:30, 20:30] += 4.0
    lum[20:30, 20:30] = 1.0
    m = scene_metrics(depth, lum)
    assert m["structure_alignment"] > 0.5


def test_structure_alignment_is_low_when_relief_and_edges_are_unrelated():
    depth = _ramp(64, 64)
    lum = np.zeros((64, 64))
    depth[10:20, 10:20] += 4.0
    lum[40:50, 40:50] = 1.0
    m = scene_metrics(depth, lum)
    assert abs(m["structure_alignment"]) < 0.2


def test_conf_texture_corr_is_nan_without_confidence():
    m = scene_metrics(_ramp(), np.zeros((32, 48)))
    assert np.isnan(m["conf_texture_corr"])


def test_luminance_uses_rec601_weights():
    rgb = np.zeros((1, 1, 3))
    rgb[0, 0] = [1.0, 0.0, 0.0]
    assert luminance(rgb)[0, 0] == pytest.approx(0.299)


def test_metrics_module_is_torch_free():
    """Load-bearing: this module must test in .venv without .venv-da3."""
    # Other tests may legitimately import the production RDAH adapter.  Check
    # the import boundary in a fresh interpreter instead of depending on test
    # collection order.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys; import viewer.metrics; assert 'torch' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "cv2" not in sys.modules
    assert "depth_anything_3" not in sys.modules
