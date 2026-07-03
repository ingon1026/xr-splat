"""test_export_web.py — 09_export_web .splat 변환 라운드트립 검증 (CPU, gsplat-free)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("export_web", ROOT / "scripts" / "09_export_web.py")
ew = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ew)

SPLAT_DTYPE = [("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)]


def _make_ply(path, rows):
    """rows: list of dict(x..rot_3 raw 값)."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.array([tuple(r[n] for n in names) for r in rows],
                   dtype=[(n, "f4") for n in names])
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def test_roundtrip_prune_and_order(tmp_path):
    big = dict(x=1.0, y=2.0, z=3.0, f_dc_0=1.0, f_dc_1=0.0, f_dc_2=-1.0,
               opacity=2.0, scale_0=-1.0, scale_1=-1.0, scale_2=-1.0,   # exp→0.368, sigmoid→0.881
               rot_0=1.0, rot_1=0.0, rot_2=0.0, rot_3=0.0)
    small = dict(big, x=9.0, scale_0=-5.0, scale_1=-5.0, scale_2=-5.0)   # importance 낮음
    invisible = dict(big, x=-9.0, opacity=-9.0)                          # sigmoid≈0 → prune
    ply = tmp_path / "t.ply"
    _make_ply(ply, [small, big, invisible])

    out = tmp_path / "t.splat"
    meta = ew.export_splat(ew.load_gaussians(ply), out, opacity_min=0.02, max_gaussians=0)

    assert meta["n_input"] == 3 and meta["n_output"] == 2          # invisible prune
    assert out.stat().st_size == 2 * ew.BYTES_PER_GAUSSIAN

    rec = np.fromfile(out, dtype=SPLAT_DTYPE)
    assert rec["pos"][0].tolist() == [1.0, 2.0, 3.0]               # importance: big 먼저
    assert rec["pos"][1][0] == 9.0
    np.testing.assert_allclose(rec["scale"][0], np.exp(-1.0), rtol=1e-6)
    # 색: 0.5 + SH_C0*f_dc → [0.782, 0.5, 0.218]*255, alpha sigmoid(2)=0.881*255
    np.testing.assert_allclose(rec["rgba"][0], [199, 127, 55, 224], atol=1)
    # 단위 quat (1,0,0,0) → u8 (255,128,128,128)
    np.testing.assert_allclose(rec["rot"][0], [255, 128, 128, 128], atol=1)


def test_max_gaussians_cap(tmp_path):
    base = dict(x=0.0, y=0.0, z=0.0, f_dc_0=0.0, f_dc_1=0.0, f_dc_2=0.0,
                opacity=2.0, scale_0=-1.0, scale_1=-1.0, scale_2=-1.0,
                rot_0=1.0, rot_1=0.0, rot_2=0.0, rot_3=0.0)
    ply = tmp_path / "t.ply"
    _make_ply(ply, [base] * 5)
    meta = ew.export_splat(ew.load_gaussians(ply), tmp_path / "t.splat",
                           opacity_min=0.02, max_gaussians=3)
    assert meta["n_output"] == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
