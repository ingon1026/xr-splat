"""Phase 3 변환 단위테스트 (SPEC 필수).

핵심: round-trip 항등성만으로는 함정#1(역변환 누락)을 못 잡는다 — forward/inverse가 서로 거울이면
역변환을 빼먹어도 통과하기 때문. 그래서 COLMAP 정의에 대한 **절대** 검증을 한다:
  카메라중심 C = -R_cw^T @ t_cw 가 원래 TUM 카메라 위치 t_wc 와 같아야 한다.
  + 손계산 케이스: 월드 (1,0,0) identity → t_cw = (-1,0,0).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.colmap_convert import twc_to_tcw, camera_center, write_images_txt  # noqa: E402


def rand_unit_quat(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)   # (x, y, z, w)


def test_hand_case():
    # 월드 (1,0,0)에 카메라, identity 회전 → Tcw: t_cw=(-1,0,0), q=identity
    out = twc_to_tcw(1, 0, 0, 0, 0, 0, 1)   # qx qy qz qw = identity
    assert np.allclose(out[4:7], [-1, 0, 0], atol=1e-12)   # t_cw
    assert np.allclose(out[0:4], [1, 0, 0, 0], atol=1e-12)  # COLMAP (qw,qx,qy,qz) identity


def test_camera_center_recovery_absolute():
    # 함정#1 방어: 변환 후 카메라중심을 COLMAP 정의로 복원 → 원래 t_wc 와 일치해야 함
    rng = np.random.default_rng(0)
    for _ in range(50):
        t_wc = rng.uniform(-3, 3, 3)
        q = rand_unit_quat(rng)
        out = twc_to_tcw(t_wc[0], t_wc[1], t_wc[2], q[0], q[1], q[2], q[3])
        C = camera_center(*out)
        assert np.allclose(C, t_wc, atol=1e-9), f"camera center {C} != t_wc {t_wc}"


def test_quat_is_Rcw():
    # COLMAP 쿼터니언이 R_cw = R_wc^T 를 나타내야 함 (순서/역변환 동시 검증)
    rng = np.random.default_rng(1)
    q = rand_unit_quat(rng)
    out = twc_to_tcw(0.5, -1.0, 2.0, *q)
    R_cw_out = Rot.from_quat([out[1], out[2], out[3], out[0]])  # COLMAP(w,x,y,z)→scipy(x,y,z,w)
    assert np.allclose(R_cw_out.as_matrix(), Rot.from_quat(q).inv().as_matrix(), atol=1e-9)


def test_roundtrip_secondary():
    # 보조: Tcw→Twc 복원 항등성 (절대검증과 함께여야 의미 있음)
    rng = np.random.default_rng(2)
    for _ in range(20):
        t = rng.uniform(-2, 2, 3)
        q = rand_unit_quat(rng)
        qw, qx, qy, qz, tx, ty, tz = twc_to_tcw(t[0], t[1], t[2], *q)
        assert np.allclose(camera_center(qw, qx, qy, qz, tx, ty, tz), t, atol=1e-9)
        R_wc_back = Rot.from_quat([qx, qy, qz, qw]).inv()
        assert np.allclose(R_wc_back.as_matrix(), Rot.from_quat(q).as_matrix(), atol=1e-9)


def test_images_txt_two_lines_per_image(tmp_path):
    # 함정#3: 이미지당 2줄 (pose + 빈 2D points 줄)
    entries = [(1, (1, 0, 0, 0, -1, 0, 0), "a.png"), (2, (1, 0, 0, 0, 0, 0, 0), "b.png")]
    p = tmp_path / "images.txt"
    write_images_txt(str(p), entries)
    data = [ln for ln in open(p) if not ln.startswith("#")]
    assert len(data) == 4, f"이미지 2개 → 4 데이터 줄이어야 (pose+빈줄), got {len(data)}"
    assert data[0].split()[-1] == "a.png" and data[1].strip() == ""
    assert data[2].split()[-1] == "b.png" and data[3].strip() == ""
