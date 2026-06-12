"""backproject.py — depth 역투영 + COLMAP/TUM 포즈 읽기 (Phase 4 공유).

좌표: X_world = R_wc @ X_cam + t_wc.
  TUM(Twc): R_wc, t_wc 직접.   COLMAP(Tcw): R_wc=R_cw^T, t_wc=-R_cw^T@t_cw.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def read_colmap_images(path):
    """images.txt → {NAME: (qw,qx,qy,qz,tx,ty,tz)} (Tcw). 매 짝수 줄=pose, 홀수=2D points."""
    data = [l for l in open(path) if not l.startswith("#")]
    out = {}
    for l in data[0::2]:
        s = l.split()
        if len(s) >= 10:
            out[s[9]] = tuple(float(x) for x in s[1:8])
    return out


def tum_world_RT(tx, ty, tz, qx, qy, qz, qw):
    """Twc → (R_wc, t_wc)."""
    return Rot.from_quat([qx, qy, qz, qw]).as_matrix(), np.array([tx, ty, tz], float)


def colmap_world_RT(qw, qx, qy, qz, tx, ty, tz):
    """Tcw → (R_wc, t_wc) = (R_cw^T, -R_cw^T@t_cw)."""
    R_cw = Rot.from_quat([qx, qy, qz, qw])
    R_wc = R_cw.inv()
    return R_wc.as_matrix(), -R_wc.apply([tx, ty, tz])


def backproject(depth_m, fx, fy, cx, cy, R_wc, t_wc, stride=4, crop=None, rgb=None):
    """depth_m: meters (H,W). crop: 중앙 크롭 비율(0~1) 또는 None.
    반환: world (M,3), pix (M,2) [u,v], color (M,3 RGB 0~1) 또는 None."""
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    uu, vv = uu.ravel(), vv.ravel()
    z = depth_m[vv, uu]
    m = z > 0
    if crop is not None:
        cw, ch = int(W * crop), int(H * crop)
        u0, u1 = (W - cw) // 2, (W + cw) // 2
        v0, v1 = (H - ch) // 2, (H + ch) // 2
        m &= (uu >= u0) & (uu < u1) & (vv >= v0) & (vv < v1)
    uu, vv, z = uu[m], vv[m], z[m]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    world = np.stack([x, y, z], 1) @ R_wc.T + t_wc
    color = None
    if rgb is not None:
        color = rgb[vv, uu][:, ::-1].astype(np.float32) / 255.0  # BGR→RGB
    return world, np.stack([uu, vv], 1), color


def nn_distances(src, dst):
    """src의 각 점에서 dst 최근접 거리 (m). scipy cKDTree."""
    from scipy.spatial import cKDTree
    if len(src) == 0 or len(dst) == 0:
        return np.array([])
    return cKDTree(dst).query(src, k=1)[0]
