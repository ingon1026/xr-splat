"""overlay_geometry.py — 키프레임 궤적 폴리라인 + 카메라 프러스텀 와이어프레임을 3D 선분으로 생성.

SLAM 이동경로 시각화 공유 모듈(Part 2). 두 출력이 같은 기하를 공유:
  - render_trajectory_overlay.py : 선분을 2D 투영해 렌더 프레임 위에 그림.
  - bake_trajectory_ply.py       : 선분을 dense 가우시안으로 샘플해 scene.ply에 합침.

좌표: COLMAP Tcw → colmap_world_RT 로 (R_wc, C). 카메라 컨벤션 +Z forward / +Y down (backproject.py 일치).
색: 궤적 cyan, 프러스텀 orange, 첫 KF green, 끝 KF red.
"""
import numpy as np
from pipeline.backproject import read_colmap_images, read_colmap_cameras, colmap_world_RT

CYAN = (0.0, 1.0, 1.0); ORANGE = (1.0, 0.5, 0.0); GREEN = (0.0, 1.0, 0.0); RED = (1.0, 0.2, 0.2)


def keyframe_centers(images_txt):
    """시간순 정렬된 [(name, C[3], R_wc[3,3])]."""
    imgs = read_colmap_images(images_txt)
    names = sorted(imgs.keys(), key=lambda n: float(n.rsplit(".", 1)[0]))
    out = []
    for n in names:
        R_wc, C = colmap_world_RT(*imgs[n])
        out.append((n, np.asarray(C, float), np.asarray(R_wc, float)))
    return out


def _frustum_edges(C, R_wc, fx, fy, cx, cy, W, H, d):
    """apex C + 이미지면 4코너(near d) → 8엣지 [(P0,P1)]."""
    px = [(0, 0), (W, 0), (W, H), (0, H)]
    corners = []
    for u, v in px:
        Xc = np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])
        corners.append(R_wc @ Xc + C)
    edges = [(C, c) for c in corners]                       # apex → 코너 4
    edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]   # 이미지면 사각형 4
    return edges


def build_segments(images_txt, cameras_txt, frustum_stride=5, frustum_size=1.0):
    """→ list[(P0[3], P1[3], color[3])]. 궤적(전체 KF) + 프러스텀(매 stride)."""
    kf = keyframe_centers(images_txt)
    W, H, fx, fy, cx, cy = read_colmap_cameras(cameras_txt)
    centers = np.array([c for _, c, _ in kf])
    # near depth = 연속 KF 중심 간 median 간격 (겹침 방지), fallback = 0.04 * 궤적 bbox 대각
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    diag = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    d = (float(np.median(steps)) if len(steps) and np.median(steps) > 1e-6 else 0.04 * diag) * frustum_size

    segs = []
    for i in range(len(centers) - 1):                       # 궤적 폴리라인
        segs.append((centers[i], centers[i + 1], CYAN))
    for i, (_, C, R_wc) in enumerate(kf):                   # 프러스텀
        if i % frustum_stride and i not in (0, len(kf) - 1):
            continue
        col = GREEN if i == 0 else RED if i == len(kf) - 1 else ORANGE
        for P0, P1 in _frustum_edges(C, R_wc, fx, fy, cx, cy, W, H, d):
            segs.append((P0, P1, col))
    return segs, d, diag                                    # diag = 궤적(카메라 중심) bbox 대각


def sample_points(segments, radius):
    """선분을 간격 ≤2*radius 로 점 샘플 → (pts[M,3], cols[M,3]). 가우시안 bake용."""
    pts, cols = [], []
    for P0, P1, col in segments:
        L = np.linalg.norm(P1 - P0)
        n = max(2, int(np.ceil(L / max(radius * 1.5, 1e-6))) + 1)
        for a in np.linspace(0, 1, n):
            pts.append((1 - a) * P0 + a * P1)
            cols.append(col)
    return np.array(pts, np.float32), np.array(cols, np.float32)
