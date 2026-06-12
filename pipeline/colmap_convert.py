"""colmap_convert.py — TUM 궤적(Twc) → COLMAP 모델(Tcw) 변환 핵심 (importable, 테스트 대상).

좌표 변환 (SPEC §3 Phase 3, 함정 체크리스트):
  TUM 한 줄  = (t, tx ty tz, qx qy qz qw) = Twc (camera→world)
  COLMAP img = (ID, qw qx qy qz, tx ty tz, CAM_ID, NAME) = Tcw (world→camera)
  R_cw = R_wc^T,  t_cw = -R_wc^T @ t_wc

함정:
  #1 Twc→Tcw 역변환 누락 → 학습이 "흐릿하게" 실패 (round-trip 테스트로는 못 잡음; 카메라중심 절대검증 필요)
  #2 쿼터니언 순서: TUM (qx qy qz qw) ↔ COLMAP (qw qx qy qz)
  #4 scipy Rotation quat 순서는 (x,y,z,w):
       - 입력: TUM (qx qy qz qw) 가 우연히 scipy (x,y,z,w) 와 같은 순서 → 재배열 불필요
       - 출력: COLMAP (qw qx qy qz) 로 명시 재배열 (여기서만)
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def twc_to_tcw(tx, ty, tz, qx, qy, qz, qw):
    """TUM Twc → COLMAP Tcw. 반환: (qw, qx, qy, qz, tx, ty, tz) — COLMAP 순서."""
    R_wc = Rot.from_quat([qx, qy, qz, qw])          # scipy (x,y,z,w) == TUM 순서 (입력 재배열 X)
    t_wc = np.array([tx, ty, tz], dtype=float)
    R_cw = R_wc.inv()                                # 함정#1: R_cw = R_wc^T (역변환 필수)
    t_cw = -R_cw.apply(t_wc)                         # t_cw = -R_wc^T @ t_wc
    qx_c, qy_c, qz_c, qw_c = R_cw.as_quat()          # scipy (x,y,z,w)
    return (qw_c, qx_c, qy_c, qz_c, t_cw[0], t_cw[1], t_cw[2])   # 함정#2,#4: COLMAP (w,x,y,z)


def camera_center(qw, qx, qy, qz, tx, ty, tz):
    """COLMAP (R_cw, t_cw) → 월드 좌표상 카메라 중심 C = -R_cw^T @ t_cw. (검증용)"""
    R_cw = Rot.from_quat([qx, qy, qz, qw])           # COLMAP (w,x,y,z) → scipy (x,y,z,w) 재배열
    return -R_cw.inv().apply(np.array([tx, ty, tz], dtype=float))


def load_tum(path):
    """[(ts, tx, ty, tz, qx, qy, qz, qw)] 로 로드 (주석/빈 줄 무시)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = line.split()
        out.append((float(v[0]),) + tuple(float(x) for x in v[1:8]))
    return out


def load_rgb_list(rgb_txt):
    """[(ts, relpath)] — associate/매칭용."""
    out = []
    for line in open(rgb_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split()[0], line.split()[1]
        out.append((float(ts), rel))
    return out


def match_nearest(ts, rgb_sorted, ts_keys, tol):
    """ts에 가장 가까운 rgb relpath (tol 이내) 또는 None. rgb_sorted: ts 정렬, ts_keys: 그 ts 리스트."""
    import bisect
    i = bisect.bisect_left(ts_keys, ts)
    cand = [j for j in (i - 1, i) if 0 <= j < len(ts_keys)]
    if not cand:
        return None
    best = min(cand, key=lambda j: abs(ts_keys[j] - ts))
    return rgb_sorted[best][1] if abs(ts_keys[best] - ts) <= tol else None


def write_cameras_txt(path, width, height, fx, fy, cx, cy, camera_id=1):
    """PINHOLE 모델 (SPEC §3). 주의: PINHOLE은 왜곡 미포함 — distortion 처리는 Phase 5/6에서."""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{camera_id} PINHOLE {int(width)} {int(height)} {fx} {fy} {cx} {cy}\n")


def write_images_txt(path, entries, camera_id=1):
    """entries: [(image_id, (qw,qx,qy,qz,tx,ty,tz), name)]. 함정#3: 이미지당 2줄(2번째=빈 2D points)."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, p, name in entries:
            f.write(f"{image_id} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {camera_id} {name}\n")
            f.write("\n")   # 2번째 줄(빈 2D points) — 반드시 존재해야 파서가 침묵 실패 안 함


def write_points3d_txt(path):
    """빈 헤더만 — 실제 포인트는 Phase 4(04_make_pointcloud.py)가 채운다."""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
