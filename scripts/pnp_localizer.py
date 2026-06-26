"""pnp_localizer.py — feature-PnP 기반 실시간 Localizer.

feature_map.npz의 ORB 맵 점에 query ORB를 매칭 → cv2.solvePnPRansac으로 Tcw 추정.
GPU/gsplat 불필요 — 순수 CPU(opencv).
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.runtime import Localizer, PoseResult, OK, LOST  # noqa: E402

# confidence 기준: inlier가 이 수 이상이면 OK
INLIER_OK   = 20     # 이상 → OK
INLIER_GOOD = 60     # conf = 1.0 기준

N_FEATURES  = 1000   # query ORB keypoints


class PnPLocalizer(Localizer):
    """feature-PnP 단발 위치추정. 전역 hint 불필요.

    Args:
        feature_map_path: feature_map.npz 경로.
        K:                [3,3] float32 ndarray — query 이미지 intrinsics.
        device:           무시(CPU만 사용).
    """

    def __init__(self, feature_map_path, K: np.ndarray, device=None):
        data = np.load(str(feature_map_path))
        self.pts3d = data["points3d"]       # [N, 3] float32
        self.map_desc = data["desc"]        # [N, 32] uint8

        self.K    = K.astype(np.float32)
        self.dist = np.zeros(4, dtype=np.float32)

        self.orb     = cv2.ORB_create(nfeatures=N_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 마지막 relocalize/track 결과 진단용 (테스트 접근용)
        self.last_n_inliers: int = 0

    # ── 내부 ──────────────────────────────────────────────────────────────

    def _extract(self, rgb: np.ndarray):
        """rgb (uint8 or float 0-1) → (kps, descs). None이면 ([], None)."""
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.shape[2] == 3 else rgb
        kps, descs = self.orb.detectAndCompute(gray, None)
        return kps, descs

    def _match(self, query_desc: np.ndarray, map_pts3d, map_desc: np.ndarray):
        """Lowe ratio 0.75 필터 → (query 2D [M,2], map 3D [M,3])."""
        matches = self.matcher.knnMatch(query_desc, map_desc, k=2)
        good_q2d = []
        good_p3d = []
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_q2d.append(m.queryIdx)
                good_p3d.append(m.trainIdx)
        return good_q2d, good_p3d

    def _pnp(self, kps, q2d_idx, pts3d, p3d_idx):
        """solvePnPRansac → (rvec, tvec, inlier_mask) or (None, None, None)."""
        if len(q2d_idx) < 6:
            return None, None, None

        obj_pts = pts3d[p3d_idx].reshape(-1, 1, 3).astype(np.float64)
        img_pts = np.array(
            [[kps[i].pt[0], kps[i].pt[1]] for i in q2d_idx],
            dtype=np.float64
        ).reshape(-1, 1, 2)

        try:
            ok_flag, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.K.astype(np.float64), self.dist.astype(np.float64),
                reprojectionError=4.0,
                confidence=0.99,
                iterationsCount=500,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None, None, None

        if not ok_flag or inliers is None:
            return None, None, None

        return rvec, tvec, inliers

    def _build_result(self, rvec, tvec, n_inliers: int) -> PoseResult:
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3,  3] = tvec.flatten().astype(np.float32)

        conf  = float(np.clip(n_inliers / INLIER_GOOD, 0.0, 1.0))
        state = OK if n_inliers >= INLIER_OK else LOST
        return PoseResult(T_map_cam=T, state=state, confidence=conf)

    # ── 공개 인터페이스 ───────────────────────────────────────────────────

    def relocalize(self, rgb: np.ndarray, hint: Optional[np.ndarray] = None) -> PoseResult:
        """전역 위치추정. hint 불필요(PnP의 강점)."""
        kps, descs = self._extract(rgb)
        if descs is None:
            return PoseResult(np.eye(4, dtype=np.float32), LOST, 0.0)

        q2d_idx, p3d_idx = self._match(descs, self.pts3d, self.map_desc)
        rvec, tvec, inliers = self._pnp(kps, q2d_idx, self.pts3d, p3d_idx)

        if rvec is None:
            self.last_n_inliers = 0
            return PoseResult(np.eye(4, dtype=np.float32), LOST, 0.0)

        self.last_n_inliers = len(inliers)
        return self._build_result(rvec, tvec, len(inliers))

    def track(self, rgb: np.ndarray, prior: "PoseResult") -> PoseResult:
        """단발 PnP — relocalize와 동일(prior로 맵 점 prefilter 가능하나 현재는 전역)."""
        return self.relocalize(rgb, hint=prior.T_map_cam)
