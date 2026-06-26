"""orchestrator.py — 01→08 단계 자동 실행 통합 진입점.

원칙: compose, not rewrite — 기존 스크립트 로직을 복제하지 않고 함수/subprocess로 호출.
  - 함수 in-process: 05(validate_poses), 06(train_loop) — 명시적 추출 함수.
    importlib으로 수치 접두 파일(05_*)을 로드하고, train_gsplat은 sys.path 경유 import.
  - subprocess: 01, 03, 04, 08, build_report — sys.exit()를 직접 호출하는 CLI.
    in-process로 부르면 SystemExit + sys.argv 패치가 필요해 "compose, not rewrite"에 역행.
  - subprocess(bash): 02(02_run_orbslam3.sh). bash 스크립트라 래핑 필요.
  - 07_evaluate: 두 scene 교차 비교 + groundtruth.txt 필요 — 단일 scene 파이프라인 외 범위.
    관련 입력 없으면 graceful skip.

resume/idempotent:
  - 각 단계 산출물이 있으면 skip (force=False 기본).
  - need_train = force or not asset_ply.exists().
    asset_ply가 있으면 05·06 모두 skip → home처럼 이미 빌드된 scene은 전 단계 skip.

게이트 (CLAUDE.md §2 / SPEC §2):
  - need_train=True일 때만 gate 실행 (05 validate + frame integrity).
  - validate.enabled=True면 validate_poses PASS 필수.
  - orb_ref_traj 있으면 frame integrity(Sim3 scale≈1·rot<2°) 확인.
  - 둘 중 하나라도 FAIL → RuntimeError, 06 학습 차단.

usage:
  python -m pipeline.orchestrator <scene_or_yaml> [--force]
"""
import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("orchestrator")
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


# ── 지연 로더 ────────────────────────────────────────────────────────────────

def _load_script_mod(name, path):
    """importlib으로 파일경로 직접 로드 (수치 접두 파일명 대응)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_validate_mod = None


def _get_validate_poses():
    """05_validate_poses.validate_poses 지연 로드 (gsplat-free 경로)."""
    global _validate_mod
    if _validate_mod is None:
        _validate_mod = _load_script_mod("v05", SCRIPTS / "05_validate_poses.py")
    return _validate_mod.validate_poses


_report_mod = None


def _get_compute_frame_integrity():
    """build_report.compute_frame_integrity 지연 로드 (gsplat-free)."""
    global _report_mod
    if _report_mod is None:
        _report_mod = _load_script_mod("build_report", SCRIPTS / "build_report.py")
    return _report_mod.compute_frame_integrity


def _get_train_loop():
    """train_gsplat.train_loop 지연 로드 — gsplat import를 06 실행 시점으로 미룸."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from train_gsplat import train_loop  # noqa: PLC0415
    return train_loop


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _run_sub(cmd, desc):
    """subprocess 실행. 실패 시 RuntimeError."""
    log.info("[run] %s", desc)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} 실패 (exit {r.returncode})")


def _skip(step, reason):
    log.info("skip: %s — %s", step, reason)


# ── 메인 빌드 ─────────────────────────────────────────────────────────────────

def build(cfg, force=False):
    """01→08 순차 실행. 산출물 있으면 skip(idempotent). 게이트 강제.

    Parameters
    ----------
    cfg : SceneConfig
        pipeline.config.load_config() 반환값.
    force : bool
        True면 모든 단계 재실행 (기존 산출물 무시).
    """
    log.info("=== build 시작: scene=%s tag=%s force=%s ===", cfg.scene, cfg.train.tag, force)

    # ── 01: RGB-D 추출 ───────────────────────────────────────────────
    assoc = cfg.proc_dir / "associations.txt"
    if not force and assoc.exists():
        _skip("01_extract_bag", f"associations.txt 이미 있음: {assoc}")
    else:
        inp = cfg.input
        if inp.type == "bag":
            _run_sub(
                [sys.executable, str(SCRIPTS / "01_extract_bag.py"),
                 "bag",
                 "--bag", str(ROOT / inp.path),
                 "--out", str(cfg.proc_dir),
                 "--depth-scale", str(inp.depth_scale)],
                "01_extract_bag",
            )
        elif inp.type == "dir":
            raise NotImplementedError(
                "01 dir 모드는 orchestrator 미지원 — 수동으로 01_extract_bag.py dir 실행 후 재시도"
            )
        else:
            raise ValueError(f"01: 알 수 없는 input.type: {inp.type!r}")

    # ── 02: ORB-SLAM3 (bash subprocess, 성공 판정=KeyFrameTrajectory.txt 존재) ──
    if not force and cfg.slam_traj.exists():
        _skip("02_run_orbslam3", f"KeyFrameTrajectory.txt 이미 있음: {cfg.slam_traj}")
    else:
        subprocess.run(
            ["bash", str(SCRIPTS / "02_run_orbslam3.sh"), cfg.scene, str(cfg.slam.nfeatures)],
            cwd=ROOT,
        )
        if not cfg.slam_traj.exists():
            raise RuntimeError(
                f"02_run_orbslam3 FAIL — {cfg.slam_traj} 미생성 "
                "(nfeatures 늘려서 재시도: SPEC Phase 2)"
            )

    # ── 03: TUM 궤적 → COLMAP 모델 ──────────────────────────────────
    images_txt = cfg.sparse_dir / "images.txt"
    if not force and images_txt.exists():
        _skip("03_tum_to_colmap", f"images.txt 이미 있음: {images_txt}")
    else:
        _run_sub(
            [sys.executable, str(SCRIPTS / "03_tum_to_colmap.py"), "--scene", cfg.scene],
            "03_tum_to_colmap",
        )

    # ── 04: 초기 포인트클라우드 ─────────────────────────────────────
    points_init_ply = cfg.proc_dir / "colmap" / "points_init.ply"
    if not force and points_init_ply.exists():
        _skip("04_make_pointcloud", f"points_init.ply 이미 있음: {points_init_ply}")
    else:
        cmd = [sys.executable, str(SCRIPTS / "04_make_pointcloud.py"), "--scene", cfg.scene]
        if cfg.exclude_list:
            cmd += ["--exclude-list", str(cfg.exclude_list)]
        _run_sub(cmd, "04_make_pointcloud")

    # ── 05 게이트 + 06 학습 ──────────────────────────────────────────
    # need_train=False면 05·06 모두 skip (asset_ply 이미 존재 = 학습 완료).
    need_train = force or not cfg.asset_ply.exists()
    if not need_train:
        _skip("05_validate_poses", f"asset_ply 이미 있음 (학습 완료): {cfg.asset_ply}")
        _skip("06_train_gsplat", f"asset_ply 이미 있음: {cfg.asset_ply}")
    else:
        # Gate 1: validate_poses
        if cfg.validate.enabled:
            log.info("[05] validate_poses 게이트 실행 (CLAUDE.md §2)")
            validate_poses = _get_validate_poses()
            passed, metrics = validate_poses(
                cfg.scene,
                root=ROOT,
                baseline=cfg.validate.baseline,
                pass_cm=cfg.validate.pass_cm,
                crop=cfg.validate.crop,
            )
            if metrics.get("status") == "cannot_validate":
                raise RuntimeError(
                    f"[05] CANNOT VALIDATE — {metrics.get('reason', '')}. "
                    "06 학습 차단 (default-PASS 금지, SPEC §2)"
                )
            if not passed:
                raise RuntimeError(
                    "[05] validate FAIL — 06 학습 차단 (CLAUDE.md §2). "
                    "오버레이 .ply로 ghost 방향 확인 후 캡처/파이프라인 수정 필요."
                )
            log.info("[05] validate PASS")
        else:
            log.info("[05] validate.enabled=False — 게이트 skip")

        # Gate 2: 프레임 무결성 (Sim3 scale≈1, rot<2°)
        orb_ref = cfg.orb_ref_traj
        if orb_ref.exists():
            compute_fi = _get_compute_frame_integrity()
            fi = compute_fi(cfg.sparse_dir / "images.txt", orb_ref)
            if fi["status"] == "fail":
                raise RuntimeError(
                    f"[fi] frame integrity FAIL — 06 학습 차단. "
                    f"scale={fi.get('scale')}, rot={fi.get('rot_deg')}° "
                    "(ORB 좌표계 불일치 — 03 배선 또는 ORB run 확인)"
                )
            log.info(
                "[fi] frame integrity: %s (scale=%s, rot=%s°, matched=%s KF)",
                fi["status"], fi.get("scale"), fi.get("rot_deg"), fi.get("matched"),
            )
        else:
            log.warning("[fi] orb_ref_traj 없음 — frame integrity 미검증: %s", orb_ref)

        # 06: train_gsplat (gsplat import은 여기서 처음 발생)
        log.info("[06] train_gsplat 실행 (strategy=%s, iters=%d, tag=%s)",
                 cfg.train.strategy, cfg.train.iters, cfg.train.tag)
        train_loop = _get_train_loop()
        import argparse as _ap  # noqa: PLC0415
        train_args = _ap.Namespace(
            scene=cfg.scene,
            root=ROOT,
            iters=cfg.train.iters,
            depth_lambda=cfg.train.depth_lambda,
            ssim_lambda=0.2,
            sh_degree=3,
            holdout_every=cfg.train.holdout_every,
            tag=cfg.train.tag,
            refine_stop=cfg.train.refine_stop,
            strategy=cfg.train.strategy,
            cap_max=cfg.train.cap_max,
            opacity_reg=0.01,
            scale_reg=0.01,
            exclude_list=Path(cfg.exclude_list) if cfg.exclude_list else None,
        )
        train_loop(train_args)
        log.info("[06] train DONE → %s", cfg.asset_ply)

    # ── 07: eval_m1 (선택 — 두 scene 교차 비교, 단일 파이프라인 외 범위) ──
    log.info(
        "[07] eval_m1 skip — 07_evaluate.py는 orb_scene vs colmap_scene 교차 비교로 "
        "단일 scene build에서 지원하지 않음. 별도 실행: python scripts/07_evaluate.py"
    )

    # ── 08: 후처리 (opacity pruning + SH 축소 + 경량 export) ─────────
    if not force and cfg.asset_lite.exists():
        _skip("08_postprocess", f"scene_lite.ply 이미 있음: {cfg.asset_lite}")
    else:
        _run_sub(
            [sys.executable, str(SCRIPTS / "08_postprocess.py"),
             "--scene", cfg.scene,
             "--tag", cfg.train.tag,
             "--opacity", str(cfg.postprocess.opacity),
             "--sh-out", str(cfg.postprocess.sh_out)],
            "08_postprocess",
        )

    # ── build_report (항상 실행 — 경량, XR-ready 판정 + asset_report.md) ──
    log.info("[report] build_report 실행")
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_report.py"),
         "--scene", cfg.scene,
         "--tag", cfg.train.tag],
        cwd=ROOT,
    )
    if r.returncode != 0:
        log.warning("[report] build_report exit=%d (산출물 확인 필요)", r.returncode)
    else:
        log.info("[report] 완료 → outputs/%s/asset_report_%s.md", cfg.scene, cfg.train.tag)

    log.info("=== build 완료: scene=%s ===", cfg.scene)


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="xr-splat 오프라인 build 오케스트레이터 (01→08 순차 실행, resume/idempotent)",
    )
    ap.add_argument("scene_or_yaml", help="scene 이름 또는 configs/<scene>.yaml 경로")
    ap.add_argument("--force", action="store_true", help="기존 산출물 무시하고 전 단계 재실행")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    sys.path.insert(0, str(ROOT))
    from pipeline.config import load_config  # noqa: PLC0415
    cfg = load_config(args.scene_or_yaml)
    build(cfg, force=args.force)
