#!/usr/bin/env python3
"""xrsplat — 단일 진입점 CLI (파편화된 실행 통합).

  build    <scene|config> [--force]   오프라인 파이프라인 01→08 + XR-ready report
  report   <scene>                    자산 XR-ready 게이트 판정
  view     <scene> [--port 8080]      3D 가우시안 뷰어(브라우저)
  localize <scene> <query.png>        한 프레임 위치추정(+선택 렌더)
  run      <scene> <frames_dir>       프레임 스트림 localize→render 루프

설계: 얇은 현관. 무거운 의존(gsplat 등)은 각 서브커맨드 내부에서 lazy import →
이 모듈 자체는 gsplat 없이 import 가능. 실제 일은 pipeline.orchestrator / pipeline.merged_map /
scripts(view_gsplat, build_report)가 한다.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def cmd_build(a):
    from pipeline.config import load_config
    from pipeline.orchestrator import build
    cfg = load_config(a.scene)
    build(cfg, force=a.force)
    if not a.no_showcase:
        _run_showcase(cfg.scene)


def _run_showcase(scene):
    """build 후 결과 팩(mp4·지표·갤러리·localize→render·궤적) 자동 생성."""
    subprocess.run([sys.executable, str(ROOT / "scripts/make_showcase.py"), scene], check=False)


def cmd_showcase(a):
    _run_showcase(a.scene)


def cmd_report(a):
    from pipeline.config import load_config
    cfg = load_config(a.scene)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_report.py"),
                    "--scene", cfg.scene, "--tag", cfg.train.tag], check=False)


def cmd_view(a):
    from pipeline.config import load_config
    cfg = load_config(a.scene)
    if not cfg.asset_ply.exists():
        sys.exit(f"자산 없음: {cfg.asset_ply} — 먼저 `xrsplat build {a.scene}`")
    subprocess.run([sys.executable, str(ROOT / "scripts/view_gsplat.py"),
                    "--ply", str(cfg.asset_ply), "--port", str(a.port)], check=False)


def cmd_localize(a):
    import cv2
    import numpy as np
    from pipeline.merged_map import MergedMap
    mm = MergedMap(a.scene, localizer=a.localizer)
    rgb = cv2.imread(a.query)[:, :, ::-1].astype(np.float32) / 255.0
    pr = mm.localize(rgb)
    print(f"[localize] state={pr.state}  confidence={pr.confidence:.3f}")
    print("[localize] T_map_cam (Tcw) =\n" + np.array2string(np.asarray(pr.T_map_cam), precision=4))
    if a.render_out and pr.state == "OK":
        im = mm.render(pr.T_map_cam)
        cv2.imwrite(a.render_out, (im[:, :, ::-1] * 255).astype(np.uint8))
        print(f"[localize] render @ found pose → {a.render_out}")


def cmd_run(a):
    import cv2
    import numpy as np
    from pipeline.merged_map import MergedMap
    d = Path(a.frames_dir)
    names = sorted([p for p in d.glob("*.png")], key=lambda p: p.name)[: a.n]
    if not names:
        sys.exit(f"프레임 없음: {d}/*.png")
    frames = [cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255.0 for p in names]
    mm = MergedMap(a.scene, localizer=a.localizer)
    res = mm.run(frames, conf_thresh=a.conf_thresh)
    ok = sum(1 for r in res if r["pose"].state == "OK")
    print(f"[run] {len(res)} frames | OK {ok}/{len(res)} | "
          f"conf median {np.median([r['pose'].confidence for r in res]):.3f}")


def main():
    ap = argparse.ArgumentParser(prog="xrsplat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="오프라인 파이프라인 01→08 + report + 결과 팩")
    b.add_argument("scene"); b.add_argument("--force", action="store_true")
    b.add_argument("--no-showcase", action="store_true", help="끝에 결과 팩 자동생성 생략")
    b.set_defaults(func=cmd_build)

    sc = sub.add_parser("showcase", help="결과 팩 생성(mp4·지표·갤러리·localize→render·궤적)")
    sc.add_argument("scene"); sc.set_defaults(func=cmd_showcase)

    r = sub.add_parser("report", help="XR-ready 게이트 판정")
    r.add_argument("scene"); r.set_defaults(func=cmd_report)

    v = sub.add_parser("view", help="3D 가우시안 뷰어")
    v.add_argument("scene"); v.add_argument("--port", type=int, default=8080)
    v.set_defaults(func=cmd_view)

    lo = sub.add_parser("localize", help="한 프레임 위치추정(+선택 렌더)")
    lo.add_argument("scene"); lo.add_argument("query")
    lo.add_argument("--localizer", choices=["pnp", "photometric"], default="pnp")
    lo.add_argument("--render-out", help="찾은 포즈로 렌더해 저장할 png")
    lo.set_defaults(func=cmd_localize)

    ru = sub.add_parser("run", help="프레임 스트림 localize→render 루프")
    ru.add_argument("scene"); ru.add_argument("frames_dir")
    ru.add_argument("--n", type=int, default=40); ru.add_argument("--conf-thresh", type=float, default=0.5)
    ru.add_argument("--localizer", choices=["pnp", "photometric"], default="pnp")
    ru.set_defaults(func=cmd_run)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
