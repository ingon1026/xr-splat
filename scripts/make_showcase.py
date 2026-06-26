#!/usr/bin/env python3
"""make_showcase.py — build 후 scene별 '결과 팩' 자동 생성 (mp4·지표·갤러리·localize→render·궤적·.ply 안내).

생성물 → outputs/<scene>/results/:
  gallery.png            렌더 N뷰 (real | render, PSNR)
  localize_to_render.png non-KF query를 PnP localize → 찾은 포즈로 렌더 (전 과정 증거)
  flythrough.mp4         KF 궤적 보간 walk-through (3D 증거)
  localization_path.png  SLAM 궤적 위 localized query (top-down)
  showcase_metrics.json/md  render-vs-real·FPS·reloc·자산 통계

scene yaml/이름으로 동작. 렌더는 항상 풀 자산(cfg.asset_ply). usage: make_showcase.py <scene> [--frames 6]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import imageio
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.config import load_config  # noqa: E402
from pipeline.backproject import read_colmap_images, read_colmap_cameras, colmap_world_RT  # noqa: E402

DEV = "cuda"


def _lab(img, t, c=(255, 255, 255)):
    img = (img.clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    cv2.rectangle(img, (0, 0), (430, 32), (0, 0, 0), -1)
    cv2.putText(img, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.scene)
    if not cfg.asset_ply.exists():
        sys.exit(f"자산 없음: {cfg.asset_ply} — 먼저 build")
    res = cfg.out_dir / "results"; res.mkdir(parents=True, exist_ok=True)

    from pipeline.gsplat_io import load_ply, render
    proc, sp = cfg.proc_dir, cfg.sparse_dir
    W, H, fx, fy, cx, cy = read_colmap_cameras(sp / "cameras.txt")
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float32)
    Kt = torch.tensor(K_np, device=DEV)
    imgs = read_colmap_images(sp / "images.txt")
    names = sorted(imgs, key=lambda x: float(Path(x).stem))
    g = load_ply(str(cfg.asset_ply), DEV)
    n_gauss = int(g["means"].shape[0])

    def vm_of(n):
        p = imgs[n]; vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = Rot.from_quat([p[1], p[2], p[3], p[0]]).as_matrix(); vm[:3, 3] = p[4:7]
        return vm

    def rd(vm):
        with torch.no_grad():
            return render(g, torch.tensor(vm, device=DEV), Kt, W, H).clamp(0, 1).cpu().numpy()

    m = {"scene": cfg.scene, "tag": cfg.train.tag, "n_gauss": n_gauss,
         "asset_ply_mb": round(cfg.asset_ply.stat().st_size / 1e6, 1)}

    # ── 1) 갤러리 (KF 골고루) ──
    rows = []
    pick = [names[int(len(names) * f)] for f in np.linspace(0.05, 0.95, args.frames)]
    psnrs = []
    for n in pick:
        gt = cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0
        im = rd(vm_of(n)); ps = -10 * np.log10(max(((im - gt) ** 2).mean(), 1e-10)); psnrs.append(ps)
        tw = 560; h = int(H * tw / W)
        rows.append(np.hstack([_lab(cv2.resize(gt, (tw, h)), "REAL"),
                               _lab(cv2.resize(im, (tw, h)), f"RENDER {ps:.1f}dB", (120, 255, 120))]))
    cv2.imwrite(str(res / "gallery.png"), np.vstack(rows))
    m["gallery_psnr_median"] = round(float(np.median(psnrs)), 2)

    # ── 2) fly-through (KF 보간 walk-through) ──
    win = names[:: max(1, len(names) // 36)][:36]
    Rs = [vm_of(n)[:3, :3] for n in win]; Cs = np.array([-vm_of(n)[:3, :3].T @ vm_of(n)[:3, 3] for n in win])
    key = np.arange(len(win)); slerp = Slerp(key, Rot.from_matrix(np.array(Rs)))
    frames = []
    for t in np.linspace(0, len(win) - 1, 120):
        Rcw = slerp([t])[0].as_matrix(); C = np.array([np.interp(t, key, Cs[:, k]) for k in range(3)])
        vm = np.eye(4, dtype=np.float32); vm[:3, :3] = Rcw; vm[:3, 3] = -Rcw @ C
        frames.append((rd(vm) * 255).astype(np.uint8))
    imageio.mimsave(str(res / "flythrough.mp4"), [cv2.resize(f, (W // 2, H // 2)) for f in frames], fps=24, quality=7)

    # ── 3) feature_map (없으면 생성) + localize→render + 궤적 plot ──
    if not cfg.feature_map.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/build_feature_map.py"), "--scene", cfg.scene], check=False)
    if cfg.feature_map.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("pnp_localizer", ROOT / "scripts/pnp_localizer.py")
        pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
        from pipeline.runtime import OK
        loc = pl.PnPLocalizer(cfg.feature_map, K_np)
        kf = set(names); allrgb = [l.split()[1].split('/')[-1] for l in open(proc / "rgb.txt") if l.strip() and not l.startswith('#')]
        klo, khi = float(names[0][:-4]), float(names[-1][:-4])
        cand = sorted([n for n in allrgb if n not in kf and klo <= float(n[:-4]) <= khi], key=lambda x: float(x[:-4]))
        qpick = [cand[int(len(cand) * f)] for f in np.linspace(0, 0.999, max(args.frames, 6))] if cand else pick
        lr_rows, qc, oks, mss, qps = [], [], 0, [], []
        for n in qpick:
            gt = cv2.imread(str(proc / "rgb" / n))[:, :, ::-1].astype(np.float32) / 255.0
            t0 = time.perf_counter(); pr = loc.relocalize(gt); ms = (time.perf_counter() - t0) * 1000; mss.append(ms)
            vm = np.asarray(pr.T_map_cam, np.float32)
            if pr.state == OK:
                oks += 1; qc.append(-vm[:3, :3].T @ vm[:3, 3])
            im = rd(vm); ps = -10 * np.log10(max(((im - gt) ** 2).mean(), 1e-10)); qps.append(ps)
            tw = 560; h = int(H * tw / W)
            lr_rows.append(np.hstack([_lab(cv2.resize(gt, (tw, h)), "REAL (non-KF)"),
                                      _lab(cv2.resize(im, (tw, h)), f"RENDER@PnP {ps:.1f}dB {ms:.0f}ms", (120, 255, 120))]))
        cv2.imwrite(str(res / "localize_to_render.png"), np.vstack(lr_rows))
        m["localize"] = {"global_reloc": f"{oks}/{len(qpick)}", "fps_median": round(1000 / np.median(mss), 1),
                         "render_vs_real_psnr": round(float(np.median(qps)), 2)}
        # 궤적 plot
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            kfc = np.array([-vm_of(n)[:3, :3].T @ vm_of(n)[:3, 3] for n in names]); qc = np.array(qc)
            ext = kfc.max(0) - kfc.min(0); a, b = sorted(np.argsort(ext)[-2:]); L = {0: "X", 1: "Y", 2: "Z"}
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.plot(kfc[:, a], kfc[:, b], "-", c="#2a9d4a", lw=1.6, label=f"SLAM trajectory ({len(names)} KF)")
            if len(qc):
                ax.scatter(qc[:, a], qc[:, b], s=70, c="red", ec="k", zorder=5, label=f"PnP-localized query ({oks} ok)")
            ax.set_xlabel(f"{L[a]} (m)"); ax.set_ylabel(f"{L[b]} (m)"); ax.set_aspect("equal")
            ax.set_title(f"Localization on SLAM map — {cfg.scene}"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(str(res / "localization_path.png"), dpi=120)
        except Exception as e:
            print(f"[showcase] path plot skip: {e}")

    (res / "showcase_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
    md = [f"# {cfg.scene} — showcase 결과 팩\n",
          f"- 자산: {n_gauss:,} gaussians, {m['asset_ply_mb']} MB ({cfg.asset_ply})",
          f"- 렌더 품질(갤러리 median): {m.get('gallery_psnr_median')} dB"]
    if "localize" in m:
        lz = m["localize"]; md.append(f"- 위치추정: 전역 {lz['global_reloc']}, {lz['fps_median']} FPS, render-vs-real {lz['render_vs_real_psnr']} dB")
    md += ["", "결과물: gallery.png · flythrough.mp4 · localize_to_render.png · localization_path.png · scene.ply/scene_lite.ply"]
    (res / "showcase.md").write_text("\n".join(md) + "\n")
    print(f"[showcase] {cfg.scene} → {res}")
    print("  " + json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
