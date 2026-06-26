#!/usr/bin/env python3
"""make_merged_summary.py — 합쳐진 맵 성능 한 장 종합 그림 (재현 가능).
4개 결과 PNG + 70/30 동작범위(envelope) 표 + 결론 텍스트 → merged_map_summary.png.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "docs/assets/report"
PANELS = [
    (R / "quality_mcmc_AB.png", "1) Render quality: 23.82 -> 27.96 dB (MCMC-2M)"),
    (R / "pose_sensitivity_curve.png", "2) Pose tolerance: 1deg = -7dB, 1cm = -2.8dB"),
    (R / "convergence_basin.png", "3) Photometric basin: ~20cm / ~12deg robust"),
    (R / "photometric_reloc_AB.png", "4) Self-localization (in-region): 5cm/3deg -> 0.18cm"),
]

fig, axes = plt.subplots(2, 3, figsize=(19, 9))
fig.suptitle("Merged-Map Performance  (decoupled SLAM x Gaussian Splatting)", fontsize=17, fontweight="bold")

for ax, (png, title) in zip(axes.flat[:4], PANELS):
    if png.exists():
        ax.imshow(plt.imread(str(png)))
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.axis("off")

# Panel 5 — 70/30 envelope table
ax = axes.flat[4]; ax.axis("off")
ax.set_title("5) Novel 70/30 -> Operating Envelope", fontsize=10.5, loc="left", color="#222")
tbl = ax.table(
    cellText=[
        ["In-capture (interp.)", "0.18 cm", "28 dB", "works"],
        ["Out-capture (extrap.)", "SfM 0.73 cm", "10 dB", "no signal"],
    ],
    colLabels=["region", "localize", "render", "photometric"],
    cellLoc="center", loc="center", bbox=[0.0, 0.45, 1.0, 0.45])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0: cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    elif r == 1: cell.set_facecolor("#e8f5e9")
    elif r == 2: cell.set_facecolor("#fdecea")
ax.text(0.5, 0.30, "Bottleneck = capture coverage, NOT localization.",
        ha="center", fontsize=10, fontweight="bold", color="#b03a2e", transform=ax.transAxes)
ax.text(0.5, 0.16, "Localization generalizes to novel views (0.73cm).\n"
        "Render does not extrapolate to uncaptured space (10dB).\n"
        "Photometric(B) = in-region refiner, not a novel-view localizer.",
        ha="center", va="top", fontsize=8.5, color="#333", transform=ax.transAxes)

# Panel 6 — takeaway
ax = axes.flat[5]; ax.axis("off")
ax.text(0.5, 0.92, "Takeaway", ha="center", fontsize=12, fontweight="bold", transform=ax.transAxes)
ax.text(0.04, 0.78,
        "PROVEN\n"
        "  - SLAM-frame ⊗ Gaussian: shared coordinate frame\n"
        "  - Photorealistic asset: 27.96 dB (in-capture)\n"
        "  - Re-entry by SfM reloc -> render: works\n"
        "  - Gaussian map self-localizes (in-region)\n\n"
        "BOUNDED BY\n"
        "  - Capture coverage (no render extrapolation)\n"
        "  - Pose budget: <1deg / <1-2cm for peak quality\n\n"
        "OPEN\n"
        "  - Real-time loop, HMD/VIO, guarded photometric\n"
        "  - Reproducibility / multi-scene",
        ha="left", va="top", fontsize=8.6, family="monospace", color="#222", transform=ax.transAxes)

fig.tight_layout(rect=[0, 0, 1, 0.96])
op = R / "merged_map_summary.png"
fig.savefig(str(op), dpi=115)
print(f"[summary] -> {op}")
