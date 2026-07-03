---
title: xr-splat — walk a SLAM-built photoreal room
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: Decoupled SLAM × Gaussian Splatting, rendered in your browser
---

# xr-splat demo

Interactive Gaussian-Splatting scene built by **[xr-splat](https://github.com/ingon1026/xr-splat)** —
a decoupled **ORB-SLAM3 × gsplat** pipeline: SLAM estimates the poses, Gaussian Splatting is
trained on them frozen, so the tracking map and the photoreal map share **one coordinate frame**.

**Scene:** Replica `office0` · 524k Gaussians (web asset, ~17 MB) · full asset scores
**45.35 dB PSNR** on held-out views.

**Controls:** drag to orbit · wheel to zoom · right-drag or WASD to move · touch supported.

Rendering is fully client-side WebGL ([gsplat.js](https://github.com/huggingface/gsplat.js)) —
no server GPU involved. The `.splat` asset is hosted on the Hugging Face Hub.
