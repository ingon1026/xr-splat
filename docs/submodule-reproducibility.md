# ORB-SLAM3 Submodule Reproducibility

`third_party/ORB_SLAM3` should point at a commit available from the public upstream
repository in `.gitmodules`.

Required local changes are kept as patches under `third_party/patches/` and are applied by:

```bash
bash third_party/patches/apply.sh
```

Active patches:

- `0001-orbslam3-cxx11-to-cxx14.patch`: build fix for newer Ubuntu toolchains.
- `0002-headless-rgbd-viewer-off.patch`: disables the Pangolin viewer for WSL/headless runs.

Optional archived patch:

- `0003-optional-rgbd-localization-runner.patch.disabled`: an abandoned ORB-SLAM3 localization-mode
  runner. The current runtime path uses COLMAP PnP instead, so this patch is not applied by
  default. Rename it to `.patch` only if that experiment is intentionally revived.

Fresh-clone invariant:

1. `git submodule update --init --recursive` must be able to check out ORB-SLAM3 from the
   public upstream remote.
2. Repo-specific ORB-SLAM3 source changes must be represented in `third_party/patches/`, not
   as unpublished submodule commits.
