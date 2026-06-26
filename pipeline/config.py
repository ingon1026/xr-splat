"""config.py — xr-splat scene 설정 단일 진실원 (통합 실행의 공유 계약).

scene별 yaml 한 파일이 input·단계별 knob·경로를 모두 정의한다. 흩어진 인자(05 임계,
06 strategy/cap, scene 이름, exclude)를 흡수. intrinsics.json(01이 만드는 카메라 진실원)은
그대로 두고 config가 참조한다.

usage:
    from pipeline.config import load_config
    cfg = load_config("configs/ros2_bag2_home_rgbd_orbframe.yaml")   # 또는 load_config(scene_name)
    cfg.asset_ply  # 풀자산 .ply 경로 (orchestrator·MergedMap·CLI가 동일하게 해석)
"""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


@dataclass
class InputCfg:
    type: str = "bag"            # bag | dir
    path: str = ""               # data/raw/<>.bag  또는 TUM/Replica 디렉토리
    depth_scale: int = 1000      # D455 DepthMapFactor (01_extract_bag)


@dataclass
class SlamCfg:
    nfeatures: int = 1250        # 02_run_orbslam3


@dataclass
class ValidateCfg:
    enabled: bool = True
    baseline: List[float] = field(default_factory=lambda: [0.2, 0.6])  # 05 페어 거리
    pass_cm: float = 3.0         # 05 임계 (대형 D455는 4.0)
    crop: float = 0.6            # 중앙 크롭


@dataclass
class TrainCfg:
    strategy: str = "mcmc"       # mcmc | default
    cap_max: int = 2_000_000     # mcmc 가우시안 상한
    iters: int = 30000
    refine_stop: int = 15000     # default 전략용
    depth_lambda: float = 0.2
    holdout_every: int = 8
    tag: str = "mcmc2m"          # 출력 gsplat_<tag>


@dataclass
class PostCfg:
    opacity: float = 0.05        # 08 prune 임계
    sh_out: int = 1              # 08 SH 축소 후 degree


@dataclass
class SceneConfig:
    scene: str
    orb_scene: Optional[str] = None      # 프레임무결성 기준 ORB run (없으면 scene-'_orbframe')
    input: InputCfg = field(default_factory=InputCfg)
    slam: SlamCfg = field(default_factory=SlamCfg)
    validate: ValidateCfg = field(default_factory=ValidateCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    postprocess: PostCfg = field(default_factory=PostCfg)
    exclude_list: Optional[str] = None   # data/processed/<scene>/exclude.txt 등

    # ── 경로 해석 (모든 통합 컴포넌트가 동일하게 사용 — 하드코딩 제거) ──
    @property
    def proc_dir(self) -> Path:
        return ROOT / "data" / "processed" / self.scene

    @property
    def sparse_dir(self) -> Path:
        return self.proc_dir / "colmap" / "sparse" / "0"

    @property
    def intrinsics(self) -> Path:
        return self.proc_dir / "intrinsics.json"

    @property
    def out_dir(self) -> Path:
        return ROOT / "outputs" / self.scene

    @property
    def slam_traj(self) -> Path:
        return self.out_dir / "slam" / "KeyFrameTrajectory.txt"

    @property
    def asset_dir(self) -> Path:
        return self.out_dir / f"gsplat_{self.train.tag}"

    @property
    def asset_ply(self) -> Path:
        return self.asset_dir / "scene.ply"

    @property
    def asset_lite(self) -> Path:
        return self.asset_dir / "scene_lite.ply"

    @property
    def feature_map(self) -> Path:
        return self.out_dir / "feature_map.npz"

    @property
    def orb_ref_traj(self) -> Path:
        """프레임무결성(build_report) 기준 ORB 궤적."""
        ref = self.orb_scene or (self.scene[:-len("_orbframe")] if self.scene.endswith("_orbframe") else self.scene)
        return ROOT / "outputs" / ref / "slam" / "KeyFrameTrajectory.txt"


def _build(cls, data):
    """중첩 dataclass에 dict를 재귀 병합(기본값 보존)."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        if f.name not in data or data[f.name] is None:
            continue
        v = data[f.name]
        # 중첩 dataclass면 재귀
        nested = {"input": InputCfg, "slam": SlamCfg, "validate": ValidateCfg,
                  "train": TrainCfg, "postprocess": PostCfg}.get(f.name)
        kwargs[f.name] = _build(nested, v) if nested else v
    return cls(**kwargs)


def load_config(path_or_scene) -> SceneConfig:
    """yaml 경로 또는 scene 이름(configs/<scene>.yaml)에서 SceneConfig 로드."""
    p = Path(path_or_scene)
    if not p.exists():
        p = CONFIG_DIR / f"{path_or_scene}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"config 없음: {path_or_scene} (또는 {p})")
    data = yaml.safe_load(p.read_text()) or {}
    if "scene" not in data:
        raise ValueError(f"config에 'scene' 필수: {p}")
    return _build(SceneConfig, data)


if __name__ == "__main__":  # 스모크
    import sys
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "ros2_bag2_home_rgbd_orbframe")
    print(f"scene={cfg.scene} tag={cfg.train.tag} strategy={cfg.train.strategy} cap={cfg.train.cap_max}")
    print(f"asset_ply={cfg.asset_ply}")
    print(f"feature_map={cfg.feature_map}")
    print(f"orb_ref_traj={cfg.orb_ref_traj}")
    assert cfg.asset_ply.name == "scene.ply"
    print("config OK")
