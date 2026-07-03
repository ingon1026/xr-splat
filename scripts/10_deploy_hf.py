#!/usr/bin/env python3
"""10_deploy_hf.py — 웹 데모를 HuggingFace에 배포 (dataset repo + Static Space).

하는 일:
  1. <hf-id>/xr-splat-assets (dataset) 생성 + .splat/.json 업로드  ← 자산 호스팅(CORS 허용)
  2. spaces/ 를 임시 스테이징으로 복사, index.html의 __HF_ID__ 치환
  3. <hf-id>/xr-splat-demo (Space, sdk=static) 생성 + 업로드
렌더링은 방문자 브라우저(WebGL) — 서버 GPU 불필요, 무료 Static Space.

사전 조건: `hf auth login` (write 토큰) 1회.

usage:
  python scripts/10_deploy_hf.py --hf-id <username> [--scene replica_office0]
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-id", required=True, help="HuggingFace 계정 id")
    ap.add_argument("--scene", default="replica_office0")
    ap.add_argument("--assets-repo", default="xr-splat-assets")
    ap.add_argument("--space-repo", default="xr-splat-demo")
    args = ap.parse_args()

    api = HfApi()
    who = api.whoami()  # 미로그인 시 여기서 명확히 실패
    print(f"[10] 로그인: {who['name']}")

    splat = ROOT / "outputs" / args.scene / "web" / f"{args.scene}.splat"
    if not splat.exists():
        sys.exit(f"[10] .splat 없음: {splat}\n     먼저: python scripts/09_export_web.py --scene {args.scene}")

    # ── 1) dataset repo: .splat 호스팅 ──
    assets_id = f"{args.hf_id}/{args.assets_repo}"
    api.create_repo(assets_id, repo_type="dataset", exist_ok=True)
    for f in (splat, splat.with_suffix(".json")):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=assets_id, repo_type="dataset")
        print(f"[10] 업로드: {f.name} → datasets/{assets_id}")
    asset_url = f"https://huggingface.co/datasets/{assets_id}/resolve/main/{splat.name}"

    # ── 2) spaces/ 스테이징 + __HF_ID__ 치환 ──
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "spaces"
        shutil.copytree(ROOT / "spaces", stage)
        idx = stage / "index.html"
        idx.write_text(idx.read_text().replace("__HF_ID__", args.hf_id))

        # ── 3) Static Space 생성 + 업로드 ──
        space_id = f"{args.hf_id}/{args.space_repo}"
        api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=space_id, repo_type="space")

    space_url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\n[10] 완료 ✅")
    print(f"[10] 자산: {asset_url}")
    print(f"[10] 데모: {space_url}")
    print(f"[10] CORS 확인: curl -sI '{asset_url}' | grep -i access-control")


if __name__ == "__main__":
    main()
