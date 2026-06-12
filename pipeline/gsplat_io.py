"""gsplat_io.py — 3DGS PLY 로드/저장 + 렌더 (Phase 6 평가·후처리 공유)."""
import numpy as np
import torch
from gsplat import rasterization


def load_ply(path, device="cuda"):
    """표준 3DGS PLY → dict(means, quats, scales[log], opacities[logit], sh[N,K,3], sh_degree)."""
    f = open(path, "rb")
    hdr = b""
    while b"end_header" not in hdr:
        hdr += f.readline()
    lines = hdr.decode().splitlines()
    props = [l.split()[-1] for l in lines if l.startswith("property")]
    N = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    arr = np.frombuffer(f.read(N * len(props) * 4), dtype=np.float32).reshape(N, len(props))
    c = {p: i for i, p in enumerate(props)}
    g = lambda *ks: torch.tensor(arr[:, [c[k] for k in ks]], device=device)
    nrest = sum(p.startswith("f_rest_") for p in props)
    sh = np.concatenate([arr[:, [c[f"f_dc_{i}"] for i in range(3)]][:, None, :],
                         arr[:, [c[f"f_rest_{i}"] for i in range(nrest)]].reshape(N, -1, 3)], 1)
    return dict(
        means=g("x", "y", "z"),
        quats=g("rot_0", "rot_1", "rot_2", "rot_3"),
        scales=g("scale_0", "scale_1", "scale_2"),
        opacities=torch.tensor(arr[:, c["opacity"]], device=device),
        sh=torch.tensor(sh, device=device),
        sh_degree=int(round((sh.shape[1]) ** 0.5)) - 1,
    )


def render(g, viewmat, K, W, H):
    """gaussians dict + viewmat(Tcw 4x4) + K → rgb [H,W,3] (0~1)."""
    out, _, _ = rasterization(
        g["means"], g["quats"], torch.exp(g["scales"]), torch.sigmoid(g["opacities"]),
        g["sh"], viewmat[None], K[None], W, H, sh_degree=g["sh_degree"], render_mode="RGB")
    return out[0, ..., :3].clamp(0, 1)


def save_ply(path, g):
    """dict(means, quats, scales[log], opacities[logit], sh[N,K,3]) → 표준 3DGS PLY."""
    p = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in g.items()}
    N = p["means"].shape[0]
    f_dc = p["sh"][:, :1, :].reshape(N, -1)
    f_rest = p["sh"][:, 1:, :].reshape(N, -1)
    fields = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(f_dc.shape[1])] \
        + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] \
        + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([p["means"], np.zeros((N, 3), np.float32), f_dc, f_rest,
                           p["opacities"].reshape(N, 1), p["scales"], p["quats"]], 1).astype(np.float32)
    with open(path, "wb") as fo:
        fo.write(b"ply\nformat binary_little_endian 1.0\n")
        fo.write(f"element vertex {N}\n".encode())
        for fld in fields:
            fo.write(f"property float {fld}\n".encode())
        fo.write(b"end_header\n")
        fo.write(data.tobytes())
