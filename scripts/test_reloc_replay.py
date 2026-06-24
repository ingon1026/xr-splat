import numpy as np
from render_reloc_replay import twc_line_to_viewmat


def test_identity_pose_gives_identity_viewmat():
    # Twc=identity(카메라 원점·정렬) → Tcw=identity
    vm = twc_line_to_viewmat(0, 0, 0, 0, 0, 0, 1)
    assert np.allclose(vm, np.eye(4), atol=1e-6)


def test_translation_only_inverts():
    # 카메라가 world (1,2,3)에 있고 회전 없음 → Tcw t = -C
    vm = twc_line_to_viewmat(1, 2, 3, 0, 0, 0, 1)
    assert np.allclose(vm[:3, 3], [-1, -2, -3], atol=1e-6)
    assert np.allclose(vm[:3, :3], np.eye(3), atol=1e-6)
