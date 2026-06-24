import numpy as np
from export_reloc_poses import colmap_image_to_twc_tum

def test_identity_tcw_gives_origin_twc():
    # Tcw=identity → 카메라 중심 원점, 회전 identity
    cx,cy,cz, qx,qy,qz,qw = colmap_image_to_twc_tum(1,0,0,0, 0,0,0)
    assert np.allclose([cx,cy,cz],[0,0,0], atol=1e-6)
    assert np.allclose([qx,qy,qz,qw],[0,0,0,1], atol=1e-6)

def test_translation_tcw_center_is_negative_R_t():
    # Tcw t=(0,0,5), R=I → 카메라 중심 C=-R^T t=(0,0,-5)
    cx,cy,cz,*_ = colmap_image_to_twc_tum(1,0,0,0, 0,0,5)
    assert np.allclose([cx,cy,cz],[0,0,-5], atol=1e-6)
