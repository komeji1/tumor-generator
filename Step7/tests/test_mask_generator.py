"""Step 5 单元测试: mask_generator.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Step5', 'src'))
import numpy as np
import mask_generator as mg
rng = np.random.default_rng(42)

shape = (40, 80, 80); center = (20, 40, 40); radius_mm = 10.0
spacing = (2.0, 1.0, 1.0)
basic_cfg = {
    'axis_ratio_range': [0.8, 1.2],
    'elastic_deformation': {'enabled': False},
    'salt_noise': {'enabled': False},
    'gaussian_filter': {'enabled': False},
    'scaling_clipping': {'enabled': True},
}
full_cfg = {
    'axis_ratio_range': [0.8, 1.2],
    'elastic_deformation': {'enabled': True, 'alpha': 10, 'sigma': 2},
    'salt_noise': {'enabled': True, 'probability': 0.02},
    'gaussian_filter': {'enabled': True, 'sigma_mm': 0.5},
    'scaling_clipping': {'enabled': True},
}

class TestRadii:
    def test_positive(self):
        rz, ry, rx = mg.compute_radii_from_mm(radius_mm, spacing, rng=rng)
        assert all(r > 0 for r in (rz, ry, rx))

class TestEllipsoid:
    def test_basic(self):
        m = mg.create_ellipsoid(shape, center, (5,5,5))
        assert m.dtype == np.uint8 and m.sum() > 100
        assert m[center[0], center[1], center[2]] == 1

class TestPipeline:
    def test_create_mask_basic(self):
        m = mg.create_mask(center, radius_mm, shape, spacing, basic_cfg, rng=rng)
        assert m.dtype == np.uint8 and m.sum() > 0

    def test_create_mask_full(self):
        m = mg.create_mask(center, radius_mm, shape, spacing, full_cfg, rng=rng)
        assert m.dtype == np.uint8 and m.sum() > 0

    def test_diversity(self):
        volumes = set()
        for i in range(5):
            r = np.random.default_rng(i)
            m = mg.create_mask(center, radius_mm, shape, spacing, full_cfg, rng=r)
            volumes.add(int(m.sum()))
        assert len(volumes) >= 3

class TestNifti:
    def test_save_and_reload(self):
        import tempfile, nibabel, shutil
        tmp = tempfile.mkdtemp()
        try:
            m = mg.create_mask(center, radius_mm, shape, spacing, full_cfg, rng=rng)
            aff = np.eye(4); aff[0,0]=aff[1,1]=1.0; aff[2,2]=2.0
            p = mg.mask_to_nifti(m, aff, os.path.join(tmp, 'test.nii.gz'))
            img = nibabel.load(p)
            assert img.get_fdata().shape == shape
            assert np.allclose(img.affine, aff)
        finally:
            shutil.rmtree(tmp)
