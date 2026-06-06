"""Step 3 单元测试: validator.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Step3', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Step1', 'src'))
import numpy as np
import validator as vd
import utils as ut
rng = np.random.default_rng(42)

organ = np.zeros((40,80,80), dtype=np.uint8); organ[5:35,10:70,10:70]=1
center = (20, 40, 40)
radii = (8, 8, 8)
tumor = (ut.compute_ellipsoid_dist(organ.shape, center, radii) <= 1.0).astype(np.uint8)

class TestCheckOrganVolume:
    def test_normal(self):
        ok, vol = vd.check_organ_volume(organ, min_voxels=100)
        assert ok and vol > 10000
    def test_empty(self):
        ok, vol = vd.check_organ_volume(np.zeros((5,5,5), dtype=np.uint8))
        assert not ok and vol == 0
    def test_too_small(self):
        small = np.zeros((10,10,10), dtype=np.uint8); small[4:6,4:6,4:6]=1
        ok, _ = vd.check_organ_volume(small, min_voxels=100)
        assert not ok

class TestCheckSizeRange:
    cfg = {"r_min_mm": 5, "r_max_mm": 10}
    def test_inside(self):
        ok, _ = vd.check_size_range(7.5, self.cfg)
        assert ok
    def test_below(self):
        ok, _ = vd.check_size_range(4.0, self.cfg)
        assert not ok
    def test_above(self):
        ok, _ = vd.check_size_range(12.0, self.cfg)
        assert not ok

class TestCheckInOrgan:
    def test_center_inside(self):
        ok, _ = vd.check_in_organ(center, radii, organ)
        assert ok
    def test_center_outside_organ(self):
        ok, msg = vd.check_in_organ((0, 5, 5), radii, organ)
        assert not ok
    def test_boundary_overflow(self):
        # 中心靠近器官表面 → 肿瘤可能超出
        ok, msg = vd.check_in_organ((6, 11, 11), (5,5,5), organ)
        # 应该检测到超出
        assert not ok

class TestCheckMaskNonzero:
    def test_nonzero(self):
        ok, c = vd.check_mask_nonzero(tumor)
        assert ok and c > 0
    def test_zero(self):
        ok, _ = vd.check_mask_nonzero(np.zeros((10,10,10), dtype=np.uint8))
        assert not ok

class TestCheckNotOverlapping:
    def test_no_overlap(self):
        a = np.zeros((20,20,20), dtype=np.uint8); a[2:8,2:8,2:8]=1
        b = np.zeros((20,20,20), dtype=np.uint8); b[12:18,12:18,12:18]=1
        ok, ov = vd.check_not_overlapping(b, [a])
        assert ok and ov == 0.0
    def test_overlap(self):
        a = np.zeros((20,20,20), dtype=np.uint8); a[2:10,2:10,2:10]=1
        b = np.zeros((20,20,20), dtype=np.uint8); b[6:14,6:14,6:14]=1
        ok, ov = vd.check_not_overlapping(b, [a])
        assert not ok

class TestValidateSample:
    def test_all_pass(self):
        r = vd.validate_sample(center, radii, 8.0, tumor, organ,
                               {"r_min_mm": 5, "r_max_mm": 20})
        assert r['passed']
    def test_multi_fail(self):
        r = vd.validate_sample((0,0,0), radii, 50.0,
                               np.zeros(organ.shape, dtype=np.uint8), organ,
                               {"r_min_mm": 5, "r_max_mm": 10})
        assert not r['passed'] and len(r['errors']) >= 2
