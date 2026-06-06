"""Step 1 单元测试: utils.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Step1', 'src'))
import numpy as np
import utils as ut
rng = np.random.default_rng(42)

class TestCoordinateTransforms:
    def test_get_spacing_isotropic(self):
        aff = np.eye(4); aff[0,0]=aff[1,1]=aff[2,2]=1.5
        assert ut.get_spacing(aff) == (1.5, 1.5, 1.5)

    def test_get_spacing_anisotropic(self):
        aff = np.diag([0.5, 0.5, 3.0, 1.0])
        assert ut.get_spacing(aff) == (3.0, 0.5, 0.5)

    def test_voxel_mm_roundtrip(self):
        aff = np.eye(4); aff[2,2]=2.0
        for _ in range(100):
            v = tuple(float(x) for x in rng.integers(0, 50, 3))
            mm = ut.voxel_to_mm(v, aff)
            v2 = ut.mm_to_voxel(mm, aff)
            assert np.allclose(v, v2, atol=1e-6)

    def test_batch_transform(self):
        aff = np.eye(4)
        coords = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
        result = ut.voxel_to_mm(coords, aff)
        assert result.shape == (3, 3)

class TestHUProcessing:
    def test_clip_lower_bound(self):
        data = np.array([-1000, -175, -174], dtype=np.int16)
        clipped = ut.clip_hu(data)
        assert clipped[0] == -175

    def test_clip_upper_bound(self):
        data = np.array([249, 250, 1000], dtype=np.int16)
        clipped = ut.clip_hu(data)
        assert clipped[-1] == 250

    def test_clip_preserves_dtype(self):
        data = np.array([0, 100], dtype=np.int16)
        assert ut.clip_hu(data).dtype == np.int16

    def test_normalize_range(self):
        data = np.array([-175, 0, 250], dtype=np.float32)
        n = ut.normalize_hu(data)
        assert abs(n[0]) < 1e-6
        assert abs(n[-1] - 1.0) < 1e-6

class TestMorphology:
    def test_erode_reduces_volume(self):
        mask = np.zeros((20,20,20), dtype=np.uint8); mask[5:15,5:15,5:15]=1
        eroded = ut.erode_mask(mask, 2.0)
        assert eroded.sum() < mask.sum()

    def test_erode_zero_radius(self):
        mask = np.ones((5,5,5), dtype=np.uint8)
        assert np.array_equal(ut.erode_mask(mask, 0), mask)

    def test_dilate_increases_volume(self):
        mask = np.zeros((20,20,20), dtype=np.uint8); mask[8:12,8:12,8:12]=1
        dilated = ut.dilate_mask(mask, 2.0)
        assert dilated.sum() > mask.sum()

class TestGeometry:
    def test_ellipsoid_center_zero(self):
        d = ut.compute_ellipsoid_dist((10,10,10),(5,5,5),(3,2,2))
        assert abs(d[5,5,5]) < 1e-6

    def test_ellipsoid_surface_one(self):
        d = ut.compute_ellipsoid_dist((10,10,10),(5,5,5),(2,2,2))
        assert 0.9 < d[5,5,7] < 1.1

    def test_volume_isotropic(self):
        v = ut.volume_from_radius(10.0, (1,1,1))
        expected = int((4/3)*np.pi*1000)
        assert abs(v - expected) < 10

    def test_valid_region_nonempty(self):
        organ = np.zeros((20,20,20), dtype=np.uint8); organ[3:17,3:17,3:17]=1
        valid = ut.compute_valid_region(organ, 2.0)
        assert valid.sum() > 0

class TestRandomSampling:
    def test_sample_in_valid(self):
        valid = np.zeros((10,10,10), dtype=np.uint8); valid[3:7,3:7,3:7]=1
        c = ut.random_sample_valid(valid, n=5, rng=rng)
        assert all(valid[z,y,x]==1 for z,y,x in c)

    def test_axis_ratios_conserve_volume(self):
        for _ in range(50):
            rz, ry, rx = ut.random_axis_ratios(rng=rng)
            assert abs(rz*ry*rx - 1.0) < 1e-6

class TestElasticDeformation:
    def test_field_shape(self):
        f = ut.generate_elastic_deformation_field((16,16,16), alpha=5, sigma=2, rng=rng)
        assert f.shape == (3, 16, 16, 16)

    def test_deformation_preserves_range(self):
        # 小 alpha 确保体积变化不大
        mask = np.zeros((32,32,32), dtype=np.uint8); mask[8:24,8:24,8:24]=1
        f = ut.generate_elastic_deformation_field((32,32,32), alpha=3, sigma=3, rng=rng)
        d = ut.apply_deformation(mask, f)
        assert set(np.unique(d)).issubset({0, 1})
        vc = abs(float(d.sum())-float(mask.sum()))/float(mask.sum())
        assert vc < 0.3, f"vol_change={vc:.1%}"

class TestMisc:
    def test_ensure_uint8(self):
        assert ut.ensure_uint8(np.array([0.2, 0.8])).dtype == np.uint8
        assert np.array_equal(ut.ensure_uint8(np.array([0, 1, 0])), [0,1,0])

    def test_bbox(self):
        mask = np.zeros((10,10,10), dtype=np.uint8); mask[2:8,3:7,4:6]=1
        zs, ys, xs = ut.get_bbox(mask)
        assert zs.start == 2 and zs.stop == 8
        assert xs.start == 4 and xs.stop == 6
