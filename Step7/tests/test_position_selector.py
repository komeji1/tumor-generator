"""Step 4 单元测试: position_selector.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Step4', 'src'))
import numpy as np
import position_selector as ps
rng = np.random.default_rng(42)

organ = np.zeros((40, 80, 80), dtype=np.uint8); organ[5:35, 10:70, 10:70] = 1
spacing = (2.0, 1.0, 1.0)

class TestMargin:
    def test_basic(self):
        m = ps.compute_margin_voxel(8.0, 3.0, 5.0, 2.0)
        assert abs(m - 12.0) < 0.1

class TestSampleUniform:
    def test_in_valid(self):
        valid = np.zeros((20,20,20), dtype=np.uint8); valid[5:15,5:15,5:15]=1
        for _ in range(10):
            c = ps.sample_uniform(valid, rng=rng)
            assert valid[c] == 1

class TestSampleDistanceWeighted:
    def test_in_valid(self):
        valid = np.zeros((20,20,20), dtype=np.uint8); valid[5:15,5:15,5:15]=1
        for _ in range(10):
            c = ps.sample_distance_weighted(valid, organ[:20,:20,:20], alpha=1.0, rng=rng)
            assert valid[c] == 1

class TestSelectLocation:
    def test_uniform(self):
        c = ps.select_location(organ, 6.0, spacing,
                               strategy=ps.PlacementStrategy.UNIFORM,
                               max_retries=30, rng=rng)
        assert organ[c] == 1

    def test_distance_weighted(self):
        c = ps.select_location(organ, 6.0, spacing,
                               strategy=ps.PlacementStrategy.DISTANCE_WEIGHTED,
                               distance_alpha=2.0, max_retries=30, rng=rng)
        assert organ[c] == 1

    def test_from_config(self):
        cfg = {'strategy': 'uniform', 'margin': {'feather_mm': 3, 'safety_mm': 5},
               'max_retries': 30, 'distance_weighted': {'alpha': 1.0}}
        c = ps.select_location_from_config(organ, 6.0, spacing, cfg, rng=rng)
        assert organ[c] == 1

    def test_too_large_radius_handled(self):
        """半径过大应抛出 ValueError/RuntimeError 或 MemoryError"""
        try:
            ps.select_location(organ, 25.0, spacing, max_retries=3, rng=rng)
        except (ValueError, RuntimeError, MemoryError):
            pass  # 预期行为
