"""Step 7 集成测试: 端到端流程"""
import os, sys, tempfile, shutil
_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for s in ['Step1','Step2','Step3','Step4','Step5','Step6']:
    p = os.path.join(_project, s, 'src')
    if p not in sys.path: sys.path.insert(0, p)

import numpy as np
import nibabel as nib
rng = np.random.default_rng(42)

class TestEndToEnd:
    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="integration_")
        # 更大的体积确保 large 肿瘤也能放下
        shape = (80, 80, 80)
        aff = np.eye(4); aff[0,0]=aff[1,1]=aff[2,2]=1.0

        ct_data = rng.integers(-175, 250, shape, dtype=np.int16)
        ctd = os.path.join(cls.tmpdir, 'ct', 'S001')
        os.makedirs(ctd, exist_ok=True)
        cls.ct_path = os.path.join(ctd, 'ct.nii.gz')
        nib.save(nib.Nifti1Image(ct_data, aff), cls.ct_path)

        organ = np.zeros(shape, dtype=np.uint8)
        organ[10:70, 10:70, 10:70] = 1  # 216,000 voxels
        lbd = os.path.join(cls.tmpdir, 'labels', 'S001', 'segmentations')
        os.makedirs(lbd, exist_ok=True)
        cls.organ_path = os.path.join(lbd, 'liver.nii.gz')
        nib.save(nib.Nifti1Image(organ, aff), cls.organ_path)

        cls.config = {
            'project': {'output_dir': os.path.join(cls.tmpdir, 'output')},
            'data': {'ct_dir': os.path.join(cls.tmpdir, 'ct'),
                     'organ_label_dir': os.path.join(cls.tmpdir, 'labels')},
            'organs': [{'name': 'liver_lesion', 'organ_label_file': 'liver.nii.gz',
                        'organ_name': 'liver', 'count': 2}],
            'size_categories': {
                'categories': {
                    'tiny': {'r_min_mm':1,'r_max_mm':5,'weight':0},
                    'small': {'r_min_mm':5,'r_max_mm':10,'weight':4},
                    'medium': {'r_min_mm':10,'r_max_mm':20,'weight':2},
                    'large': {'r_min_mm':20,'r_max_mm':50,'weight':1},
                }
            },
            'shape': {
                'axis_ratio_range': [0.8, 1.2],
                'elastic_deformation': {'enabled': True, 'alpha': 10, 'sigma': 2},
                'salt_noise': {'enabled': True, 'probability': 0.02},
                'gaussian_filter': {'enabled': True, 'sigma_mm': 0.5},
                'scaling_clipping': {'enabled': True},
            },
            'placement': {
                'strategy': 'uniform',
                'margin': {'feather_mm': 2, 'safety_mm': 3},
                'max_retries': 30,
                'distance_weighted': {'alpha': 1.0},
            },
            'preprocessing': {'hu_min': -175, 'hu_max': 250},
            'output': {
                'format': 'nifti', 'dtype': 'uint8', 'value_range': [0, 1],
                'compress': True,
                'naming_pattern': '{organ_type}_{sample_id}.nii.gz',
            },
            'logging': {
                'log_file': os.path.join(cls.tmpdir, 'output', 'log.json'),
                'stats_file': os.path.join(cls.tmpdir, 'output', 'stats.json'),
            },
        }
        cls.aff = aff
        cls.shape = shape

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)

    def test_e2e_generate_one(self):
        from main import generate_one
        meta = generate_one(self.ct_path, self.organ_path, 'liver_lesion',
                           'liver.nii.gz', 'E2E_001', self.config, rng=rng)
        assert meta['success'], f"Failed: {meta.get('error')}"
        assert os.path.exists(meta['output_path'])
        img = nib.load(meta['output_path'])
        data = img.get_fdata()
        assert data.shape == self.shape
        assert set(np.unique(data)).issubset({0, 1})
        assert data.sum() > 0
        assert np.allclose(img.affine, self.aff)

    def test_e2e_batch(self):
        from main import generate_batch
        results = generate_batch(self.config, rng_seed=123)
        success = [r for r in results if r['success']]
        assert len(success) >= 1, f"All {len(results)} failed: {[r.get('error','?')[:60] for r in results]}"
        for r in success:
            assert os.path.exists(r['output_path'])

    def test_e2e_deterministic(self):
        from main import generate_one
        m1 = generate_one(self.ct_path, self.organ_path, 'liver_lesion',
                         'liver.nii.gz', 'DET_001', self.config,
                         rng=np.random.default_rng(42))
        m2 = generate_one(self.ct_path, self.organ_path, 'liver_lesion',
                         'liver.nii.gz', 'DET_001', self.config,
                         rng=np.random.default_rng(42))
        assert m1['success'] == m2['success']
        if m1['success'] and m2['success']:
            d1 = nib.load(m1['output_path']).get_fdata()
            d2 = nib.load(m2['output_path']).get_fdata()
            assert np.array_equal(d1, d2)

    def test_e2e_config_valid(self):
        from main import load_config
        cfg = load_config(os.path.join(_project, 'Step0', 'config', 'generation_config.json'))
        names = [o['name'] for o in cfg['organs']]
        expected = ['liver_lesion','pancreatic_lesion','kidney_lesion',
                    'colon_lesion','esophagus_tumor','endometrioma_tumor']
        assert names == expected
        assert all(o['count'] == 20 for o in cfg['organs'])
