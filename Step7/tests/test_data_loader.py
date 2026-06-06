"""Step 2 单元测试: data_loader.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Step2', 'src'))
import tempfile, shutil
import numpy as np
import nibabel as nib
import data_loader as dl

tmpdir = tempfile.mkdtemp(prefix="dl_test_")
shape = (20, 40, 40)
affine = np.eye(4); affine[0,0]=affine[1,1]=1.0; affine[2,2]=2.0

def setup_module():
    ct = np.random.default_rng(0).integers(-175, 250, shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(ct, affine), os.path.join(tmpdir, 'ct.nii.gz'))
    organ = np.zeros(shape, dtype=np.uint8); organ[3:17, 5:35, 5:35] = 1
    nib.save(nib.Nifti1Image(organ, affine), os.path.join(tmpdir, 'organ.nii.gz'))
    empty = np.zeros(shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(empty, affine), os.path.join(tmpdir, 'empty.nii.gz'))

def teardown_module():
    shutil.rmtree(tmpdir)

class TestLoadCT:
    def test_basic(self):
        ct = dl.load_ct(os.path.join(tmpdir, 'ct.nii.gz'))
        assert ct.shape == shape and ct.spacing == (2.0, 1.0, 1.0)

    def test_file_not_found(self):
        try:
            dl.load_ct(os.path.join(tmpdir, 'nonexistent.nii.gz'))
            assert False, "should raise"
        except FileNotFoundError:
            pass

    def test_hu_clipped(self):
        ct = dl.load_ct(os.path.join(tmpdir, 'ct.nii.gz'), hu_min=-175, hu_max=250)
        assert ct.array.min() >= -175 and ct.array.max() <= 250

class TestLoadOrganMask:
    def test_basic(self):
        om = dl.load_organ_mask(os.path.join(tmpdir, 'organ.nii.gz'), 'liver_lesion')
        assert om.array.dtype == np.uint8 and om.array.sum() > 0

    def test_empty_mask(self):
        try:
            dl.load_organ_mask(os.path.join(tmpdir, 'empty.nii.gz'), 'liver_lesion')
            assert False, "should raise"
        except ValueError:
            pass

class TestLoadSample:
    def test_basic(self):
        s = dl.load_sample(os.path.join(tmpdir, 'ct.nii.gz'),
                          os.path.join(tmpdir, 'organ.nii.gz'), 'liver_lesion')
        assert s.ct is not None and s.organ_mask is not None

class TestValidate:
    def test_compatible(self):
        ct = dl.load_ct(os.path.join(tmpdir, 'ct.nii.gz'))
        om = dl.load_organ_mask(os.path.join(tmpdir, 'organ.nii.gz'), 'liver_lesion')
        dl.validate_compatibility(ct, om)  # no exception

    def test_shape_mismatch(self):
        ct = dl.load_ct(os.path.join(tmpdir, 'ct.nii.gz'))
        bad_data = np.zeros((10, 40, 40), dtype=np.uint8)
        bad_data[2:8, 5:35, 5:35] = 1
        bad_path = os.path.join(tmpdir, 'bad_shape.nii.gz')
        nib.save(nib.Nifti1Image(bad_data, affine), bad_path)
        bad_om = dl.load_organ_mask(bad_path, 'liver_lesion')
        try:
            dl.validate_compatibility(ct, bad_om)
            assert False, "should raise"
        except ValueError:
            pass

class TestManifest:
    def test_build(self):
        ctd = os.path.join(tmpdir, 'ct_dir')
        lbd = os.path.join(tmpdir, 'label_dir')
        for sid in ['S001', 'S002']:
            os.makedirs(os.path.join(ctd, sid), exist_ok=True)
            seg = os.path.join(lbd, sid, 'segmentations')
            os.makedirs(seg, exist_ok=True)
            nib.save(nib.Nifti1Image(np.random.default_rng(0).integers(-175,250,shape,dtype=np.int16), affine),
                     os.path.join(ctd, sid, 'ct.nii.gz'))
            nib.save(nib.Nifti1Image(np.ones(shape,dtype=np.uint8), affine),
                     os.path.join(seg, 'liver.nii.gz'))
        cfg = [{'name': 'liver_lesion', 'organ_label_file': 'liver.nii.gz'}]
        m = dl.build_manifest(ctd, lbd, cfg)
        assert len(m) == 2
        assert all(x['exists'] for x in m)
