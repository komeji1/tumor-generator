"""项目自检脚本 — 检查 Step0~Step2 所有文件完整性"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print('=' * 60)
print('项目自检报告 — Tumor Mask Generator')
print(f'日期: 2026-06-04')
print('=' * 60)

errors, warnings = [], []
base = os.path.dirname(os.path.abspath(__file__))
os.chdir(base)

# ═══════════════════════════════════════════════════════
# 1. 目录结构
# ═══════════════════════════════════════════════════════
print('\n[1] 目录结构')
dirs = ['Step0/config','Step0/src','Step0/tests','Step0/help',
        'Step1/src','Step1/help',
        'Step2/src','Step2/help',
        'Step3/src','Step3/help',
        'Step4/src','Step4/help',
        'data/ct','data/organ_labels','output']
for d in dirs:
    ok = os.path.isdir(d)
    print(f'  {"OK" if ok else "MISS"}  {d}/')
    if not ok: errors.append(f'Missing dir: {d}')

# ═══════════════════════════════════════════════════════
# 2. 文件完整性 (Step0)
# ═══════════════════════════════════════════════════════
print('\n[2] Step0 文件')
s0 = {
    'Step0/config/generation_config.json': 'JSON配置',
    'Step0/src/__init__.py': '包入口',
    'Step0/tests/__init__.py': '测试入口',
    'Step0/help/generation_config.json.md': '配置文档',
    'Step0/help/src___init__.py.md': '包文档',
    'Step0/help/tests___init__.py.md': '测试文档',
}
for p, d in s0.items():
    ok = os.path.exists(p)
    sz = os.path.getsize(p) if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {p} ({sz:,}B) - {d}')
    if not ok: errors.append(f'Missing: {p}')

# ═══════════════════════════════════════════════════════
# 3. 文件完整性 (Step1)
# ═══════════════════════════════════════════════════════
print('\n[3] Step1 文件')
s1 = {
    'Step1/src/__init__.py': '包入口',
    'Step1/src/utils.py': '工具模块 (16函数)',
    'Step1/help/utils.py.md': '工具文档',
}
for p, d in s1.items():
    ok = os.path.exists(p)
    sz = os.path.getsize(p) if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {p} ({sz:,}B) - {d}')
    if not ok: errors.append(f'Missing: {p}')

# ═══════════════════════════════════════════════════════
# 4. 文件完整性 (Step2)
# ═══════════════════════════════════════════════════════
print('\n[4] Step2 文件')
s2 = {
    'Step2/src/__init__.py': '包入口',
    'Step2/src/data_loader.py': '数据加载 (3 dataclass + 7函数)',
    'Step2/help/data_loader.py.md': '数据加载文档',
}
for p, d in s2.items():
    ok = os.path.exists(p)
    sz = os.path.getsize(p) if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {p} ({sz:,}B) - {d}')
    if not ok: errors.append(f'Missing: {p}')

# ═══════════════════════════════════════════════════════
# 5. 文件完整性 (Step3)
# ═══════════════════════════════════════════════════════
print('\n[5] Step3 文件')
s3 = {
    'Step3/src/__init__.py': '包入口',
    'Step3/src/validator.py': '校验模块 (6函数)',
    'Step3/help/validator.py.md': '校验文档',
}
for p, d in s3.items():
    ok = os.path.exists(p)
    sz = os.path.getsize(p) if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {p} ({sz:,}B) - {d}')
    if not ok: errors.append(f'Missing: {p}')

# ═══════════════════════════════════════════════════════
# 6. 文件完整性 (Step4)
# ═══════════════════════════════════════════════════════
print('\n[6] Step4 文件')
s4 = {
    'Step4/src/__init__.py': '包入口',
    'Step4/src/position_selector.py': '位置选择 (枚举+6函数)',
    'Step4/help/position_selector.py.md': '位置选择文档',
}
for p, d in s4.items():
    ok = os.path.exists(p)
    sz = os.path.getsize(p) if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {p} ({sz:,}B) - {d}')
    if not ok: errors.append(f'Missing: {p}')

# ═══════════════════════════════════════════════════════
# 7. JSON 配置验证
# ═══════════════════════════════════════════════════════
print('\n[6] generation_config.json')
cfg = json.load(open('Step0/config/generation_config.json', 'r', encoding='utf-8'))
for k in ['project','data','organs','size_categories','shape','placement','preprocessing','output','logging']:
    assert k in cfg
    print(f'  OK  cfg.{k}')

organs = cfg['organs']
exp = ['liver_lesion','pancreatic_lesion','kidney_lesion','colon_lesion','esophagus_tumor','endometrioma_tumor']
assert [o['name'] for o in organs] == exp
print(f'  OK  {len(organs)} organs: {exp}')

cats = cfg['size_categories']['categories']
w = {k: cats[k]['weight'] for k in cats}
assert w == {'tiny':0,'small':4,'medium':2,'large':1}
print(f'  OK  weights 4:2:1 = {w}')

for s in ['elastic_deformation','salt_noise','gaussian_filter','scaling_clipping']:
    print(f'  OK  shape.{s}: enabled={cfg["shape"][s]["enabled"]}')

hu = cfg['preprocessing']
assert hu['hu_min'] == -175 and hu['hu_max'] == 250
print(f'  OK  HU range: [{hu["hu_min"]}, {hu["hu_max"]}]')

# ═══════════════════════════════════════════════════════
# 6. 模块导入
# ═══════════════════════════════════════════════════════
print('\n[6] 模块导入')

sys.path.insert(0, 'Step1/src')
import utils as ut
uf = ['get_spacing','voxel_to_mm','mm_to_voxel','clip_hu','normalize_hu',
      'erode_mask','dilate_mask','compute_ellipsoid_dist','volume_from_radius',
      'compute_valid_region','random_sample_valid','random_axis_ratios',
      'generate_elastic_deformation_field','apply_deformation','ensure_uint8','get_bbox']
ok_u = sum(1 for f in uf if hasattr(ut, f) and callable(getattr(ut, f)))
print(f'  OK  utils: {ok_u}/{len(uf)} functions')
if ok_u < len(uf): errors.append(f'utils missing {len(uf)-ok_u} functions')

sys.path.insert(0, 'Step2/src')
import data_loader as dl
df = ['load_ct','load_organ_mask','load_sample','validate_compatibility',
      'get_organ_bbox','build_manifest','save_manifest_csv','load_manifest_csv']
ok_d = sum(1 for f in df if hasattr(dl, f) and callable(getattr(dl, f)))
print(f'  OK  data_loader: {ok_d}/{len(df)} functions')
for c in ['CTVolume','OrganMask','Sample']:
    if not hasattr(dl, c): errors.append(f'data_loader missing {c}')
print(f'  OK  data_loader dataclasses: CTVolume, OrganMask, Sample')
if ok_d < len(df): errors.append(f'data_loader missing {len(df)-ok_d} functions')

# validator (Step3)
sys.path.insert(0, 'Step3/src')
import validator as vd
vf = ['check_organ_volume','check_size_range','check_in_organ',
      'check_mask_nonzero','check_not_overlapping','validate_sample']
ok_v = sum(1 for f in vf if hasattr(vd, f) and callable(getattr(vd, f)))
print(f'  OK  validator: {ok_v}/{len(vf)} functions')
if ok_v < len(vf): errors.append(f'validator missing {len(vf)-ok_v} functions')

# position_selector (Step4)
sys.path.insert(0, 'Step4/src')
import position_selector as ps
pf = ['compute_margin_voxel','sample_uniform','sample_distance_weighted',
      'select_location','select_location_from_config']
ok_p = sum(1 for f in pf if hasattr(ps, f) and callable(getattr(ps, f)))
print(f'  OK  position_selector: {ok_p}/{len(pf)} functions')
if hasattr(ps, 'PlacementStrategy'):
    print(f'  OK  PlacementStrategy enum: {[s.value for s in ps.PlacementStrategy]}')
else:
    errors.append('position_selector missing PlacementStrategy')
if ok_p < len(pf): errors.append(f'position_selector missing {len(pf)-ok_p} functions')

# ═══════════════════════════════════════════════════════
# 7. 功能测试
# ═══════════════════════════════════════════════════════
print('\n[7] 功能测试')
import numpy as np; rng = np.random.default_rng(42)

# utils
aff = np.eye(4); aff[2,2] = 2.0
assert ut.get_spacing(aff) == (2.0, 1.0, 1.0); print('  OK  get_spacing')
v = np.array([5.,10.,20.])
assert np.allclose(v, ut.mm_to_voxel(ut.voxel_to_mm(v, aff), aff)); print('  OK  voxel<->mm')
c = ut.clip_hu(np.array([-500,-175,0,250,500]))
assert c[0]==-175 and c[-1]==250; print('  OK  clip_hu')
d = ut.compute_ellipsoid_dist((10,10,10),(5,5,5),(3,2,2))
assert abs(d[5,5,5])<1e-6 and 0.9<d[5,5,7]<1.1; print('  OK  ellipsoid_dist')
r = ut.random_axis_ratios(rng=rng)
assert abs(r[0]*r[1]*r[2]-1.0)<1e-6; print('  OK  axis_ratios')
mask = np.zeros((32,32,32),dtype=np.uint8); mask[8:24,8:24,8:24]=1
field = ut.generate_elastic_deformation_field((32,32,32),alpha=10,sigma=3,rng=rng)
deformed = ut.apply_deformation(mask, field)
vc = abs(float(deformed.sum())-float(mask.sum()))/float(mask.sum())
assert vc<0.3; print(f'  OK  elastic (vol_change={vc:.1%})')
organ = np.zeros((20,20,20),dtype=np.uint8); organ[3:17,3:17,3:17]=1
valid = ut.compute_valid_region(organ, 3.0)
assert valid.sum()>0; print(f'  OK  valid_region ({valid.sum()} voxels)')
# validator quick test
ok, vol = vd.check_organ_volume(organ)
assert ok and vol > 0; print(f'  OK  check_organ_volume ({vol} voxels)')
ok, _ = vd.check_size_range(7.5, {"r_min_mm":5,"r_max_mm":10})
assert ok; print('  OK  check_size_range')
ok, _ = vd.check_mask_nonzero(organ)
assert ok; print('  OK  check_mask_nonzero')

# ═══════════════════════════════════════════════════════
# 8. 文档
# ═══════════════════════════════════════════════════════
print('\n[8] 文档完整性')
docs = ['PROJECT_OVERVIEW.md','IMPLEMENTATION_PLAN.md']
for d in docs:
    ok = os.path.exists(d)
    sz = os.path.getsize(d)/1024 if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {d} ({sz:.1f} KB)')
    if not ok: errors.append(f'Missing: {d}')

for h in ['Step0/help/generation_config.json.md','Step0/help/src___init__.py.md',
          'Step0/help/tests___init__.py.md','Step1/help/utils.py.md',
          'Step2/help/data_loader.py.md','Step3/help/validator.py.md',
          'Step4/help/position_selector.py.md']:
    ok = os.path.exists(h)
    sz = os.path.getsize(h)/1024 if ok else 0
    print(f'  {"OK" if ok else "MISS"}  {h} ({sz:.1f} KB)')
    if not ok: errors.append(f'Missing: {h}')

# ═══════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('汇总')
print('=' * 60)
print(f'  已完成步骤: Step0, Step1, Step2, Step3, Step4')
print(f'  待完成步骤: Step5 (mask_generator), Step6 (main), Step7 (tests)')
print(f'  进度: 5/7 (71%)')
print(f'  错误: {len(errors)}')
print(f'  警告: {len(warnings)}')

if errors:
    print(f'\n[ERRORS]')
    for e in errors: print(f'  X  {e}')
if warnings:
    print(f'\n[WARNINGS]')
    for w in warnings: print(f'  !  {w}')
print(f'\n=> {"所有检查通过" if not errors else f"发现{len(errors)}个错误需修复"}')
