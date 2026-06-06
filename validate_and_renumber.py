"""
Comprehensive validation and renumbering of all tumor masks.
"""
import os, glob, numpy as np, nibabel as nib, shutil, json

OUTPUT_BASE = 'output/real_ct'
LABEL_DIR = 'data/organ_labels'
CT_DIR = 'data/ct'

organ_map = {
    'liver_lesion': 'liver',
    'pancreatic_lesion': 'pancreas',
    'kidney_lesion': 'kidney_left',
    'colon_lesion': 'colon',
    'esophagus_tumor': 'esophagus',
    'endometrioma_tumor': 'uterus',
}

ct_ids = sorted([d for d in os.listdir(CT_DIR) if d.startswith('BDMAP')])

# Load CT shapes
ct_shapes = {}
for ct_id in ct_ids:
    ct_path = os.path.join(CT_DIR, ct_id, 'ct.nii.gz')
    if os.path.exists(ct_path):
        ct_shapes[ct_id] = nib.load(ct_path).shape

print('=' * 60)
print('FULL VALIDATION & RENUMBERING')
print('=' * 60)

all_issues = []
validation_results = []

for organ_type, organ_name in organ_map.items():
    organ_dir = os.path.join(OUTPUT_BASE, organ_type)
    masks = sorted(glob.glob(os.path.join(organ_dir, '*.nii.gz')))

    print(f'\n{organ_type}: {len(masks)} masks')
    organ_issues = []
    valid_masks = []

    for mask_path in masks:
        mask_name = os.path.basename(mask_path)
        tumor = nib.load(mask_path).get_fdata()
        tumor_shape = tumor.shape
        tumor_vol = int(np.sum(tumor > 0))

        # Issue 1: zero volume
        if tumor_vol == 0:
            organ_issues.append(f'{mask_name}: ZERO VOLUME')
            continue

        # Find matching CT
        matching_cts = [ct for ct, shape in ct_shapes.items() if shape == tumor_shape]
        if not matching_cts:
            organ_issues.append(f'{mask_name}: no matching CT shape')
            continue

        ct_id = matching_cts[0]
        organ_path = os.path.join(LABEL_DIR, ct_id, 'segmentations', f'{organ_name}.nii.gz')

        if not os.path.exists(organ_path):
            organ_issues.append(f'{mask_name}: organ mask not found ({ct_id}/{organ_name})')
            continue

        organ = nib.load(organ_path).get_fdata()

        # Issue 2: tumor outside organ
        inside = int(np.sum((tumor > 0) & (organ > 0)))
        outside = int(np.sum((tumor > 0) & (organ == 0)))
        pct_out = outside / max(1, tumor_vol) * 100

        if pct_out > 0:
            organ_issues.append(f'{mask_name}: {pct_out:.1f}% outside organ (ct={ct_id})')
            if pct_out > 5:
                continue  # Skip this mask if >5% outside
            # Small amounts (<5%) are acceptable (clipping in code handles this)

        valid_masks.append((mask_path, ct_id, tumor_vol, pct_out))

    # Report
    if organ_issues:
        print(f'  Issues: {len(organ_issues)}')
        for issue in organ_issues:
            print(f'    {issue}')
    else:
        print(f'  All OK!')

    all_issues.extend(organ_issues)
    validation_results.append((organ_type, organ_dir, valid_masks))

# Summary
print(f'\n{"=" * 60}')
print(f'VALIDATION SUMMARY')
print(f'{"=" * 60}')
total_valid = sum(len(vm) for _, _, vm in validation_results)
total_masks = sum(len(glob.glob(os.path.join(OUTPUT_BASE, o, '*.nii.gz'))) for o in organ_map)
print(f'Valid masks: {total_valid}/{total_masks}')
print(f'Issues found: {len(all_issues)}')

if all_issues:
    print('\nAll issues:')
    for issue in all_issues:
        print(f'  {issue}')

# Renumber valid masks
print(f'\n{"=" * 60}')
print(f'RENUMBERING')
print(f'{"=" * 60}')

# Step 1: Delete invalid masks
for organ_type, organ_dir, valid_masks in validation_results:
    valid_paths = set(vm[0] for vm in valid_masks)
    all_masks = glob.glob(os.path.join(organ_dir, '*.nii.gz'))
    for f in all_masks:
        if f not in valid_paths:
            print(f'  DELETE invalid: {os.path.basename(f)}')
            os.remove(f)

# Step 2: Rename to temp names first to avoid overwrites, then to final
for organ_type, organ_dir, valid_masks in validation_results:
    print(f'\n{organ_type}: {len(valid_masks)} masks')
    valid_masks.sort(key=lambda x: -x[2])  # Sort by volume descending

    # Phase A: rename all to temp
    temp_map = {}  # temp_name -> final_name
    for i, (old_path, ct_id, vol, pct_out) in enumerate(valid_masks):
        temp_name = f'{organ_type}_tmp{i:03d}.nii.gz'
        final_name = f'{organ_type}_t{i:02d}.nii.gz'
        temp_path = os.path.join(organ_dir, temp_name)
        final_path = os.path.join(organ_dir, final_name)
        shutil.move(old_path, temp_path)
        temp_map[temp_path] = final_path

    # Phase B: rename temp to final
    for temp_path, final_path in temp_map.items():
        if os.path.exists(temp_path):
            shutil.move(temp_path, final_path)

    print(f'  => {organ_type}_t00.nii.gz .. {organ_type}_t{len(valid_masks)-1:02d}.nii.gz')

print(f'\n{"=" * 60}')
print('Done: all masks validated and renumbered')
print('=' * 60)
