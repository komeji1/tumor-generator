"""
Run mask generation with real-time progress.
Usage: python -u run_generation.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Step6', 'src'))
from main import generate_batch, load_config

config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'Step0', 'config', 'generation_config.json'))

print("Starting 120-mask generation...", flush=True)
t0 = time.time()
results = generate_batch(config, rng_seed=42)
elapsed = time.time() - t0

success = sum(1 for r in results if r['success'])
total = len(results)
print(f"\n{'='*60}")
print(f"COMPLETE: {success}/{total} masks in {elapsed:.0f}s ({elapsed/total:.1f}s/mask)")
if success < total:
    failed = [(r['organ_type'], r['sample_id'], r.get('error', '?')[:120])
              for r in results if not r['success']]
    print(f"Failures ({len(failed)}):")
    for org, sid, err in failed:
        print(f"  {org}/{sid}: {err}")
print(f"{'='*60}")
