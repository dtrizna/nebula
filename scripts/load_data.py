import json
import os
from collections import Counter
from glob import glob

EMULATION_EXAMPLES = "emulation"

print("="*70)
print("DEMONSTRATING THE THREAD ISSUE WITH LOCAL EXAMPLE REPORTS")
print("="*70)

ep_type_counts = Counter()
samples_with_threads = []

json_files = glob(os.path.join(EMULATION_EXAMPLES, "report_example_*.json"))
print(f"\nAnalyzing {len(json_files)} example report files...\n")

for filepath in json_files:
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            entry_points = data
            sha256 = filename
        elif isinstance(data, dict) and 'entry_points' in data:
            entry_points = data.get('entry_points', [])
            sha256 = data.get('sha256', filename)
        else:
            continue
        
        print(f"File: {filename}")
        print(f"  SHA256: {sha256[:40]}...")
        print(f"  Number of entry_points: {len(entry_points)}")
        
        for i, ep in enumerate(entry_points):
            ep_type = ep.get('ep_type', 'unknown')
            ep_type_counts[ep_type] += 1
            api_count = len(ep.get('apis', []))
            print(f"    [{i}] ep_type={ep_type}, apis={api_count}")
        
        if len(entry_points) > 1:
            samples_with_threads.append({
                'file': filename,
                'count': len(entry_points),
                'types': [ep.get('ep_type') for ep in entry_points]
            })
        print()
        
    except Exception as e:
        print(f"  Error reading {filename}: {e}\n")

print("="*70)
print("SUMMARY")
print("="*70)
print(f"\nEntry type distribution:")
for ep_type, count in ep_type_counts.items():
    print(f"  {ep_type}: {count}")

print(f"\n{'*'*70}")
print(f"KEY FINDING: {len(samples_with_threads)} samples have MULTIPLE entry points!")
print(f"{'*'*70}")
print("""
ISSUE EXPLANATION:
-----------------
If the HuggingFace dataset stores each entry_point as a SEPARATE ROW,
then a sample with 2 entry_points (1 module_entry + 1 thread) will
produce 2 rows -> 2 separate predictions instead of 1.

The Nebula inference pipeline (nebula.preprocess) AGGREGATES all
entry_points into a single record, but direct dataset loading does NOT.
""")
