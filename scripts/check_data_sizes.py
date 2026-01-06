import json
import os
import glob

data_dir = r"c:\Users\valno\Dev\spectral-fingerprints\data\rigorous_comparison"
json_files = glob.glob(os.path.join(data_dir, "*.json"))

print(f"Checking {len(json_files)} files in {data_dir}...")

for fpath in json_files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"{os.path.basename(fpath)}: {len(data)} samples")
            else:
                 print(f"{os.path.basename(fpath)}: Not a list (Type: {type(data)})")
    except Exception as e:
        print(f"{os.path.basename(fpath)}: Error {e}")
