import json
import os

target_file = r"c:\Users\valno\Dev\spectral-fingerprints\data\rigorous_comparison\all_mixed.json"

try:
    with open(target_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if len(data) > 200:
        print(f"Truncating {os.path.basename(target_file)} from {len(data)} to 200 samples.")
        truncated_data = data[:200]
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(truncated_data, f, indent=2)
        print("Done.")
    else:
        print(f"{os.path.basename(target_file)} has {len(data)} samples. No truncation needed.")

except Exception as e:
    print(f"Error: {e}")
