
import sys
import os
import argparse
import json
import glob

# Ensure root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.experiments import MODELS
from src.plotting import plot_multilingual_benchmark, plot_tokenization_topology

def main():
    parser = argparse.ArgumentParser(description="Spectral Visualization CLI")
    
    parser.add_argument("--mode", type=str, required=True, choices=['multilingual', 'topology'], 
                        help="Plotting mode: 'multilingual' (benchmark) or 'topology' (script swap)")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Input file or directory (containing results.json)")
    
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save plots")
    
    parser.add_argument("--model-name", type=str, help="Override model name label")
    
    args = parser.parse_args()
    
    if args.mode == 'multilingual':
        # Expecting directory with results.json or direct file
        if os.path.isdir(args.input):
            json_path = os.path.join(args.input, "results.json")
        else:
            json_path = args.input
            
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found.")
            return
            
        with open(json_path, 'r') as f:
            results = json.load(f)
            
        # Infer model name if not provided
        m_name = args.model_name
        if not m_name:
            # Try to reverse lookup from path
            norm_path = args.input.replace("\\", "/").rstrip("/")
            dirname = os.path.basename(norm_path)
            for k, v in MODELS.items():
                if v == dirname:
                    m_name = k
                    break
        if not m_name: m_name = "Unknown Model"
        
        plot_multilingual_benchmark(results, m_name, args.output_dir)
        
    elif args.mode == 'topology':
        # Expecting direct json file usually
        if os.path.isdir(args.input):
            json_path = os.path.join(args.input, "results.json")
        else:
            json_path = args.input
            
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found.")
            return

        plot_tokenization_topology(json_path, args.output_dir)

if __name__ == "__main__":
    main()
