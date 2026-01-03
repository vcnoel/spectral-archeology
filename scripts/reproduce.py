
import sys
import os
import argparse
import json

# Ensure root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis import run_spectral_analysis
from src.experiments import MODELS, TOPOLOGY_SAMPLES, SYNTACTIC_SAMPLES

def main():
    parser = argparse.ArgumentParser(description="Experiment Reproduction CLI")
    
    parser.add_argument("--experiment", type=str, required=True, choices=['topology', 'scar'],
                        help="Experiment to run: 'topology' (Script Swap) or 'scar' (Passive Voice)")
    
    parser.add_argument("--model", type=str, required=True, 
                        help=f"Model shortname ({', '.join(MODELS.keys())}) or full HuggingFace ID")
    
    parser.add_argument("--output-dir", type=str, default="output_reproduce", help="Output directory")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quant-4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Resolve Model
    model_id = MODELS.get(args.model, args.model)
    print(f"Running Experiment: {args.experiment}")
    print(f"Target Model: {model_id}")
    
    # Configure Inputs
    inputs = []
    if args.experiment == 'topology':
        for k, v in TOPOLOGY_SAMPLES.items():
            inputs.append({"id": k, "lang": k, "text": v, "type": "topology"})
            
    elif args.experiment == 'scar':
        inputs.append({"id": "active", "lang": "en", "text": SYNTACTIC_SAMPLES['active'], "type": "active"})
        inputs.append({"id": "passive", "lang": "en", "text": SYNTACTIC_SAMPLES['passive'], "type": "passive"})
        
    # Run Analysis
    results = run_spectral_analysis(
        model_name=model_id,
        inputs=inputs,
        device=args.device,
        quantization_4bit=args.quant_4bit,
        runs=1
    )
    
    # Save
    out_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "results.json")
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Experiment complete. Results saved to {out_file}")
    print(f"To visualize, run: python scripts/visualize.py --mode { 'multilingual' if args.experiment == 'scar' else 'topology' } --input {out_file}")

if __name__ == "__main__":
    main()
