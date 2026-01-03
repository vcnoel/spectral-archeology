
import sys
import os
import argparse
import json

# Ensure root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis import run_spectral_analysis, load_dataset, sample_inputs, LANG_PROMPTS

def main():
    parser = argparse.ArgumentParser(description="Spectral Diagnostics CLI")
    
    # Model & Execution
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu, cuda)")
    parser.add_argument("--offline", action="store_true", help="Use local cached files only")
    parser.add_argument("--load-in-8bit", action="store_true", help="8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantization")
    
    # Input Data
    parser.add_argument("--text", type=str, help="Single text input")
    parser.add_argument("--lang", type=str, help="Preset language prompt (e.g., 'en', 'fr')")
    parser.add_argument("--dataset", nargs="+", help="Path to dataset folders or JSON files")
    parser.add_argument("--max-samples", type=int, help="Resample limit")
    
    # Output
    parser.add_argument("--results-file", type=str, default="results.json", help="Output JSON path")
    parser.add_argument("--runs", type=int, default=1, help="Runs per input")
    
    # Ablation
    parser.add_argument("--ablate", type=str, help="Ablation mask: 'layer,head;layer,head'")
    
    args = parser.parse_args()
    
    # Prepare Inputs
    inputs = []
    if args.dataset:
        inputs.extend(load_dataset(args.dataset))
    elif args.text:
        inputs.append({"id": "cli_text", "text": args.text, "lang": "user"})
    elif args.lang:
        if args.lang.lower() == "all":
             for lg, txt in LANG_PROMPTS.items():
                 inputs.append({"id": lg, "text": txt, "lang": lg})
        else:
             txt = LANG_PROMPTS.get(args.lang)
             if txt:
                 inputs.append({"id": args.lang, "text": txt, "lang": args.lang})
             else:
                 print(f"Language {args.lang} not found.")
                 return

    if args.max_samples:
        inputs = sample_inputs(inputs, args.max_samples)

    if not inputs:
        print("No inputs provided. Use --text, --lang, or --dataset.")
        return

    # Parse Ablation
    ablation_mask = None
    if args.ablate:
        ablation_mask = {}
        for p in args.ablate.split(";"):
            if "," in p:
                l, h = map(int, p.split(","))
                if l not in ablation_mask: ablation_mask[l] = []
                ablation_mask[l].append(h)
    
    # Run
    results = run_spectral_analysis(
        model_name=args.model,
        inputs=inputs,
        device=args.device,
        offline=args.offline,
        quantization_8bit=args.load_in_8bit,
        quantization_4bit=args.load_in_4bit,
        ablation_mask=ablation_mask,
        runs=args.runs
    )
    
    # Save
    if results:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.results_file}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
