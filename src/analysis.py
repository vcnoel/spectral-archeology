
import os
import json
import glob
import random
import traceback
import numpy as np
from tqdm import tqdm
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

# Placeholder prompts for multilingual support
LANG_PROMPTS = {
    "en": "The capital of France is Paris.",
    "fr": "La capitale de la France est Paris.",
    "es": "La capital de Francia es París.",
    "de": "Die Hauptstadt von Frankreich ist Paris.",
    "it": "La capitale della Francia è Parigi.",
    "pt": "A capital da França é Paris.",
    "ru": "Столица Франции - Париж.",
    "zh": "法国的首都是巴黎。",
    "ja": "フランスの首都はパリです。",
    "ar": "عاصمة فرنسا هي باريس."
}

def load_dataset(dataset_paths):
    inputs = []
    if dataset_paths:
        files = []
        for d in dataset_paths:
            if os.path.isdir(d):
                files.extend(glob.glob(os.path.join(d, "*.json")))
            else:
                matches = glob.glob(d)
                if matches:
                    files.extend(matches)
                else:
                    files.append(d)
            
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        inputs.extend(data)
                    else:
                        inputs.append(data)
            except Exception as e:
                print(f"Failed to load dataset {fpath}: {e}")
    return inputs

def sample_inputs(inputs, max_samples):
    if not inputs: return []
    if max_samples and len(inputs) > max_samples:
        print(f"Limiting inputs from {len(inputs)} to {max_samples} (Stratified Sampling)")
        groups = {}
        for x in inputs:
            key = (x.get('lang', 'unknown'), x.get('type', 'unknown'))
            if key not in groups: groups[key] = []
            groups[key].append(x)
            
        n_groups = len(groups)
        if n_groups > 0:
            per_group = max_samples // n_groups
            if per_group == 0 and max_samples > 0: per_group = 1
            
            selected = []
            for key, items in groups.items():
                random.shuffle(items)
                selected.extend(items[:per_group])
            
            if len(selected) < max_samples:
                remaining = []
                seen = set(id(x) for x in selected)
                for x in inputs:
                     if id(x) not in seen: remaining.append(x)
                
                random.shuffle(remaining)
                needed = max_samples - len(selected)
                selected.extend(remaining[:needed])
            
            inputs = selected
            random.shuffle(inputs)
        else:
            random.shuffle(inputs)
            inputs = inputs[:max_samples]
    return inputs

def register_ablation_hooks(model, ablation_mask):
    hooks = []
    
    def get_ablation_hook(layer_idx, heads_to_mask):
        def hook(module, args):
            x = args[0]
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
            head_dim = hidden_size // num_heads
            
            x_mod = x.clone()
            for h in heads_to_mask:
                start = h * head_dim
                end = (h + 1) * head_dim
                x_mod[:, :, start:end] = 0.0
            return (x_mod,)
        return hook

    for name, module in model.named_modules():
        if "self_attn.o_proj" in name or "attention.out_proj" in name: 
             parts = name.split(".")
             try:
                 l_idx = int(parts[2])
             except:
                 continue
                 
             if l_idx in ablation_mask:
                 print(f"  -> Registering ablation hook on {name} (Heads: {ablation_mask[l_idx]})")
                 h = module.register_forward_pre_hook(get_ablation_hook(l_idx, ablation_mask[l_idx]))
                 hooks.append(h)
    return hooks

def run_spectral_analysis(model_name, inputs, device="cuda", offline=False, 
                          quantization_8bit=False, quantization_4bit=False, 
                          ablation_mask=None, runs=1):
    
    print(f"Loading model: {model_name}")
    
    model_kwargs = {
        "output_attentions": True, 
        "output_hidden_states": True,
        "use_safetensors": True 
    }
    
    if quantization_8bit:
        model_kwargs["load_in_8bit"] = True
    elif quantization_4bit:
        model_kwargs["load_in_4bit"] = True
        
    attempts = [True, False] if not offline else [True]
    
    last_exception = None
    for attempt_offline in attempts:
        try:
            mode_str = "OFFLINE" if attempt_offline else "ONLINE"
            print(f"[{mode_str}] Attempting to load model: {model_name}")
            
            config = GSPConfig(
                model_name=model_name, 
                device=device, 
                local_files_only=attempt_offline,
                model_kwargs=model_kwargs
            )
            
            results_data = []

            with GSPDiagnosticsFramework(config) as framework:
                framework.instrumenter.load_model(model_name)
                
                if ablation_mask:
                    hooks = register_ablation_hooks(framework.instrumenter.model, ablation_mask)
                    if hooks: print(f"Registered {len(hooks)} ablation hooks.")
                
                for item in tqdm(inputs, desc="Inference"):
                    for r in range(runs):
                        try:
                            analysis = framework.analyze_text(item['text'], save_results=False)
                            traj = []
                            if 'layer_diagnostics' in analysis:
                                for idx, ld in enumerate(analysis['layer_diagnostics']):
                                    traj.append({
                                        "layer": idx,
                                        "fiedler_value": float(ld.fiedler_value) if ld.fiedler_value is not None else None,
                                        "hfer": float(ld.hfer) if ld.hfer is not None else None,
                                        "smoothness": float(ld.smoothness_index) if ld.smoothness_index is not None else None,
                                        "entropy": float(ld.spectral_entropy) if ld.spectral_entropy is not None else None,
                                    })
                            
                            results_data.append({
                                "id": item.get('id', 'unknown'),
                                "lang": item.get('lang', 'unknown'),
                                "type": item.get('type', 'unknown'),
                                "run": r,
                                "trajectory": traj
                            })
                        except Exception as e:
                            print(f"Error analyzing {item.get('id', '?')}: {e}")
            
            # If we reached here, success!
            return results_data
            
        except Exception as e:
            last_exception = e
            if attempt_offline and not offline:
                print(f"Offline load failed for {model_name}. Retrying online...")
            else:
                print(f"Failed to load {model_name} (Offline={attempt_offline}): {e}")

    # If all attempts failed
    if last_exception:
        raise last_exception
    return []
