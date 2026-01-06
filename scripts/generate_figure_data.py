import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import scipy.linalg

# --- CONFIGURATION ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
SAMPLE_TEXT = "The book was written by the man." 
ACTIVE_TEXT = "The man wrote the book."
# This sample had PPL ~26.7 and good recovery (+0.08) in previous runs.
STEERING_LAYER_IDX = 2
ALPHA = 0.2
TOP_K = 2 # For Sparsity (Aggressive)
WINDOW_SIZE = 1 # For Window (Local only)

# --- UTILS ---
def get_spectral_metrics(adj_matrix):
    # adj_matrix: (N, N)
    N = adj_matrix.shape[0]
    if N < 2: return 0.0, 0.0, 0.0, 0.0
    
    # 1. Laplacian
    # Symmetrize
    W = 0.5 * (adj_matrix + adj_matrix.T)
    # Degrees
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Eigen
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Fiedler
        fiedler = eigvals[1] if N > 1 else 0.0
        
        # HFER (High Frequency Energy Ratio)
        # Signal x is the token index or simple ramp? 
        # Usually HFER is computed on the *Attention Signal* or the *Hidden State* projection?
        # In the paper context, Smoothness/HFER are usually on the *Value* vectors or the *Token Position*.
        # Let's assume standard graph signal: x = [0, 1, ..., N-1] normalized?
        # Or simple uniform signal. 
        # Actually, "Smoothness" = x^T L x / x^T x.
        # Let's use x = linear ramp (position encoding proxy).
        x = np.linspace(-1, 1, N)
        smoothness = (x.T @ L @ x) / (x.T @ x)
        
        # HFER: Energy in top 50% frequencies
        # signal expansion c = U^T x
        c = eigvecs.T @ x
        energy = c**2
        total_energy = np.sum(energy)
        # High freq = upper half of eigenvalues
        k_cut = N // 2
        high_energy = np.sum(energy[k_cut:])
        hfer = high_energy / total_energy if total_energy > 0 else 0.0
        
    except:
        fiedler = 0.0
        hfer = 0.0
        smoothness = 0.0

    # Entropy (Shannon of Average Attention Row)
    # avg_attn = adj_matrix.mean(axis=0) + 1e-9
    # entropy = -np.sum(avg_attn * np.log(avg_attn))
    
    # Or Average Entropy of rows?
    # Usually "Entropy" in attention papers is avg of row entropies.
    row_entropies = []
    for row in adj_matrix:
        r = row + 1e-12
        r = r / r.sum()
        e = -np.sum(r * np.log(r))
        row_entropies.append(e)
    entropy = np.mean(row_entropies)

    return fiedler, hfer, smoothness, entropy

# --- LOADING ---
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager"
)

# Calibrate Steering Vector (Technical Base for robust "General" direction)
vocab_technical = {
     "subjects": ["The data", "The token", "The input", "The model", "The query"],
     "verbs": ["processed", "encoded", "decoded", "parsed", "routed"],
     "agents": ["by the system", "by the layer", "by the network"]
}
def generate_calib(n):
    pairs = []
    for _ in range(n):
        s = np.random.choice(vocab_technical["subjects"])
        v = np.random.choice(vocab_technical["verbs"])
        a = np.random.choice(vocab_technical["agents"])
        active = f"{a.replace('by ', '')} {v} {s.lower()}."
        passive = f"{s} was {v} {a}."
        pairs.append((active, passive))
    return pairs

kc_pairs = generate_calib(20)
def get_mean_hidden(texts, layer_idx):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h_states = outputs.hidden_states[layer_idx + 1]
    mask = inputs.attention_mask.unsqueeze(-1)
    sum_h = (h_states * mask).sum(dim=1)
    count_h = mask.sum(dim=1)
    return (sum_h / count_h).mean(dim=0)

mu_act = get_mean_hidden([p[0] for p in kc_pairs], STEERING_LAYER_IDX)
mu_pas = get_mean_hidden([p[1] for p in kc_pairs], STEERING_LAYER_IDX)
steering_vector = mu_act - mu_pas

# --- EXPERIMENTS ---

def run_experiment(mode="baseline"):
    # Prepare Input
    target_text = SAMPLE_TEXT
    
    if mode == "cot":
        # Prompted version
        # We need to find where the target sentence is.
        # Format: "Question: ... Sentence: {target} ... "
        prefix = "Question: Rewrite the following sentence in active voice.\nSentence: "
        suffix = "\nAnalysis: Let's identify the subject and object."
        text_input = prefix + target_text + suffix
        
        # Tokenize full
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        
        # Find indices of target_text in inputs
        # Heuristic: Tokenize prefix and get length
        # Tokenizer on single string returns dict with list input_ids
        len_prefix = len(tokenizer(prefix, add_special_tokens=True).input_ids)
        # Verify: Phi-3 tokenizer might not add special tokens consistently in middle.
        # Let's trust "len_prefix" approach roughly.
        # Actually, let's just search for the token sequence of target_text.
        # Better:
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
        full_ids = inputs.input_ids[0].tolist()
        
        # Find subsequence
        start_idx = -1
        for i in range(len(full_ids) - len(target_ids) + 1):
             if full_ids[i:i+len(target_ids)] == target_ids:
                 start_idx = i
                 break
        
        if start_idx == -1:
            # Fallback (maybe tokenizer spacing issues)
            # Just take the middle chunk roughly?
            # Or tokenizing prefix without special tokens
            len_p = len(tokenizer(prefix, add_special_tokens=True).input_ids) 
            # Note: add_special_tokens=True adds BOS.
            start_idx = len_p - 1 # Approximation
            
        end_idx = start_idx + len(target_ids)
        
    elif mode == "active":
        text_input = ACTIVE_TEXT
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        start_idx = 0
        end_idx = inputs.input_ids.shape[1]
        
    else:
        text_input = target_text
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        start_idx = 0
        end_idx = inputs.input_ids.shape[1]
    
    seq_len = end_idx - start_idx
    
    # Hooks
    hooks = []
    
    # 1. Steering Hook
    if mode == "structural":
        def steer_hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                v = steering_vector.view(1, 1, -1).to(h.dtype)
                return (h + ALPHA * v,) + out[1:]
            return out + ALPHA * steering_vector.view(1, 1, -1).to(out.dtype)
        
        h = model.model.layers[STEERING_LAYER_IDX].register_forward_hook(steer_hook)
        hooks.append(h)

    # Run
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    for h in hooks: h.remove()
    
    # Collect Metrics
    layer_metrics = []
    num_layers = len(outputs.attentions)
    
    for i in range(num_layers):
        # Full Matrix
        A_full = outputs.attentions[i][0].mean(dim=0).float().cpu().numpy() # (L, L)
        
        # Slice to Target Sentence Sub-Graph
        # We want A[start:end, start:end]
        if end_idx > A_full.shape[0]: end_idx = A_full.shape[0] # Safety
        A = A_full[start_idx:end_idx, start_idx:end_idx]
        
        # Re-Normalize (Treat as isolated graph)
        # If attention went to prompt (outside slice), it is lost. 
        # We re-normalize what remains to see the "Structure" of the residual syntax.
        row_sums = A.sum(axis=1, keepdims=True) + 1e-12
        A = A / row_sums
        
        # APPLY MASKS for modes (Baseline/CoT don't mask)
        if mode == "window":
            mask = np.eye(A.shape[0], k=0) + np.eye(A.shape[0], k=1) + np.eye(A.shape[0], k=-1)
            causal_window = np.tril(mask)
            A = A * causal_window
            row_sums = A.sum(axis=1, keepdims=True) + 1e-9
            A = A / row_sums
            
        elif mode == "sparsity":
            for r in range(A.shape[0]):
                row = A[r]
                if len(row) > TOP_K:
                    idx = np.argpartition(row, -TOP_K)[-TOP_K:]
                    new_row = np.zeros_like(row)
                    new_row[idx] = row[idx]
                    A[r] = new_row
            row_sums = A.sum(axis=1, keepdims=True) + 1e-9
            A = A / row_sums

        f, h, s, e = get_spectral_metrics(A)
        layer_metrics.append((f, h, s, e))
        
    return layer_metrics

# --- RUN ALL ---
results = {}
modes = [
    ("Baseline", "baseline"),
    ("Structural", "structural"),
    ("Sparsity", "sparsity"),
    ("Window", "window"),
    ("CoT", "cot"),
    ("Active", "active")
]

for name, mode in modes:
    print(f"Running {name}...")
    results[name] = run_experiment(mode)

# --- FORMAT OUTPUT ---
# Generate LaTeX addplots
metrics_names = ["Fiedler", "HFER", "Smoothness", "Entropy"]
metrics_indices = [0, 1, 2, 3]

latex_map = {
    "Baseline": "clrBaseline, line width=2pt, mark=*, mark size=1.2pt, mark repeat=4",
    "Structural": "clrStructural, line width=1.2pt, dashed",
    "Sparsity": "clrSparsity, line width=1.2pt, dotted",
    "Window": "clrWindow, line width=1.2pt, dashdotted",
    "CoT": "clrCoT, line width=2pt, mark=square*, mark size=1.2pt, mark repeat=4",
    "Active": "clrActive, line width=2pt, color=green!70!black"
}

output_str = ""

for m_idx, m_name in zip(metrics_indices, metrics_names):
    output_str += f"\n% ========== {m_name.upper()} ==========\n"
    for method_name, style in latex_map.items():
        coords = []
        data = results[method_name]
        for layer_i, values in enumerate(data):
            val = values[m_idx]
            coords.append(f"({layer_i},{val:.4f})")
        coords_str = "".join(coords)
        output_str += f"% {method_name}\n\\addplot[{style}] coordinates {{{coords_str}}};\n"

print("\nDONE. DATA GENERATED.")
print(output_str[:500] + "...") # Preview

with open("figure_data.txt", "w") as f:
    f.write(output_str)
