import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer

# Ensure pandas prints all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager"
)

# --- CONFIG ---
config = GSPConfig(model_name=MODEL_NAME, device=device)
gc = GraphConstructor(config)
sa = SpectralAnalyzer(config)

# --- EXTENSIVE DOMAIN DEFINITIONS ---
DOMAINS = {
    "Standard (Simple)": [
        ("The man ate the apple.", "The apple was eaten by the man."),
        ("The boy threw the ball.", "The ball was thrown by the boy."),
        ("The dog chased the cat.", "The cat was chased by the dog."),
        ("The bird built the nest.", "The nest was built by the bird.")
    ],
    "Complex NP (Adjectives)": [
        ("The angry old man ate the red apple.", "The red apple was eaten by the angry old man."),
        ("The small happy boy threw the big blue ball.", "The big blue ball was thrown by the small happy boy."),
        ("The wild feral dog chased the scared cat.", "The scared cat was chased by the wild feral dog."),
        ("The tiny colorful bird built the large nest.", "The large nest was built by the tiny colorful bird.")
    ],
    "Abstract/Corporate (Formal)": [
        ("The board approved the merger.", "The merger was approved by the board."),
        ("The committee rejected the proposal.", "The proposal was rejected by the committee."),
        ("The policy changed the workflow.", "The workflow was changed by the policy."),
        ("The audit revealed the error.", "The error was revealed by the audit.")
    ],
    "Scientific (Technical-Lite)": [
        ("The experiment proved the theory.", "The theory was proved by the experiment."),
        ("The friction stopped the motion.", "The motion was stopped by the friction."),
        ("The reaction released the energy.", "The energy was released by the reaction."),
        ("The virus infected the cell.", "The cell was infected by the virus.")
    ],
    "Mythological (Epic)": [
        ("Zeus defeated the Titans.", "The Titans were defeated by Zeus."),
        ("Arthur pulled the sword.", "The sword was pulled by Arthur."),
        ("Thor threw the hammer.", "The hammer was thrown by Thor."),
        ("The dragon guarded the gold.", "The gold was guarded by the dragon.")
    ],
    "No-Agent (Truncated Passive)": [
         # Testing if steering works when 'by X' is missing. 
         # Compare 'The man ate the apple' vs 'The apple was eaten.'
         ("The man ate the apple.", "The apple was eaten."), 
         ("The boy threw the ball.", "The ball was thrown."),
         ("The committee rejected the proposal.", "The proposal was rejected."),
         ("The experiment proved the theory.", "The theory was proved.")
    ]
}

# --- CALIBRATION (Standard Technical) ---
LAYER_IDX = 2
CALIB_SET = [
    ("The system processed the data.", "The data was processed by the system."),
    ("The server hosted the website.", "The website was hosted by the server."),
    ("The code compiled the program.", "The program was compiled by the code."),
    ("The layer encoded the input.", "The input was encoded by the layer."),
    ("The model predicted the token.", "The token was predicted by the model."),
    ("The node routed the packet.", "The packet was routed by the node."),
    ("The script executed the command.", "The command was executed by the script."),
    ("The database stored the record.", "The record was stored by the database."),
    ("The program calculated the sum.", "The sum was calculated by the program."),
    ("The analyzer parsed the text.", "The text was parsed by the analyzer."),
    ("The sensor detected the motion.", "The motion was detected by the sensor."),
    ("The filter blocked the spam.", "The spam was blocked by the filter."),
    ("The api returned the response.", "The response was returned by the api."),
    ("The drive saved the file.", "The file was saved by the drive."),
    ("The logic validated the user.", "The user was validated by the logic."),
    ("The screen displayed the image.", "The image was displayed by the screen.")
]

def get_mean_hidden(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad(): outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[LAYER_IDX + 1]
    mask = inputs.attention_mask.unsqueeze(-1)
    return ((h * mask).sum(dim=1) / mask.sum(dim=1)).mean(dim=0)

v_act = get_mean_hidden([x[0] for x in CALIB_SET])
v_pas = get_mean_hidden([x[1] for x in CALIB_SET])
STEERING_VECTOR = v_act - v_pas
print("Steering Vector Calibrated.")

# --- UTILS ---
def get_fiedler_value(adj_matrix):
    if adj_matrix.shape[0] < 2: return 0.0
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device).unsqueeze(0)
    L = gc.construct_laplacian(adj_tensor)
    L = L.cpu()
    evals, _ = sa.compute_eigendecomposition(L)
    return evals[0, 1].item() if evals.shape[1] > 1 else 0.0

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad(): outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

# --- RUN MASSIVE SWEEP ---
# High res alpha sweep
alphas = np.arange(0.01, 0.51, 0.01) # 50 steps
RESULTS = []

print("Starting Exhaustive Sweep...")
for domain_name, pairs in DOMAINS.items():
    print(f"Processing Domain: {domain_name}")
    
    # Pre-compute baselines to save time? No, alpha loop changes nothing for baseline but good for structure.
    # Actually, we can pre-compute baselines once per domain.
    ppl_bases_cache = []
    f_bases_cache = []
    for active, passive in pairs:
        ppl_bases_cache.append(calculate_perplexity(passive))
        inp = tokenizer(passive, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
            f_bases_cache.append(get_fiedler_value(out.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy()))
            
    # Iterate Alphas
    for alpha in alphas:
        alpha = round(alpha, 2)
        
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                v = STEERING_VECTOR.view(1, 1, -1).to(h.dtype)
                return (h + alpha * v,) + out[1:]
            return out + alpha * STEERING_VECTOR.view(1, 1, -1).to(out.dtype)
            
        ppl_steers = []
        f_steers = []
        
        # Batching steering? No, keep simple loops.
        h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
        
        for i, (active, passive) in enumerate(pairs):
             # Steer
            ppl_steer = calculate_perplexity(passive)
            inp = tokenizer(passive, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                f_steer = get_fiedler_value(out.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy())
            
            ppl_steers.append(ppl_steer)
            f_steers.append(f_steer)
            
        h.remove()
        
        # Calculate Aggregates
        mean_ppl_base = np.mean(ppl_bases_cache)
        mean_ppl_steer = np.mean(ppl_steers)
        mean_f_base = np.mean(f_bases_cache)
        mean_f_steer = np.mean(f_steers)
        
        pct_gain = (mean_ppl_base - mean_ppl_steer) / mean_ppl_base * 100
        # Simple recovery metric: Increase in Fiedler
        fiedler_gain = mean_f_steer - mean_f_base
        
        RESULTS.append({
            "Domain": domain_name,
            "Alpha": alpha,
            "Mean PPL Gain %": pct_gain,
            "mean_ppl_base": mean_ppl_base,
            "mean_ppl_steer": mean_ppl_steer,
            "Mean Fiedler Gain": fiedler_gain,
            "Mean Fiedler Steered": mean_f_steer
        })

df = pd.DataFrame(RESULTS)
df.to_csv("exhaustive_sweep_results.csv", index=False)

# Analysis
print("\n--- TOP PPL GAIN BY DOMAIN ---")
for domain in DOMAINS.keys():
    subset = df[df["Domain"] == domain]
    best = subset.sort_values("Mean PPL Gain %", ascending=False).iloc[0]
    print(f"{domain}: PPL Gain {best['Mean PPL Gain %']:.2f}% @ Alpha {best['Alpha']}")

print("\n--- TOP FIEDLER GAIN BY DOMAIN ---")
for domain in DOMAINS.keys():
    subset = df[df["Domain"] == domain]
    best = subset.sort_values("Mean Fiedler Gain", ascending=False).iloc[0]
    print(f"{domain}: Fiedler Gain {best['Mean Fiedler Gain']:.4f} @ Alpha {best['Alpha']}")

print("\n--- GLOBAL OPTIMUM (PPL) ---")
print(df.sort_values("Mean PPL Gain %", ascending=False).head(5))
