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
print(f"Using device: {device}")

# --- Load Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager"
)

# --- Initialize Spectral Tools ---
config = GSPConfig(model_name=MODEL_NAME, device=device)
gc = GraphConstructor(config)
sa = SpectralAnalyzer(config)

# --- DATA DEFINITIONS ---
NATURAL_PAIRS = [
    ("The man ate the apple.", "The apple was eaten by the man."),
    ("The boy threw the ball.", "The ball was thrown by the boy."),
    ("The woman drove the car.", "The car was driven by the woman."),
    ("The dog chased the cat.", "The cat was chased by the dog."),
    ("The bird built the nest.", "The nest was built by the bird.")
]

TECHNICAL_PAIRS = [
    ("The router forwarded the packet.", "The packet was forwarded by the router."),
    ("The compiler optimized the code.", "The code was optimized by the compiler."),
    ("The server validated the request.", "The request was validated by the server."),
    ("The function returned the value.", "The value was returned by the function."),
    ("The browser rendered the page.", "The page was rendered by the browser.")
]

MIXED_PAIRS = NATURAL_PAIRS + TECHNICAL_PAIRS

# --- HELPER FUNCTIONS ---
def get_fiedler_value(adj_matrix):
    if adj_matrix.shape[0] < 2: return 0.0
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device).unsqueeze(0)
    L = gc.construct_laplacian(adj_tensor)
    L = L.cpu() # Fix for SciPy
    evals, _ = sa.compute_eigendecomposition(L)
    return evals[0, 1].item() if evals.shape[1] > 1 else 0.0

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

# Pre-calculate Steering Vector (Expanded Technical Set)
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
print("Steering Vector Calibrated (16 pairs).")

def run_sweep(name, pairs, alphas):
    print(f"\n--- Running Scenario: {name} ---")
    results = []
    
    for alpha in alphas:
        alpha = round(alpha, 2)
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                v = STEERING_VECTOR.view(1, 1, -1).to(h.dtype)
                return (h + alpha * v,) + out[1:]
            return out + alpha * STEERING_VECTOR.view(1, 1, -1).to(out.dtype)
            
        ppl_bases, ppl_steers = [], []
        f_bases, f_steers, f_actives = [], [], []
        
        for active, passive in pairs:
            # Base
            ppl_base = calculate_perplexity(passive)
            inp = tokenizer(passive, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                f_base = get_fiedler_value(out.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy())
            
            # Active Ref
            inp_a = tokenizer(active, return_tensors="pt").to(device)
            with torch.no_grad():
                out_a = model(**inp_a, output_attentions=True)
                f_act = get_fiedler_value(out_a.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy())
                
            # Steer
            h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
            ppl_steer = calculate_perplexity(passive)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                f_steer = get_fiedler_value(out.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy())
            h.remove()
            
            ppl_bases.append(ppl_base)
            ppl_steers.append(ppl_steer)
            f_bases.append(f_base)
            f_steers.append(f_steer)
            f_actives.append(f_act)
            
        # Aggregation
        mean_ppl_base = np.mean(ppl_bases)
        mean_ppl_steer = np.mean(ppl_steers)
        mean_f_base = np.mean(f_bases)
        mean_f_steer = np.mean(f_steers)
        mean_f_act = np.mean(f_actives)
        
        pct_gain = (mean_ppl_base - mean_ppl_steer) / mean_ppl_base * 100
        loss_gap = mean_f_act - mean_f_base
        rec_pct = (mean_f_steer - mean_f_base) / loss_gap * 100 if loss_gap > 0 else 0
        
        results.append({
            "Alpha": alpha,
            "PPL Gain %": pct_gain,
            "Recovery %": rec_pct
        })
        
    return pd.DataFrame(results)

# --- EXECUTE SCENARIOS ---
alphas = np.arange(0.15, 0.46, 0.05)
# Scenario 1: Natural
df_natural = run_sweep("Natural", NATURAL_PAIRS, alphas)
print("\nTop PPL Gain (Natural):")
print(df_natural.sort_values("PPL Gain %", ascending=False).head(1))
print("\nTop Recovery (Natural):")
print(df_natural.sort_values("Recovery %", ascending=False).head(1))
print("\nFull Natural Table:")
print(df_natural)

# Scenario 2: Technical
low_alphas = np.arange(0.01, 0.16, 0.01)
df_technical = run_sweep("Technical", TECHNICAL_PAIRS, low_alphas)
print("\nTop PPL Gain (Technical):")
print(df_technical.sort_values("PPL Gain %", ascending=False).head(1))
print("\nFull Technical Table:")
print(df_technical)

# Scenario 3: Mixed
df_mixed = run_sweep("Mixed", MIXED_PAIRS, alphas)
print("\nTop Balanced (Mixed, PPL > 0):")
print(df_mixed[df_mixed['PPL Gain %'] > -100].sort_values("PPL Gain %", ascending=False).head(1))
print("\nFull Mixed Table:")
print(df_mixed)

# Scenario 4: Optimization
max_ppl_nat = df_natural['PPL Gain %'].max()
max_rec_nat = df_natural['Recovery %'].max()
try:
    max_ppl_tech = df_technical['PPL Gain %'].max()
except:
    max_ppl_tech = -999
max_rec_tech = df_technical['Recovery %'].max()

print("\n=== OPTIMIZATION SUMMARY ===")
print(f"Max PPL Gain (Natural):   {max_ppl_nat:.2f}%")
print(f"Max PPL Gain (Technical): {max_ppl_tech:.2f}%")
print(f"Max Recovery (Natural):   {max_rec_nat:.2f}%")
print(f"Max Recovery (Technical): {max_rec_tech:.2f}%")

winner_ppl = "Natural" if max_ppl_nat > max_ppl_tech else "Technical"
print(f"\n1. Sentence Type Maximizing PPL: **{winner_ppl}**")

combined_score_nat = max_ppl_nat + max_rec_nat
combined_score_tech = max_ppl_tech + max_rec_tech
winner_combined = "Natural" if combined_score_nat > combined_score_tech else "Technical"
print(f"2. Sentence Type Maximizing Both (Joint Utility): **{winner_combined}**")
