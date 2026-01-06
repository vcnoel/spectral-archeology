import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer

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

# --- DOMAINS ---
DOMAINS = {
    "Natural (Baseline)": [
        ("The man ate the apple.", "The apple was eaten by the man."),
        ("The boy threw the ball.", "The ball was thrown by the boy."),
        ("The woman drove the car.", "The car was driven by the woman.")
    ],
    "Abstract/Concepts": [
        ("The memory haunted the man.", "The man was haunted by the memory."),
        ("The truth shocked the nation.", "The nation was shocked by the truth."),
        ("The idea changed the world.", "The world was changed by the idea."),
        ("The silence filled the room.", "The room was filled by the silence."),
        ("The fear paralyzed the group.", "The group was paralyzed by the fear.")
    ],
    "Emotional/Social": [
        ("The mother loved the child.", "The child was loved by the mother."),
        ("The fan admired the actor.", "The actor was admired by the fan."),
        ("The crowd cheered the player.", "The player was cheered by the crowd."),
        ("The friend helped the neighbor.", "The neighbor was helped by the friend."),
        ("The teacher praised the student.", "The student was praised by the teacher.")
    ],
    "Physical/Destructive": [
        ("The fire burned the house.", "The house was burned by the fire."),
        ("The wind broke the window.", "The window was broken by the wind."),
        ("The storm destroyed the city.", "The city was destroyed by the storm."),
        ("The car hit the tree.", "The tree was hit by the car."),
        ("The rock crushed the glass.", "The glass was crushed by the rock.")
    ]
}

# --- CALIBRATION (Technical - Standard Repair Tool) ---
LAYER_IDX = 2
CALIB_SET = [
    ("The system processed the data.", "The data was processed by the system."),
    ("The server hosted the website.", "The website was hosted by the server."),
    ("The code compiled the program.", "The program was compiled by the code."),
    ("The layer encoded the input.", "The input was encoded by the layer.")
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

# --- RUN SWEEP ---
alphas = [0.15, 0.20, 0.25, 0.30, 0.35]
RESULTS = []

for domain_name, pairs in DOMAINS.items():
    print(f"\nProcessing Domain: {domain_name}")
    for alpha in alphas:
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                v = STEERING_VECTOR.view(1, 1, -1).to(h.dtype)
                return (h + alpha * v,) + out[1:]
            return out + alpha * STEERING_VECTOR.view(1, 1, -1).to(out.dtype)
            
        ppl_gains, recovs = [], []
        
        for active, passive in pairs:
            # Base
            ppl_base = calculate_perplexity(passive)
            inp = tokenizer(passive, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                f_base = get_fiedler_value(out.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy())
            
            # Steer
            h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
            ppl_steer = calculate_perplexity(passive)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                f_steer = get_fiedler_value(out.attentions[LAYER_IDX+1][0].mean(dim=0).float().cpu().numpy())
            h.remove()
            
            pct_gain = (ppl_base - ppl_steer) / ppl_base * 100
            # Approx recov for speed (using fixed bounds for relative comparison)
            # Or just raw f_steer - f_base
            recov = (f_steer - f_base) * 100 # Scaling for readability
            
            ppl_gains.append(pct_gain)
            recovs.append(recov)
            
        RESULTS.append({
            "Domain": domain_name,
            "Alpha": alpha,
            "Mean PPL Gain": np.mean(ppl_gains),
            "Mean Fiedler Delta": np.mean(recovs)
        })

df = pd.DataFrame(RESULTS)
print(df.sort_values(["Mean PPL Gain", "Mean Fiedler Delta"], ascending=False))
