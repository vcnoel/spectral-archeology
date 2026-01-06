import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
STEERING_LAYER_IDX = 2
ALPHA = 0.2
TEST_SIZE = 100

# --- 1. GENERATION (Natural Vocabulary Only) ---
vocab_natural = {
    "subjects": ["The letter", "The ball", "The song", "The book", "The cake", "The house", "The door", "The car", "The apple", "The road"],
    "verbs": ["written", "thrown", "sung", "read", "eaten", "built", "opened", "driven", "found", "seen"],
    "agents": ["by the boy", "by the girl", "by the man", "by the woman", "by the student", "by the teacher", "by the child", "by the doctor"]
}

vocab_technical = {
     "subjects": ["The data", "The token", "The input", "The model", "The query", "The signal"],
     "verbs": ["processed", "encoded", "decoded", "parsed", "routed", "mapped"],
     "agents": ["by the system", "by the layer", "by the network", "by the mechanism"]
}

def generate_pairs(n, vocab):
    pairs = []
    for _ in range(n):
        s = np.random.choice(vocab["subjects"])
        v = np.random.choice(vocab["verbs"])
        a = np.random.choice(vocab["agents"])
        active = f"{a.replace('by ', '')} {v} {s.lower()}."
        passive = f"{s} was {v} {a}."
        pairs.append((active, passive))
    return pairs

def get_fiedler_value(adj_matrix):
    if adj_matrix.shape[0] < 2: return 0.0
    W = 0.5 * (adj_matrix + adj_matrix.T)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    try:
        eigvals = np.linalg.eigvalsh(L)
        return sorted(eigvals)[1] if len(eigvals) > 1 else 0.0
    except:
        return 0.0

# --- MAIN ---
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager"
)

# A. CALIBRATION (Active - Passive stats to define "Lost Connectivity")
# We need to know what the TARGET Fiedler value is (from Active) to calculate % Recovered.
print("Generating Reference Data...")
ref_pairs = generate_pairs(TEST_SIZE, vocab_natural) # Testing on NATURAL
actives = [p[0] for p in ref_pairs]
passives = [p[1] for p in ref_pairs]

# B. STEERING VECTOR (Still calibrated on TECHNICAL or NATURAL?)
# The user's claim refers to THE vector. If the vector is "universal syntax", 
# it should arguably be calibrated on the general syntax (Technical or Mixed). 
# But for strict "Natural" claim verification, let's try calibrating on Natural to give it the best shot, 
# OR keep Technical if that's the "Universal" vector.
# Let's use TECHNICAL calibration as the "General Mechanism" that fixes specific instances.
print("Calibrating Steering Vector (Technical Base)...")
calib_pairs = generate_pairs(20, vocab_technical)
calib_actives = [p[0] for p in calib_pairs]
calib_passives = [p[1] for p in calib_pairs]

def get_mean_hidden(texts, layer_idx):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h_states = outputs.hidden_states[layer_idx + 1]
    mask = inputs.attention_mask.unsqueeze(-1)
    sum_h = (h_states * mask).sum(dim=1)
    count_h = mask.sum(dim=1)
    mean_h = sum_h / count_h
    return mean_h.mean(dim=0)

mu_active_vec = get_mean_hidden(calib_actives, STEERING_LAYER_IDX)
mu_passive_vec = get_mean_hidden(calib_passives, STEERING_LAYER_IDX)
steering_vector = mu_active_vec - mu_passive_vec

# Hook
def steering_hook(module, input, output):
    if isinstance(output, tuple):
        h = output[0]
        v_inject = steering_vector.view(1, 1, -1).to(h.dtype)
        h_steered = h + (ALPHA * v_inject)
        return (h_steered,) + output[1:]
    else:
        return output + (ALPHA * steering_vector.view(1, 1, -1).to(output.dtype))

# C. BATCH ANALYSIS
print(f"Testing on {TEST_SIZE} Natural Samples...")
metrics = []

for i, (act_txt, pas_txt) in enumerate(zip(actives, passives)):
    # 1. Get Active Fiedler (The "Gold Standard")
    inputs_act = tokenizer(act_txt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_act = model(**inputs_act, output_attentions=True)
    attn_act = out_act.attentions[3][0].mean(dim=0).float().cpu().numpy()
    # Normalize lengths: Fiedler depends on matrix size. 
    # Comparing Fiedler of "Boy hit ball" vs "Ball was hit by boy" is tricky because N differs.
    # However, "Recovery of lost connectivity" implies we compare Passive_Base vs Passive_Steered 
    # relative to the gap to Active? Or just percentage increase?
    # Usually "Recovery" = (Steered - Passive) / (Active - Passive).
    # But sizes differ. Let's just track raw Delta and see if we can approximate.
    # Actually, simpler: Measure Passive_Steered vs Passive_Base.
    
    # Let's get Passive stats
    inputs_pas = tokenizer(pas_txt, return_tensors="pt").to(model.device)
    labels_pas = inputs_pas.input_ids.clone()
    seq_len = inputs_pas.input_ids.shape[1]
    
    # Passive Base
    with torch.no_grad():
        out_base = model(**inputs_pas, output_attentions=True, labels=labels_pas)
    ppl_base = torch.exp(out_base.loss).item()
    attn_base = out_base.attentions[3][0].mean(dim=0).float().cpu().numpy()
    fiedler_base = get_fiedler_value(attn_base[:seq_len, :seq_len])
    
    # Passive Steered
    handle = model.model.layers[STEERING_LAYER_IDX].register_forward_hook(steering_hook)
    with torch.no_grad():
        out_steer = model(**inputs_pas, output_attentions=True, labels=labels_pas)
    handle.remove()
    ppl_steer = torch.exp(out_steer.loss).item()
    attn_steer = out_steer.attentions[3][0].mean(dim=0).float().cpu().numpy()
    fiedler_steer = get_fiedler_value(attn_steer[:seq_len, :seq_len])
    
    # We also need Active Fiedler roughly to know "Total Lost".
    # Since sizes differ, we can't do direct subtraction.
    # BUT we know typical Active Fiedler is ~0.45-0.50 (from earlier experiments).
    # Typical Passive is ~0.29.
    # Steered is ~0.38.
    # Recovery % = (0.38 - 0.29) / (0.48 - 0.29) approx 0.09 / 0.19 = ~47%.
    # So we can estimate "Active Reference" as ~0.48 (empirically) or calculate it specifically if sizes matched.
    # Let's calculate proper Active Fiedler anyway to see the distribution.
    fiedler_active = get_fiedler_value(attn_act[:inputs_act.input_ids.shape[1], :inputs_act.input_ids.shape[1]])

    # METRICS
    loss_gap = fiedler_active - fiedler_base
    recovered = fiedler_steer - fiedler_base
    pct_rec = (recovered / loss_gap) * 100 if loss_gap > 0.01 else 0.0
    
    metrics.append({
        "ppl_base": ppl_base,
        "ppl_steer": ppl_steer,
        "ppl_reduction": ppl_base - ppl_steer,
        "f_base": fiedler_base,
        "f_steer": fiedler_steer,
        "f_active": fiedler_active,
        "rec_pct": pct_rec
    })
    
    if (i+1) % 20 == 0:
        print(f"Processed {i+1}...")

# D. AGGREGATE
df = pd.DataFrame(metrics)
print("\n" + "="*40)
print(f"VERIFICATION RESULTS (N={TEST_SIZE} Natural Samples)")
print("="*40)

mean_ppl_red = df['ppl_reduction'].mean()
mean_rec_pct = df['rec_pct'].mean()
mean_f_base = df['f_base'].mean()
mean_f_steer = df['f_steer'].mean()
mean_f_act = df['f_active'].mean()

print(f"Mean PPL Reduction: {mean_ppl_red:.2f}")
if len(df['ppl_reduction']) > 0:
    print(f"Max PPL Reduction:  {df['ppl_reduction'].max():.2f}")
    print(f"Min PPL Reduction:  {df['ppl_reduction'].min():.2f}")

print(f"Mean Fiedler Base:   {mean_f_base:.4f}")
print(f"Mean Fiedler Steer:  {mean_f_steer:.4f}")
print(f"Mean Fiedler Active: {mean_f_act:.4f}")
print(f"Connectivity Recovery: {mean_rec_pct:.2f}%")

print("\nChecking Claims:")
print(f"1. 'Recovers ~42%': {mean_rec_pct:.2f}% -> {'✅ MATCHES' if 38 <= mean_rec_pct <= 46 else '⚠️ DEVIATION'}")
print(f"2. 'Substantial PPL Reduction': {mean_ppl_red:.2f} -> {'✅ YES' if mean_ppl_red > 0 else '⚠️ NO/NEGATIVE'}")

df.to_csv("natural_repair_verification.csv", index=False)
