
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer

# Models to test
MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 1. TEST SAMPLES (Canonical & Natural) ---
TEST_PAIRS = [
    # Natural
    ("The man ate the apple.", "The apple was eaten by the man."),
    ("The boy threw the ball.", "The ball was thrown by the boy."),
    ("The woman drove the car.", "The car was driven by the woman."),
    ("The dog chased the cat.", "The cat was chased by the dog."),
    ("The bird built the nest.", "The nest was built by the bird."),
    # Technical (New, disjoint from calibration)
    ("The router forwarded the packet.", "The packet was forwarded by the router."),
    ("The compiler optimized the code.", "The code was optimized by the compiler."),
    ("The server validated the request.", "The request was validated by the server."),
    ("The function returned the value.", "The value was returned by the function."),
    ("The browser rendered the page.", "The page was rendered by the browser.")
]

# --- 2. SPECTRAL UTILS (Using spectral-trust) ---
def get_fiedler_value(adj_matrix, graph_constructor, spectral_analyzer):
    """Compute Fiedler value using spectral-trust library."""
    if adj_matrix.shape[0] < 2: return 0.0
    
    # Convert to Tensor [1, seq, seq]
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Construct Laplacian (handles symmetrization)
    L = graph_constructor.construct_laplacian(adj_tensor)
    
    # FIX: Move to CPU for SciPy based eigendecomposition
    L = L.cpu()
    
    # Compute Eigenvalues
    evals, _ = spectral_analyzer.compute_eigendecomposition(L)
    
    # Return lambda_2 (batch index 0)
    return evals[0, 1].item() if evals.shape[1] > 1 else 0.0

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

# --- 3. EXPERIMENT LOOP ---
RESULTS = []

for model_name in MODELS:
    print(f"\nProcessing {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            attn_implementation="eager"
        )
        
        # Initialize Spectral Trust Tools
        config = GSPConfig(model_name=model_name, device=device)
        gc = GraphConstructor(config)
        sa = SpectralAnalyzer(config)
        
    except Exception as e:
        print(f"Skipping {model_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

    # CALIBRATE STEERING (Robust Technical Set)
    LAYER_IDX = 2
    technical_calibration = [
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
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad(): outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[LAYER_IDX + 1]
        mask = inputs.attention_mask.unsqueeze(-1)
        return ((h * mask).sum(dim=1) / mask.sum(dim=1)).mean(dim=0)
    
    # Calibration: Expanded Technical Set
    v_act = get_mean_hidden([x[0] for x in technical_calibration])
    v_pas = get_mean_hidden([x[1] for x in technical_calibration])
    steering_vector = v_act - v_pas
    print("Steering calibrated on Expanded Technical Set.")
    
    # Run with alpha sweep (Lower Range for Peak PPL)
    alphas = [0.18, 0.20, 0.22, 0.24, 0.25]
    for alpha in alphas:
        alpha = round(alpha, 2)
        print(f"Testing alpha={alpha}...")
        ppl_bases, ppl_steers = [], []
        f_bases, f_steers, f_actives = [], [], []
    
        def steer_hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                v = steering_vector.view(1, 1, -1).to(h.dtype)
                return (h + alpha * v,) + out[1:]
            return out + alpha * steering_vector.view(1, 1, -1).to(out.dtype)
        
        for active, passive in TEST_PAIRS:
            # Baseline
            ppl_base = calculate_perplexity(model, tokenizer, passive)
            inp = tokenizer(passive, return_tensors="pt").to(model.device)
            with torch.no_grad(): 
                out = model(**inp, output_attentions=True)
                # FIX: Measure effect at LAYER_IDX + 1 (the layer AFFECTED by the steer)
                adj = out.attentions[LAYER_IDX + 1][0].mean(dim=0).float().cpu().numpy()
                f_base = get_fiedler_value(adj, gc, sa)
                
            # Active Ref
            inp_a = tokenizer(active, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_a = model(**inp_a, output_attentions=True)
                adj_a = out_a.attentions[LAYER_IDX + 1][0].mean(dim=0).float().cpu().numpy()
                f_act = get_fiedler_value(adj_a, gc, sa)

            # Steered
            h = model.model.layers[LAYER_IDX].register_forward_hook(steer_hook)
            ppl_steer = calculate_perplexity(model, tokenizer, passive)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                adj = out.attentions[LAYER_IDX + 1][0].mean(dim=0).float().cpu().numpy()
                f_steer = get_fiedler_value(adj, gc, sa)
            h.remove()
            
            ppl_bases.append(ppl_base)
            ppl_steers.append(ppl_steer)
            f_bases.append(f_base)
            f_steers.append(f_steer)
            f_actives.append(f_act)

        # METRICS
        mean_ppl_base = np.mean(ppl_bases)
        mean_ppl_steer = np.mean(ppl_steers)
        mean_f_base = np.mean(f_bases)
        mean_f_steer = np.mean(f_steers)
        mean_f_act = np.mean(f_actives)
        
        pct_gain = (mean_ppl_base - mean_ppl_steer) / mean_ppl_base * 100
        loss_gap = mean_f_act - mean_f_base
        rec_pct = (mean_f_steer - mean_f_base) / loss_gap * 100 if loss_gap > 0 else 0
        
        print(f"  [Alpha {alpha}] Recovery: {rec_pct:.2f}% | PPL Gain: +{pct_gain:.2f}%")
        
        RESULTS.append({
            "Model": model_name,
            "Alpha": alpha,
            "Recov %": rec_pct
        })

# --- 4. DISPLAY TABLE ---
df = pd.DataFrame(RESULTS)
print(df)
