
# Models Dictionary
MODELS = {
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "Phi-3.5": "microsoft/Phi-3.5-mini-instruct",
    "Phi-4": "microsoft/phi-4",
    "Phi-1.5": "microsoft/phi-1_5",
    "Phi-2": "microsoft/phi-2",
    "Orca-2": "microsoft/Orca-2-7b",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen-2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1"
}

# Paths
DATA_DIR_SYNTHETIC = "data/synthetic_multilingual"
OUTPUT_DIR_TOPO = "output_phi4_topology_full"

# Tokenization Topology Experiment Data (The "Killer Control")
TOPOLOGY_SAMPLES = {
    "ja": "猫は犬を追いかける。",         # Dense (Kana)
    "ja_romaji": "Neko wa inu o oikakeru.", # Sparse (Latin Script) - Fixed Punctuation
    "en_trans": "The cat chases the dog."   # Sparse (English Control)
}

# Syntactic Integrity Experiment Data (Passive Voice)
SYNTACTIC_SAMPLES = {
    "active": "The scientist discovered a new particle.",
    "passive": "A new particle was discovered by the scientist."
}
