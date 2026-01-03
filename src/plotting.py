
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    sns.set_context("poster") 
    sns.set_style("whitegrid", {"font.family": "serif"})
    
    # PDF config
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

def extract_metric(results, metric_name):
    traces = []
    for item in results:
        traj = item.get('trajectory', [])
        trace = [step.get(metric_name) for step in traj]
        if trace:
            traces.append(trace)
    
    if not traces:
        return np.array([])
        
    max_len = max(len(t) for t in traces)
    norm = []
    for t in traces:
        norm.append(t + [np.nan]*(max_len - len(t)))
        
    return np.array(norm)

def plot_multilingual_benchmark(results, model_name, output_dir="figures"):
    setup_style()
    
    TITLE_SIZE = 34
    LABEL_SIZE = 28
    TICK_SIZE = 24
    LEGEND_SIZE = 26 

    unique_langs = sorted(list(set(r.get('lang', 'unknown') for r in results)))
    metrics = [
        ("fiedler_value", "Fiedler Value"),
        ("hfer", "HFER"),
        ("smoothness", "Smoothness Index"),
        ("entropy", "Spectral Entropy")
    ]
    
    if len(unique_langs) > 10:
        palette = sns.color_palette("husl", len(unique_langs))
    else:
        palette = sns.color_palette("tab10", len(unique_langs))
    lang_color_map = {lang: palette[i] for i, lang in enumerate(unique_langs)}

    fig, axes = plt.subplots(2, 4, figsize=(32, 16), sharex=True) 
    
    handles = []
    labels = []
    
    en_delta_min = 0.0
    en_delta_layer = 0

    for idx, (key, title) in enumerate(metrics):
        ax_raw = axes[0][idx]
        ax_diff = axes[1][idx]
        
        for lang in unique_langs:
            color = lang_color_map.get(lang, 'black')
            
            # Active (Solid)
            active_items = [r for r in results if r.get('lang') == lang and r.get('type') == 'active']
            mean_active = None
            if active_items:
                data = extract_metric(active_items, key)
                if data.size > 0:
                    mean_active = np.nanmean(data, axis=0)
                    x = np.arange(len(mean_active))
                    l, = ax_raw.plot(x, mean_active, color=color, linestyle='-', linewidth=5, alpha=0.9)
                    
                    if idx == 0: 
                        handles.append(l)
                        labels.append(lang)
                        
            # Passive (Dotted)
            passive_items = [r for r in results if r.get('lang') == lang and r.get('type') == 'passive']
            mean_passive = None
            if passive_items:
                data = extract_metric(passive_items, key)
                if data.size > 0:
                    mean_passive = np.nanmean(data, axis=0)
                    x = np.arange(len(mean_passive))
                    ax_raw.plot(x, mean_passive, color=color, linestyle=':', linewidth=5, alpha=0.8)

            # Delta
            if mean_active is not None and mean_passive is not None:
                min_len = min(len(mean_active), len(mean_passive))
                diff = mean_passive[:min_len] - mean_active[:min_len]
                x = np.arange(len(diff))
                ax_diff.plot(x, diff, color=color, linewidth=5, alpha=0.9)
                ax_diff.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=2)
                
                if lang == 'en' and key == 'fiedler_value':
                    min_val = np.min(diff)
                    min_idx = np.argmin(diff)
                    en_delta_min = min_val
                    en_delta_layer = min_idx

        ax_raw.set_title(title, fontweight="bold", fontsize=TITLE_SIZE)
        ax_raw.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        if idx == 0:
            ax_raw.set_ylabel("Metric Value", fontsize=LABEL_SIZE)
        else:
            ax_raw.set_ylabel("", fontsize=LABEL_SIZE)
             
        ax_diff.set_xlabel("Layer", fontsize=LABEL_SIZE)
        ax_diff.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        
        if idx == 0:
            ax_diff.set_ylabel(r"$\Delta$ (Passive - Active)", fontsize=LABEL_SIZE)
            if en_delta_min < -0.3:
                ax_diff.annotate('Synthetic Scar', 
                                 xy=(en_delta_layer + 1.5, en_delta_min), 
                                 xytext=(en_delta_layer + 5, -0.4), 
                                 arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=10),
                                 fontsize=LABEL_SIZE, fontweight='bold', color='black')
        else:
             ax_diff.set_ylabel("", fontsize=LABEL_SIZE)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=4, label='Active'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=4, label='Passive')
    ]
    style_labels = [r"$\bf{Active}$", r"$\bf{Passive}$"]
    
    bold_lang_labels = [r"$\bf{" + l.replace(" ", r"\ ") + "}$" for l in labels]
        
    final_handles = style_handles + handles
    final_labels = style_labels + bold_lang_labels
    
    plt.tight_layout(rect=[0, 0.08, 1, 1.0]) 
    
    fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=len(final_handles), frameon=False, fontsize=LEGEND_SIZE)
    
    os.makedirs(output_dir, exist_ok=True)
    sanitized_name = model_name.replace(" ", "_").replace(".", "")
    out_pdf = os.path.join(output_dir, f"{sanitized_name}_spectral.pdf")
    out_png = os.path.join(output_dir, f"{sanitized_name}_spectral.png")
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")

def plot_tokenization_topology(result_path, output_dir="figures"):
    setup_style()
    
    with open(result_path, 'r') as f:
        data = json.load(f)
        
    groups = { 'ja': [], 'ja_romaji': [], 'en_trans': [] }
    for item in data:
        lang = item.get('lang')
        traj = item.get('trajectory', [])
        fvals = [step.get('fiedler_value') for step in traj]
        
        # Simple fix for None
        fvals = [f if f is not None else np.nan for f in fvals]
        
        if lang in groups:
            groups[lang].append(fvals)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'ja': '#e74c3c',       # Red (Asian/Dense)
        'ja_romaji': '#9b59b6',# Purple (Conflict Zone)
        'en_trans': '#3498db'  # Blue (Western/Sparse)
    }
    
    labels = {
        'ja': 'Japanese (Kana)',
        'ja_romaji': 'Japanese (Romaji)',
        'en_trans': 'English (Translation)'
    }

    for lang, trajectories in groups.items():
        if not trajectories: continue
        trajectories = np.array(trajectories)
        layers = np.arange(trajectories.shape[1])
        
        for traj in trajectories:
            ax.plot(layers, traj, color=colors.get(lang, 'gray'), alpha=0.3, linewidth=1.0)
            
        mean_traj = np.nanmean(trajectories, axis=0)
        ax.plot(layers, mean_traj, color=colors.get(lang, 'black'), linewidth=4.0, label=labels.get(lang, lang))
        
    ax.set_xlabel('Layer', fontweight='bold', fontsize=18)
    ax.set_ylabel(r'Fiedler Value ($\lambda_2$)', fontweight='bold', fontsize=18)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=16)
    ax.set_ylim(0, 1.0)
    
    ax.text(39, 0.98, 'Dense Regime', color='#e74c3c', ha='right', va='top', fontweight='bold', fontsize=16)
    ax.text(35, 0.08, 'Sparse Regime', color='#3498db', ha='right', va='bottom', fontweight='bold', fontsize=16)

    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "topology_law.png")
    
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.pdf'), dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
