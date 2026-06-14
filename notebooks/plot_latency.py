import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Define paths
RES_MD_PATH = r"f:\APPS_PJ\mamba-forecast-ad\notebooks\final\res.md"
OUTPUT_DIR = r"f:\APPS_PJ\mamba-forecast-ad\Hydrid-Mamba-for-Predictive-Bearing-Fault\artifacts\figures"

# Resolve paths for WSL if running in WSL
if not os.path.exists(RES_MD_PATH):
    wsl_res_path = RES_MD_PATH.replace("f:\\", "/mnt/f/").replace("\\", "/")
    if os.path.exists(wsl_res_path):
        RES_MD_PATH = wsl_res_path
        
if "/mnt/f/" in RES_MD_PATH or not os.path.exists(r"f:\APPS_PJ"):
    OUTPUT_DIR = OUTPUT_DIR.replace("f:\\", "/mnt/f/").replace("\\", "/")

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hinh_7.png")


def parse_res_md(file_path):
    """Parses res.md and extracts latency data for specific models and batch sizes."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using fallback values.")
        return get_fallback_values()
        
    current_bs = "64" # Default fallback BS
    parsed_data = {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Track batch size section
            bs_match = re.search(r"Batch Size\s*=\s*(\d+)", line, re.IGNORECASE)
            if bs_match:
                current_bs = bs_match.group(1)
                continue
                
            if not line.startswith("|"):
                continue
                
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 13:
                continue
                
            model_name_raw = cols[1]
            if "model" in model_name_raw.lower() or "---" in model_name_raw:
                continue
                
            model_name_clean = re.sub(r"\*\*|\*", "", model_name_raw).strip()
            
            # Extract BS from model name if present, e.g. "Mamba-Hybrid (BS=64)"
            bs_in_name_match = re.search(r"\((?:BS|batch\s*size)\s*=\s*(\d+)\)", model_name_clean, re.IGNORECASE)
            if bs_in_name_match:
                bs = bs_in_name_match.group(1)
                model_name_base = re.sub(r"\s*\((?:BS|batch\s*size)\s*=\s*\d+\)", "", model_name_clean, flags=re.IGNORECASE).strip()
            else:
                bs = current_bs
                model_name_base = model_name_clean
            
            # Map known models
            if "lstm" in model_name_base.lower():
                model_key = "LSTM"
            elif "simple-mamba" in model_name_base.lower():
                model_key = "Simple-Mamba"
            elif "patchtst" in model_name_base.lower():
                model_key = "PatchTST"
            elif "mamba-hybrid" in model_name_base.lower() or "hybridmamba" in model_name_base.lower():
                model_key = "Mamba-Hybrid"
            else:
                continue
                
            try:
                def clean_val(val_str):
                    return float(re.sub(r"\*\*|\*", "", val_str).strip())
                lat_inf = clean_val(cols[12])
                
                parsed_data[(model_key, bs)] = {
                    "inference": lat_inf,
                    "bs": bs
                }
            except ValueError:
                continue
                
    fallback = get_fallback_values()
    final_data = {}
    
    # We want these specific combinations:
    targets = [
        ("LSTM", "64"),
        ("Simple-Mamba", "64"),
        ("PatchTST", "64"),
        ("Mamba-Hybrid", "64"),
        ("Mamba-Hybrid", "1024")
    ]
    
    for model, bs in targets:
        key_str = f"{model} (BS={bs})"
        if (model, bs) in parsed_data:
            final_data[key_str] = parsed_data[(model, bs)]
        else:
            final_data[key_str] = fallback[key_str]
            
    return final_data


def get_fallback_values():
    """Provides fallback values based on the expected res.md contents."""
    return {
        "LSTM (BS=64)": {"inference": 14.1646, "bs": "64"},
        "Simple-Mamba (BS=64)": {"inference": 5.6641, "bs": "64"},
        "PatchTST (BS=64)": {"inference": 2.0382, "bs": "64"},
        "Mamba-Hybrid (BS=64)": {"inference": 3.9245, "bs": "64"},
        "Mamba-Hybrid (BS=1024)": {"inference": 0.8155, "bs": "1024"}
    }


def main():
    print("Parsing latencies from res.md...")
    data = parse_res_md(RES_MD_PATH)
    
    models = [
        "LSTM (BS=64)",
        "Simple-Mamba (BS=64)",
        "PatchTST (BS=64)",
        "Mamba-Hybrid (BS=64)",
        "Mamba-Hybrid (BS=1024)"
    ]
    latencies = [data[m]["inference"] for m in models]
    batch_sizes = [data[m]["bs"] for m in models]
    
    print("Parsed values:")
    for m, lat, bs in zip(models, latencies, batch_sizes):
        print(f" - {m}: {lat:.4f} ms")
        
    # Set up matplotlib style for professional academic output
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'text.usetex': False,
        'svg.fonttype': 'none'
    })
    
    # Calculate slowdown relative to Mamba-Hybrid (BS=1024)
    baseline_key = "Mamba-Hybrid (BS=1024)"
    mamba_lat = data[baseline_key]["inference"]
    slowdowns = [lat / mamba_lat for lat in latencies]
    
    # Color palette matching professional light-theme style
    # LSTM (steel blue), Simple-Mamba (orange), PatchTST (bright blue), Mamba-Hybrid BS=64 (teal), Mamba-Hybrid BS=1024 (vibrant green)
    colors = ['#7a82ab', '#f39c12', '#3498db', '#1abc9c', '#2ecc71']
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Grid configuration
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc', zorder=0)
    
    # X axis labels showing model and batch size symmetrically
    x_labels = [
        "LSTM\n(BS=64)",
        "Simple-Mamba\n(BS=64)",
        "PatchTST\n(BS=64)",
        "Mamba-Hybrid\n(BS=64)",
        "Mamba-Hybrid\n(BS=1024)"
    ]
    
    # Plot bars
    bars = ax.bar(x_labels, latencies, color=colors, edgecolor='grey', linewidth=0.7, width=0.55, zorder=3)
    
    # Add values and slowdown labels on top of the bars (no text inside the bars)
    for bar, lat, slowdown, bs, model_name in zip(bars, latencies, slowdowns, batch_sizes, models):
        yval = bar.get_height()
        
        if model_name == baseline_key:
            label_text = f"{lat:.3f} ms\n(Baseline)"
        else:
            label_text = f"{lat:.3f} ms\n{slowdown:.1f}x slower"
            
        ax.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + 0.15, 
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=11.0, 
            fontweight='bold', 
            color='black',
            zorder=5
        )
            
    # Set titles and labels with academic scaling (larger fonts)
    ax.set_title("Real-Time Inference Latency per Sample (ms/sample)\n(Model Architecture Comparison)", 
                 fontsize=15, fontweight='bold', pad=20, color='black')
    ax.set_ylabel("Inference Latency (ms)", fontsize=13, fontweight='bold', color='black')
    ax.set_xlabel("Model Architecture & Configuration", fontsize=13, fontweight='bold', color='black')
    
    # Configure ticks and spine styles
    ax.tick_params(colors='black', labelsize=12)
    ax.set_ylim(0, max(latencies) * 1.25)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Tight layout to avoid truncation
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save high-resolution figure
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Successfully saved plot to → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
