import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Add src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

def analyze_and_plot_bearing(data_dir, output_dir):
    bearing_name = os.path.basename(data_dir)
    print(f"\nAnalyzing bearing: {bearing_name}...")
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    if not files:
        print(f"No .pt files found in {data_dir}. Skipping.")
        return
    
    rms_list = []
    kurtosis_list = []
    file_names = []
    
    # Process files
    for idx, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        try:
            signal = torch.load(fpath, map_location='cpu', weights_only=True)
            
            # Compute RMS
            rms = torch.sqrt(torch.mean(signal ** 2, dim=-1)).mean().item()
            
            # Compute Kurtosis
            mean = signal.mean(dim=-1, keepdim=True)
            std = signal.std(dim=-1, keepdim=True) + 1e-8
            kurt = torch.mean(((signal - mean) / std) ** 4, dim=-1).mean().item()
            
            rms_list.append(rms)
            kurtosis_list.append(kurt)
            file_names.append(fname)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            
    if not rms_list:
        return
    
    # Save the index-to-filename mapping as a CSV
    mapping_df = pd.DataFrame({
        'Index': range(len(file_names)),
        'Filename': file_names,
        'RMS': rms_list,
        'Kurtosis': kurtosis_list
    })
    
    mapping_path = os.path.join(output_dir, f"{bearing_name}_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved mapping CSV to: {mapping_path}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    indices = np.arange(len(rms_list))
    
    # Plot RMS
    ax1.plot(indices, rms_list, color='tab:blue', linewidth=2, label='RMS')
    ax1.set_ylabel('RMS Value', fontsize=12)
    ax1.set_title(f'RMS Trend - Bearing {bearing_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # Plot Kurtosis
    ax2.plot(indices, kurtosis_list, color='tab:green', linewidth=2, label='Kurtosis')
    ax2.axhline(y=3.0, color='gray', linestyle=':', label='Normal distribution (Kurtosis = 3.0)')
    ax2.set_ylabel('Kurtosis Value', fontsize=12)
    ax2.set_xlabel('File Sequence Index', fontsize=12)
    ax2.set_title(f'Kurtosis Trend - Bearing {bearing_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    
    # Customize x-axis ticks to show filenames at intervals
    tick_interval = max(1, len(indices) // 10)
    tick_indices = indices[::tick_interval]
    tick_labels = [file_names[i] for i in tick_indices]
    
    plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{bearing_name}_trend.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trend plot to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot RMS and Kurtosis trends for bearings.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed directories")
    parser.add_argument("--output_dir", type=str, default="results/visualizations/bearing_trends", help="Output folder")
    parser.add_argument("--bearing", type=str, default=None, help="Name of specific bearing to plot (plots all if None)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.bearing:
        target_dir = os.path.join(args.data_dir, args.bearing)
        if os.path.exists(target_dir):
            analyze_and_plot_bearing(target_dir, args.output_dir)
        else:
            print(f"Bearing directory {target_dir} does not exist.")
    else:
        # Loop through all subfolders
        subdirs = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
        for subdir in subdirs:
            # Skip hidden folders and PRONOSTIA/PHM datasets
            if subdir.startswith('.') or 'pronostia' in subdir.lower() or 'phm' in subdir.lower():
                continue
            analyze_and_plot_bearing(os.path.join(args.data_dir, subdir), args.output_dir)

if __name__ == "__main__":
    main()
