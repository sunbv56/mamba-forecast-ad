import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.mamba import MambaTSOfficial, MambaTSConfig, HybridMambaCNN
from src.data import B02Dataset
from src.evaluation.anomaly_scorer import calculate_anomaly_score

def visualize_full_lifecycle(model_path, config_path, device, skip_first_pct=0.0):
    """
    Vẽ xu hướng toàn bộ vòng đời của vòng bi (tất cả các file trong processed_dir).
    """
    # --- Toggles ---
    USE_PHYSICAL_STATS = False  
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    train_ratio = config['data'].get('train_ratio', 0.5)
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    highpass_freq = config['data'].get('highpass_freq', 0)
    sampling_rate = config['data'].get('sampling_rate', 128000)
    patch_size = config['data'].get('patch_size', 2048)
    stride = config['data'].get('stride', 1024)
    processed_dir = config['data']['processed_dir']
    
    # 1. Load Model
    if "hybrid" in model_path.lower():
        model = HybridMambaCNN({
            'model': {
                'mamba_version': 1,
                'mamba_d_model': config['model'].get('mamba_d_model', 64), 
                'mamba_n_layer': config['model'].get('mamba_n_layer', 4),
                'mamba_d_state': config['model'].get('mamba_d_state', 16), 
                'mamba_d_conv': config['model'].get('mamba_d_conv', 4), 
                'mamba_expand': config['model'].get('mamba_expand', 2),
                'forecast_len': horizon, 
                'patch_size': patch_size, 
                'stride': stride,
                'in_channels': 2, 'lookback': lookback,
                'decomp_kernel': config['model'].get('decomp_kernel', 25), 
                'use_multiscale': True,
            },
            'data': {'patch_size': patch_size, 'stride': stride, 'lookback': lookback}
        })
    else:
        mamba_config = MambaTSConfig(
            in_channels=2, lookback=lookback, forecast_len=horizon,
            patch_size=config['data'].get('patch_size', 64), 
            stride=config['data'].get('stride', 32), 
            d_model=config['model'].get('mamba_d_model', 64), 
            n_layers=config['model'].get('mamba_n_layer', 4),
            VPT_mode=1, ATSP_solver='SA'
        )
        model = MambaTSOfficial(mamba_config)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 2. Get All Files using B02Dataset logic
    dataset = B02Dataset(processed_dir, lookback, horizon, stride, split='test',
                          file_sample_ratio=1, normalize=False, 
                          train_ratio=train_ratio, skip_ratio=skip_ratio,
                          highpass_freq=highpass_freq, sampling_rate=sampling_rate)
    
    all_files = dataset.files
    n_files_total = len(all_files)
    
    # Bỏ qua X% dữ liệu đầu tiên (thường bị nhiễu khởi động)
    start_idx = int(n_files_total * skip_first_pct)
    if start_idx > 0:
        print(f"Skipping first {skip_first_pct*100:.1f}% ({start_idx} files)...")
        all_files = all_files[start_idx:]
    
    n_files = len(all_files)
    
    # Identify split boundaries for visualization (Dựa trên index gốc)
    skip_end_abs  = int(n_files_total * skip_ratio)
    train_end_abs = int(n_files_total * (skip_ratio + train_ratio))
    
    file_indices = []
    avg_errors = []
    file_rms = []
    file_kurtosis = []
    file_crest_factor = []
    
    print(f"Evaluating Full Life-cycle trend across {n_files} files...")
    
    with torch.no_grad():
        for i, f_name in enumerate(tqdm(all_files)):
            actual_idx = i + start_idx
            f_path = os.path.join(processed_dir, f_name)
            signal = torch.load(f_path, map_location='cpu', weights_only=True)
            
            if highpass_freq > 0:
                from scipy import signal as scipy_signal
                nyq = 0.5 * sampling_rate
                normal_cutoff = highpass_freq / nyq
                b, a = scipy_signal.butter(4, normal_cutoff, btype='high', analog=False)
                sig_np = signal.numpy()
                sig_filtered = scipy_signal.lfilter(b, a, sig_np, axis=1)
                signal = torch.from_numpy(sig_filtered.copy()).float()

            # Sample 5 windows per file
            n_samples = signal.shape[1]
            win_starts = np.linspace(0, n_samples - lookback - horizon, 5, dtype=int)
            
            file_win_kurt = []
            file_win_crest = []
            errs = []
            
            for start in win_starts:
                x = signal[:, start:start+lookback]
                y = signal[:, start+lookback:start+lookback+horizon]
                
                x_gpu = x.unsqueeze(0).to(device)
                y_gpu = y.unsqueeze(0).to(device)
                
                # Physical stats for model
                mean = x.mean(dim=-1, keepdim=True)
                std  = x.std(dim=-1, keepdim=True)
                rms  = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
                peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
                z = (x - mean) / (std + 1e-8)
                skew = torch.mean(z**3, dim=-1, keepdim=True)
                kurt = torch.mean(z**4, dim=-1, keepdim=True)
                crest = peak / (rms + 1e-8)
                shape = rms / (torch.mean(torch.abs(x), dim=-1, keepdim=True) + 1e-8)
                
                stats = torch.cat([mean, std, rms, peak, skew, kurt, crest, shape], dim=-1)
                stats = stats.unsqueeze(0).to(device)
                
                # Metrics for plotting
                file_win_kurt.append(kurt.mean().item())
                file_win_crest.append(crest.mean().item())

                if not USE_PHYSICAL_STATS:
                    stats = torch.zeros_like(stats)
                
                y_pred = model(x_gpu, stats)
                errs.append(calculate_anomaly_score(y_gpu, y_pred, metric='mse', normalized=False).item())
            
            avg_errors.append(np.mean(errs))
            file_rms.append(dataset.file_rms[f_name])
            file_kurtosis.append(np.mean(file_win_kurt))
            file_crest_factor.append(np.mean(file_win_crest))
            file_indices.append(actual_idx)

    # 3. Plot Combined Trends
    fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    color_err = 'tab:red'
    label_err = "Anomaly Score (MSE)"
    
    # Helper to add split regions
    def add_regions(ax):
        ax.axvspan(0, skip_end_abs, color='gray', alpha=0.1, label='Skip/Init')
        ax.axvspan(skip_end_abs, train_end_abs, color='blue', alpha=0.05, label='Train/Val Area')
        ax.axvspan(train_end_abs, n_files_total, color='red', alpha=0.05, label='Test/Fault Area')

    # Subplot 1: RMS vs Anomaly Score
    color_rms = 'tab:blue'
    ax1.set_ylabel('Signal RMS', color=color_rms, fontsize=12)
    ax1.plot(file_indices, file_rms, color=color_rms, linewidth=2, label='RMS')
    ax1.tick_params(axis='y', labelcolor=color_rms)
    ax1.set_title("Full Life-cycle Trend: RMS vs. Anomaly Score", fontsize=14)
    ax1.grid(True, alpha=0.3)
    add_regions(ax1)
    ax1.set_xlim(start_idx, n_files_total)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel(label_err, color=color_err, fontsize=12)
    ax1_twin.plot(file_indices, avg_errors, color=color_err, linewidth=2, alpha=0.6, label=label_err)
    ax1_twin.tick_params(axis='y', labelcolor=color_err)
    ax1_twin.set_yscale('log')

    # Subplot 2: Kurtosis vs Anomaly Score
    color_kurt = 'tab:green'
    ax3.set_ylabel('Kurtosis', color=color_kurt, fontsize=12)
    ax3.plot(file_indices, file_kurtosis, color=color_kurt, linewidth=2, label='Kurtosis')
    ax3.tick_params(axis='y', labelcolor=color_kurt)
    ax3.set_title("Full Life-cycle Trend: Kurtosis vs. Anomaly Score", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5)
    add_regions(ax3)

    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel(label_err, color=color_err, fontsize=12)
    ax3_twin.plot(file_indices, avg_errors, color=color_err, linewidth=1.5, alpha=0.4)
    ax3_twin.tick_params(axis='y', labelcolor=color_err)
    ax3_twin.set_yscale('log')

    # Subplot 3: Crest Factor vs Anomaly Score
    color_crest = 'tab:orange'
    ax4.set_ylabel('Crest Factor', color=color_crest, fontsize=12)
    ax4.plot(file_indices, file_crest_factor, color=color_crest, linewidth=2, label='Crest Factor')
    ax4.tick_params(axis='y', labelcolor=color_crest)
    ax4.set_xlabel('File Sequence (Time)', fontsize=12)
    ax4.set_title("Full Life-cycle Trend: Crest Factor vs. Anomaly Score", fontsize=14)
    ax4.grid(True, alpha=0.3)
    add_regions(ax4)

    ax4_twin = ax4.twinx()
    ax4_twin.set_ylabel(label_err, color=color_err, fontsize=12)
    ax4_twin.plot(file_indices, avg_errors, color=color_err, linewidth=1.5, alpha=0.4)
    ax4_twin.tick_params(axis='y', labelcolor=color_err)
    ax4_twin.set_yscale('log')

    plt.tight_layout()
    dataset_name = os.path.basename(processed_dir.rstrip(os.sep))
    save_path = f"results/visualizations/full_lifecycle_trend_{dataset_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Full life-cycle trend visualization saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/models/mamba1_hybrid_best.pth", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--skip_first_pct", type=float, default=0.05, help="Skip the first X% of files (default 0.05 = 5%)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    visualize_full_lifecycle(args.model_path, args.config, args.device, skip_first_pct=args.skip_first_pct)
