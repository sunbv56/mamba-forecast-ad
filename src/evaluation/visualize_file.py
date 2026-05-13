import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

# Add src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.mamba import MambaTSOfficial, MambaTSConfig
from src.data import B02Dataset
from src.evaluation.anomaly_scorer import calculate_anomaly_score

def visualize_file(file_name, model_path, config_path, device):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lookback = config['data'].get('lookback', 512)
    horizon = config['data'].get('forecast_len', 128)
    stride = 32 # Use smaller stride for smoother visualization
    
    # 2. Load Model
    mamba_config = MambaTSConfig(
        in_channels=2,
        lookback=lookback,
        forecast_len=horizon,
        patch_size=64,
        stride=32,
        d_model=64,
        n_layers=4,
        dropout=0.2,
        VPT_mode=1,
        ATSP_solver='SA'
    )
    model = MambaTSOfficial(mamba_config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 3. Load Data File
    data_path = os.path.join(config['data']['processed_dir'], file_name)
    signal = torch.load(data_path, map_location='cpu', weights_only=True) # (2, L)
    
    # Load Operating Conditions
    oc_path = os.path.join(config['data']['processed_dir'], 'operating_conditions.csv')
    import pandas as pd
    oc_df = pd.read_csv(oc_path)
    # The file names are in the format data_B02_Mxxxx.pt
    # We need to find the index to get the OC
    # B02Dataset skips 51 files, but the CSV might have all.
    # In B02Dataset: self.files = sorted(...) [51:]
    # self.oc_df = pd.read_csv(...) [51:]
    # So we need to match the file name to the index in sorted(os.listdir)
    all_files = sorted([f for f in os.listdir(config['data']['processed_dir']) if f.endswith('.pt')])
    file_idx = all_files.index(file_name)
    oc = oc_df.iloc[file_idx].values[1:].astype('float32')
    oc = torch.from_numpy(oc).unsqueeze(0).to(device) # (1, 6)
    
    # [IMPORTANT] Load RMS stats to match labeling logic
    # We need to know if this file is considered "Faulty" by the dataset
    dataset_temp = B02Dataset(config['data']['processed_dir'], lookback, horizon, stride, split='test')
    healthy_baseline = dataset_temp.healthy_rms_baseline
    fault_threshold = healthy_baseline * 3.0
    file_rms = dataset_temp.file_rms[file_name]
    is_faulty_file = file_rms > fault_threshold
    
    print(f"File: {file_name}")
    print(f"File RMS: {file_rms:.4f} | Healthy Baseline: {healthy_baseline:.4f} | Fault Threshold: {fault_threshold:.4f}")
    print(f"Labeled as Faulty: {is_faulty_file}")

    # 4. Sliding Window Inference
    n_samples = signal.shape[1]
    window_indices = range(0, n_samples - lookback - horizon, stride)
    
    errors = []
    rms_values = []
    timestamps = []
    
    with torch.no_grad():
        for start in window_indices:
            end_x = start + lookback
            end_y = end_x + horizon
            
            x = signal[:, start:end_x].unsqueeze(0).to(device) # (1, 2, lookback)
            y = signal[:, end_x:end_y].unsqueeze(0).to(device) # (1, 2, horizon)
            
            # Predict
            y_pred = model(x, oc)
            
            # Score
            score = calculate_anomaly_score(y, y_pred, metric='mse', normalized=True).item()
            errors.append(score)
            
            # RMS of the current window
            win_rms = torch.sqrt(torch.mean(x**2)).item()
            rms_values.append(win_rms)
            
            timestamps.append(end_x) # Plot error at the end of lookback
            
    # 5. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Raw Signal (Channel 0)
    axes[0].plot(signal[0].numpy(), color='gray', alpha=0.5, label='Vibration Ch0')
    axes[0].set_title(f"Raw Signal - {file_name} ({'FAULTY' if is_faulty_file else 'HEALTHY'})")
    axes[0].legend()
    
    # Plot 2: RMS
    axes[1].plot(timestamps, rms_values, color='blue', label='Window RMS')
    axes[1].axhline(y=fault_threshold, color='red', linestyle='--', label='Fault Threshold (3x Baseline)')
    axes[1].set_title("Window RMS")
    axes[1].legend()
    
    # Plot 3: Reconstruction Error
    axes[2].plot(timestamps, errors, color='red', label='MSE Reconstruction Error')
    axes[2].set_title("MambaTS Reconstruction Error")
    axes[2].set_xlabel("Sample Index")
    axes[2].set_yscale('log') # Errors can span multiple orders of magnitude
    axes[2].legend()
    
    plt.tight_layout()
    save_path = f"results/visualizations/ad_check_{file_name.split('.')[0]}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data_B02_M1100.pt")
    parser.add_argument("--model_path", type=str, default="results/models/mambats_official_best.pth")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_file(args.file, args.model_path, args.config, device)
