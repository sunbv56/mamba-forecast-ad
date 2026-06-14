import os
import sys
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import MultiBearingDataset
from src.models.mamba import HybridMambaCNN
from src.evaluation.anomaly_scorer import calculate_anomaly_score
from src.evaluation.metrics import calculate_threshold_pot, calculate_metrics

# 8 physical features in dataset order:
FEATURE_NAMES = [
    "Mean", 
    "Std", 
    "RMS", 
    "Peak-to-Peak", 
    "Skewness", 
    "Kurtosis", 
    "Crest Factor", 
    "Shape Factor"
]

def run_bearing_inference(model, ds, config, device, mask_idx=None, method='zero'):
    """
    Run inference on a single bearing dataset and return MSE list and anomaly scores.
    """
    model.eval()
    loader = DataLoader(ds, batch_size=config['training'].get('batch_size', 128), shuffle=False)
    
    bearing_scores = []
    bearing_labels = []
    mse_list = []
    
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            
            # Apply masking or permutation if requested
            if stats is not None and mask_idx is not None:
                stats = stats.clone()
                if method == 'zero':
                    stats[:, :, mask_idx] = 0.0
                elif method == 'permutation':
                    # Shuffle along batch dimension
                    B = stats.shape[0]
                    perm = torch.randperm(B, device=device)
                    stats[:, :, mask_idx] = stats[perm, :, mask_idx]
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                y_pred = model(x, stats) if stats is not None else model(x)
            
            # Calculate forecasting MSE
            y_cpu = y.detach().cpu().numpy()
            y_pred_cpu = y_pred.detach().cpu().numpy()
            batch_mse = np.mean((y_cpu - y_pred_cpu) ** 2)
            mse_list.append(batch_mse)
            
            # Calculate anomaly scores (MSE metric)
            scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
            bearing_scores.extend(scores.tolist())
            
            if len(batch) > 3:
                bearing_labels.extend(batch[3].cpu().numpy().tolist())
            else:
                bearing_labels.extend([0] * x.size(0))
                
    bearing_scores = np.array(bearing_scores)
    bearing_labels = np.array(bearing_labels, dtype=int)
    avg_mse = float(np.mean(mse_list))
    
    # Calculate POT threshold and F1 score
    n_total = len(bearing_labels)
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    train_ratio = config['data'].get('train_ratio', 0.5)
    
    skip_end = int(n_total * skip_ratio)
    train_end = int(n_total * (skip_ratio + train_ratio))
    
    normal_indices = np.where(bearing_labels == 0)[0]
    if len(normal_indices) > 0:
        train_end = min(train_end, normal_indices[-1] + 1)
        
    if train_end > skip_end:
        healthy_subset = bearing_scores[skip_end:train_end]
        healthy_labels = bearing_labels[skip_end:train_end]
        healthy_scores = healthy_subset[healthy_labels == 0]
        if len(healthy_scores) == 0:
            healthy_scores = bearing_scores[:max(1, int(n_total * 0.1))]
    else:
        healthy_scores = bearing_scores[:max(1, int(n_total * 0.1))]
        
    try:
        pot_th = calculate_threshold_pot(healthy_scores, q=1e-3)
        metrics = calculate_metrics(bearing_scores, bearing_labels, pot_th)
        f1_score = metrics.get('F1', 0.0)
    except Exception:
        f1_score = 0.0
        
    return avg_mse, f1_score

def main():
    parser = argparse.ArgumentParser(description="Zero-out and Permutation Feature Sensitivity Analysis")
    parser.add_argument("--config", type=str, default="configs/nano.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="results/models/mamba_hybrid_withstats_nano_with_stats_best.pth", help="Path to best model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save report and charts")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 70)
    print("      LAUNCHING INFERENCE-TIME FEATURE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train_dirs = config['data'].get('train_datasets', [config['data']['processed_dir']])
    test_dirs = config['data'].get('test_datasets', [config['data']['processed_dir']])
    
    window_stride = config['data'].get('window_stride', 1024)
    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    sampling_rate = config['data'].get('sampling_rate', 128000)
    highpass_freq = 0
    label_strategy = config['data'].get('label_strategy', 'rms')
    train_ratio = config['data'].get('train_ratio', 0.5)
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    
    patch_size = config['model'].get('patch_size', 64)
    patch_stride = config['model'].get('patch_stride', 32)
    trend_downsample = config['model'].get('trend_downsample', 1)
    
    # Setup datasets (Need train dataset to extract operating conditions stats)
    print("\n--- Loading Datasets ---")
    train_dataset = MultiBearingDataset(
        train_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='train',
        file_sample_ratio=10, train_ratio=train_ratio, skip_ratio=skip_ratio, 
        normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate, label_strategy=label_strategy
    )
    oc_stats = train_dataset.oc_stats
    
    test_dataset = MultiBearingDataset(
        test_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='test',
        file_sample_ratio=1, oc_stats=oc_stats, train_ratio=train_ratio, skip_ratio=skip_ratio, 
        normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate, label_strategy=label_strategy
    )
    
    test_datasets = test_dataset.datasets if hasattr(test_dataset, 'datasets') else [test_dataset]
    bearing_names = [os.path.basename(ds.data_dir) if hasattr(ds, 'data_dir') else f"Bearing_{i}" for i, ds in enumerate(test_datasets)]
    print(f"Loaded {len(test_datasets)} test bearings: {bearing_names}")
    
    # Initialize and load the model
    model_config = {
        'model': {
            'mamba_version': 1,
            'mamba_d_model': config['model'].get('mamba_d_model', 64), 
            'mamba_n_layer': config['model'].get('mamba_n_layer', 4),
            'mamba_d_state': config['model'].get('mamba_d_state', 16), 
            'mamba_d_conv': config['model'].get('mamba_d_conv', 4), 
            'mamba_expand': config['model'].get('mamba_expand', 2),
            'forecast_len': horizon, 
            'patch_size': patch_size, 
            'stride': patch_stride,
            'trend_downsample': trend_downsample,
            'in_channels': 2, 'lookback': lookback,
            'decomp_alpha': config['model'].get('decomp_alpha', 0.1),
            'decomp_learnable': config['model'].get('decomp_learnable', True),
            'use_multiscale': False,
            'use_revin': False,
            'use_decomposition': True,
            'use_stats': True,
        },
        'data': {
            'patch_size': patch_size, 
            'stride': patch_stride, 
            'lookback': lookback
        }
    }
    
    model = HybridMambaCNN(model_config)
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Run Baseline Evaluations
    print("\n--- Running Baseline Evaluation (No Masking) ---")
    baselines = {}
    for name, ds in zip(bearing_names, test_datasets):
        mse, f1 = run_bearing_inference(model, ds, config, device, mask_idx=None)
        baselines[name] = {"mse": mse, "f1": f1}
        print(f"[{name}] Baseline - MSE: {mse:.6f} | F1-Score: {f1:.4f}")
        
    # Run Zero-out & Permutation loop
    print("\n--- Running Feature Sensitivity Loop ---")
    results = {
        "zero": {name: {} for name in bearing_names},
        "permutation": {name: {} for name in bearing_names}
    }
    
    for method in ["zero", "permutation"]:
        print(f"\nEvaluating sensitivity via method: {method.upper()}")
        for i, feat_name in enumerate(FEATURE_NAMES):
            print(f"  Processing feature {i}: {feat_name}...")
            for name, ds in zip(bearing_names, test_datasets):
                mse, f1 = run_bearing_inference(model, ds, config, device, mask_idx=i, method=method)
                
                # Calculate absolute values and delta percentages
                base_mse = baselines[name]["mse"]
                base_f1 = baselines[name]["f1"]
                
                delta_mse_pct = ((mse - base_mse) / base_mse) * 100
                delta_f1 = f1 - base_f1
                
                results[method][name][feat_name] = {
                    "mse": mse,
                    "f1": f1,
                    "delta_mse_pct": delta_mse_pct,
                    "delta_f1": delta_f1
                }
                
    # Compile results and generate report
    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, "sensitivity_analysis_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Inference-time Feature Sensitivity Analysis Report\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: `Mamba-Hybrid-WithStats` ({args.model_path})\n")
        f.write(f"Config: `{args.config}`\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write("This report evaluates the sensitivity and feature importance of the 8 physical statistics features fed into the forecasting head.\n")
        f.write("Tầm quan trọng được định nghĩa bằng mức độ suy giảm của mô hình khi thiếu đi đặc trưng vật lý tương ứng:\n")
        f.write("- **Zero-out**: Gán giá trị đặc trưng = 0.\n")
        f.write("- **Permutation**: Xáo trộn giá trị đặc trưng giữa các mẫu nhằm phá vỡ tương quan thông tin.\n\n")
        
        for method in ["zero", "permutation"]:
            method_title = "Zero-out Masking" if method == "zero" else "Permutation Shuffle"
            f.write(f"## 2. Sensitivity Analysis - {method_title}\n\n")
            
            # Print comparative table across all bearings
            f.write("| Feature | " + " | ".join([f"{name} ΔMSE (%)" for name in bearing_names]) + " | Mean ΔMSE (%) | " + " | ".join([f"{name} ΔF1" for name in bearing_names]) + " | Mean ΔF1 |\n")
            f.write("| :--- | " + " | ".join([":---:" for _ in range(2 * len(bearing_names) + 2)]) + " |\n")
            
            # Aggregate stats across bearings
            avg_mse_deltas = []
            
            for feat_name in FEATURE_NAMES:
                row = [feat_name]
                mse_deltas = []
                f1_deltas = []
                
                for name in bearing_names:
                    d_mse = results[method][name][feat_name]["delta_mse_pct"]
                    d_f1 = results[method][name][feat_name]["delta_f1"]
                    row.append(f"{d_mse:+.2f}%")
                    mse_deltas.append(d_mse)
                    f1_deltas.append(d_f1)
                    
                mean_d_mse = np.mean(mse_deltas)
                mean_d_f1 = np.mean(f1_deltas)
                
                row.append(f"{mean_d_mse:+.2f}%")
                for d_f1 in f1_deltas:
                    row.append(f"{d_f1:+.4f}")
                row.append(f"{mean_d_f1:+.4f}")
                
                f.write(" | ".join(row) + " |\n")
                avg_mse_deltas.append((feat_name, mean_d_mse))
                
            f.write("\n")
            
            # Xếp hạng Importance
            avg_mse_deltas.sort(key=lambda x: x[1], reverse=True)
            f.write(f"### Feature Importance Ranking ({method_title} by Mean ΔMSE):\n")
            for rank, (feat, val) in enumerate(avg_mse_deltas, 1):
                f.write(f"{rank}. **{feat}**: Mean ΔMSE = `{val:+.2f}%`\n")
            f.write("\n---\n\n")
            
    print(f"\n[Success] Sensitivity report written to {report_path}")
    
    # ── PLOTTING CHARTS ───────────────────────────────────────────────────
    figures_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Bar Chart: Overall Feature Importance (Mean delta MSE)
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.35
    index = np.arange(len(FEATURE_NAMES))
    
    mean_zero_mse = []
    mean_perm_mse = []
    for feat in FEATURE_NAMES:
        mean_zero_mse.append(np.mean([results["zero"][name][feat]["delta_mse_pct"] for name in bearing_names]))
        mean_perm_mse.append(np.mean([results["permutation"][name][feat]["delta_mse_pct"] for name in bearing_names]))
        
    plt.bar(index - bar_width/2, mean_zero_mse, bar_width, label='Zero-out', color='#1f77b4', alpha=0.85)
    plt.bar(index + bar_width/2, mean_perm_mse, bar_width, label='Permutation', color='#ff7f0e', alpha=0.85)
    
    plt.xlabel('Physical Features')
    plt.ylabel('Mean Increase in Test MSE (%)')
    plt.title('Feature Sensitivity Analysis - Mamba-Hybrid-WithStats')
    plt.xticks(index, FEATURE_NAMES, rotation=30, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    bar_chart_path = os.path.join(figures_dir, "sensitivity_importance.png")
    plt.savefig(bar_chart_path, dpi=150)
    plt.close()
    print(f"[Success] Importance Bar Chart saved to {bar_chart_path}")
    
    # 2. Heatmap: Feature sensitivity across different Bearings (Zero-out method)
    plt.figure(figsize=(10, 8))
    
    # Construct matrix
    heatmap_matrix = np.zeros((len(FEATURE_NAMES), len(bearing_names)))
    for f_idx, feat in enumerate(FEATURE_NAMES):
        for b_idx, name in enumerate(bearing_names):
            heatmap_matrix[f_idx, b_idx] = results["zero"][name][feat]["delta_mse_pct"]
            
    sns.heatmap(
        heatmap_matrix, 
        annot=True, 
        fmt="+.2f", 
        cmap="Oranges", 
        xticklabels=bearing_names, 
        yticklabels=FEATURE_NAMES,
        cbar_kws={'label': 'Test MSE Increase (%)'}
    )
    plt.title('Feature Sensitivity Heatmap (Zero-out Method)')
    plt.ylabel('Physical Features')
    plt.xlabel('Bearings')
    plt.tight_layout()
    
    heatmap_path = os.path.join(figures_dir, "sensitivity_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[Success] Sensitivity Heatmap saved to {heatmap_path}")
    print("\n" + "=" * 70)
    print("      SENSITIVITY ANALYSIS LIFECYCLE COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
