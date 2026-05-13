import os
import sys
import yaml
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc as auc_score_func

# Add src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import B02Dataset
from src.models.mamba import HybridMambaCNN, MambaTS, MambaTSOfficial, MambaTSConfig
from src.evaluation.anomaly_scorer import calculate_anomaly_score
from src.evaluation.metrics import (
    calculate_threshold_3sigma, calculate_threshold_robust, 
    calculate_threshold_percentile, calculate_threshold_gmm, 
    find_best_threshold, calculate_metrics, calculate_threshold_pot
)

def evaluate_model(name, model, train_loader, test_loader, config, device):
    model.to(device)
    model.eval()

    # --- 1. Calculate Thresholds from Train Set (Healthy) ---
    print(f"\n>>> Calculating thresholds from training samples (Healthy Baseline)...")
    train_scores = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Train Scores"):
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            oc = batch[4].to(device) if len(batch) > 4 else None
            
            if stats is not None and isinstance(model, HybridMambaCNN):
                y_pred = model(x, stats)
            elif oc is not None and isinstance(model, MambaTSOfficial):
                y_pred = model(x, oc)
            else:
                y_pred = model(x)
                
            scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
            train_scores.extend(scores.tolist())

    train_scores = np.array(train_scores)
    threshold_3s  = calculate_threshold_3sigma(train_scores)
    threshold_rb  = calculate_threshold_robust(train_scores)
    threshold_pc  = calculate_threshold_percentile(train_scores, q=99.7)
    threshold_pot = calculate_threshold_pot(train_scores, q=1e-3)

    print(f"Thresholds calculated:")
    print(f"  3-Sigma:    {threshold_3s:.6f}")
    print(f"  Robust:     {threshold_rb:.6f}")
    print(f"  Percentile: {threshold_pc:.6f}")
    print(f"  POT (q=1e-3): {threshold_pot:.6f}")

    # --- 2. Evaluate on Test Set ---
    print(f"\n>>> Evaluating on Test Set...")
    test_scores = []
    test_labels = []
    test_latencies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            oc = batch[4].to(device) if len(batch) > 4 else None

            start_inf = time.time()
            if stats is not None and isinstance(model, HybridMambaCNN):
                y_pred = model(x, stats)
            elif oc is not None and isinstance(model, MambaTSOfficial):
                y_pred = model(x, oc)
            else:
                y_pred = model(x)
            test_latencies.append((time.time() - start_inf) / x.size(0))

            scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
            test_scores.extend(scores.tolist())

            if len(batch) > 3:
                test_labels.extend(batch[3].numpy().tolist())
            else:
                test_labels.extend([0] * x.size(0))

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels, dtype=int)
    test_latency_avg = np.mean(test_latencies) * 1000

    # Optimal Threshold (Oracle)
    threshold_opt, _ = find_best_threshold(test_scores, test_labels)
    
    # GMM Threshold
    combined_for_gmm = np.concatenate([train_scores, test_scores])
    threshold_gmm = calculate_threshold_gmm(combined_for_gmm)

    print(f"\n>>> TEST RESULTS ({name})")
    print(f"{'Method':<12} | {'F1':<8} | {'FAR':<8} | {'Thresh':<10}")
    print("-" * 45)
    
    for t_name, t_val in [
        ("3-Sigma", threshold_3s), 
        ("Robust", threshold_rb), 
        ("Percentile", threshold_pc),
        ("POT", threshold_pot), 
        ("Self-Learn", threshold_gmm), 
        ("Optimal", threshold_opt)
    ]:
        m = calculate_metrics(test_scores, test_labels, t_val)
        print(f"{t_name:<12} | {m.get('F1', 0):.4f} | {m.get('FAR', 0):.4f} | {t_val:.6f}")

    if len(np.unique(test_labels)) > 1:
        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        auprc = auc_score_func(recall, precision)
        print(f"\nAUPRC: {auprc:.4f}")
    
    print(f"Avg Latency: {test_latency_avg:.4f} ms/sample")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--model_type", type=str, default="MambaTS-Official", help="Model type (e.g., MambaTS-Official).")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--subset_ratio", type=int, default=10, help="Ratio for training subset to compute thresholds.")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Setup
    processed_dir = config['data']['processed_dir']
    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    stride = config['data'].get('stride', 1024)
    patch_size = config['data'].get('patch_size', 2048)
    
    print(f"Loading datasets...")
    train_dataset = B02Dataset(processed_dir, lookback, horizon, stride, split='train', 
                               file_sample_ratio=args.subset_ratio, normalize=False)
    oc_stats = getattr(train_dataset, 'oc_stats', None)
    test_dataset = B02Dataset(processed_dir, lookback, horizon, stride, split='test', 
                              file_sample_ratio=1, oc_stats=oc_stats, normalize=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model Setup
    if args.model_type == "MambaTS-Official":
        model = MambaTSOfficial(MambaTSConfig(
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
        ))
    elif args.model_type == "Mamba1-Hybrid":
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
            'data': {
                'patch_size': patch_size, 
                'stride': stride, 
                'lookback': lookback
            }
        })
    else:
        # Extend as needed for other model types
        print(f"Unsupported model type: {args.model_type}")
        return

    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    
    evaluate_model(args.model_type, model, train_loader, test_loader, config, device)

if __name__ == "__main__":
    main()
