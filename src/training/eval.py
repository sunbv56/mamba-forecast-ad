import os
import sys
import yaml
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc as auc_score_func

# Add src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import MultiBearingDataset
from src.models.mamba import HybridMambaCNN, MambaTSOfficial, MambaTSConfig
from src.evaluation.anomaly_scorer import calculate_anomaly_score
from src.evaluation.metrics import (
    calculate_threshold_3sigma, calculate_threshold_robust, 
    calculate_threshold_percentile, calculate_threshold_gmm, 
    find_best_threshold, calculate_metrics, calculate_threshold_pot
)

def evaluate_model(name, model, test_loader, config, device):
    model.to(device)
    model.eval()

    print(f"\n>>> EVALUATION ON TEST (Per-Bearing Anomaly Detection Performance)...")
    
    macro_metrics = {
        t_name: {"F1": [], "FAR": [], "AUC": [], "AUPRC": []} 
        for t_name in ["3-Sigma", "Robust", "Percentile", "POT", "Self-Learn", "Optimal"]
    }
    
    total_test_latencies = []
    
    # Lấy thông số từ dataset cấu hình
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    train_ratio = config['data'].get('train_ratio', 0.5)

    test_datasets = test_loader.dataset.datasets if hasattr(test_loader.dataset, 'datasets') else [test_loader.dataset]

    model.eval()
    for test_idx, ds in enumerate(test_datasets):
        bearing_name = os.path.basename(ds.data_dir) if hasattr(ds, 'data_dir') else f"Dataset_{test_idx}"
        
        # Tạo temp loader cho 1 vòng bi duy nhất
        loader = DataLoader(ds, batch_size=config['training'].get('batch_size', 128), shuffle=False)
        
        bearing_scores = []
        bearing_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Inference {bearing_name}", leave=False):
                x, y = batch[0].to(device), batch[1].to(device)
                stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
                # Check for OC (USE_OPERATING_CONDITIONS is true by default for MambaTSOfficial)
                oc = batch[4].to(device) if len(batch) > 4 else None

                start_inf = time.time()
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    if stats is not None and isinstance(model, HybridMambaCNN):
                        y_pred = model(x, stats)
                    elif oc is not None and isinstance(model, MambaTSOfficial):
                        y_pred = model(x, oc)
                    else:
                        y_pred = model(x)
                
                total_test_latencies.append((time.time() - start_inf) / x.size(0))
                
                scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
                bearing_scores.extend(scores.tolist())

                if len(batch) > 3:
                    labels_batch = batch[3]
                    if hasattr(labels_batch, 'numpy'):
                        bearing_labels.extend(labels_batch.numpy().tolist())
                    else:
                        bearing_labels.extend([int(l) for l in labels_batch])
                else:
                    bearing_labels.extend([0] * x.size(0))

        bearing_scores = np.array(bearing_scores)
        bearing_labels = np.array(bearing_labels, dtype=int)
        
        n_fault = int(bearing_labels.sum())
        n_total = len(bearing_labels)
        print(f"\n  [{bearing_name}] Label distribution: {n_fault}/{n_total} anomaly windows ({n_fault/n_total:.1%})")

        # Tính LOCAL THRESHOLD (Self-learning)
        skip_end = int(n_total * skip_ratio)
        train_end = int(n_total * (skip_ratio + train_ratio))
        
        # Ngăn chặn rò rỉ dữ liệu lỗi vào tập tính ngưỡng POT (Leakage fix)
        normal_indices = np.where(bearing_labels == 0)[0]
        if len(normal_indices) > 0:
            train_end = min(train_end, normal_indices[-1] + 1)
            
        if train_end > skip_end:
            # Lọc chắc chắn chỉ lấy nhãn 0
            healthy_subset = bearing_scores[skip_end:train_end]
            healthy_labels = bearing_labels[skip_end:train_end]
            healthy_scores = healthy_subset[healthy_labels == 0]
            
            if len(healthy_scores) == 0:
                healthy_scores = bearing_scores[:max(1, int(n_total * 0.1))]
        else:
            healthy_scores = bearing_scores[:max(1, int(n_total * 0.1))] # Fallback 10%
            
        local_th_3s  = calculate_threshold_3sigma(healthy_scores)
        local_th_rb  = calculate_threshold_robust(healthy_scores)
        local_th_pc  = calculate_threshold_percentile(healthy_scores, q=99.7)
        local_th_pot = calculate_threshold_pot(healthy_scores, q=1e-3)
        local_th_gmm = calculate_threshold_gmm(bearing_scores)
        local_th_opt, _ = find_best_threshold(bearing_scores, bearing_labels)

        # AUPRC cho riêng vòng bi
        if len(np.unique(bearing_labels)) > 1:
            precision_path, recall_path, _ = precision_recall_curve(bearing_labels, bearing_scores)
            bearing_auprc = auc_score_func(recall_path, precision_path)
        else:
            bearing_auprc = 0.0

        thresholds = [
            ("3-Sigma", local_th_3s), ("Robust", local_th_rb), ("Percentile", local_th_pc),
            ("POT", local_th_pot), ("Self-Learn", local_th_gmm), ("Optimal", local_th_opt)
        ]
        
        for t_name, t_val in thresholds:
            m = calculate_metrics(bearing_scores, bearing_labels, t_val)
            macro_metrics[t_name]["F1"].append(m.get('F1', 0))
            macro_metrics[t_name]["FAR"].append(m.get('FAR', 0))
            macro_metrics[t_name]["AUC"].append(m.get('AUC', 0))
            macro_metrics[t_name]["AUPRC"].append(bearing_auprc)
            
            # In log tóm tắt cho từng vòng bi
            if t_name in ["Robust", "POT", "Optimal"]:
                print(f"     - {t_name:<7}: F1={m.get('F1',0):.4f} | FAR={m.get('FAR',0):.4f} | Thresh={t_val:.4f}")

    test_latency_avg = np.mean(total_test_latencies) * 1000

    print(f"\n============================================================")
    print(f">>> MACRO-AVERAGE PERFORMANCE ({len(test_datasets)} Bearings)")
    for t_name in macro_metrics.keys():
        avg_f1 = np.mean(macro_metrics[t_name]["F1"])
        avg_far = np.mean(macro_metrics[t_name]["FAR"])
        avg_auc = np.mean(macro_metrics[t_name]["AUC"])
        avg_auprc = np.mean(macro_metrics[t_name]["AUPRC"])
        print(f"   [{t_name:<10}] > F1: {avg_f1:.4f} | FAR: {avg_far:.4f} | AUC: {avg_auc:.4f} | AUPRC: {avg_auprc:.4f}")
    print(f"   > Avg Latency: {test_latency_avg:.4f} ms/sample")
    print(f"============================================================")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--model_type", type=str, default="Mamba1-Hybrid", help="Model type (e.g., MambaTS-Official, Mamba1-Hybrid).")
    parser.add_argument("--batch_size", type=int, help="Override batch_size from config")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # --- Dataset Setup ---
    train_dirs = config['data'].get('train_datasets')
    test_dirs = config['data'].get('test_datasets')
    
    if train_dirs is None:
        train_dirs = [config['data']['processed_dir']]
    if test_dirs is None:
        test_dirs = [config['data']['processed_dir']]

    window_stride = config['data'].get('window_stride', 1024)
    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    sampling_rate = config['data'].get('sampling_rate', 128000)
    
    # Model patching params
    patch_size = config['model'].get('patch_size', 64)
    patch_stride = config['model'].get('patch_stride', 32)
    trend_downsample = config['model'].get('trend_downsample', 1)
    
    train_ratio = config['data'].get('train_ratio', 0.5)
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    highpass_freq = config['data'].get('highpass_freq', 1000)
    label_strategy = config['data'].get('label_strategy', 'rms')
    
    # We only need test_dataset for evaluation, but sometimes oc_stats comes from train
    print(f"Loading datasets...")
    train_dataset = MultiBearingDataset(train_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='train',
                                         file_sample_ratio=1, train_ratio=train_ratio, skip_ratio=skip_ratio, 
                                         normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate, label_strategy=label_strategy)
    oc_stats = train_dataset.oc_stats
    
    test_dataset  = MultiBearingDataset(test_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='test',
                                         file_sample_ratio=1, oc_stats=oc_stats, train_ratio=train_ratio, skip_ratio=skip_ratio, 
                                         normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate, label_strategy=label_strategy)
        
    batch_size = int(config['training'].get('batch_size', 128))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Model Initialization ---
    if args.model_type == "MambaTS-Official":
        model = MambaTSOfficial(MambaTSConfig(
            in_channels=2,
            lookback=lookback,
            forecast_len=horizon,
            patch_size=patch_size,
            stride=patch_stride,
            d_model=config['model'].get('mamba_d_model', 64),
            n_layers=config['model'].get('mamba_n_layer', 4),
            dropout=0.2, # Paper suggested 0.2-0.3
            VPT_mode=1, # Enable Variable-Aware Scanning
            ATSP_solver='SA',
            oc_dim=2
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
                'stride': patch_stride,
                'trend_downsample': trend_downsample,
                'in_channels': 2, 'lookback': lookback,
                'decomp_kernel': config['model'].get('decomp_kernel', 25), 
                'use_multiscale': True,
            },
            'data': {
                'patch_size': patch_size, 
                'stride': patch_stride, 
                'lookback': lookback
            }
        })
    else:
        print(f"Unsupported model type: {args.model_type}")
        return

    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    
    evaluate_model(args.model_type, model, test_loader, config, device)

if __name__ == "__main__":
    main()
