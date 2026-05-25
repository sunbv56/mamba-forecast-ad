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
from src.models.baselines.lstm import LSTMForecaster
from src.models.baselines.modern_tcn import ModernTCNForecaster
from src.models.baselines.patch_models import PatchTST, PatchLSTM
from src.evaluation.anomaly_scorer import calculate_anomaly_score
from src.evaluation.metrics import (
    calculate_threshold_3sigma, calculate_threshold_robust, 
    calculate_threshold_percentile, calculate_threshold_gmm, 
    find_best_threshold, calculate_metrics, calculate_threshold_pot
)

comparison_results = {}

def evaluate_model(name, model, test_loader, config, device):
    model.to(device)
    model.eval()

    print(f"\n>>> EVALUATION ON TEST (Per-Bearing Anomaly Detection Performance)...")
    
    # Khởi tạo lại thống kê bộ nhớ đỉnh nếu dùng GPU
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    macro_metrics = {
        t_name: {"F1": [], "FAR": [], "AUC": [], "AUPRC": []} 
        for t_name in ["3-Sigma", "Robust", "Percentile", "POT", "Self-Learn", "Optimal"]
    }
    
    macro_forecasting = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "MAPE": []
    }
    
    transfer_latencies = []
    inf_latencies = []
    scoring_latencies = []
    decision_latencies = []
    total_eval_latencies = []
    
    calib_times = {
        "3-Sigma": [],
        "Robust": [],
        "Percentile": [],
        "POT": [],
        "Self-Learn": [],
        "Optimal": []
    }
    
    # Lấy thông số từ dataset cấu hình
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    train_ratio = config['data'].get('train_ratio', 0.5)

    test_datasets = test_loader.dataset.datasets if hasattr(test_loader.dataset, 'datasets') else [test_loader.dataset]
    last_bearing_data = {}

    model.eval()
    for test_idx, ds in enumerate(test_datasets):
        bearing_name = os.path.basename(ds.data_dir) if hasattr(ds, 'data_dir') else f"Dataset_{test_idx}"
        
        # Tạo temp loader cho 1 vòng bi duy nhất
        loader = DataLoader(ds, batch_size=config['training'].get('batch_size', 128), shuffle=False)
        
        bearing_scores = []
        bearing_labels = []
        
        bearing_mae_list = []
        bearing_mse_list = []
        bearing_mape_list = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Inference {bearing_name}", leave=False)
            for batch in pbar:
                # 1. Đo lường thời gian truyền dữ liệu (CPU -> GPU)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                
                x = batch[0].to(device)
                y = batch[1].to(device)
                stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
                # Check for OC (USE_OPERATING_CONDITIONS is true by default for MambaTSOfficial)
                oc = batch[4].to(device) if len(batch) > 4 else None
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.time()
                
                # 2. Đo lường thời gian Model Inference (Forward Pass)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    if stats is not None and isinstance(model, HybridMambaCNN):
                        y_pred = model(x, stats)
                    elif oc is not None and isinstance(model, MambaTSOfficial):
                        y_pred = model(x, oc)
                    else:
                        y_pred = model(x)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t2 = time.time()
                
                # 3. Đo lường thời gian Tính điểm bất thường (Anomaly Scoring)
                scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t3 = time.time()
                
                # 4. Đo lường thời gian Ra quyết định (Threshold Decision)
                dummy_threshold = 0.5
                is_anomaly = scores > dummy_threshold
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t4 = time.time()
                
                bearing_scores.extend(scores.tolist())

                if len(batch) > 3:
                    labels_batch = batch[3]
                    if hasattr(labels_batch, 'numpy'):
                        bearing_labels.extend(labels_batch.numpy().tolist())
                    else:
                        bearing_labels.extend([int(l) for l in labels_batch])
                else:
                    bearing_labels.extend([0] * x.size(0))

                # Tính toán tốc độ / độ trễ trên mỗi sample (ms/sample)
                n_samples = x.size(0)
                transfer_ms = (t1 - t0) * 1000 / n_samples
                inf_ms = (t2 - t1) * 1000 / n_samples
                score_ms = (t3 - t2) * 1000 / n_samples
                decision_ms = (t4 - t3) * 1000 / n_samples
                total_ms = (t4 - t0) * 1000 / n_samples
                
                transfer_latencies.append(transfer_ms)
                inf_latencies.append(inf_ms)
                scoring_latencies.append(score_ms)
                decision_latencies.append(decision_ms)
                total_eval_latencies.append(total_ms)

                # Hiển thị VRAM hoạt động thực tế trên tqdm của từng vòng bi
                if device.type == 'cuda':
                    vram_active = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    pbar.set_postfix({'vram_mb': f'{vram_active:.0f}'})

                # Compute Forecasting Metrics for the batch
                y_cpu = y.detach().cpu().numpy()
                y_pred_cpu = y_pred.detach().cpu().numpy()
                
                batch_mae = np.mean(np.abs(y_cpu - y_pred_cpu))
                batch_mse = np.mean((y_cpu - y_pred_cpu) ** 2)
                
                epsilon = 1e-5
                batch_mape = np.mean(np.abs(y_cpu - y_pred_cpu) / (np.abs(y_cpu) + epsilon)) * 100
                
                bearing_mae_list.append(batch_mae)
                bearing_mse_list.append(batch_mse)
                bearing_mape_list.append(batch_mape)

        bearing_scores = np.array(bearing_scores)
        bearing_labels = np.array(bearing_labels, dtype=int)
        
        n_fault = int(bearing_labels.sum())
        n_total = len(bearing_labels)
        print(f"\n  [{bearing_name}] Label distribution: {n_fault}/{n_total} anomaly windows ({n_fault/n_total:.1%})")

        # Compute Forecasting Metrics for this bearing
        bearing_mae = np.mean(bearing_mae_list)
        bearing_mse = np.mean(bearing_mse_list)
        bearing_rmse = np.sqrt(bearing_mse)
        bearing_mape = np.mean(bearing_mape_list)
        
        macro_forecasting["MAE"].append(bearing_mae)
        macro_forecasting["MSE"].append(bearing_mse)
        macro_forecasting["RMSE"].append(bearing_rmse)
        macro_forecasting["MAPE"].append(bearing_mape)
        
        print(f"     - [Forecasting] MAE: {bearing_mae:.6f} | MSE: {bearing_mse:.6f} | RMSE: {bearing_rmse:.6f} | MAPE: {bearing_mape:.4f}%")

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
            
        # Đo lường thời gian hiệu chuẩn từng thuật toán ngưỡng
        t_cal = time.time()
        local_th_3s  = calculate_threshold_3sigma(healthy_scores)
        calib_times["3-Sigma"].append((time.time() - t_cal) * 1000)
        
        t_cal = time.time()
        local_th_rb  = calculate_threshold_robust(healthy_scores)
        calib_times["Robust"].append((time.time() - t_cal) * 1000)
        
        t_cal = time.time()
        local_th_pc  = calculate_threshold_percentile(healthy_scores, q=99.7)
        calib_times["Percentile"].append((time.time() - t_cal) * 1000)
        
        t_cal = time.time()
        local_th_pot = calculate_threshold_pot(healthy_scores, q=1e-3)
        calib_times["POT"].append((time.time() - t_cal) * 1000)
        
        t_cal = time.time()
        local_th_gmm = calculate_threshold_gmm(bearing_scores)
        calib_times["Self-Learn"].append((time.time() - t_cal) * 1000)
        
        t_cal = time.time()
        local_th_opt, _ = find_best_threshold(bearing_scores, bearing_labels)
        calib_times["Optimal"].append((time.time() - t_cal) * 1000)

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

        # Lưu trữ dữ liệu của vòng bi cuối cùng để trực quan hóa
        last_bearing_data = {
            'bearing_name': bearing_name,
            'scores': bearing_scores,
            'labels': bearing_labels,
            'th_3sigma': local_th_3s,
            'th_robust': local_th_rb,
            'th_pot': local_th_pot
        }

        # Tích hợp lưu trữ kết quả để so sánh 2x2
        global comparison_results
        if bearing_name not in comparison_results:
            comparison_results[bearing_name] = {}
        
        file_scores = {}
        for i_score, score in enumerate(bearing_scores):
            f_idx = ds.samples[i_score][0]
            if f_idx not in file_scores:
                file_scores[f_idx] = []
            file_scores[f_idx].append(score)
            
        indices = sorted(file_scores.keys())
        mse_vals = [np.mean(file_scores[f_idx]) for f_idx in indices]
        rms_vals = [ds.file_rms[ds.files[f_idx]] for f_idx in indices]
        
        cache_key = f"_kurtosis_cache_{bearing_name}"
        if cache_key not in globals():
            globals()[cache_key] = {}
        kurt_cache = globals()[cache_key]
        
        kurt_vals = []
        for f_idx in indices:
            if f_idx not in kurt_cache:
                f_path = os.path.join(ds.data_dir, ds.files[f_idx])
                signal_data = torch.load(f_path, map_location='cpu', weights_only=True)
                mean_sig = signal_data.mean(dim=-1, keepdim=True)
                std_sig = signal_data.std(dim=-1, keepdim=True)
                z_sig = (signal_data - mean_sig) / (std_sig + 1e-8)
                kurt_cache[f_idx] = torch.mean(z_sig**4, dim=-1).mean().item()
            kurt_vals.append(kurt_cache[f_idx])
            
        comparison_results[bearing_name][name] = {
            'mse': mse_vals,
            'indices': indices,
            'rms': rms_vals,
            'kurtosis': kurt_vals
        }

    # Tính toán trung bình các chỉ số thời gian
    avg_transfer_lat = float(np.mean(transfer_latencies))
    avg_inf_lat = float(np.mean(inf_latencies))
    avg_scoring_lat = float(np.mean(scoring_latencies))
    avg_decision_lat = float(np.mean(decision_latencies))
    avg_total_eval_lat = float(np.mean(total_eval_latencies))
    
    avg_calib_times = {k: float(np.mean(v)) for k, v in calib_times.items()}
    
    avg_mae = np.mean(macro_forecasting["MAE"])
    avg_mse = np.mean(macro_forecasting["MSE"])
    avg_rmse = np.mean(macro_forecasting["RMSE"])
    avg_mape = np.mean(macro_forecasting["MAPE"])

    # Tính toán bộ nhớ GPU đỉnh sử dụng suốt phiên đánh giá
    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == 'cuda' else 0.0

    print("\n" + "=" * 60)
    print(f">>> MACRO-AVERAGE PERFORMANCE ({len(test_datasets)} Bearings)")
    print(f"   [Forecasting Metrics] > MAE: {avg_mae:.6f} | MSE: {avg_mse:.6f} | RMSE: {avg_rmse:.6f} | MAPE: {avg_mape:.4f}%")
    if device.type == 'cuda':
        print(f"   [Hardware Metrics]    > Peak GPU VRAM: {peak_vram_mb:.1f} MB")
    print("-" * 60)
    print("   [Real-time Evaluation Latency Breakdown (per sample)]:")
    print(f"     * Data Transfer (CPU->GPU): {avg_transfer_lat:.4f} ms")
    print(f"     * Model Inference (Forward): {avg_inf_lat:.4f} ms")
    print(f"     * Anomaly Scoring:           {avg_scoring_lat:.4f} ms")
    print(f"     * Decision (Threshold Cmp):  {avg_decision_lat:.4f} ms")
    print(f"     * TOTAL REAL LATENCY:        {avg_total_eval_lat:.4f} ms/sample")
    print("-" * 60)
    print("   [Threshold Calibration Overhead (per bearing)]:")
    for k, v in avg_calib_times.items():
        print(f"     * {k:<12}: {v:.4f} ms")
    print("=" * 60)

    return {
        'name': name,
        'f1_3s': float(np.mean(macro_metrics["3-Sigma"]["F1"])),
        'f1_rb': float(np.mean(macro_metrics["Robust"]["F1"])),
        'f1_pot': float(np.mean(macro_metrics["POT"]["F1"])),
        'auc': float(np.mean(macro_metrics["POT"]["AUC"])),
        'far_pot': float(np.mean(macro_metrics["POT"]["FAR"])),
        'auprc': float(np.mean(macro_metrics["POT"]["AUPRC"])),
        'latency': float(avg_total_eval_lat),
        'mae': float(avg_mae),
        'mse': float(avg_mse),
        'rmse': float(avg_rmse),
        'mape': float(avg_mape),
        'last_bearing': last_bearing_data,
        'peak_vram_mb': peak_vram_mb,
        'latency_breakdown': {
            'data_transfer': avg_transfer_lat,
            'inference': avg_inf_lat,
            'scoring': avg_scoring_lat,
            'decision': avg_decision_lat,
            'total': avg_total_eval_lat
        },
        'calib_latencies': avg_calib_times
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--model_path", type=str, help="Path to .pth checkpoint (optional in multi-model mode).")
    parser.add_argument("--model_type", type=str, default="Mamba1-Hybrid", help="Model type (e.g., MambaTS-Official, Mamba1-Hybrid, LSTM, PatchLSTM, ModernTCN, PatchTST).")
    parser.add_argument("--batch_size", type=int, help="Override batch_size from config")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to evaluate (e.g., 'LSTM,PatchLSTM,ModernTCN,PatchTST,Mamba1-Hybrid' or 'all').")
    parser.add_argument("--models_dir", type=str, default="results/models", help="Directory where trained model checkpoints are saved.")
    
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
    
    # Extract config name from filename (e.g., 'configs/small.yaml' -> 'small')
    config_name = os.path.basename(args.config).replace('.yaml', '')

    def initialize_model(model_type):
        if model_type == "MambaTS-Official":
            return MambaTSOfficial(MambaTSConfig(
                in_channels=2,
                lookback=lookback,
                forecast_len=horizon,
                patch_size=patch_size,
                stride=patch_stride,
                d_model=config['model'].get('mamba_d_model', 64),
                n_layers=config['model'].get('mamba_n_layer', 4),
                dropout=0.2,
                VPT_mode=1,
                ATSP_solver='SA',
                oc_dim=2
            ))
        elif model_type == "Mamba1-Hybrid":
            return HybridMambaCNN({
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
                    'use_multiscale': config['model'].get('use_multiscale', True),
                    'use_revin': config['model'].get('use_revin', True),
                    'use_decomposition': config['model'].get('use_decomposition', True),
                    'use_stats': config['model'].get('use_stats', True),
                },
                'data': {
                    'patch_size': patch_size, 
                    'stride': patch_stride, 
                    'lookback': lookback
                }
            })
        elif model_type == "LSTM":
            print(f"Initializing LSTM with hardcoded parameters: hidden_dim=122, num_layers=3, horizon={horizon}")
            return LSTMForecaster(input_dim=2, hidden_dim=122, num_layers=3, horizon=horizon)
        elif model_type == "PatchLSTM":
            print(f"Initializing PatchLSTM with hardcoded parameters: patch_size=64, stride=64, d_model=112, num_layers=3, horizon={horizon}")
            return PatchLSTM(in_channels=2, patch_size=64, stride=64, d_model=112, num_layers=3, horizon=horizon)
        elif model_type == "ModernTCN":
            print(f"Initializing ModernTCN with hardcoded parameters: d_model=144, num_layers=3, kernel_size=17, horizon={horizon}")
            return ModernTCNForecaster(input_dim=2, d_model=144, num_layers=3, kernel_size=17, horizon=horizon)
        elif model_type == "PatchTST":
            print(f"Initializing PatchTST with standard paper parameters: d_model=128, nhead=16, num_layers=3, horizon={horizon}, lookback={lookback}, patch_size=16, stride=8")
            return PatchTST(in_channels=2, lookback=lookback, patch_size=16, stride=8, d_model=128, nhead=16, num_layers=3, horizon=horizon)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Determine models to evaluate
    results = {}
    if args.models:
        if args.models == "all":
            models_to_eval = ["LSTM", "PatchLSTM", "ModernTCN", "PatchTST", "Mamba1-Hybrid"]
        else:
            models_to_eval = [m.strip() for m in args.models.split(',')]
            
        print(f"\n>>> Running multi-model evaluation for: {models_to_eval}")
        for model_type in models_to_eval:
            model_slug = model_type.lower().replace('-', '_')
            checkpoint_path = os.path.join(args.models_dir, f"{model_slug}_{config_name}_best.pth")
            
            if not os.path.exists(checkpoint_path):
                print(f"\n⚠️ WARNING: Checkpoint not found for {model_type} at {checkpoint_path}. Skipping...")
                continue
                
            print(f"\n>>> Initializing & evaluating {model_type} from {checkpoint_path}...")
            try:
                model = initialize_model(model_type)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
                res = evaluate_model(model_type, model, test_loader, config, device)
                results[model_type] = res
            except Exception as e:
                print(f"❌ Error evaluating {model_type}: {e}")
                
        # Print final comparison summary tables
        if results:
            print("\n" + "="*85)
            print(">>> FINAL MULTI-MODEL COMPARISON: FORECASTING PERFORMANCE")
            print("="*85)
            print(f"{'Model':<20} | {'MAE':<10} | {'MSE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'Latency (ms)':<12}")
            print("-" * 85)
            for name, res in results.items():
                print(f"{name:<20} | {res['mae']:<10.6f} | {res['mse']:<10.6f} | {res['rmse']:<10.6f} | {res['mape']:<10.4f}% | {res['latency']:<12.4f}")
            print("="*85)
            
            print("\n" + "="*87)
            print(">>> FINAL MULTI-MODEL COMPARISON: ANOMALY DETECTION PERFORMANCE")
            print("="*87)
            print(f"{'Model':<20} | {'F1 (3s)':<8} | {'F1 (rb)':<8} | {'F1 (POT)':<8} | {'AUC (POT)':<9} | {'FAR (POT)':<9} | {'Latency (ms)':<12}")
            print("-" * 87)
            for name, res in results.items():
                print(f"{name:<20} | {res['f1_3s']:<8.4f} | {res['f1_rb']:<8.4f} | {res.get('f1_pot', 0):<8.4f} | {res['auc']:<9.4f} | {res['far_pot']:<9.4f} | {res['latency']:<12.4f}")
            print("="*87)
        else:
            print("\n❌ No models were successfully evaluated.")
            
    else:
        # Single model evaluation (Backward compatible)
        if not args.model_path:
            parser.error("Either --model_path (for single model) or --models (for multiple models) must be provided.")
            
        print(f"\n>>> Running single model evaluation for {args.model_type}...")
        model = initialize_model(args.model_type)
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        evaluate_model(args.model_type, model, test_loader, config, device)

    # Save comparison results for visualization
    if comparison_results:
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"eval_comparison_results_{config_name}.pth")
        torch.save(comparison_results, save_path)
        print(f"\n[Info] Saved comparison results to {save_path}")

if __name__ == "__main__":
    main()
