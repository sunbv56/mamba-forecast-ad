import os
import sys
import torch
import yaml

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("--- Đang khởi tạo các mô hình để kiểm tra tham số ---")
from src.models.mamba import HybridMambaCNN
from src.models.baselines.lstm import LSTMForecaster
from src.models.baselines.modern_tcn import ModernTCNForecaster
from src.models.baselines.patch_models import PatchTST, PatchLSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_mamba_core_parameters(model):
    # Total params minus trend_head params
    total = count_parameters(model)
    trend_params = sum(p.numel() for p in model.trend_head.parameters() if p.requires_grad)
    return total - trend_params

def find_closest_lstm(target, horizon):
    best_dim = 8
    best_params = 0
    min_diff = float('inf')
    for h in range(8, 1024, 2):
        model = LSTMForecaster(input_dim=2, hidden_dim=h, num_layers=3, horizon=horizon)
        p = count_parameters(model)
        diff = abs(p - target)
        if diff < min_diff:
            min_diff = diff
            best_dim = h
            best_params = p
        if p > target and diff > min_diff:
            break
    return best_dim, best_params

def find_closest_patch_lstm(target, patch_size, stride, horizon):
    best_dim = 8
    best_params = 0
    min_diff = float('inf')
    for d in range(8, 1024, 2):
        model = PatchLSTM(in_channels=2, patch_size=patch_size, stride=stride, d_model=d, num_layers=3, horizon=horizon)
        p = count_parameters(model)
        diff = abs(p - target)
        if diff < min_diff:
            min_diff = diff
            best_dim = d
            best_params = p
        if p > target and diff > min_diff:
            break
    return best_dim, best_params

def find_closest_modern_tcn(target, horizon):
    best_dim = 8
    best_params = 0
    min_diff = float('inf')
    for d in range(8, 1024, 2):
        model = ModernTCNForecaster(input_dim=2, d_model=d, num_layers=3, kernel_size=17, horizon=horizon)
        p = count_parameters(model)
        diff = abs(p - target)
        if diff < min_diff:
            min_diff = diff
            best_dim = d
            best_params = p
        if p > target and diff > min_diff:
            break
    return best_dim, best_params

def find_closest_patchtst(target, lookback, patch_size, stride, horizon):
    best_dim = 8
    best_params = 0
    min_diff = float('inf')
    for d in range(8, 1024, 4):
        nhead = 4 if d >= 4 else 1
        if d % 4 != 0:
            nhead = 2 if d % 2 == 0 else 1
        model = PatchTST(in_channels=2, lookback=lookback, patch_size=patch_size, stride=stride, d_model=d, nhead=nhead, num_layers=4, horizon=horizon)
        p = count_parameters(model)
        diff = abs(p - target)
        if diff < min_diff:
            min_diff = diff
            best_dim = d
            best_params = p
        if p > target and diff > min_diff:
            break
    return best_dim, best_params

def main():
    config_files = ["snano.yaml", "nano.yaml"]
    
    results_actual = []
    results_fair = []
    results_budget = []

    for config_file in config_files:
        config_path = os.path.join(project_root, "configs", config_file)
        if not os.path.exists(config_path):
            print(f"Bỏ qua {config_file}: File không tồn tại.")
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        lookback = config['data'].get('lookback', 4096)
        horizon = config['data'].get('horizon', 1024)
        
        # Model patching & trend params
        patch_size = config['model'].get('patch_size', 64)
        patch_stride = config['model'].get('patch_stride', 32)
        trend_downsample = config['model'].get('trend_downsample', 1)
        
        # Mamba dimension configs
        mamba_d_model = config['model'].get('mamba_d_model', 64)
        mamba_n_layer = config['model'].get('mamba_n_layer', 4)

        # 1. Mamba1-Hybrid (Target model created first)
        hybrid_model = HybridMambaCNN({
            'model': {
                'mamba_version': 1,
                'mamba_d_model': mamba_d_model,
                'mamba_n_layer': mamba_n_layer,
                'mamba_d_state': config['model'].get('mamba_d_state', 16),
                'mamba_d_conv': config['model'].get('mamba_d_conv', 4),
                'mamba_expand': config['model'].get('mamba_expand', 2),
                'forecast_len': horizon,
                'patch_size': patch_size,
                'stride': patch_stride,
                'trend_downsample': trend_downsample,
                'in_channels': 2,
                'lookback': lookback,
                'decomp_kernel': config['model'].get('decomp_kernel', 25),
                'use_multiscale': True,
            },
            'data': {'patch_size': patch_size, 'stride': patch_stride, 'lookback': lookback}
        })
        mamba_total_params = count_parameters(hybrid_model)
        mamba_core_params = count_mamba_core_parameters(hybrid_model)

        # ------------------------------------------------------------------
        # Setup 1: Actual Sizing (Hardcoded in train.py)
        # ------------------------------------------------------------------
        results_actual.append({
            "Config": config_file, "Model": "Mamba1-Hybrid (Total)",
            "Params": mamba_total_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={mamba_d_model}"
        })
        results_actual.append({
            "Config": config_file, "Model": "Mamba1-Hybrid (Core)",
            "Params": mamba_core_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={mamba_d_model} (No Trend Head)"
        })
        results_actual.append({
            "Config": config_file, "Model": "LSTM",
            "Params": count_parameters(LSTMForecaster(input_dim=2, hidden_dim=140, num_layers=3, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": "hidden_dim=140"
        })
        results_actual.append({
            "Config": config_file, "Model": "PatchLSTM",
            "Params": count_parameters(PatchLSTM(in_channels=2, patch_size=64, stride=64, d_model=120, num_layers=3, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": "d_model=120"
        })
        results_actual.append({
            "Config": config_file, "Model": "ModernTCN",
            "Params": count_parameters(ModernTCNForecaster(input_dim=2, d_model=160, num_layers=3, kernel_size=17, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": "d_model=160"
        })
        results_actual.append({
            "Config": config_file, "Model": "PatchTST",
            "Params": count_parameters(PatchTST(in_channels=2, lookback=lookback, patch_size=16, stride=8, d_model=128, nhead=16, num_layers=3, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": "d_model=128 (Paper)"
        })

        # ------------------------------------------------------------------
        # Setup 2: Fair Scaling (Cùng chiều ẩn d_model / hidden_dim) - TRULY FAIR
        # ------------------------------------------------------------------
        results_fair.append({
            "Config": config_file, "Model": "Mamba1-Hybrid (Total)",
            "Params": mamba_total_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={mamba_d_model}"
        })
        results_fair.append({
            "Config": config_file, "Model": "Mamba1-Hybrid (Core)",
            "Params": mamba_core_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={mamba_d_model} (No Trend Head)"
        })
        results_fair.append({
            "Config": config_file, "Model": "LSTM",
            "Params": count_parameters(LSTMForecaster(input_dim=2, hidden_dim=mamba_d_model, num_layers=mamba_n_layer, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": f"hidden_dim={mamba_d_model}"
        })
        results_fair.append({
            "Config": config_file, "Model": "PatchLSTM",
            "Params": count_parameters(PatchLSTM(in_channels=2, patch_size=patch_size, stride=patch_stride, d_model=mamba_d_model, num_layers=mamba_n_layer, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": f"d_model={mamba_d_model}"
        })
        results_fair.append({
            "Config": config_file, "Model": "ModernTCN",
            "Params": count_parameters(ModernTCNForecaster(input_dim=2, d_model=mamba_d_model, num_layers=mamba_n_layer, kernel_size=17, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": f"d_model={mamba_d_model}"
        })
        nhead = 4 if mamba_d_model >= 4 else 1
        if mamba_d_model % 4 != 0:
            nhead = 2 if mamba_d_model % 2 == 0 else 1
        results_fair.append({
            "Config": config_file, "Model": "PatchTST",
            "Params": count_parameters(PatchTST(in_channels=2, lookback=lookback, patch_size=patch_size, stride=patch_stride, d_model=mamba_d_model, nhead=nhead, num_layers=mamba_n_layer, horizon=horizon)),
            "Lookback": lookback, "Horizon": horizon, "Details": f"d_model={mamba_d_model}"
        })

        # ------------------------------------------------------------------
        # Setup 3: Parameter Budget Sizing (Khớp tổng tham số) - UNFAIR TO MAMBA ENCODER
        # ------------------------------------------------------------------
        results_budget.append({
            "Config": config_file, "Model": "Mamba1-Hybrid (Target)",
            "Params": mamba_total_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={mamba_d_model} (Target)"
        })
        
        # LSTM
        lstm_dim, lstm_params = find_closest_lstm(mamba_total_params, horizon)
        results_budget.append({
            "Config": config_file, "Model": "LSTM",
            "Params": lstm_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"hidden_dim={lstm_dim}"
        })
        
        # PatchLSTM
        pl_dim, pl_params = find_closest_patch_lstm(mamba_total_params, patch_size, patch_stride, horizon)
        results_budget.append({
            "Config": config_file, "Model": "PatchLSTM",
            "Params": pl_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={pl_dim}"
        })
        
        # ModernTCN
        tcn_dim, tcn_params = find_closest_modern_tcn(mamba_total_params, horizon)
        results_budget.append({
            "Config": config_file, "Model": "ModernTCN",
            "Params": tcn_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={tcn_dim}"
        })
        
        # PatchTST
        pt_dim, pt_params = find_closest_patchtst(mamba_total_params, lookback, 16, 8, horizon)
        results_budget.append({
            "Config": config_file, "Model": "PatchTST",
            "Params": pt_params, "Lookback": lookback, "Horizon": horizon,
            "Details": f"d_model={pt_dim}"
        })

    # Print results
    print("\n" + "=" * 115)
    print(" BẢNG 1: SO SÁNH THỰC TẾ (Cấu hình đang dùng trong train.py - Baselines bị Hardcode)")
    print("=" * 115)
    print(f"{'Config File':<15} | {'Model':<25} | {'Parameters':<15} | {'Lookback':<10} | {'Horizon':<10} | {'Hyperparams'}")
    print("-" * 115)
    for res in results_actual:
        print(f"{res['Config']:<15} | {res['Model']:<25} | {res['Params']:,} | {res['Lookback']:<10} | {res['Horizon']:<10} | {res['Details']}")
    print("-" * 115)

    print("\n" + "=" * 115)
    print(" BẢNG 2: SO SÁNH CÙNG QUY MÔ BIỂU DIỄN (Khớp chiều ẩn d_model/hidden_dim) - [CÔNG BẰNG NHẤT]")
    print("=" * 115)
    print(f"{'Config File':<15} | {'Model':<25} | {'Parameters':<15} | {'Lookback':<10} | {'Horizon':<10} | {'Fair Dimensions'}")
    print("-" * 115)
    for res in results_fair:
        print(f"{res['Config']:<15} | {res['Model']:<25} | {res['Params']:,} | {res['Lookback']:<10} | {res['Horizon']:<10} | {res['Details']}")
    print("-" * 115)

    print("\n" + "=" * 115)
    print(" BẢNG 3: SO SÁNH THEO TỔNG NGÂN SÁCH (Tự động scale Baselines để khớp tổng tham số của Mamba1-Hybrid)")
    print("=" * 115)
    print(f"{'Config File':<15} | {'Model':<25} | {'Parameters':<15} | {'Lookback':<10} | {'Horizon':<10} | {'Scale Dimensions'}")
    print("-" * 115)
    for res in results_budget:
        print(f"{res['Config']:<15} | {res['Model']:<25} | {res['Params']:,} | {res['Lookback']:<10} | {res['Horizon']:<10} | {res['Details']}")
    print("-" * 115)
    print()

if __name__ == "__main__":
    main()
