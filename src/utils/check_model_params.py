import os
import sys
import torch
import yaml

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("--- Đang khởi tạo mô hình để kiểm tra tham số ---")
from src.models.mamba import MambaTSOfficial, MambaTSConfig, HybridMambaCNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    config_files = ["default.yaml", "small.yaml", "tiny.yaml", "nano.yaml", "expanded.yaml"]
    results = []

    for config_file in config_files:
        config_path = os.path.join(project_root, "configs", config_file)
        if not os.path.exists(config_path):
            print(f"Bỏ qua {config_file}: File không tồn tại.")
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        window_stride = config['data'].get('window_stride', 1024)
        lookback = config['data'].get('lookback', 4096)
        horizon = config['data'].get('horizon', 1024)
        
        # Model patching & trend params
        patch_size = config['model'].get('patch_size', 64)
        patch_stride = config['model'].get('patch_stride', 32)
        trend_downsample = config['model'].get('trend_downsample', 1)

        model_name = config_file.replace(".yaml", "").capitalize()
        
        # 1. Initialize Hybrid-Mamba-V1
        hybrid_model = HybridMambaCNN({
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
                'in_channels': 2,
                'lookback': lookback,
                'decomp_kernel': config['model'].get('decomp_kernel', 25),
                'use_multiscale': True,
            },
            'data': {'patch_size': patch_size, 'stride': patch_stride, 'lookback': lookback}
        })
        
        results.append({
            "Config": config_file,
            "Model": f"Hybrid-Mamba ({model_name})",
            "Params": count_parameters(hybrid_model)
        })

        # 2. Initialize MambaTS-Official
        mambats_model = MambaTSOfficial(MambaTSConfig(
            in_channels=2,
            lookback=lookback,
            forecast_len=horizon,
            patch_size=patch_size,
            stride=patch_stride,
            d_model=config['model'].get('mamba_d_model', 64),
            n_layers=config['model'].get('mamba_n_layer', 4)
        ))

        results.append({
            "Config": config_file,
            "Model": f"MambaTS-Official ({model_name})",
            "Params": count_parameters(mambats_model)
        })

    print("-" * 75)
    print(f"{'Config File':<15} | {'Model Variant':<35} | {'Parameters':<15}")
    print("-" * 75)
    for res in results:
        print(f"{res['Config']:<15} | {res['Model']:<35} | {res['Params']:,}")
    print("-" * 75)

if __name__ == "__main__":
    main()
