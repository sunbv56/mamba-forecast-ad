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
    config_path = os.path.join(project_root, "configs/default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    patch_size = config['data'].get('patch_size', 2048)
    stride = config['data'].get('stride', 1024)

    models = {
        "Hybrid-Mamba-V1": HybridMambaCNN({
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
                'in_channels': 2,
                'lookback': lookback,
                'decomp_kernel': config['model'].get('decomp_kernel', 25),
                'use_multiscale': True,
            },
            'data': {'patch_size': patch_size, 'stride': stride, 'lookback': lookback}
        }),
        "MambaTS-Official": MambaTSOfficial(MambaTSConfig(
            in_channels=2,
            lookback=lookback,
            forecast_len=horizon,
            patch_size=64,
            stride=32,
            d_model=config['model'].get('mamba_d_model', 64),
            n_layers=config['model'].get('mamba_n_layer', 4)
        ))
    }

    print("-" * 50)
    print(f"{'Model Name':<25} | {'Parameters':<15}")
    print("-" * 50)
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name:<25} | {params:,}")
    print("-" * 50)

if __name__ == "__main__":
    main()
