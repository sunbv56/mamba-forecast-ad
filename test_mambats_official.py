import torch
from src.models.mamba.mamba_ts_official import MambaTSOfficial, MambaTSConfig

def test_mamba_ts_official():
    # Setup config
    config = MambaTSConfig(
        in_channels=2, # x_vibration, y_vibration
        lookback=256,
        forecast_len=64,
        patch_size=64,
        stride=32,
        d_model=64,
        n_layers=2,
        VPT_mode=1 # Enable Variable-Aware Scanning
    )
    
    # Initialize model
    model = MambaTSOfficial(config)
    print("Model initialized successfully.")
    
    # Create dummy input: (Batch, Channels, Length)
    x = torch.randn(4, 2, 256)
    
    # Forward pass
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    assert out.shape == (4, 2, 64), f"Wrong output shape: {out.shape}"
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_mamba_ts_official()
    except ImportError as e:
        print(f"Import Error (Expected if deps not installed): {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
